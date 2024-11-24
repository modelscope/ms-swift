# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import datetime as dt
import multiprocessing
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import json
from evalscope.backend.opencompass import OpenCompassBackendManager
from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
from evalscope.config import TaskConfig
from evalscope.constants import DEFAULT_ROOT_CACHE_DIR
from evalscope.models.custom import CustomModel
from evalscope.run import run_task
from evalscope.summarizer import Summarizer
from evalscope.utils import EvalBackend
from openai import APIConnectionError
from tqdm import tqdm
from transformers import GenerationConfig

from swift.utils import append_to_jsonl, get_logger, get_main, seed_everything
from .infer import merge_lora, prepare_model_template
from .utils import DeployArguments, EvalArguments, XRequestConfig, inference, inference_client_async

logger = get_logger()


def run_custom_model(args: EvalArguments):
    from swift.llm import deploy_main

    port = args.port
    args = args.__dict__
    attrs = dir(DeployArguments)
    for key in list(args.keys()):
        if key not in attrs:
            args.pop(key)
    args['verbose'] = False
    deploy_args = DeployArguments(**args)
    deploy_args.port = port
    deploy_main(deploy_args)


class EvalDatasetContext:

    def __init__(self):
        self.cache_dir = self.prepare_evalscope_dataset()

    def __enter__(self):
        data_dir = os.path.join(self.cache_dir, 'data')
        local_dir = os.path.join(os.getcwd(), 'data')
        if os.path.exists(local_dir) and not os.path.islink(local_dir):
            raise AssertionError('Please promise your pwd dir does not contain a `data` dir.')
        if os.path.islink(local_dir):
            os.remove(os.path.join(local_dir))
        os.symlink(data_dir, local_dir)

    def __exit__(self, *args, **kwargs):
        pass

    @staticmethod
    def prepare_evalscope_dataset():
        from swift.llm.utils.media import MediaCache

        return MediaCache.download(
            'https://www.modelscope.cn/api/v1/datasets/swift/evalscope_resource/'
            'repo?Revision=master&FilePath=eval.zip',
            'evalscope',
        )


def get_model_type(port, timeout):
    cnt = 0
    while True:
        from openai import OpenAI

        client = OpenAI(
            api_key='EMPTY',
            base_url=f'http://localhost:{port}/v1',
        )
        try:
            return client.models.list().data
        except APIConnectionError as e:
            cnt += 1
            if cnt > timeout:
                logger.error('Cannot get model_type from the deploy service, please check the error to continue eval')
                raise e
            else:
                time.sleep(1)


def opencompass_runner(args: EvalArguments, dataset: List[str], model_type: str, is_chat: bool, url: str):
    eval_limit = args.eval_limit
    if eval_limit is not None and '[' not in eval_limit:
        eval_limit = int(eval_limit)
    limit_config = {'limit': eval_limit} if eval_limit else {}
    task_cfg = dict(
        eval_backend='OpenCompass',
        eval_config={
            'datasets':
            dataset,
            'reuse':
            'latest' if args.eval_use_cache else None,
            'batch_size':
            args.eval_batch_size,
            'work_dir':
            args.eval_output_dir,
            'models': [
                {
                    'path': model_type,
                    'openai_api_base': url,
                    'is_chat': is_chat,
                    'key': args.eval_token,
                    'temperature': args.temperature,
                },
            ],
            **limit_config,
        },
    )
    with EvalDatasetContext():
        run_task(task_cfg=task_cfg)

    return Summarizer.get_report_from_cfg(task_cfg=task_cfg)


def vlmeval_runner(args: EvalArguments, dataset: List[str], model_type: str, is_chat: bool, url: str):
    eval_limit = args.eval_limit
    if eval_limit is not None and '[' not in eval_limit:
        eval_limit = int(eval_limit)
    limit_config = {'limit': eval_limit} if eval_limit else {}
    if args.eval_batch_size or args.eval_use_cache:
        logger.warn('VLMEval does not support `eval_batch_size` or `eval_use_cache`')
    task_cfg = dict(
        eval_backend='VLMEvalKit',
        eval_config={
            'data':
            dataset,
            'work_dir':
            args.eval_output_dir,
            'model': [
                {
                    'name': 'CustomAPIModel',
                    'api_base': url,
                    'key': args.eval_token,
                    'type': model_type,
                    'temperature': args.temperature,
                },
            ],
            **limit_config,
            'nproc':
            args.eval_nproc,
        },
    )
    run_task(task_cfg=task_cfg)
    return Summarizer.get_report_from_cfg(task_cfg=task_cfg)


@contextmanager
def deploy_context(args):
    from swift.utils import find_free_port

    process = None
    try:
        if not args.eval_url:
            port = _find_free_port()
            args.port = port
            mp = multiprocessing.get_context('spawn')
            process = mp.Process(target=run_custom_model, args=(args, ))
            process.start()
        yield
    finally:
        if process is not None:
            process.kill()
            process.join()
            logger.info('The deployment process has been terminated.')


def eval_opencompass(args: EvalArguments) -> List[Dict[str, Any]]:
    logger.info(f'args: {args}')
    with deploy_context(args):
        if not args.eval_url:
            port = args.port
            # health check: try to get model_type until raises
            get_model_type(port, args.deploy_timeout)
            model_type = ('default-lora'
                          if args.sft_type in ('lora', 'longlora') and not args.merge_lora else args.model_type)
            from .deploy import is_generation_template

            if is_generation_template(args.template_type):
                url = f'http://127.0.0.1:{port}/v1/completions'
            else:
                url = f'http://127.0.0.1:{port}/v1/chat/completions'
            is_chat = not is_generation_template(args.template_type)
        else:
            url = args.eval_url
            url = url.rstrip('/')
            if args.eval_is_chat_model:
                url += '/chat/completions'
            else:
                url += '/completions'
            model_type = args.model_type
            is_chat = args.eval_is_chat_model

        nlp_datasets = set(OpenCompassBackendManager.list_datasets()) & set(args.eval_dataset)
        mm_datasets = set(VLMEvalKitBackendManager.list_supported_datasets()) & set(args.eval_dataset)

        final_report = []
        for dataset, runner in zip(
            [list(nlp_datasets), list(mm_datasets)],
            [opencompass_runner, vlmeval_runner],
        ):
            if not dataset:
                continue

            report = runner(args, dataset, model_type, is_chat, url)
            logger.info(f'Final report:{report}\n')
            final_report.extend(report)
    if not final_report:
        raise ValueError(f'Cannot load final report, please check your dataset: {args.eval_dataset} and the eval log')
    return final_report


def llm_eval(args: EvalArguments) -> List[Dict[str, Any]]:
    logger.info(f'args: {args}')
    seed_everything(args.seed)
    args.eval_output_dir = os.path.join(args.eval_output_dir, args.name or 'default')
    if args.custom_eval_config:
        args.eval_backend = EvalBackend.NATIVE.value
        if args.eval_dataset:
            logger.warn('--custom_eval_config cannot use together with --eval_dataset')
            args.eval_dataset = []
    return eval_opencompass(args)


eval_main = get_main(EvalArguments, llm_eval)
