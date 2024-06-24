# Copyright (c) Alibaba, Inc. and its affiliates.
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List

from openai import APIConnectionError

from swift.utils import get_logger, get_main, seed_everything
from . import DeployArguments
from .utils import EvalArguments

logger = get_logger()
mp.set_start_method('spawn', force=True)


def run_custom_model(args: EvalArguments):
    from swift.llm.deploy import llm_deploy
    port = args.port
    args = args.__dict__
    attrs = dir(DeployArguments)
    for key in list(args.keys()):
        if key not in attrs:
            args.pop(key)
    deploy_args = DeployArguments(**args)
    deploy_args.port = port
    llm_deploy(deploy_args)


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

    @staticmethod
    def prepare_evalscope_dataset():
        from swift.llm.utils.media import MediaCache
        return MediaCache.download(
            'https://www.modelscope.cn/api/v1/datasets/swift/evalscope_resource/'
            'repo?Revision=master&FilePath=eval.zip', 'evalscope')


def get_model_type(port):
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
            if cnt > 60:
                logger.error('Cannot get model_type from the deploy service, please check the error to continue eval')
                raise e
            else:
                time.sleep(1)


def llm_eval(args: EvalArguments) -> List[Dict[str, Any]]:
    from llmuses.run import run_task
    from swift.utils.torch_utils import _find_free_port
    logger.info(f'args: {args}')
    seed_everything(args.seed)
    port = _find_free_port()
    args.port = port
    process = mp.Process(target=run_custom_model, args=(args, ))
    process.start()

    # health check: try to get model_type until raises
    get_model_type(port)
    model_type = 'default-lora' if args.sft_type in ('lora', 'longlora') and not args.merge_lora else args.model_type

    task_cfg = dict(
        eval_backend='OpenCompass',
        eval_config={
            'datasets': args.eval_dataset,
            'models': [
                {
                    'path': model_type,
                    'openai_api_base': f'http://127.0.0.1:{port}/v1/chat/completions'
                },
            ]
        },
    )

    with EvalDatasetContext():
        run_task(task_cfg=task_cfg)

    # task_configs = TaskConfig.load(custom_model=eval_model, tasks=args.eval_dataset + custom_names)
    # for task_config in task_configs:
    #     task_config.use_cache = args.eval_use_cache
    #     if args.eval_limit is not None:
    #         task_config.limit = args.eval_limit
    #     if args.eval_few_shot is not None:
    #         for dataset in task_config.datasets:
    #             if not task_config.dataset_args.get(dataset):
    #                 task_config.dataset_args[dataset] = {}
    #             task_config.dataset_args[dataset]['few_shot_num'] = args.eval_few_shot

    # run_task(task_cfg=task_configs)
    # final_report: List[dict] = Summarizer.get_report_from_cfg(task_cfg=task_configs)
    # logger.info(f'Final report:{final_report}\n')
    #
    # if args.save_result:
    #     result_dir = args.ckpt_dir
    #     if result_dir is None:
    #       result_dir = eval_model.llm_engine.model_dir if args.infer_backend == 'vllm' else eval_model.model.model_dir
    #     assert result_dir is not None
    #     jsonl_path = os.path.join(result_dir, 'eval_result.jsonl')
    #     result = {report['name']: report['score'] for report in final_report}
    #     logger.info(f'result: {result}')
    #     result_info = {
    #         'result': result,
    #         'time': dt.datetime.now().strftime('%Y%m%d-%H%M%S'),
    #     }
    #     append_to_jsonl(jsonl_path, result_info)
    #     logger.info(f'save_result_path: {jsonl_path}')
    # return final_report


eval_main = get_main(EvalArguments, llm_eval)
