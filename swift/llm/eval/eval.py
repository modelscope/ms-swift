# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from contextlib import nullcontext
from typing import List, Optional, Union

from evalscope.constants import EvalBackend, EvalType
from evalscope.run import TaskConfig, run_task
from evalscope.summarizer import Summarizer

from swift.utils import append_to_jsonl, get_logger
from .. import MediaResource
from ..argument import EvalArguments
from ..base import SwiftPipeline
from ..infer import run_deploy

logger = get_logger()


class SwiftEval(SwiftPipeline):
    args_class = EvalArguments
    args: args_class

    def run(self):
        args = self.args
        eval_report = {}
        deploy_context = nullcontext() if args.eval_url else run_deploy(args, return_url=True)
        with deploy_context as base_url:
            base_url = args.eval_url or base_url

            task_cfg = self.get_task_cfg(args.eval_dataset, args.eval_backend, base_url)
            result = self.get_task_result(task_cfg)
            eval_report[args.eval_backend] = result

        eval_report.update({
            'time': args.time,
            'model': args.model,
            'adapters': args.adapters,
            'result_path': args.result_path,
            'eval_output_dir': args.eval_output_dir,
            'eval_limit': args.eval_limit
        })

        if args.result_jsonl:
            append_to_jsonl(args.result_jsonl, eval_report)
            logger.info(f'The eval result have been saved to result_jsonl: `{args.result_jsonl}`.')
        return eval_report

    def get_task_result(self, task_cfg: TaskConfig):
        run_task(task_cfg=task_cfg)
        reports = Summarizer.get_report_from_cfg(task_cfg=task_cfg)
        result = {}
        if task_cfg.eval_backend == EvalBackend.OPEN_COMPASS:
            for report in reports:
                if report[self.args.model_suffix] != '-':
                    result[report['dataset']] = {report['metric']: report[self.args.model_suffix]}
        elif task_cfg.eval_backend == EvalBackend.VLM_EVAL_KIT:
            for report in reports:
                splited_key = next(iter(report)).rsplit('_', 2)
                if len(splited_key) == 3:
                    _, dataset, metric = splited_key
                else:
                    dataset, metric = '-', '-'
                result[dataset] = {metric: list(report.values())[0]}
        else:
            result = reports
        return result

    def get_task_cfg(self, dataset: List[str], eval_backend: str, url: str):
        assert eval_backend in {EvalBackend.NATIVE, EvalBackend.OPEN_COMPASS, EvalBackend.VLM_EVAL_KIT}
        if eval_backend == EvalBackend.OPEN_COMPASS:
            if self.args.local_dataset:
                if os.path.exists('data'):
                    if not os.path.exists(os.path.join('data', 'CMB')):
                        raise RuntimeError('Opencompass need a `data` folder in your work dir('
                                           'which will be created automatically by swift eval), '
                                           'but a local path named `data` already exists, '
                                           'please consider moving the dir to another location.')
                else:
                    local_dir = MediaResource.download(
                        'https://modelscope.cn/datasets/'
                        'opencompass/OpenCompassDataComplete/'
                        'resolve/master/OpenCompassData-complete-20240207.zip', 'OpenCompassData')
                    os.symlink(os.path.join(local_dir, 'data'), 'data')

            task_cfg = self.get_opencompass_task_cfg(dataset, url)
        elif eval_backend == EvalBackend.VLM_EVAL_KIT:
            task_cfg = self.get_vlmeval_task_cfg(dataset, url)
        else:
            task_cfg = self.get_native_task_cfg(dataset, url)
        return task_cfg

    def get_native_task_cfg(self, dataset: List[str], url: str):
        args = self.args
        work_dir = os.path.join(args.eval_output_dir, 'native')
        return TaskConfig(
            model=args.model_suffix,
            eval_type=EvalType.SERVICE,
            api_url=url,
            api_key=args.api_key or 'EMPTY',
            datasets=dataset,
            work_dir=work_dir,
            limit=args.eval_limit,
            eval_batch_size=args.eval_num_proc,
            dataset_args=args.eval_dataset_args,
            generation_config=args.eval_generation_config,
            **args.extra_eval_args)

    def get_opencompass_task_cfg(self, dataset: List[str], url: str):
        # Must use chat/completion endpoint
        url = f"{url.rstrip('/')}/chat/completions"

        args = self.args
        work_dir = os.path.join(args.eval_output_dir, 'opencompass')
        return TaskConfig(
            eval_backend=EvalBackend.OPEN_COMPASS,
            eval_config={
                'datasets':
                dataset,
                'batch_size':
                args.eval_num_proc,
                'work_dir':
                work_dir,
                'models': [{
                    'path': args.model_suffix,
                    'openai_api_base': url,
                    'key': args.api_key or 'EMPTY',
                    'is_chat': args.use_chat_template
                }],
                'limit':
                args.eval_limit
            },
            work_dir=work_dir)

    def get_vlmeval_task_cfg(self, dataset: List[str], url: str):
        # Must use chat/completion endpoint
        url = f"{url.rstrip('/')}/chat/completions"

        args = self.args
        work_dir = os.path.join(args.eval_output_dir, 'vlmeval')
        return TaskConfig(
            eval_backend=EvalBackend.VLM_EVAL_KIT,
            eval_config={
                'data':
                dataset,
                'model': [{
                    'type': args.model_suffix,
                    'name': 'CustomAPIModel',
                    'api_base': url,
                    'key': args.api_key or 'EMPTY',
                    **args.eval_generation_config
                }],
                'nproc':
                args.eval_num_proc,
                'limit':
                args.eval_limit
            },
            work_dir=work_dir)


def eval_main(args: Optional[Union[List[str], EvalArguments]] = None):
    return SwiftEval(args).main()
