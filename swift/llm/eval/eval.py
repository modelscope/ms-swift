# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
from contextlib import nullcontext
from typing import List, Union

from evalscope.run import run_task
from evalscope.summarizer import Summarizer

from swift.utils import append_to_jsonl, get_logger
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
        deploy_context = nullcontext() if args.eval_url else run_deploy(self.args, return_url=True)
        with deploy_context as url:
            url = args.eval_url or url
            if args.eval_dataset_oc:
                reports = self.run_task(args.eval_dataset_oc, 'opencompass', url)
                result = {}
                for report in reports:
                    if report[args.model_suffix] != '-':
                        result[report['dataset']] = {report['metric']: report[args.model_suffix]}
                eval_report['opencompass'] = result
            if args.eval_dataset_vlm:
                reports = self.run_task(args.eval_dataset_vlm, 'vlmeval', url)
                result = {}
                for dataset, report in zip(args.eval_dataset_vlm, reports):
                    metric = next(iter(report)).rsplit('_')[-1]
                    result[dataset] = {metric: list(report.values())[0]}
                eval_report['vlmeval'] = result
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

    def run_task(self, dataset: List[str], eval_backend: str, url: str):
        args = self.args
        assert eval_backend in {'opencompass', 'vlmeval'}
        if eval_backend == 'opencompass':
            task_cfg = self.get_opencompass_task_cfg(dataset, url)
        else:
            task_cfg = self.get_vlmeval_task_cfg(dataset, url)
        if args.eval_limit:
            task_cfg['eval_config']['limit'] = args.eval_limit

        run_task(task_cfg=task_cfg)
        return Summarizer.get_report_from_cfg(task_cfg=task_cfg)

    def get_opencompass_task_cfg(self, dataset: List[str], url: str):
        args = self.args
        return {
            'eval_backend': 'OpenCompass',
            'eval_config': {
                'datasets':
                dataset,
                'batch_size':
                args.max_batch_size or 256,
                'work_dir':
                os.path.join(args.eval_output_dir, 'opencompass'),
                'models': [{
                    'path': args.model_suffix,
                    'openai_api_base': url,
                    'key': args.api_key or 'EMPTY',
                    'is_chat': args.use_chat_template
                }]
            }
        }

    def get_vlmeval_task_cfg(self, dataset: List[str], url: str):
        args = self.args
        task_cfg = {
            'eval_backend': 'VLMEvalKit',
            'eval_config': {
                'data':
                dataset,
                'work_dir':
                os.path.join(args.eval_output_dir, 'vlmeval',
                             dt.datetime.now().strftime('%Y%m%d-%H%M%S')),
                'model': [{
                    'type': args.model_suffix,
                    'name': 'CustomAPIModel',
                    'api_base': url,
                    'key': args.api_key or 'EMPTY',
                }],
                'nproc':
                args.max_batch_size or 16,
            }
        }
        task_cfg['work_dir'] = task_cfg['eval_config']['work_dir']  # compat evalscope 0.8.1
        return task_cfg


def eval_main(args: Union[List[str], EvalArguments, None] = None):
    return SwiftEval(args).main()
