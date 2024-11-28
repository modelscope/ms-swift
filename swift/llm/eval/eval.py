import datetime as dt
import inspect
import multiprocessing
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from typing import Any, Dict, List, Union

from aiohttp.client_exceptions import ClientConnectorError
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

from swift.utils import append_to_jsonl, get_logger
from ..argument import DeployArguments, EvalArguments
from ..base import SwiftPipeline
from ..dataset import MediaResource
from ..infer import InferClient, deploy_main

logger = get_logger()


class SwiftEval(SwiftPipeline):
    args_class = EvalArguments
    args: args_class

    @staticmethod
    @contextmanager
    def run_deploy(args):
        if isinstance(args, DeployArguments) and args.__class__.__name__ == 'DeployArguments':
            deploy_args = args
        else:
            args_dict = asdict(args)
            parameters = inspect.signature(DeployArguments.__init__).parameters
            for k in list(args_dict.keys()):
                if k not in parameters:
                    args_dict.pop(k)
            deploy_args = DeployArguments(**args_dict)

        mp = multiprocessing.get_context('spawn')
        process = mp.Process(target=deploy_main, args=(deploy_args, ))
        process.start()
        try:
            while not SwiftEval._is_accessible(deploy_args.port):
                time.sleep(1)
            yield f'http://127.0.0.1:{deploy_args.port}/v1/chat/completions'
        finally:
            process.kill()
            logger.info('The deployment process has been terminated.')

    @staticmethod
    def _is_accessible(port: int):
        infer_client = InferClient(port=port)
        try:
            infer_client.get_model_list()
        except ClientConnectorError:
            return False
        return True

    def run(self):
        args = self.args
        eval_report = {
            'time': args.time,
            'model': args.ckpt_dir or args.model,
            'result_path': args.result_path,
            'eval_output_dir': args.eval_output_dir,
            'eval_limit': args.eval_limit
        }
        deploy_context = nullcontext() if args.eval_url else self.run_deploy(self.args)
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
                    result[dataset] = {metric: list(report.values())[0]['Overall']}
                eval_report['vlmeval'] = result

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
            task_cfg['limit'] = args.eval_limit

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
                args.max_batch_size,
                'work_dir':
                os.path.join(args.eval_output_dir, 'opencompass'),
                'models': [{
                    'path': args.model_suffix,
                    'openai_api_base': url,
                    'key': args.api_key,
                    'is_chat': args.use_chat_template
                }]
            }
        }

    def get_vlmeval_task_cfg(self, dataset: List[str], url: str):
        args = self.args
        time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        return {
            'eval_backend': 'VLMEvalKit',
            'eval_config': {
                'data': dataset,
                'work_dir': os.path.join(args.eval_output_dir, 'vlmeval', time),
                'model': [{
                    'type': args.model_suffix,
                    'name': 'CustomAPIModel',
                    'api_base': url,
                    'key': args.api_key,
                }],
                'nproc': args.max_batch_size,
            }
        }


def eval_main(args: Union[List[str], EvalArguments, None] = None):
    return SwiftEval(args).main()
