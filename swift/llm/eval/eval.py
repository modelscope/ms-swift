import inspect
import multiprocessing
import time
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Dict, List, Union

from aiohttp.client_exceptions import ClientConnectorError
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

from swift.utils import get_logger
from ..argument import DeployArguments, EvalArguments
from ..base import SwiftPipeline
from ..dataset import MediaResource
from ..infer import InferClient, deploy_main

logger = get_logger()


class SwiftEval(SwiftPipeline):
    args_class = EvalArguments
    args: args_class

    @contextmanager
    def run_deploy(self):

        args = self.args
        args_dict = asdict(args)
        parameters = inspect.signature(DeployArguments.__init__).parameters
        for k in list(args_dict.keys()):
            if k not in parameters:
                args_dict.pop(k)

        mp = multiprocessing.get_context('spawn')
        process = mp.Process(target=deploy_main, args=(DeployArguments(**args_dict), ))
        process.start()
        try:
            while not self._is_accessible(args.port):
                time.sleep(1)
            yield
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
        eval_report = []
        with self.run_deploy():
            if args.eval_dataset_oc:
                eval_report += self.run_task(args.eval_dataset_oc, 'opencompass')
            if args.eval_dataset_vlm:
                eval_report += self.run_task(args.eval_dataset_vlm, 'vlmeval')

        if args.eval_result_dir is not None:
            logger.info(f'The eval results have been saved to eval_result_dir: `{args.eval_result_dir}`.')
        return eval_report

    def run_task(self, dataset: List[str], eval_backend: str):
        args = self.args
        assert eval_backend in {'opencompass', 'vlmeval'}
        if eval_backend == 'opencompass':
            task_cfg = self.get_opencompass_task_cfg(dataset)
        else:
            task_cfg = self.get_vlmeval_task_cfg(dataset)
        if args.eval_limit:
            task_cfg['limit'] = args.eval_limit

        run_task(task_cfg=task_cfg)
        return Summarizer.get_report_from_cfg(task_cfg=task_cfg)

    def get_opencompass_task_cfg(self, dataset: List[str]):
        args = self.args
        return {
            'eval_backend': 'OpenCompass',
            'eval_config': {
                'datasets': dataset,
                'batch_size': args.max_batch_size,
                'work_dir': args.eval_result_dir,
                'models': [{
                    'path': args.model_name,
                    'openai_api_base': args.url,
                }]
            }
        }

    def get_vlmeval_task_cfg(self, dataset: List[str]):
        args = self.args
        return {
            'eval_backend': 'VLMEvalKit',
            'eval_config': {
                'data': dataset,
                'work_dir': args.eval_result_dir,
                'model': [{
                    'name': 'CustomAPIModel',
                    'api_base': args.url,
                    'type': args.model_name,
                }],
                'nproc': args.max_batch_size,
            }
        }


def eval_main(args: Union[List[str], EvalArguments, None] = None):
    return SwiftEval(args).main()
