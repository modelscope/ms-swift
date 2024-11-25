import multiprocessing
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Union

from aiohttp.client_exceptions import ClientConnectorError
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

from swift.utils import get_logger
from ..argument import EvalArguments
from ..base import SwiftPipeline
from ..dataset import MediaResource
from ..infer import InferClient

logger = get_logger()


class SwiftEval(SwiftPipeline):
    args_class = EvalArguments
    args: args_class

    def __init__(self, args: Union[List[str], args_class, None] = None):
        super().__init__(args)
        self._download_eval_dataset()
        self.url = f'http://127.0.0.1:{args.port}/v1/chat/completions'

    def run(self):
        args = self.args
        eval_report = []
        with self.run_deploy():
            for oc_dataset in args.eval_dataset_oc:
                eval_report.append(self.run_task(oc_dataset, 'opencompass'))
            for vlm_dataset in args.eval_dataset_vlm:
                eval_report.append(self.run_task(vlm_dataset, 'vlmeval'))
        return eval_report

    @contextmanager
    def run_deploy(self):
        from swift.llm import deploy_main
        args = self.args
        mp = multiprocessing.get_context('spawn')
        process = mp.Process(target=deploy_main, args=(args, ))
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

    def run_task(self, dataset: str, eval_backend):
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

    def get_opencompass_task_cfg(self, dataset: str):
        args = self.args
        return {
            'eval_backend': 'OpenCompass',
            'eval_config': {
                'datasets': dataset,
                'batch_size': args.max_batch_size,
                'work_dir': args.eval_result_dir,
                'models': [{
                    'url': self.url,
                }]
            }
        }

    def get_vlmeval_task_cfg(self, dataset: str):
        args = self.args
        return {
            'eval_backend': 'VLMEvalKit',
            'eval_config': {
                'data': dataset,
                'work_dir': args.eval_result_dir,
                'models': [{
                    'api_base': self.url,
                }],
                'nproc': args.max_batch_size,
            }
        }

    def _download_eval_dataset(self):
        self.cache_dir = MediaResource.download(
            'https://www.modelscope.cn/api/v1/datasets/swift/evalscope_resource/'
            'repo?Revision=master&FilePath=eval.zip',
            'evalscope',
        )


def eval_main(args: Union[List[str], EvalArguments, None] = None):
    return SwiftEval(args).main()
