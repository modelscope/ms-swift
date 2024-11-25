import multiprocessing
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Union

from aiohttp.client_exceptions import ClientConnectorError
from evalscope.backend.opencompass import OpenCompassBackendManager
from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager

from swift.utils import get_logger
from ..argument import EvalArguments
from ..base import SwiftPipeline
from ..infer import InferClient

logger = get_logger()


class SwiftEval(SwiftPipeline):
    args_class = EvalArguments
    args: args_class

    def run(self):
        args = self.args
        with self.run_deploy():
            for oc_dataset in args.eval_dataset_oc:
                self.run_opencompass(oc_dataset)
            for vlm_dataset in args.eval_dataset_vlm:
                self.run_vlmeval(oc_dataset)

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

    def run_opencompass(self, dataset: str):
        pass

    def run_vlmeval(self, dataset: str):
        pass


def eval_main(args: Union[List[str], EvalArguments, None] = None):
    return SwiftEval(args).main()
