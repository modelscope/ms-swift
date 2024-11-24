import multiprocessing
from contextlib import contextmanager
from typing import Any, Dict, List, Union

from evalscope.backend.opencompass import OpenCompassBackendManager
from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager

from ..argument import EvalArguments
from ..base import SwiftPipeline


class SwiftEval(SwiftPipeline):
    args_class = EvalArguments
    args: args_class

    def run(self):
        args = self.args
        with self.run_deploy():
            for oc_dataset in self.eval_dataset_oc:
                self.run_opencompass(oc_dataset)
            for vlm_dataset in self.eval_dataset_vlm:
                self.run_vlmeval(oc_dataset)

    @contextmanager
    def run_deploy(self):
        from swift.llm import deploy_main
        process = multiprocessing.Process(target=deploy_main, args=(self.args, ))
        process.start()
        try:
            yield
        finally:
            process.kill()
            logger.info('The deployment process has been terminated.')

    def run_opencompass(self):
        pass

    def run_vlmeval(self):
        pass


def eval_main(args: Union[List[str], EvalArguments, None] = None):
    return SwiftEval(args).main()
