


from ..base import SwiftPipeline
from ..argument import EvalArguments

from typing import Union, List, Dict, Any

from evalscope.backend.opencompass import OpenCompassBackendManager
from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager

class SwiftEval(SwiftPipeline[EvalArguments]):
    args_class = EvalArguments

    def run(self):
        args = self.args

def eval_main(args: Union[List[str], EvalArguments, None] = None) -> List[Dict[str, Any]]:
    return SwiftEval(args).main()
