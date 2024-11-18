from typing import Any, Dict, List, Union


from ..argument import RLHFArguments
from .sft import SwiftSft


class SwiftRLHF(SwiftSft[RLHFArguments]):
    args_class = RLHFArguments

    def run(self):
        args = self.args


def rlhf_main(args: Union[List[str], RLHFArguments, None] = None) -> List[Dict[str, Any]]:
    return SwiftRLHF(args).main()
