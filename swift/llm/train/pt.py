from typing import Any, Dict, List, Union

from ..argument import TrainArguments
from .sft import SwiftSft


class SwiftPt(SwiftSft):
    args_class = TrainArguments
    args: args_class


def pt_main(args: Union[List[str], TrainArguments, None] = None):
    return SwiftPt(args).main()
