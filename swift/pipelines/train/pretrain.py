# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Optional, Union

from swift.arguments import PretrainArguments
from swift.utils import get_logger
from .sft import SwiftSft

logger = get_logger()


class SwiftPretrain(SwiftSft):
    args_class = PretrainArguments
    args: args_class


def pretrain_main(args: Optional[Union[List[str], PretrainArguments]] = None):
    return SwiftPretrain(args).main()
