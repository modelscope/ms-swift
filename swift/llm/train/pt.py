# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Union

from swift.utils import get_logger
from ..argument import TrainArguments
from .sft import SwiftSft

logger = get_logger()


class SwiftPt(SwiftSft):
    args_class = TrainArguments
    args: args_class

    def _prepare_template(self) -> None:
        self.args.use_chat_template = False
        self.args.loss_scale = 'all'
        logger.info('Setting args.use_chat_template: False')
        logger.info("Setting args.loss_scale: 'all'")
        super()._prepare_template()


def pt_main(args: Union[List[str], TrainArguments, None] = None):
    return SwiftPt(args).main()
