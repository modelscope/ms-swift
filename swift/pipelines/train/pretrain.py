# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Optional, Union

from swift.arguments import PretrainArguments
from swift.utils import get_logger
from .sft import SwiftSft

logger = get_logger()


class SwiftPretrain(SwiftSft):
    args_class = PretrainArguments
    args: args_class

    def _prepare_template(self) -> None:
        self.args.use_chat_template = False
        self.args.loss_scale = 'all'
        logger.info('Setting args.use_chat_template: False')
        logger.info("Setting args.loss_scale: 'all'")
        super()._prepare_template()


def pretrain_main(args: Optional[Union[List[str], PretrainArguments]] = None):
    return SwiftPretrain(args).main()
