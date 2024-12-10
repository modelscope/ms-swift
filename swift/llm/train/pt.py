# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Union

from ..argument import TrainArguments
from .sft import SwiftSft


class SwiftPt(SwiftSft):
    args_class = TrainArguments
    args: args_class

    def _prepare_template(self, use_chat_template: bool) -> None:
        super()._prepare_template(use_chat_template=False)
        self.template.loss_scale = 'all'


def pt_main(args: Union[List[str], TrainArguments, None] = None):
    return SwiftPt(args).main()
