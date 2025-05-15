# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Union

from ..argument import MegatronTrainArguments
from .sft import MegatronSft


class MegatronPt(MegatronSft):
    args_class = MegatronTrainArguments
    args: args_class

    def _prepare_template(self) -> None:
        self.args.use_chat_template = False
        self.args.loss_scale = 'all'
        super()._prepare_template()


def megatron_pt_main(args: Union[List[str], MegatronTrainArguments, None] = None):
    return MegatronPt(args).main()
