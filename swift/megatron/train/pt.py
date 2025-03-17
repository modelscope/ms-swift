# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Union

from ..argument import MegatronTrainArguments
from .sft import MegatronSft


class MegatronPt(MegatronSft):
    args_class = MegatronTrainArguments
    args: args_class

    def _prepare_template(self) -> None:
        self.args.use_chat_template = False
        super()._prepare_template()
        self.template.loss_scale = 'all'


def megatron_pt_main(args: Union[List[str], MegatronTrainArguments, None] = None):
    return MegatronPt(args).main()
