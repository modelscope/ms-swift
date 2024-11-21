from typing import Any, Dict, List, Union

from ..argument import TrainArguments
from .sft import SwiftSft


class SwiftPt(SwiftSft):
    args_class = TrainArguments
    args: args_class

    def _prepare_train(self):
        self.template.set_mode('train')
        self.template.loss_scale = 'all'

    def _prepare_template(self, **template_kwargs) -> None:
        super()._prepare_template(use_chat_template=False)


def pt_main(args: Union[List[str], TrainArguments, None] = None):
    return SwiftPt(args).main()
