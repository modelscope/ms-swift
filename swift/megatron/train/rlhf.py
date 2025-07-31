# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Optional, Union

from swift.utils import get_logger
from ..argument import MegatronRLHFArguments
from ..trainers import MegatronDPOTrainer
from .sft import MegatronSft

logger = get_logger()


class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    args: args_class

    def prepare_trainer(self):
        args = self.args
        if args.rlhf_type == 'dpo':
            trainer_cls = MegatronDPOTrainer
        else:
            raise ValueError(f'The current Megatron-SWIFT does not support rlhf_type: {args.rlhf_type}.')
        return trainer_cls(args)

    def _prepare_template(self) -> None:
        super()._prepare_template()
        self.template.set_mode('rlhf')


def megatron_rlhf_main(args: Optional[Union[List[str], MegatronRLHFArguments]] = None):
    return MegatronRLHF(args).main()
