# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Union

from swift.utils import get_logger
from ..argument import MegatronRLHFArguments
from .sft import MegatronSft
from .trainers import MegatronDPOTrainer

logger = get_logger()


class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    args: args_class
    trainer_cls = MegatronDPOTrainer

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


def megatron_rlhf_main(args: Union[List[str], MegatronRLHFArguments, None] = None):
    return MegatronRLHF(args).main()
