# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Optional, Union

from swift.trainers.rlhf_trainer.utils import identity_data_collator
from swift.utils import get_logger
from ..argument import MegatronRLHFArguments
from ..trainers import MegatronDPOTrainer, MegatronGRPOTrainer
from .sft import MegatronSft

logger = get_logger()


class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    args: args_class

    def prepare_trainer(self):
        args = self.args
        if args.rlhf_type == 'dpo':
            trainer_cls = MegatronDPOTrainer
        elif args.rlhf_type == 'grpo':
            trainer_cls = MegatronGRPOTrainer
        else:
            raise ValueError(f'The current Megatron-SWIFT does not support rlhf_type: {args.rlhf_type}.')
        return trainer_cls(args, self.template)

    def _prepare_template(self) -> None:
        super()._prepare_template()
        model_mapping = {'grpo': 'train'}
        self.template.set_mode(model_mapping.get(self.args.rlhf_type, 'rlhf'))

    def _get_data_collator(self):
        if self.args.rlhf_type == 'grpo':
            super()._get_data_collator()
            return identity_data_collator
        return super()._get_data_collator()


def megatron_rlhf_main(args: Optional[Union[List[str], MegatronRLHFArguments]] = None):
    return MegatronRLHF(args).main()
