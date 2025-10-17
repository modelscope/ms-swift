# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Optional, Union

from swift.llm.train.kto import prepare_kto_dataset
from swift.utils import get_logger
from ..argument import MegatronRLHFArguments
from ..trainers import MegatronDPOTrainer, MegatronKTOTrainer, MegatronRewardTrainer
from .sft import MegatronSft

logger = get_logger()


class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    args: args_class

    def prepare_trainer(self):
        args = self.args
        trainer_mapping = {'dpo': MegatronDPOTrainer, 'kto': MegatronKTOTrainer, 'rm': MegatronRewardTrainer}
        trainer_cls = trainer_mapping.get(args.rlhf_type)
        if trainer_cls is None:
            raise ValueError(f'The current Megatron-SWIFT does not support rlhf_type: {args.rlhf_type}.')
        return trainer_cls(args, self.template)

    def _prepare_template(self) -> None:
        super()._prepare_template()
        if self.args.rlhf_type == 'kto':
            self.template.set_mode('kto')
        else:
            self.template.set_mode('rlhf')

    def _get_dataset(self):
        args = self.args
        train_dataset, val_dataset = super()._get_dataset()
        if args.rlhf_type == 'kto':
            train_dataset, val_dataset = prepare_kto_dataset(args, train_dataset, val_dataset)
        return train_dataset, val_dataset


def megatron_rlhf_main(args: Optional[Union[List[str], MegatronRLHFArguments]] = None):
    return MegatronRLHF(args).main()
