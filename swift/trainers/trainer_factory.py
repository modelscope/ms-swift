# Copyright (c) Alibaba, Inc. and its affiliates.
from .cpo_trainer import CPOTrainer
from .dpo_trainer import DPOTrainer
from .kto_trainer import KTOTrainer
from .orpo_trainer import ORPOTrainer
from .trainers import Seq2SeqTrainer


class TrainerFactory:
    TRAINERS_MAPPING = {
        'sft': Seq2SeqTrainer,
        'dpo': DPOTrainer,
        'simpo': CPOTrainer,
        'orpo': ORPOTrainer,
        'kto': KTOTrainer,
        'cpo': CPOTrainer
    }

    @classmethod
    def get_trainer_info(cls, args):
        trainer_cls = cls.TRAINERS_MAPPING[args.train_type]
        trainer_kwargs = {}
        if args.train_type == 'sft':
            trainer_kwargs['sequence_parallel_size'] = args.sequence_parallel_size
        elif args.train_type == 'dpo':
            trainer_kwargs['rpo_alpha'] = args.rpo_alpha
        return trainer_cls, trainer_kwargs

    @staticmethod
    def patch_template(args, template) -> None:
        from swift.llm import RLHFTemplateMixin
        if args.train_type == 'sft':
            return None
        template_mixin = RLHFTemplateMixin
        template.__class__._old_encode = template.__class__.encode
        template.__class__._old_data_collator = template.__class__.data_collator
        template.__class__.encode = template_mixin.encode
        template.__class__.data_collator = template_mixin.encode
