# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import inspect
from contextlib import contextmanager
from typing import Dict

from swift.utils import dataclass_to_dict, get_logger

logger = get_logger()


class TrainerFactory:
    TRAINER_MAPPING = {
        'sft': 'swift.trainers.Seq2SeqTrainer',
        'dpo': 'swift.trainers.DPOTrainer',
        'orpo': 'swift.trainers.ORPOTrainer',
        'kto': 'swift.trainers.KTOTrainer',
        'cpo': 'swift.trainers.CPOTrainer',
        'rm': 'swift.trainers.RewardTrainer',
        'ppo': 'swift.trainers.PPOTrainer',
    }

    TRAINING_ARGS_MAPPING = {
        'sft': 'swift.trainers.Seq2SeqTrainingArguments',
        'dpo': 'swift.trainers.DPOConfig',
        'orpo': 'swift.trainers.ORPOConfig',
        'kto': 'swift.trainers.KTOConfig',
        'cpo': 'swift.trainers.CPOConfig',
        'rm': 'swift.trainers.RewardConfig',
        'ppo': 'swift.trainers.PPOConfig',
    }

    @staticmethod
    def get_cls(train_stage: str, mapping: Dict[str, str]):
        module_path, class_name = mapping[train_stage].rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def get_trainer_cls(cls, args):
        return cls.get_cls(args.train_stage, cls.TRAINER_MAPPING)

    @classmethod
    def get_training_args(cls, args):
        training_args_cls = cls.get_cls(args.train_stage, cls.TRAINING_ARGS_MAPPING)
        parameters = dataclass_to_dict(args)
        train_args_parameters = inspect.signature(training_args_cls.__init__).parameters

        training_args_kwargs = {}
        for k, v in parameters.items():
            if k in train_args_parameters:
                training_args_kwargs[k] = v

        return training_args_cls(**training_args_kwargs)

    @staticmethod
    @contextmanager
    def patch_template(args, template):
        from swift.llm import RLHFTemplateMixin, KTOTemplateMixin, PPOTemplateMixin
        if args.train_stage == 'sft':
            yield
            return
        _old_compute_per_round_loss = template.compute_per_round_loss
        _old_output_prompt_answer = template.output_prompt_answer
        if args.train_stage == 'kto':
            template_mixin = KTOTemplateMixin
            template.output_prompt_answer = True
        elif args.train_stage == 'ppo':
            template_mixin = PPOTemplateMixin
        else:
            template_mixin = RLHFTemplateMixin
        if args.train_stage != 'orpo' or args.is_multimodal:
            template.compute_per_round_loss = False
        logger.info(f'template.compute_per_round_loss: {template.compute_per_round_loss}')
        logger.info(f'template.output_prompt_answer: {template.output_prompt_answer}')
        template.__class__._old_encode = template.__class__.encode
        template.__class__._old_data_collator = template.__class__.data_collator
        template.__class__.encode = template_mixin.encode
        template.__class__.data_collator = template_mixin.data_collator
        yield
        template.compute_per_round_loss = _old_compute_per_round_loss
        template.output_prompt_answer = _old_output_prompt_answer
        template.__class__.encode = template.__class__._old_encode
        template.__class__.data_collator = template.__class__._old_data_collator
        del template.__class__._old_encode, template.__class__._old_data_collator
