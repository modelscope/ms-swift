# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import inspect
from contextlib import contextmanager
from typing import Dict

from swift.utils import get_logger

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
    def get_cls(train_type: str, mapping: Dict[str, str]):
        module_path, class_name = mapping[train_type].rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def get_trainer_info(cls, args):
        trainer_cls = cls.get_cls(args.train_type, cls.TRAINER_MAPPING)
        trainer_kwargs = {}
        if args.train_type == 'sft':
            trainer_kwargs['sequence_parallel_size'] = args.sequence_parallel_size
        return trainer_cls, trainer_kwargs

    @classmethod
    def get_training_args_info(cls, args):
        training_args_cls = cls.get_cls(args.train_type, cls.TRAINING_ARGS_MAPPING)
        training_args_kwargs = {}
        if args.train_type == 'sft':
            training_args_kwargs['predict_with_generate'] = args.predict_with_generate
        check_parameters = [
            'beta', 'label_smoothing', 'loss_type', 'rpo_alpha', 'cpo_alpha', 'simpo_gamma', 'desirable_weight',
            'undesirable_weight'
        ]
        parameters = inspect.signature(training_args_cls.__init__).parameters
        for p_name in check_parameters:
            if p_name in parameters:
                training_args_kwargs[p_name] = getattr(args, p_name)
        return training_args_cls, training_args_kwargs

    @staticmethod
    @contextmanager
    def patch_template(args, template):
        from swift.llm import RLHFTemplateMixin, KTOTemplateMixin, PPOTemplateMixin
        if args.train_type == 'sft':
            yield
            return
        _old_compute_per_round_loss = template.compute_per_round_loss
        _old_output_prompt_answer = template.output_prompt_answer
        if args.train_type == 'kto':
            template_mixin = KTOTemplateMixin
            template.output_prompt_answer = True
        elif args.train_type == 'ppo':
            template_mixin = PPOTemplateMixin
        else:
            template_mixin = RLHFTemplateMixin
        if args.train_type != 'orpo' or args.is_multimodal:
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
