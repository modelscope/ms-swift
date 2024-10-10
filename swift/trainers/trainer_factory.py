# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import inspect
from contextlib import contextmanager
from types import MethodType
from typing import Dict

from swift.plugin.custom_trainer import custom_trainer_class
from swift.utils import get_logger

logger = get_logger()


class TrainerFactory:
    TRAINER_MAPPING = {
        'sft': 'swift.trainers.Seq2SeqTrainer',
        'dpo': 'swift.trainers.DPOTrainer',
        'orpo': 'swift.trainers.ORPOTrainer',
        'kto': 'swift.trainers.KTOTrainer',
        'cpo': 'swift.trainers.CPOTrainer'
    }

    TRAINING_ARGS_MAPPING = {
        'sft': 'swift.trainers.Seq2SeqTrainingArguments',
        'dpo': 'swift.trainers.DPOConfig',
        'orpo': 'swift.trainers.ORPOConfig',
        'kto': 'swift.trainers.KTOConfig',
        'cpo': 'swift.trainers.CPOConfig'
    }

    custom_trainer_class(TRAINER_MAPPING, TRAINING_ARGS_MAPPING)

    @staticmethod
    def _get_cls(train_stage: str, mapping: Dict[str, str]):
        module_path, class_name = mapping[train_stage].rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @staticmethod
    def get_trainer(train_stage: str, args):
        trainer_cls = TrainerFactory._get_cls(train_stage, TrainerFactory.TRAINER_MAPPING)
        trainer_kwargs = {}
        if train_stage == 'sft':
            trainer_kwargs['sequence_parallel_size'] = args.sequence_parallel_size
        return trainer_cls, trainer_kwargs

    @staticmethod
    def get_training_args(train_stage: str, args):
        training_args_cls = TrainerFactory._get_cls(train_stage, TrainerFactory.TRAINING_ARGS_MAPPING)
        training_args_kwargs = {}
        if train_stage == 'sft':
            training_args_kwargs['predict_with_generate'] = args.predict_with_generate
        check_parameters = ['beta', 'label_smoothing', 'loss_type', 'rpo_alpha', 'cpo_alpha', 'simpo_gamma']
        parameters = inspect.signature(training_args_cls.__init__).parameters
        for p_name in check_parameters:
            if p_name in parameters:
                training_args_kwargs[p_name] = getattr(args, p_name)
        return training_args_cls, training_args_kwargs

    @staticmethod
    @contextmanager
    def patch_template(train_stage, args, template):
        if train_stage == 'sft':
            yield
            return
        _old_loss_scale = template.loss_scale
        _old_output_prompt_answer = template.output_prompt_answer
        if train_stage == 'kto':
            from swift.llm.template.template import KTOTemplateMixin
            template_mixin = KTOTemplateMixin
            template.output_prompt_answer = True
        else:
            from swift.llm.template.template import RLHFTemplateMixin
            template_mixin = RLHFTemplateMixin
        if args.train_type != 'orpo' or args.is_multimodal:
            template.loss_scale = 'last_round'
        logger.info(f'template.loss_scale: {template.loss_scale}')
        logger.info(f'template.output_prompt_answer: {template.output_prompt_answer}')
        template._old_encode = template.encode
        template._old_data_collator = template.data_collator
        template.encode = MethodType(template_mixin.encode, template)
        template.data_collator = MethodType(template_mixin.data_collator, template)
        yield
        template.loss_scale = _old_loss_scale
        template.output_prompt_answer = _old_output_prompt_answer
        template.encode = template._old_encode
        template.data_collator = template._old_data_collator
        del template._old_encode, template._old_data_collator
