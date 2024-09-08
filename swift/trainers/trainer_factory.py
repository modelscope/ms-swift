# Copyright (c) Alibaba, Inc. and its affiliates.

import importlib.util
from typing import Dict


class TrainerFactory:
    TRAINER_MAPPING = {
        'sft': 'swift.trainers.Seq2SeqTrainer',
        'dpo': 'swift.trainers.DPOTrainer',
        'simpo': 'swift.trainers.CPOTrainer',
        'orpo': 'swift.trainers.ORPOTrainer',
        'kto': 'swift.trainers.KTOTrainer',
        'cpo': 'swift.trainers.CPOTrainer'
    }

    TRAINING_ARGS_MAPPING = {
        'sft': 'swift.trainers.Seq2SeqTrainingArguments',
        'dpo': 'swift.trainers.DPOConfig',
        'simpo': 'swift.trainers.CPOConfig',
        'orpo': 'swift.trainers.ORPOConfig',
        'kto': 'swift.trainers.KTOConfig',
        'cpo': 'swift.trainers.CPOConfig'
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
            if args.predict_with_generate:
                trainer_kwargs['compute_metrics'] = partial(compute_nlg_metrics, tokenizer=tokenizer)
            else:
                compute_metrics = partial(
                    compute_acc_metrics, acc_strategy=args.acc_strategy, is_encoder_decoder=is_encoder_decoder)
                trainer_kwargs['compute_metrics'] = compute_metrics
                trainer_kwargs['preprocess_logits_for_metrics'] = preprocess_logits_for_metrics
        elif args.train_type == 'dpo':
            training_args_kwargs['rpo_alpha'] = args.rpo_alpha
        return training_args_cls, training_args_kwargs

    @staticmethod
    def patch_template(args, template) -> None:
        from swift.llm import RLHFTemplateMixin
        if args.train_type == 'sft':
            return None
        template_mixin = RLHFTemplateMixin
        template.__class__._old_encode = template.__class__.encode
        template.__class__._old_data_collator = template.__class__.data_collator
        template.__class__.encode = template_mixin.encode
        template.__class__.data_collator = template_mixin.data_collator
