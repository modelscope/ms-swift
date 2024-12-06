# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import inspect
from dataclasses import asdict
from typing import Dict

from swift.plugin import custom_trainer_class
from swift.utils import get_logger

logger = get_logger()


class TrainerFactory:
    TRAINER_MAPPING = {
        'train': 'swift.trainers.Seq2SeqTrainer',
        'dpo': 'swift.trainers.DPOTrainer',
        'orpo': 'swift.trainers.ORPOTrainer',
        'kto': 'swift.trainers.KTOTrainer',
        'cpo': 'swift.trainers.CPOTrainer',
        'rm': 'swift.trainers.RewardTrainer',
        'ppo': 'swift.trainers.PPOTrainer',
    }

    TRAINING_ARGS_MAPPING = {
        'train': 'swift.trainers.Seq2SeqTrainingArguments',
        'dpo': 'swift.trainers.DPOConfig',
        'orpo': 'swift.trainers.ORPOConfig',
        'kto': 'swift.trainers.KTOConfig',
        'cpo': 'swift.trainers.CPOConfig',
        'rm': 'swift.trainers.RewardConfig',
        'ppo': 'swift.trainers.PPOConfig',
    }

    custom_trainer_class(TRAINER_MAPPING, TRAINING_ARGS_MAPPING)

    @staticmethod
    def get_cls(args, mapping: Dict[str, str]):
        if hasattr(args, 'rlhf_type'):
            train_method = args.rlhf_type
        else:
            train_method = 'train'
        module_path, class_name = mapping[train_method].rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def get_trainer_cls(cls, args):
        return cls.get_cls(args, cls.TRAINER_MAPPING)

    @classmethod
    def get_training_args(cls, args):
        training_args_cls = cls.get_cls(args, cls.TRAINING_ARGS_MAPPING)
        args_dict = asdict(args)
        parameters = inspect.signature(training_args_cls.__init__).parameters

        for k in list(args_dict.keys()):
            if k not in parameters:
                args_dict.pop(k)

        return training_args_cls(**args_dict)
