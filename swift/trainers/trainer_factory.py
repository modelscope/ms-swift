# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import inspect
from contextlib import contextmanager
from typing import Dict

from swift.utils import dataclass_to_dict, get_logger

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
        parameters = dataclass_to_dict(args)
        train_args_parameters = inspect.signature(training_args_cls.__init__).parameters

        training_args_kwargs = {}
        for k, v in parameters.items():
            if k in train_args_parameters:
                training_args_kwargs[k] = v

        return training_args_cls(**training_args_kwargs)
