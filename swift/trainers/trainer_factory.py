# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import inspect
from dataclasses import asdict
from typing import Dict

from swift.utils import get_logger

logger = get_logger()


class TrainerFactory:
    TRAINER_MAPPING = {
        'causal_lm': 'swift.trainers.Seq2SeqTrainer',
        'seq_cls': 'swift.trainers.Trainer',
        'embedding': 'swift.trainers.EmbeddingTrainer',
        'reranker': 'swift.trainers.RerankerTrainer',
        'generative_reranker': 'swift.trainers.RerankerTrainer',
        # rlhf
        'dpo': 'swift.rlhf_trainers.DPOTrainer',
        'orpo': 'swift.rlhf_trainers.ORPOTrainer',
        'kto': 'swift.rlhf_trainers.KTOTrainer',
        'cpo': 'swift.rlhf_trainers.CPOTrainer',
        'rm': 'swift.rlhf_trainers.RewardTrainer',
        'ppo': 'swift.rlhf_trainers.PPOTrainer',
        'grpo': 'swift.rlhf_trainers.GRPOTrainer',
        'gkd': 'swift.rlhf_trainers.GKDTrainer',
    }

    TRAINING_ARGS_MAPPING = {
        'causal_lm': 'swift.trainers.Seq2SeqTrainingArguments',
        'seq_cls': 'swift.trainers.TrainingArguments',
        'embedding': 'swift.trainers.TrainingArguments',
        'reranker': 'swift.trainers.TrainingArguments',
        'generative_reranker': 'swift.trainers.TrainingArguments',
        # rlhf
        'dpo': 'swift.rlhf_trainers.DPOConfig',
        'orpo': 'swift.rlhf_trainers.ORPOConfig',
        'kto': 'swift.rlhf_trainers.KTOConfig',
        'cpo': 'swift.rlhf_trainers.CPOConfig',
        'rm': 'swift.rlhf_trainers.RewardConfig',
        'ppo': 'swift.rlhf_trainers.PPOConfig',
        'grpo': 'swift.rlhf_trainers.GRPOConfig',
        'gkd': 'swift.rlhf_trainers.GKDConfig',
    }

    @staticmethod
    def get_cls(args, mapping: Dict[str, str]):
        if hasattr(args, 'rlhf_type'):
            train_method = args.rlhf_type
        else:
            train_method = args.task_type
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
        parameters = inspect.signature(training_args_cls).parameters

        for k in list(args_dict.keys()):
            if k not in parameters:
                args_dict.pop(k)

        args._prepare_training_args(args_dict)
        training_args = training_args_cls(**args_dict)
        return training_args
