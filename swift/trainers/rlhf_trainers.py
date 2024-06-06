# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib
from typing import Any, Dict

from swift.llm.utils import RLHFArguments


def filter_args(func, args: RLHFArguments) -> Dict[str, Any]:
    import inspect
    func_signature = inspect.signature(func)
    args_dict = args.__dict__
    valid_args = {k: v for k, v in args_dict.items() if k in func_signature.parameters}
    return valid_args


class TrainerFactory:
    TRAINERS_MAPPING = {
        'dpo': 'swift.trainers.dpo_trainers.DPOTrainer',
        'simpo': 'swift.trainers.simpo_trainers.SimPOTrainer',
        'orpo': 'swift.trainers.orpo_trainers.ORPOTrainer',
    }

    @staticmethod
    def get_training_args(args: RLHFArguments):
        # get trainer kwargs
        trainer_kwargs = {}
        # common
        trainer_kwargs['args'] = args.rlhf_config_args
        trainer_kwargs['check_model'] = args.check_model_is_latest
        trainer_kwargs['test_oom_error'] = args.test_oom_error

        if args.rlhf_type in ['dpo', 'simpo']:
            trainer_kwargs['beta'] = args.beta
            trainer_kwargs['label_smoothing'] = args.label_smoothing
            trainer_kwargs['loss_type'] = args.loss_type
            trainer_kwargs['sft_beta'] = args.sft_beta
            trainer_kwargs['max_length'] = args.max_length
            trainer_kwargs['max_prompt_length'] = args.max_prompt_length

        if args.rlhf_type == 'simpo':
            trainer_kwargs['gamma'] = args.gamma

        return trainer_kwargs

    @staticmethod
    def get_trainer(*args, **kwargs):

        if args.rlhf_type not in TrainerFactory.TRAINERS_MAPPING:
            raise ValueError(f'Unknown rlhf type: {args.rlhf_type}')

        module_path, class_name = TrainerFactory.TRAINERS_MAPPING[args.rlhf_type].rsplit('.', 1)
        module = importlib.import_module(module_path)
        trainer_class = getattr(module, class_name)
        filtered_args = filter_args(trainer_class.__init__, args)
        return trainer_class(**filtered_args)
