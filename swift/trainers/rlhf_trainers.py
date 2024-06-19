# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib

from swift.llm.utils import RLHFArguments


class RLHFTrainerFactory:
    TRAINERS_MAPPING = {
        'dpo': 'swift.trainers.dpo_trainer.DPOTrainer',
        'simpo': 'swift.trainers.cpo_trainer.CPOTrainer',
        'orpo': 'swift.trainers.orpo_trainer.ORPOTrainer',
        'kto': 'swift.trainers.kto_trainer.KTOTrainer',
        'cpo': 'swift.trainers.cpo_trainer.CPOTrainer'
    }

    @staticmethod
    def get_training_args(args: RLHFArguments):
        # get trainer kwargs
        trainer_kwargs = {}

        trainer_kwargs['args'] = args.training_args
        trainer_kwargs['check_model'] = args.check_model_is_latest
        trainer_kwargs['test_oom_error'] = args.test_oom_error

        if args.rlhf_type in ['dpo']:
            trainer_kwargs['beta'] = args.beta
            trainer_kwargs['label_smoothing'] = args.label_smoothing
            trainer_kwargs['loss_type'] = args.loss_type
            trainer_kwargs['sft_beta'] = args.sft_beta
            trainer_kwargs['max_length'] = args.max_length
            trainer_kwargs['max_prompt_length'] = args.max_prompt_length

        if args.rlhf_type == 'simpo':
            trainer_kwargs['gamma'] = args.simpo_gamma

        return trainer_kwargs

    @staticmethod
    def get_trainer(rlhf_type):
        module_path, class_name = RLHFTrainerFactory.TRAINERS_MAPPING[rlhf_type].rsplit('.', 1)
        if rlhf_type == 'simpo':
            import trl
            from packaging import version
            if version.parse(trl.__version__) <= version.parse('0.9.4'):
                module_path = 'swift.trainers.simpo_trainer'
                class_name = 'SimPOTrainer'
        module = importlib.import_module(module_path)
        trainer_class = getattr(module, class_name)
        return trainer_class
