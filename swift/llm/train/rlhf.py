# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Union

from swift.llm import safe_snapshot_download
from swift.utils import get_logger, get_model_parameter_info
from ..argument import BaseArguments, RLHFArguments
from ..model import HfConfigFactory
from .kto import prepare_kto_dataset
from .sft import SwiftSft

logger = get_logger()


class SwiftRLHF(SwiftSft):
    args_class = RLHFArguments
    args: args_class

    def _prepare_model_tokenizer(self):
        if self.args.sequence_parallel_size > 1:
            # Duplicate calling is allowd to promise this function will
            # be called before model initializing.
            from swift.trainers.sequence_parallel import sequence_parallel
            sequence_parallel.init_sequence_parallel(self.args.sequence_parallel_size)
        # prepare ref/reward/value model
        from swift.llm.infer.utils import prepare_adapter
        args = self.args

        def prepare_single_model(key, origin_key, model_type, model_revision):
            origin_key = origin_key or key
            model_id_or_path = getattr(args, f'{key}_model')
            if model_id_or_path is None:
                return None
            if isinstance(model_id_or_path, list):
                # value model in PPO
                model_id_or_path = model_id_or_path[0]
            model_dir = safe_snapshot_download(
                model_id_or_path=model_id_or_path,
                revision=model_revision,
                download_model=False,
                use_hf=args.use_hf,
                hub_token=args.hub_token,
            )
            task_type = None
            num_labels = None
            if os.path.exists(os.path.join(model_dir, 'args.json')):
                model_args = BaseArguments.from_pretrained(model_dir)
                if hasattr(model_args, 'task_type'):
                    task_type = model_args.task_type
                if hasattr(model_args, 'num_labels'):
                    num_labels = model_args.num_labels
                if task_type == 'seq_cls' and num_labels is None:
                    num_labels = 1
            else:
                from transformers import AutoConfig
                model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
                if hasattr(model_config, 'num_labels'):
                    num_labels = model_config.num_labels

                # PretrainedConfig default num_labels = 2
                if num_labels == 1:
                    task_type = 'seq_cls'

            model, processor = args.get_model_processor(
                model=model_id_or_path,
                model_type=model_type,
                model_revision=model_revision,
                task_type=task_type,
                num_labels=num_labels)

            adapters = args.adapters if key == 'ref' else args.reward_adapters
            model = prepare_adapter(args, model, adapters)
            if origin_key in {'ref', 'reward', 'teacher'}:
                if self.args.sequence_parallel_size > 1:
                    sequence_parallel.prepare_model(model, processor)
                model.requires_grad_(False).eval()
            else:
                model = self.prepare_model(args, model, task_type=task_type)
                logger.info(f'value_model: {model}')
                model_parameter_info = get_model_parameter_info(model)
                self.train_msg['value_model_parameter_info'] = model_parameter_info
                logger.info(f'value_model_parameter_info: {model_parameter_info}')

            HfConfigFactory.set_model_config_attr(model, 'use_cache', False)
            return model, processor

        # Handle ref and value models
        for key in ['ref', 'value', 'teacher']:
            setattr(self, f'{key}_model', None)
            if key == 'ref' and args.rlhf_type == 'gkd':
                continue
            if key == 'value' and args.rlhf_type != 'ppo':
                continue
            if key == 'teacher' and args.rlhf_type != 'gkd':
                continue
            model_key = 'reward' if key == 'value' else key
            model_type = getattr(args, f'{model_key}_model_type')
            model_revision = getattr(args, f'{model_key}_model_revision')
            if key == 'value':
                model_type = model_type[0] if model_type else None
                model_revision = model_revision[0] if model_revision else None

            result = prepare_single_model(model_key, key, model_type, model_revision)
            if result is not None:
                model, _ = result
                setattr(self, f'{key}_model', model)

        # Handle reward model(s)
        self.reward_model = None
        if hasattr(args, 'reward_model') and args.reward_model is not None:
            rms = args.reward_model if isinstance(args.reward_model, list) else [args.reward_model]
            num_rms = len(rms)
            rm_types = args.reward_model_type if args.reward_model_type else [None] * num_rms
            rm_revisions = args.reward_model_revision if args.reward_model_revision else [None] * num_rms
            assert len(rms) == len(rm_types) == len(rm_revisions)

            self.reward_model = []
            if args.rlhf_type == 'grpo':
                self.reward_template = []

            for reward_model_path, rm_type, rm_revision in zip(rms, rm_types, rm_revisions):
                args.reward_model = reward_model_path  # Temporarily set for prepare_single_model
                result = prepare_single_model('reward', None, rm_type, rm_revision)
                if result is not None:
                    model, processor = result
                    self.reward_model.append(model)

                    if args.rlhf_type == 'grpo':
                        reward_template = self.args.get_template(processor, processor.model_meta.template)
                        if reward_template.use_model:
                            reward_template.model = model
                        self.reward_template.append(reward_template)
                args.reward_model = rms  # Restore original value
                if args.rlhf_type != 'grpo' and self.reward_model:
                    assert len(self.reward_model) <= 1
                    self.reward_model = self.reward_model[0]

        super()._prepare_model_tokenizer()

    def _prepare_template(self) -> None:
        args = self.args
        super()._prepare_template()
        model_mapping = {'kto': 'kto', 'gkd': 'gkd', 'ppo': 'pt', 'grpo': 'pt'}
        self.template.set_mode(model_mapping.get(args.rlhf_type, 'rlhf'))

        if args.rlhf_type == 'ppo':
            args.training_args.stop_token_id = self.template.template_meta.stop_token_id

    def _get_dataset(self):
        args = self.args
        train_dataset, val_dataset = super()._get_dataset()
        if args.rlhf_type == 'kto':
            train_dataset, val_dataset = prepare_kto_dataset(args, train_dataset, val_dataset)
        return train_dataset, val_dataset

    def _get_trainer_kwargs(self):
        trainer_kwargs = {}
        for key in ['ref', 'reward', 'value', 'teacher']:
            key = f'{key}_model'
            model = getattr(self, key, None)
            if model or self.args.rlhf_type == 'ppo' and key != 'teacher_model':
                trainer_kwargs[key] = model
        if hasattr(self, 'reward_template'):
            trainer_kwargs['reward_template'] = self.reward_template
        if self.args.rlhf_type == 'grpo':
            trainer_kwargs['reward_funcs'] = self.args.reward_funcs
            trainer_kwargs['vllm_client'] = self.args.vllm_client
        return trainer_kwargs


def rlhf_main(args: Union[List[str], RLHFArguments, None] = None):
    return SwiftRLHF(args).main()
