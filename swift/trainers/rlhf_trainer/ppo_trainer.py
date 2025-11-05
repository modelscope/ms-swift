# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
from contextlib import contextmanager
from typing import Optional

from torch.utils.data import DataLoader
from transformers import PreTrainedModel, Trainer
from trl import PPOTrainer as HFPPOTrainer

from swift.utils import patch_getattr
from ..mixin import SwiftMixin

ppo_trainer_init = HFPPOTrainer.__init__
del HFPPOTrainer.__init__


class PPOTrainer(SwiftMixin, HFPPOTrainer):

    @staticmethod
    @contextmanager
    def _patch_dataloader(collate_fn):
        __init__ = DataLoader.__init__

        def __new_init__(self, *args, **kwargs):
            kwargs['collate_fn'] = collate_fn
            __init__(self, *args, **kwargs)

        DataLoader.__init__ = __new_init__
        try:
            yield
        finally:
            DataLoader.__init__ = __init__

    def __init__(self, model: PreTrainedModel, ref_model: PreTrainedModel, *_args, **kwargs):
        super().__init__(model, *_args, **{k: v for k, v in kwargs.items() if k not in {'reward_model', 'value_model'}})
        with self._patch_dataloader(kwargs['data_collator']):
            new_kwargs = {
                k: v
                for k, v in kwargs.items() if k in [
                    'train_dataset',
                    'data_collator',
                    'reward_model',
                    'value_model',
                    'eval_dataset',
                    'callbacks',
                ]
            }
            parameters = inspect.signature(ppo_trainer_init).parameters
            if 'config' in parameters:
                new_kwargs['config'] = kwargs['args']
            else:
                new_kwargs['args'] = kwargs['args']
            if 'processing_class' in parameters:
                new_kwargs['processing_class'] = self.tokenizer
            else:
                new_kwargs['tokenizer'] = self.tokenizer
            ppo_trainer_init(self, model=model, ref_model=ref_model, **new_kwargs)
        unwrap_model = self.accelerator.unwrap_model(self.model)
        patch_getattr(unwrap_model.__class__, 'policy')

    def create_loss_and_metric(self, args):
        return {}

    def train(self, *args, **kwargs):
        # remove args that are not needed for the HFPPOTrainer
        super().train()

    def _save_checkpoint(self, *args, **kwargs):
        kwargs.pop('metrics', None)

        backup_model = self.model
        try:
            # Unwrap model if needed
            self.model = self.accelerator.unwrap_model(self.model)
            return super()._save_checkpoint(*args, **kwargs)
        finally:
            self.model = backup_model

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        # https://github.com/huggingface/trl/issues/2122
        backup_model = self.model

        # Unwrap model if needed to access the policy
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.model = unwrapped_model.policy  # save only the policy

        Trainer.save_model(self, output_dir, _internal_call)

        self.model = backup_model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.is_deepspeed_enabled:
            state_dict = {
                name.removeprefix('policy.'): param
                for name, param in state_dict.items() if name.startswith('policy.')
            }

        super()._save(output_dir, state_dict)

    def _prepare_gradient_checkpointing(self, model):
        # Be consistent with TRL
        # models = list(set([self.model.policy, self.model.value_model]))
        # for model in models:
        #     SwiftMixin._prepare_gradient_checkpointing(self, model)
        pass

    def generate_completions(self, *args, **kwargs):
        if self.eval_dataset is None:
            return
        return super().generate_completions(*args, **kwargs)
