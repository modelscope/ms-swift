# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
from contextlib import contextmanager

import transformers
from packaging import version
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
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
                for k, v in kwargs.items()
                if k in ['train_dataset', 'data_collator', 'reward_model', 'value_model', 'eval_dataset']
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

    def train(self, *args, **kwargs):
        # remove args that are not needed for the HFPPOTrainer
        super().train()

    def _save_checkpoint(self, *args, **kwargs):
        if version.parse(transformers.__version__) >= version.parse('4.47'):
            metrics = kwargs.pop('metrics', None)
            trial = kwargs.get('trial')
            self._determine_best_metric(metrics=metrics, trial=trial)
        return super()._save_checkpoint(*args, **kwargs)
