# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager

from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from trl import PPOv2Trainer as HFPPOv2Trainer

from swift.utils import patch_getattr
from ..mixin import SwiftMixin

ppo_trainer_init = HFPPOv2Trainer.__init__
del HFPPOv2Trainer.__init__


class PPOTrainer(SwiftMixin, HFPPOv2Trainer):

    @staticmethod
    @contextmanager
    def _patch_dataloader(collate_fn):
        __init__ = DataLoader.__init__

        def __new_init__(self, *args, **kwargs):
            kwargs['collate_fn'] = collate_fn
            __init__(self, *args, **kwargs)

        DataLoader.__init__ = __new_init__
        yield
        DataLoader.__init__ = __init__

    def __init__(self, model: PreTrainedModel, ref_model: PreTrainedModel, *_args, **kwargs):
        super().__init__(model, *_args, **kwargs)
        with self._patch_dataloader(kwargs['data_collator']):
            new_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ['train_dataset', 'data_collator', 'reward_model', 'value_model', 'eval_dataset']
            }
            ppo_trainer_init(
                self, config=kwargs['args'], tokenizer=self.tokenizer, policy=model, ref_policy=ref_model, **new_kwargs)
        unwrap_model = self.accelerator.unwrap_model(self.model)
        patch_getattr(unwrap_model, 'policy')

    def train(self, *args, **kwargs):
        # remove args that are not needed for the HFPPOTrainer
        super().train()
