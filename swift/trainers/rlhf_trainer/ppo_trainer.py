# Copyright (c) Alibaba, Inc. and its affiliates.
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from trl import PPOv2Trainer as HFPPOTrainer

from swift.trainers import PushToMsHubMixin, RLHFTrainerMixin, SwiftMixin


class PPOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFPPOTrainer):

    def __init__(self, model: PreTrainedModel, ref_model: PreTrainedModel, *_args, **kwargs):
        kwargs['policy'] = model
        kwargs['ref_policy'] = ref_model
        super().__init__(model, ref_model, *_args, **kwargs)
        # reset dataloader
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=kwargs['data_collator'],
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        self.accelerator.prepare(self.data_collator)
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=kwargs['data_collator'],
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

    def train(self, *args, **kwargs):
        # remove args that are not needed for the HFPPOTrainer
        HFPPOTrainer.train(self)


def patched_init(self, **kwargs):
    kwargs_to_pop = ['model', 'model_init', 'compute_metrics', 'preprocess_logits_for_metrics']
    for kwarg in kwargs_to_pop:
        kwargs.pop(kwarg, None)
    kwargs['config'] = kwargs.pop('args')
    original_init(self, **kwargs)


original_init = HFPPOTrainer.__init__
HFPPOTrainer.__init__ = patched_init
