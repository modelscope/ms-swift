# Copyright (c) Alibaba, Inc. and its affiliates.
from trl import AutoModelForCausalLMWithValueHead
from trl.trainer import PPOv2Trainer as HFPPOTrainer

from swift.trainers import PushToMsHubMixin, RLHFTrainerMixin, SwiftMixin


class PPOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFPPOTrainer):

    def __init__(self, model: AutoModelForCausalLMWithValueHead, ref_model: AutoModelForCausalLMWithValueHead, *_args,
                 **kwargs):
        kwargs['policy'] = model
        kwargs['ref_policy'] = ref_model
        super().__init__(model, ref_model, *_args, **kwargs)

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
