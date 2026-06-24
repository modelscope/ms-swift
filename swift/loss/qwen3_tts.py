# Copyright (c) ModelScope Contributors. All rights reserved.
import torch

from .base import BaseLoss


class Qwen3TTSLoss(BaseLoss):
    """Custom loss for Qwen3-TTS SFT training.

    Combines the talker codec_0 cross-entropy loss with the sub-talker loss
    using a fixed weighting factor of 0.3.

    The model forward is expected to return:
        - outputs.loss: codec_0 prediction loss from the talker
        - outputs.sub_talker_loss: multi-codebook prediction loss from the sub-talker
    """

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs) -> torch.Tensor:
        talker_loss = outputs.loss
        sub_talker_loss = getattr(outputs, 'sub_talker_loss', None)
        if sub_talker_loss is not None:
            loss = talker_loss + 0.3 * sub_talker_loss
        else:
            loss = talker_loss
        return loss
