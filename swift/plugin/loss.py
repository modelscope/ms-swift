# Copyright (c) Alibaba, Inc. and its affiliates.
from enum import Enum
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


class LossType:
    loss_scale = 'loss_scale'
    cosine_similarity = 'cosine_similarity'
    contrastive = 'contrastive'
    online_contrastive = 'online_contrastive'
    cosent = 'cosent'


LOSS_MAPPING = {}


def register_loss_func(loss_type: str, loss_func: Optional[Callable] = None):
    loss_info = {}

    if loss_func is not None:
        loss_info['loss_func'] = loss_func
        LOSS_MAPPING[loss_type] = loss_info
        return

    def _register_loss_func(loss_func: Callable) -> Callable:
        loss_info['loss_func'] = loss_func
        LOSS_MAPPING[loss_type] = loss_info
        return loss_func

    return _register_loss_func


def ce_loss_func(outputs, labels):
    logits = outputs.logits
    device = logits.device
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:].to(device)
    # Save memory
    masks = shift_labels != -100
    shift_logits = shift_logits[masks]
    shift_labels = shift_labels[masks]
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits, shift_labels)
    return loss, masks


# Use @register_loss_func to decorate your own loss, use --loss_type xxx to train
@register_loss_func(LossType.loss_scale)
def loss_scale_func(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    """Loss func

    Args:
        outputs: The model outputs
        labels: The labels
        loss_scale: The loss scale
        num_items_in_batch: Number of tokens in the labels of gradient accumulation round that are not -100.

    Returns:

    """
    loss, masks = ce_loss_func(outputs, labels)
    if loss_scale is not None:
        shift_scale = loss_scale[..., 1:].to(masks.device)
        shift_scale = shift_scale[masks]
        loss = (shift_scale * loss)
    if num_items_in_batch is None:
        loss = loss.mean()
    else:
        # compat transformers>=4.46
        loss = loss.sum() / num_items_in_batch
    return loss


def _parse_pair_sentence(outputs):
    last_hidden_state = outputs['last_hidden_state']
    batch_size = last_hidden_state.shape[0]
    shape_len = len(last_hidden_state.shape)
    if shape_len == 3:
        sentence1 = last_hidden_state[0:batch_size // 2][:, 0].squeeze(dim=1)
        sentence2 = last_hidden_state[batch_size // 2:][:, 0].squeeze(dim=1)
    else:
        sentence1 = last_hidden_state[0:batch_size // 2]
        sentence2 = last_hidden_state[batch_size // 2:]
    return sentence1, sentence2


# Code borrowed from sentence_transformers
class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)  # noqa
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)  # noqa
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)  # noqa


@register_loss_func(LossType.cosine_similarity)
def cosine_similarity_func(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    cos_score_transformation = nn.Identity()
    loss_fct = MSELoss()
    sentence1, sentence2 = _parse_pair_sentence(outputs)
    output = cos_score_transformation(torch.cosine_similarity(sentence1, sentence2))
    return loss_fct(output, labels.to(output.dtype).view(-1))


@register_loss_func(LossType.contrastive)
def contrastive_loss(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    sentence1, sentence2 = _parse_pair_sentence(outputs)
    distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
    distances = distance_metric(sentence1, sentence2)
    margin = 0.5
    labels = labels.to(sentence1.dtype)
    losses = 0.5 * (labels * distances.pow(2) + (1 - labels) * F.relu(margin - distances).pow(2))
    return losses.mean()


@register_loss_func(LossType.online_contrastive)
def online_contrastive_loss(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    sentence1, sentence2 = _parse_pair_sentence(outputs)
    distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
    distance_matrix = distance_metric(sentence1, sentence2)
    negs = distance_matrix[labels == 0]
    poss = distance_matrix[labels == 1]

    # select hard positive and hard negative pairs
    negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
    positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

    positive_loss = positive_pairs.pow(2).sum()
    margin = 0.5
    negative_loss = F.relu(margin - negative_pairs).pow(2).sum()
    loss = positive_loss + negative_loss
    return loss


def get_loss_func(loss_type: Optional[str]) -> Optional[Callable]:
    if loss_type is None:
        return None
    return LOSS_MAPPING[loss_type]['loss_func']
