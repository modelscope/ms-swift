# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from enum import Enum
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import gather_object
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.utils import strtobool


class LossType:
    loss_scale = 'loss_scale'
    cosine_similarity = 'cosine_similarity'
    contrastive = 'contrastive'
    online_contrastive = 'online_contrastive'
    infonce = 'infonce'


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
    if isinstance(outputs, dict):
        last_hidden_state = outputs['last_hidden_state']
    else:
        last_hidden_state = outputs
    batch_size = last_hidden_state.shape[0]
    shape_len = len(last_hidden_state.shape)
    first_sentence = list(range(0, batch_size, 2))
    second_sentence = list(range(1, batch_size, 2))
    if shape_len == 3:
        sentence1 = last_hidden_state[first_sentence][:, 0].squeeze(dim=1)
        sentence2 = last_hidden_state[second_sentence][:, 0].squeeze(dim=1)
    else:
        sentence1 = last_hidden_state[first_sentence]
        sentence2 = last_hidden_state[second_sentence]
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


def calculate_paired_metrics(embeddings, labels):
    from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, \
        paired_manhattan_distances
    from scipy.stats import pearsonr, spearmanr

    embeddings1, embeddings2 = _parse_pair_sentence(embeddings)
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
    dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
    eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

    eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
    eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

    eval_pearson_dot, _ = pearsonr(labels, dot_products)
    eval_spearman_dot, _ = spearmanr(labels, dot_products)

    return {
        'pearson_cosine': eval_pearson_cosine,
        'pearson_euclidean': eval_pearson_manhattan,
        'pearson_manhattan': eval_pearson_euclidean,
        'pearson_dot_product': eval_pearson_dot,
        'spearman_cosine': eval_spearman_cosine,
        'spearman_euclidean': eval_spearman_manhattan,
        'spearman_manhattan': eval_spearman_euclidean,
        'spearman_dot_product': eval_spearman_dot,
    }


def calculate_infonce_metrics(embeddings, labels):
    from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, \
        paired_manhattan_distances
    from scipy.stats import pearsonr, spearmanr
    hard_negatives = os.environ.get('INFONCE_HARD_NEGATIVES', None)
    use_batch = strtobool(os.environ.get('INFONCE_USE_BATCH', 'True'))
    split_tensors = _parse_multi_negative_sentences(torch.tensor(embeddings), torch.tensor(labels), hard_negatives)
    split_tensors = [t.numpy() for t in split_tensors]
    can_batched = hard_negatives is not None
    if hard_negatives is None and len(set([s.shape[0] for s in split_tensors])) == 1:
        can_batched = True
    all_similarity_matrix = []
    all_labels = []
    pos_neg_margins = []
    if not use_batch:
        if can_batched:
            sentences = np.stack(split_tensors, axis=0)
            similarity_matrix = np.matmul(sentences[:, 0:1], sentences[:, 1:].transpose((0, 2, 1))).squeeze(1)
            all_similarity_matrix.append(similarity_matrix)
            labels = np.zeros_like(similarity_matrix)
            labels[:, 0] = 1
            all_labels.append(labels)
        else:
            for tensor in split_tensors:
                similarity_matrix = np.matmul(tensor[0], tensor[1:].T)
                all_similarity_matrix.append(similarity_matrix)
                labels = np.zeros_like(similarity_matrix)
                labels[0] = 1
                all_labels.append(labels)
                max_neg_scores = np.max(similarity_matrix[labels == 0], axis=-1)
                pos_neg_margins.append(np.mean(similarity_matrix[labels == 1] - max_neg_scores).item())
    else:
        if can_batched:
            sentences = np.stack(split_tensors, axis=0)
            similarity_matrix = np.matmul(sentences[:, 0], sentences[:, 1:].reshape(-1, sentences.shape[2]).T)
            all_similarity_matrix.append(similarity_matrix)
            labels = np.zeros_like(similarity_matrix)
            for row, col in enumerate(range(0, sentences.shape[0] * (sentences.shape[1] - 1), sentences.shape[1] - 1)):
                labels[row, col] = 1
            all_labels.append(labels)
        else:
            all_tensors = []
            for tensor in split_tensors:
                all_tensors.append(tensor[1:])
            sentences = np.concatenate(all_tensors, axis=0)
            length = 0
            for idx, tensor in enumerate(split_tensors):
                similarity_matrix = np.matmul(tensor[0], sentences.T)
                all_similarity_matrix.append(similarity_matrix)
                labels = np.zeros_like(similarity_matrix)
                labels[length] = 1
                all_labels.append(labels)
                length += tensor.shape[0] - 1
                max_neg_scores = np.max(similarity_matrix[labels == 0], axis=-1)
                pos_neg_margins.append(np.mean(similarity_matrix[labels == 1] - max_neg_scores).item())

    similarity_matrix = np.concatenate(all_similarity_matrix, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if can_batched:
        pos_scores = similarity_matrix[labels == 1].reshape(similarity_matrix.shape[0], -1)
        neg_scores = similarity_matrix[labels == 0].reshape(similarity_matrix.shape[0], -1)
        max_neg_scores = np.max(neg_scores, axis=-1)
        pos_neg_margin = np.mean(pos_scores - max_neg_scores).item()
    else:
        pos_scores = similarity_matrix[labels == 1]
        neg_scores = similarity_matrix[labels == 0]
        pos_neg_margin = np.mean(pos_neg_margins)

    mean_neg = np.mean(neg_scores)
    mean_pos = np.mean(pos_scores)
    return {'margin': pos_neg_margin, 'mean_neg': mean_neg, 'mean_pos': mean_pos}


def _parse_multi_negative_sentences(sentences, labels, hard_negatives=None):
    split_indices = torch.nonzero(labels, as_tuple=False).squeeze().tolist()
    if isinstance(split_indices, int):
        split_indices = [split_indices]
    split_indices.append(len(labels))
    split_indices = np.array(split_indices) + np.array(list(range(len(split_indices))))
    split_tensors = []

    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        split_part = sentences[start:end]
        if hard_negatives is not None:
            negatives = len(split_part) - 2
            assert negatives > 0
            if negatives > hard_negatives:
                split_part = split_part[:hard_negatives + 2]
            elif negatives < hard_negatives:
                selected = np.random.choice(list(range(negatives)), size=hard_negatives - negatives, replace=True)
                selected += 1  # skip positive
                split_part = torch.cat((split_part, split_part[selected]), dim=0)
        split_tensors.append(split_part)
    return split_tensors


@register_loss_func(LossType.infonce)
def infonce_loss(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    temperature = float(os.environ.get('INFONCE_TEMPERATURE', '0.01'))  # temperature
    # calculate CE across the batch, meaning all samples will be negative except the matching positive
    use_batch = strtobool(os.environ.get('INFONCE_USE_BATCH', 'True'))
    hard_negatives = os.environ.get('INFONCE_HARD_NEGATIVES', None)  # how many negative prompts kept in one sample
    # mask out fake negatives
    infonce_mask_fake_negative = strtobool(os.environ.get('INFONCE_MASK_FAKE_NEGATIVE', 'False'))
    if hard_negatives is not None:
        hard_negatives = int(hard_negatives)
    from swift.utils import get_dist_setting
    rank, _, world_size, _ = get_dist_setting()
    # repeat of anchor(1)+positive(1)+negatives(n)
    sentences = outputs['last_hidden_state']

    if world_size > 1 and use_batch:
        # gather all the sentences and labels across the gpus when calculate loss across all batches of all gpus
        all_sentences = gather_object(sentences.unsqueeze(0))
        labels = gather_object(labels)
        # override the gathered one
        all_sentences[rank] = sentences
        for idx in range(len(all_sentences)):
            if idx == rank:
                continue
            # we don't calculate grad from other gpus
            all_sentences[idx] = all_sentences[idx].detach().to(sentences.device)
        sentences = torch.cat(all_sentences, dim=0)
        labels = [tensor.to(sentences.device) for tensor in labels]
        labels = torch.stack(labels, dim=0)

    # split tensors into single sample
    # for example: batch_size=2 with tensor anchor(1)+positive(1)+negatives(3) + anchor(1)+positive(1)+negatives(2)
    # labels will be [1,0,0,0,1,0,0], meaning 1 positive, 3 negatives, 1 positive, 2 negatives
    split_tensors = _parse_multi_negative_sentences(sentences, labels, hard_negatives)
    loss = 0
    can_batched = hard_negatives is not None
    if hard_negatives is None and len(set([s.shape[0] for s in split_tensors])) == 1:
        # all tensors have the same batch size
        can_batched = True
    if not use_batch:
        # only calculate loss inside one sample
        if can_batched:
            # negative numbers are equal
            # [B, neg+2, D]
            sentences = torch.stack(split_tensors, dim=0)
            # [B, 1, D] * [B, neg+1, D]
            similarity_matrix = torch.matmul(sentences[:, 0:1], sentences[:, 1:].transpose(1, 2)) / temperature
            # The positive one is the first element
            labels = torch.zeros(len(split_tensors), dtype=torch.int64).to(sentences.device)
            loss = nn.CrossEntropyLoss()(similarity_matrix.squeeze(1), labels)
        else:
            # the negative numbers may be different, use for loop
            for tensor in split_tensors:
                # [D] * [neg+1, D]
                similarity_matrix = torch.matmul(tensor[0], tensor[1:].T) / temperature
                # The positive one is the first element
                labels = torch.tensor(0).to(tensor.device)
                loss += nn.CrossEntropyLoss()(similarity_matrix, labels)
            # avg between all batches in one gpu
            loss /= len(split_tensors)
    else:

        def mask_fake_negative(sim_matrix, sim_labels):
            thresholds = sim_matrix[torch.arange(sim_matrix.size(0)), sim_labels].view(-1, 1) + 0.1
            thresholds = thresholds.detach()
            mask = sim_matrix > thresholds
            sim_matrix[mask] = float('-inf')

        if can_batched:
            # [B, neg+2, D]
            sentences = torch.stack(split_tensors, dim=0)
            # [B, D] * [B*(neg+1), D]
            similarity_matrix = torch.matmul(sentences[:, 0].squeeze(1), sentences[:,
                                                                                   1:].reshape(-1, sentences.size(2)).T)
            labels = torch.tensor(range(0,
                                        sentences.size(0) * (sentences.size(1) - 1),
                                        sentences.size(1) - 1)).view(-1).to(sentences.device)
            if infonce_mask_fake_negative:
                mask_fake_negative(similarity_matrix, labels)
            similarity_matrix = similarity_matrix / temperature
            # every neg+1 is positive start from 0
            loss = nn.CrossEntropyLoss()(similarity_matrix, labels) / world_size  # avoid duplicate
        else:
            all_tensors = []
            for tensor in split_tensors:
                all_tensors.append(tensor[1:])
            # cat all neg+1 tensors
            sentences = torch.cat(all_tensors, dim=0)
            length = 0
            for idx, tensor in enumerate(split_tensors):
                # [D] * [B*(neg+1), D], neg numbers are different
                similarity_matrix = torch.matmul(tensor[0], sentences.T) / temperature
                labels = torch.tensor(length).to(tensor.device)
                loss += nn.CrossEntropyLoss()(similarity_matrix, labels)
                # next positive is neg+1
                length += tensor.size(0) - 1
            loss /= len(split_tensors)
            loss /= world_size  # avoid duplicate
    return loss


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
