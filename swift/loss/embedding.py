# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import gather_object
from torch import nn
from torch.nn import MSELoss
from transformers.utils import strtobool

from .base import BaseLoss


# Code borrowed from sentence_transformers
class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)  # noqa
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)  # noqa
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)  # noqa


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


class CosineSimilarityLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        # You need to return a scalar representing the loss.
        cos_score_transformation = nn.Identity()
        loss_fct = MSELoss()
        sentence1, sentence2 = _parse_pair_sentence(outputs)
        output = cos_score_transformation(torch.cosine_similarity(sentence1, sentence2))
        return loss_fct(output, labels.to(output.dtype).view(-1))


class ContrastiveLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        sentence1, sentence2 = _parse_pair_sentence(outputs)
        distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
        distances = distance_metric(sentence1, sentence2)
        margin = 0.5
        labels = labels.to(sentence1.dtype)
        losses = 0.5 * (labels * distances.pow(2) + (1 - labels) * F.relu(margin - distances).pow(2))
        return losses.mean()


class OnlineContrastiveLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
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


class InfonceLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        temperature = float(os.environ.get('INFONCE_TEMPERATURE', '0.1'))  # temperature
        # calculate CE across the batch, meaning all samples will be negative except the matching positive
        use_batch = strtobool(os.environ.get('INFONCE_USE_BATCH', 'True'))
        hard_negatives = os.environ.get('INFONCE_HARD_NEGATIVES', None)  # how many negative prompts kept in one sample
        # mask out fake negatives
        infonce_mask_fake_negative = strtobool(os.environ.get('INFONCE_MASK_FAKE_NEGATIVE', 'False'))
        fake_neg_margin = float(os.environ.get('INFONCE_FAKE_NEG_MARGIN', '0.1'))
        # enhanced components to align with Qwen3-Embedding denominator; controlled individually
        # defaults set to False for backward compatibility
        infonce_include_qq = strtobool(os.environ.get('INFONCE_INCLUDE_QQ', 'False'))
        infonce_include_dd = strtobool(os.environ.get('INFONCE_INCLUDE_DD', 'False'))
        if hard_negatives is not None:
            hard_negatives = int(hard_negatives)
        from swift.utils import get_dist_setting
        rank, _, world_size, _ = get_dist_setting()
        # repeat of anchor(1)+positive(1)+negatives(n)
        sentences = outputs['last_hidden_state']

        if world_size > 1 and use_batch:
            from swift.sequence_parallel import sequence_parallel

            if getattr(sequence_parallel, 'dp_group', None) is not None:
                all_sentences = sequence_parallel._gather_object_dp(sentences.unsqueeze(0))
                labels = sequence_parallel._gather_object_dp(labels)
                rank = sequence_parallel.dp_rank
            else:
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
            if can_batched:
                # [B, neg+2, D]
                sentences = torch.stack(split_tensors, dim=0)
                # base q->d similarities (includes own positive and all in-batch documents)
                queries = sentences[:, 0].squeeze(1)  # [B, D]
                docs_all = sentences[:, 1:].reshape(-1, sentences.size(2))  # [B*(neg+1), D]
                qd_matrix = torch.matmul(queries, docs_all.T)  # [B, B*(neg+1)]
                # target indices: start of each group's document block (its positive)
                labels = torch.tensor(range(0,
                                            sentences.size(0) * (sentences.size(1) - 1),
                                            sentences.size(1) - 1)).view(-1).to(sentences.device)

                logits_list = [qd_matrix]

                if infonce_include_qq:
                    # q->q similarities; exclude self via -inf on diagonal to avoid accidental positives
                    qq_matrix = torch.matmul(queries, queries.T)  # [B, B]
                    qq_matrix = qq_matrix.clone()
                    qq_matrix.fill_diagonal_(float('-inf'))
                    logits_list.append(qq_matrix)

                if infonce_include_dd:
                    # d+ -> d (doc-doc) similarities; exclude self-positive column per row
                    pos_docs = sentences[:, 1].squeeze(1)  # [B, D]
                    dd_matrix = torch.matmul(pos_docs, docs_all.T)  # [B, B*(neg+1)]
                    # mask self positive per row: column index = row_idx * (neg+1)
                    block = sentences.size(1) - 1  # (neg+1)
                    if block > 0:
                        row_idx = torch.arange(dd_matrix.size(0), device=dd_matrix.device)
                        col_idx = row_idx * block
                        dd_matrix[row_idx, col_idx] = float('-inf')
                    logits_list.append(dd_matrix)

                if infonce_mask_fake_negative:
                    # thresholds derived from positive q->d scores per row
                    row_idx = torch.arange(qd_matrix.size(0), device=qd_matrix.device)
                    pos_scores = qd_matrix[row_idx, labels]
                    thresholds = pos_scores.view(-1, 1).detach() + fake_neg_margin

                    # qd block mask
                    qd_block = qd_matrix.clone()
                    qd_mask = qd_block > thresholds
                    qd_block[qd_mask] = float('-inf')

                    components = [qd_block]

                    # qq block mask (if present)
                    if infonce_include_qq:
                        qq_block = qq_matrix.clone()
                        qq_mask = qq_block > thresholds
                        qq_block[qq_mask] = float('-inf')
                        # diagonal already masked unconditionally at construction time
                        components.append(qq_block)

                    # dd block (if present): self-positive column already masked unconditionally
                    if infonce_include_dd:
                        # align with Qwen3-Embedding, no threshold masking for d-d
                        components.append(dd_matrix)

                    similarity_matrix = torch.cat(components, dim=1)
                else:
                    # concatenate all components without masking
                    similarity_matrix = torch.cat(logits_list, dim=1)
                # temperature scaling and CE
                similarity_matrix = similarity_matrix / temperature
                loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
            else:
                all_tensors = []
                for tensor in split_tensors:
                    all_tensors.append(tensor[1:])
                # cat all neg+1 tensors
                sentences = torch.cat(all_tensors, dim=0)
                # prepare query anchors list if q-q is included
                if infonce_include_qq:
                    queries_all = torch.stack([t[0] for t in split_tensors], dim=0)  # [B, D]
                length = 0
                for idx, tensor in enumerate(split_tensors):
                    # [D] * [B*(neg+1), D], neg numbers are different
                    qd_vec = torch.matmul(tensor[0], sentences.T)
                    target = torch.tensor(length).to(tensor.device)
                    logits_parts = []

                    # compute threshold from positive q->d score
                    threshold = (qd_vec[target].detach() + fake_neg_margin)

                    # qd part with masking
                    if infonce_mask_fake_negative:
                        qd_masked = torch.where(qd_vec > threshold, torch.tensor(float('-inf'), device=qd_vec.device),
                                                qd_vec)
                    else:
                        qd_masked = qd_vec
                    logits_parts.append(qd_masked)

                    # qq part
                    if infonce_include_qq:
                        qq_vec = torch.matmul(tensor[0], queries_all.T)  # [B]
                        # exclude self
                        qq_vec = qq_vec.clone()
                        qq_vec[idx] = float('-inf')
                        if infonce_mask_fake_negative:
                            qq_vec = torch.where(qq_vec > threshold, torch.tensor(float('-inf'), device=qq_vec.device),
                                                 qq_vec)
                        logits_parts.append(qq_vec)

                    # dd part
                    if infonce_include_dd:
                        dd_vec = torch.matmul(tensor[1], sentences.T)  # [B*(neg+1)]
                        # mask self positive column for this row only (no threshold masking for d-d)
                        block = split_tensors[idx].size(0) - 1  # (neg+1) for this group
                        dd_vec[length] = float('-inf')
                        logits_parts.append(dd_vec)

                    logits_row = torch.cat(logits_parts, dim=-1)
                    logits_row = logits_row / temperature
                    loss += nn.CrossEntropyLoss()(logits_row.unsqueeze(0), target.unsqueeze(0))
                    # next positive is neg+1
                    length += tensor.size(0) - 1
                loss /= len(split_tensors)
        return loss
