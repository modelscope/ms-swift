# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.utils import gather_object
from enum import Enum
from torch import nn
from torch.nn import MSELoss
from transformers.utils import strtobool

from swift.sequence_parallel import sequence_parallel
from swift.utils import get_dist_setting
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


class InfoNCELoss(BaseLoss):

    def _parse_config(self):
        """Parse InfoNCE loss configuration from environment variables."""
        hard_negatives = os.environ.get('INFONCE_HARD_NEGATIVES', None)
        if hard_negatives is not None:
            hard_negatives = int(hard_negatives)
        return {
            'temperature': float(os.environ.get('INFONCE_TEMPERATURE', '0.1')),
            'use_batch': strtobool(os.environ.get('INFONCE_USE_BATCH', 'True')),
            'hard_negatives': hard_negatives,
            # mask out fake negatives
            'mask_fake_negative': strtobool(os.environ.get('INFONCE_MASK_FAKE_NEGATIVE', 'False')),
            'fake_neg_margin': float(os.environ.get('INFONCE_FAKE_NEG_MARGIN', '0.1')),
            # enhanced components to align with Qwen3-Embedding denominator; controlled individually
            # defaults set to False for backward compatibility
            'include_qq': strtobool(os.environ.get('INFONCE_INCLUDE_QQ', 'False')),
            'include_dd': strtobool(os.environ.get('INFONCE_INCLUDE_DD', 'False')),
        }

    def _gather_distributed(self, sentences, labels, rank, world_size):
        """Gather sentences and labels across ranks for cross-batch negatives."""
        if getattr(sequence_parallel, 'dp_group', None) is not None:
            all_sentences = sequence_parallel._gather_object_dp(sentences.unsqueeze(0))
            labels = sequence_parallel._gather_object_dp(labels)
            rank = sequence_parallel.dp_rank
        elif self.is_megatron:
            from megatron.core import mpu
            dp_group = mpu.get_data_parallel_group()
            # Gather sentences
            shapes = [sentences.new_empty((2, ), dtype=torch.long) for _ in range(world_size)]
            dist.all_gather(
                shapes,
                sentences.new_tensor(sentences.shape, dtype=torch.long),
                group=dp_group,
            )
            all_sentences = [sentences.new_empty(shape.tolist()) for shape in shapes]
            dist.all_gather(all_sentences, sentences, group=dp_group)
            # Gather labels (must also be gathered in megatron path)
            all_labels = [labels.new_empty_like(labels) for _ in range(world_size)]
            dist.all_gather(all_labels, labels, group=dp_group)
            labels = all_labels
        else:
            # gather all the sentences and labels across the gpus when calculate loss across all batches of all gpus
            all_sentences = gather_object(sentences.unsqueeze(0))
            labels = gather_object(labels)

        # Override with local sentences to preserve gradient flow
        all_sentences[rank] = sentences
        for idx in range(len(all_sentences)):
            if idx == rank:
                continue
            # we don't calculate grad from other gpus
            all_sentences[idx] = all_sentences[idx].detach().to(sentences.device)
        sentences = torch.cat(all_sentences, dim=0)
        labels = [tensor.to(sentences.device) for tensor in labels]
        labels = torch.stack(labels, dim=0)
        return sentences, labels, rank

    def _compute_local_loss(self, split_tensors, can_batched, temperature):
        """Compute loss using only within-sample negatives (no cross-batch)."""
        if can_batched:
            # negative numbers are equal
            sentences = torch.stack(split_tensors, dim=0)  # [B, neg+2, D]
            # [B, 1, D] * [B, D, neg+1] -> [B, 1, neg+1]
            similarity = torch.matmul(sentences[:, 0:1], sentences[:, 1:].transpose(1, 2)) / temperature
            # The positive one is the first element
            labels = torch.zeros(len(split_tensors), dtype=torch.int64, device=sentences.device)
            return nn.CrossEntropyLoss()(similarity.squeeze(1), labels)
        # the negative numbers may be different, use for loop
        loss = 0
        for tensor in split_tensors:
            # [D] * [neg+1, D]
            similarity = torch.matmul(tensor[0], tensor[1:].T) / temperature
            target = torch.tensor(0, device=tensor.device)
            loss += nn.CrossEntropyLoss()(similarity, target)
        return loss / len(split_tensors)

    @staticmethod
    def _mask_fake_negatives(logits, threshold):
        """Mask logits exceeding the threshold (fake negatives) with -inf."""
        return torch.where(logits > threshold, torch.tensor(float('-inf'), device=logits.device), logits)

    def _compute_cross_batch_loss_batched(self, split_tensors, config):
        """Compute cross-batch loss when all samples have equal numbers of negatives."""
        temperature = config['temperature']
        sentences = torch.stack(split_tensors, dim=0)  # [B, neg+2, D]
        # base q->d similarities (includes own positive and all in-batch documents)
        queries = sentences[:, 0].squeeze(1)  # [B, D]
        docs_all = sentences[:, 1:].reshape(-1, sentences.size(2))  # [B*(neg+1), D]
        block_size = sentences.size(1) - 1  # neg + 1

        qd_matrix = torch.matmul(queries, docs_all.T)  # [B, B*(neg+1)]
        # target indices: start of each group's document block (its positive)
        labels = torch.arange(0, sentences.size(0) * block_size, block_size, device=sentences.device)

        logits_list = [qd_matrix]

        # Optional q->q similarity; exclude self via -inf on diagonal
        qq_matrix = None
        if config['include_qq']:
            qq_matrix = torch.matmul(queries, queries.T)  # [B, B]
            qq_matrix = qq_matrix.clone()
            qq_matrix.fill_diagonal_(float('-inf'))
            logits_list.append(qq_matrix)

        # Optional d+->d (doc-doc) similarity; exclude self-positive column per row
        dd_matrix = None
        if config['include_dd']:
            pos_docs = sentences[:, 1].squeeze(1)  # [B, D]
            dd_matrix = torch.matmul(pos_docs, docs_all.T)  # [B, B*(neg+1)]
            if block_size > 0:
                row_idx = torch.arange(dd_matrix.size(0), device=dd_matrix.device)
                dd_matrix[row_idx, row_idx * block_size] = float('-inf')
            logits_list.append(dd_matrix)

        # Build final similarity matrix with optional fake-negative masking
        if config['mask_fake_negative']:
            row_idx = torch.arange(qd_matrix.size(0), device=qd_matrix.device)
            pos_scores = qd_matrix[row_idx, labels]
            thresholds = pos_scores.view(-1, 1).detach() + config['fake_neg_margin']

            components = [self._mask_fake_negatives(qd_matrix, thresholds)]
            if qq_matrix is not None:
                # diagonal already masked unconditionally at construction time
                components.append(self._mask_fake_negatives(qq_matrix, thresholds))
            if dd_matrix is not None:
                # align with Qwen3-Embedding, no threshold masking for d-d
                components.append(dd_matrix)
            similarity_matrix = torch.cat(components, dim=1)
        else:
            similarity_matrix = torch.cat(logits_list, dim=1)

        similarity_matrix = similarity_matrix / temperature
        return nn.CrossEntropyLoss()(similarity_matrix, labels)

    def _compute_cross_batch_loss_unbatched(self, split_tensors, config):
        """Compute cross-batch loss when samples have varying numbers of negatives."""
        temperature = config['temperature']
        # Concatenate all documents (positive + negatives) across samples
        all_docs = torch.cat([t[1:] for t in split_tensors], dim=0)  # [total_docs, D]

        queries_all = None
        if config['include_qq']:
            queries_all = torch.stack([t[0] for t in split_tensors], dim=0)  # [B, D]

        loss = 0
        offset = 0  # tracks position of current sample's positive in all_docs
        for idx, tensor in enumerate(split_tensors):
            query = tensor[0]  # [D]
            target = torch.tensor(offset, device=tensor.device)

            # q->d similarity
            qd_vec = torch.matmul(query, all_docs.T)  # [total_docs]
            threshold = qd_vec[target].detach() + config['fake_neg_margin']

            if config['mask_fake_negative']:
                qd_vec = self._mask_fake_negatives(qd_vec, threshold)
            logits_parts = [qd_vec]

            # Optional q->q
            if config['include_qq']:
                qq_vec = torch.matmul(query, queries_all.T)  # [B]
                qq_vec = qq_vec.clone()
                qq_vec[idx] = float('-inf')  # exclude self
                if config['mask_fake_negative']:
                    qq_vec = self._mask_fake_negatives(qq_vec, threshold)
                logits_parts.append(qq_vec)

            # Optional d+->d (no threshold masking for d-d)
            if config['include_dd']:
                dd_vec = torch.matmul(tensor[1], all_docs.T)  # [total_docs]
                dd_vec[offset] = float('-inf')  # mask self-positive
                logits_parts.append(dd_vec)

            logits_row = torch.cat(logits_parts, dim=-1) / temperature
            loss += nn.CrossEntropyLoss()(logits_row.unsqueeze(0), target.unsqueeze(0))
            offset += tensor.size(0) - 1

        return loss / len(split_tensors)

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        config = self._parse_config()

        if self.is_megatron:
            from megatron.core import mpu
            rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        else:
            rank, _, world_size, _ = get_dist_setting()

        # repeat of anchor(1)+positive(1)+negatives(n)
        sentences = outputs['last_hidden_state']

        if world_size > 1 and config['use_batch']:
            sentences, labels, rank = self._gather_distributed(sentences, labels, rank, world_size)

        # split tensors into single sample
        # for example: batch_size=2 with tensor anchor(1)+positive(1)+negatives(3) + anchor(1)+positive(1)+negatives(2)
        # labels will be [1,0,0,0,1,0,0], meaning 1 positive, 3 negatives, 1 positive, 2 negatives
        split_tensors = _parse_multi_negative_sentences(sentences, labels, config['hard_negatives'])

        # Determine if all samples can be batched (equal negative counts)
        can_batched = config['hard_negatives'] is not None or len(set(s.shape[0] for s in split_tensors)) == 1

        if not config['use_batch']:
            return self._compute_local_loss(split_tensors, can_batched, config['temperature'])
        elif can_batched:
            return self._compute_cross_batch_loss_batched(split_tensors, config)
        else:
            return self._compute_cross_batch_loss_unbatched(split_tensors, config)
