# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Dict

import numpy as np
import torch
from transformers import EvalPrediction
from transformers.utils import strtobool

from swift.loss.embedding import _parse_multi_negative_sentences, _parse_pair_sentence
from .base import EvalMetrics


class PairedMetrics(EvalMetrics):

    def compute_metrics(self, eval_prediction: EvalPrediction):
        from sklearn.metrics.pairwise import (paired_cosine_distances, paired_euclidean_distances,
                                              paired_manhattan_distances)
        from scipy.stats import pearsonr, spearmanr
        embeddings = eval_prediction.predictions
        labels = eval_prediction.label_ids
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
            'pearson_euclidean': eval_pearson_euclidean,
            'pearson_manhattan': eval_pearson_manhattan,
            'pearson_dot_product': eval_pearson_dot,
            'spearman_cosine': eval_spearman_cosine,
            'spearman_euclidean': eval_spearman_euclidean,
            'spearman_manhattan': eval_spearman_manhattan,
            'spearman_dot_product': eval_spearman_dot,
        }


class InfonceMetrics(EvalMetrics):

    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        embeddings = eval_prediction.predictions
        labels = eval_prediction.label_ids
        hard_negatives = os.environ.get('INFONCE_HARD_NEGATIVES', None)
        use_batch = strtobool(os.environ.get('INFONCE_USE_BATCH', 'True'))
        if hard_negatives is not None:
            hard_negatives = int(hard_negatives)
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
                for row, col in enumerate(
                        range(0, sentences.shape[0] * (sentences.shape[1] - 1), sentences.shape[1] - 1)):
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
