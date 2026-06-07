# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
from transformers import EvalPrediction
from typing import Dict

from swift.utils import get_logger
from .base import EvalMetrics
from .utils import Metric

logger = get_logger()


class RerankerMetrics(EvalMetrics, Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Metric.__init__(self)
        self.add_state('logits', default_factory=list)
        self.add_state('labels', default_factory=list)
        self.add_state('group_sizes', default_factory=list)

    def update(self, logits, labels, group_sizes=None):
        self.logits.append(logits.cpu().numpy())
        self.labels.append(labels.cpu().numpy())
        if group_sizes is not None:
            self.group_sizes.append(group_sizes.cpu().numpy())

    def compute(self):
        predictions = np.concatenate(self.logits)
        labels = np.concatenate(self.labels)
        group_sizes = np.concatenate(self.group_sizes) if self.group_sizes else None
        return self._calculate_metrics(predictions, labels, group_sizes)

    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        label_ids = eval_prediction.label_ids
        group_sizes = None
        if isinstance(label_ids, (tuple, list)):
            labels = label_ids[0]
            if len(label_ids) > 1:
                group_sizes = label_ids[1]
        else:
            labels = label_ids
        return self._calculate_metrics(eval_prediction.predictions, labels, group_sizes)

    @staticmethod
    def _split_query_groups(logits, labels, group_sizes=None):
        if group_sizes is not None:
            group_sizes = np.array(group_sizes).astype(int).flatten()
            total_size = int(group_sizes.sum())
            if total_size == len(labels):
                query_groups = []
                start = 0
                for group_size in group_sizes:
                    if group_size <= 0:
                        continue
                    end = start + group_size
                    query_groups.append((logits[start:end], labels[start:end]))
                    start = end
                return query_groups
            logger.warning('The sum of group_sizes does not match the number of labels. Falling back to label-based '
                           'query boundary inference.')

        positive_indices = np.where(labels == 1)[0]
        if len(positive_indices) == 0:
            return []

        query_groups = []
        for i, pos_idx in enumerate(positive_indices):
            group_start = pos_idx
            if i + 1 < len(positive_indices):
                group_end = positive_indices[i + 1]
            else:
                group_end = len(labels)
            query_groups.append((logits[group_start:group_end], labels[group_start:group_end]))
        return query_groups

    @staticmethod
    def _calculate_classification_metrics(logits, labels):
        preds = (logits > 0).astype(int)
        labels = labels.astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        acc = np.mean(preds == labels) if len(labels) > 0 else 0.0
        return {
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def _calculate_metrics(self, logits, labels, group_sizes=None):
        """
        Calculate MRR and NDCG metrics for reranker.

        This function first groups the data based on query boundaries (identified by
        positive samples), then calculates MRR and NDCG for each group independently,
        and finally returns the mean across all queries.

        Data format:
        - Each query group starts with a positive sample (label=1) followed by negatives (label=0)
        - Example: [1,0,0,1,0,0,0] represents 2 queries: query1=[1,0,0], query2=[1,0,0,0]

        Args:
            logits: Model output scores [batch_size] (numpy array or can be converted to numpy)
            labels: Binary labels (1 for positive, 0 for negative) [batch_size]

        Returns:
            dict: Dictionary containing MRR and NDCG metrics averaged across all queries
        """
        # Convert to numpy if needed
        if hasattr(logits, 'numpy'):
            logits = logits.numpy()
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()

        logits = np.array(logits).flatten()
        labels = np.array(labels).flatten()

        metrics = {}
        if getattr(self.args, 'loss_type', None) == 'pointwise_reranker':
            metrics.update(self._calculate_classification_metrics(logits, labels))

        query_groups = self._split_query_groups(logits, labels, group_sizes)
        metrics['query_count'] = float(len(query_groups))

        # Step 3: Calculate metrics for each query independently
        mrr_scores = []
        ndcg_scores = []
        negative_only_query_count = 0
        skipped_query_count = 0

        for query_idx, (query_logits, query_labels) in enumerate(query_groups):
            if len(query_logits) < 2:
                logger.info(f'Query {query_idx}: Skipped (too small: {len(query_logits)} items)')
                skipped_query_count += 1
                continue

            if np.sum(query_labels == 1) == 0:
                negative_only_query_count += 1
                skipped_query_count += 1
                continue

            # Step 3a: Calculate ranking within this query
            ranking = np.argsort(-query_logits)  # Sort by logits descending

            # Step 3b: Find the rank of the highest-ranked positive document.
            positive_mask = query_labels[ranking] == 1
            pos_rank = np.where(positive_mask)[0][0] + 1  # +1 for 1-based ranking

            # Step 3c: Calculate MRR for this query
            mrr = 1.0 / pos_rank
            mrr_scores.append(mrr)

            # Step 3d: Calculate NDCG for this query
            def calculate_ndcg_single_query(relevance_scores, ranking):
                """Calculate NDCG for a single query"""
                # Calculate DCG (Discounted Cumulative Gain)
                dcg = 0.0
                for rank_pos, doc_idx in enumerate(ranking):
                    relevance = relevance_scores[doc_idx]
                    dcg += (2**relevance - 1) / np.log2(rank_pos + 2)  # rank_pos+2 because log2(1) undefined

                # Calculate IDCG (Ideal DCG)
                ideal_relevance = np.sort(relevance_scores)[::-1]  # Sort relevance descending
                idcg = 0.0
                for rank_pos, relevance in enumerate(ideal_relevance):
                    idcg += (2**relevance - 1) / np.log2(rank_pos + 2)

                # NDCG = DCG / IDCG
                if idcg == 0:
                    return 0.0
                return dcg / idcg

            # Create relevance scores (1 for positive, 0 for negative)
            relevance_scores = query_labels.astype(float)
            ndcg = calculate_ndcg_single_query(relevance_scores, ranking)
            ndcg_scores.append(ndcg)

        # Step 4: Calculate mean metrics across all valid queries
        metrics['ranking_query_count'] = float(len(mrr_scores))
        metrics['negative_only_query_count'] = float(negative_only_query_count)
        metrics['skipped_query_count'] = float(skipped_query_count)
        if len(mrr_scores) == 0:
            logger.warning('No valid queries found for metric calculation')
            metrics.update({'mrr': 0.0, 'ndcg': 0.0})
            return metrics

        mean_mrr = np.mean(mrr_scores)
        mean_ndcg = np.mean(ndcg_scores)

        metrics.update({
            'mrr': mean_mrr,
            'ndcg': mean_ndcg,
        })
        return metrics
