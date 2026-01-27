# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict

import numpy as np
from transformers import EvalPrediction

from .base import EvalMetrics
from .utils import Metric


class RerankerMetrics(EvalMetrics, Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Metric.__init__(self)
        self.add_state('logits', default_factory=list)
        self.add_state('labels', default_factory=list)

    def update(self, logits, labels):
        self.logits.append(logits.cpu().numpy())
        self.labels.append(labels.cpu().numpy())

    def compute(self):
        predictions = np.concatenate(self.logits)
        labels = np.concatenate(self.labels)
        return self._calculate_metrics(predictions, labels)

    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        return self._calculate_metrics(eval_prediction.predictions, eval_prediction.label_ids)

    def _calculate_metrics(self, logits, labels):
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

        # Step 1: Find all positive sample indices (query boundaries)
        positive_indices = np.where(labels == 1)[0]

        if len(positive_indices) == 0:
            return {'mrr': 0.0, 'ndcg': 0.0}

        # Step 2: Split into groups (queries)
        query_groups = []
        for i, pos_idx in enumerate(positive_indices):
            # Each group starts at a positive index
            group_start = pos_idx

            # Group ends at the next positive index or end of data
            if i + 1 < len(positive_indices):
                group_end = positive_indices[i + 1]
            else:
                group_end = len(labels)

            # Extract this query's data
            query_logits = logits[group_start:group_end]
            query_labels = labels[group_start:group_end]

            query_groups.append((query_logits, query_labels))

        # Step 3: Calculate metrics for each query independently
        mrr_scores = []
        ndcg_scores = []

        for query_idx, (query_logits, query_labels) in enumerate(query_groups):
            # Skip groups that are too small (need at least 1 positive + 1 negative)
            if len(query_logits) < 2:
                print(f'Query {query_idx}: Skipped (too small: {len(query_logits)} items)')
                continue

            # Verify that the first sample is positive (data format validation)
            if query_labels[0] != 1:
                print(f'Query {query_idx}: Skipped (first sample not positive)')
                continue

            # Step 3a: Calculate ranking within this query
            ranking = np.argsort(-query_logits)  # Sort by logits descending

            # Step 3b: Find position of positive document (should be at index 0 in query)
            pos_rank = np.where(ranking == 0)[0][0] + 1  # +1 for 1-based ranking

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
        if len(mrr_scores) == 0:
            print('No valid queries found for metric calculation')
            return {'mrr': 0.0, 'ndcg': 0.0}

        mean_mrr = np.mean(mrr_scores)
        mean_ndcg = np.mean(ndcg_scores)

        return {
            'mrr': mean_mrr,
            'ndcg': mean_ndcg,
        }
