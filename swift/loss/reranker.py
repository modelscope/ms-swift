# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from swift.utils import get_last_valid_indices
from .base import BaseLoss


class RerankerLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        logits = outputs.logits
        logits = logits.squeeze(1)
        labels = labels.to(logits.dtype)
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return loss


class ListwiseRerankerLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs):
        """
        List-wise reranker loss function.

        This loss function groups samples by query based on the pattern where each group
        consists of 1 positive document followed by n negative documents. It treats the
        ranking task as a classification problem within each group, using cross-entropy
        loss to identify the positive document among all candidates.

        Data format expected:
        - labels: [1, 0, 0, 0, 1, 0, 0, ...] where 1 indicates positive, 0 indicates negative
        - Each 1 is followed by its corresponding negative documents until the next 1

        Environment variables for configuration:
        - LISTWISE_RERANKER_TEMPERATURE: Temperature for softmax (default: 1.0)
        - LISTWISE_RERANKER_MIN_GROUP_SIZE: Minimum group size to include (default: 2)

        Args:
            outputs: Model outputs containing logits [batch_size, 1]
            labels: Binary labels (1 for positive, 0 for negative) [batch_size]

        Returns:
            torch.Tensor: Cross entropy loss for ranking classification
        """
        logits = outputs.logits.squeeze(-1)  # [batch_size]
        labels = labels.float()

        # Configuration from environment variables
        temperature = float(os.environ.get('LISTWISE_RERANKER_TEMPERATURE', '1.0'))
        min_group_size = int(os.environ.get('LISTWISE_RERANKER_MIN_GROUP_SIZE', '2'))

        # Find positive sample indices to determine group boundaries
        positive_indices = torch.nonzero(labels == 1, as_tuple=False).squeeze(-1)

        if len(positive_indices) == 0:
            # No positive samples in this batch, return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Ensure positive_indices is 1D
        if positive_indices.dim() == 0:
            positive_indices = positive_indices.unsqueeze(0)

        total_loss = 0.0
        num_groups = 0

        for i, pos_idx in enumerate(positive_indices):
            # Determine group boundaries
            group_start = pos_idx.item()

            # Find the end of current group (start of next group or end of batch)
            if i + 1 < len(positive_indices):
                group_end = positive_indices[i + 1].item()
            else:
                group_end = len(labels)

            # Extract group logits and labels
            group_logits = logits[group_start:group_end]  # [group_size]
            group_labels = labels[group_start:group_end]  # [group_size]

            # Skip groups that are too small
            if len(group_logits) < min_group_size:
                continue

            # Verify that the first sample in the group is positive
            if group_labels[0] != 1:
                continue  # Skip malformed groups

            # Apply temperature scaling for better training dynamics
            scaled_logits = group_logits / temperature

            # The positive document is always at index 0 within the group
            target = torch.tensor(0, dtype=torch.long, device=logits.device)

            # Apply cross-entropy loss: positive document should have highest score
            loss_fct = CrossEntropyLoss()
            group_loss = loss_fct(scaled_logits.unsqueeze(0), target.unsqueeze(0))

            total_loss += group_loss
            num_groups += 1

        if num_groups == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Return average loss across all groups
        return total_loss / num_groups
