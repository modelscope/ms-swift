# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import os
import numpy as np
import torch
from typing import Callable, List, Optional

from twinkle.data_format import InputFeature
from twinkle.processor import InputProcessor as TwinkleInputProcessor


class InputProcessor(TwinkleInputProcessor):
    """Swift extension of twinkle InputProcessor.

    Adds template collate_mm_data hook for model-specific VLM collation, the embedding / reranker
    group-row flattening, and post-forward gather helpers for the transformers framework.
    """

    def __init__(self,
                 *,
                 collate_fn: Optional[Callable] = None,
                 cp_partition_mode: str = 'zigzag',
                 task_type: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._external_collate_fn = collate_fn
        self._cp_partition_mode = cp_partition_mode
        self._template = None
        #: The task_type the template encoded with. embedding / reranker rows are GROUP-shaped (one
        #: row holds an anchor + candidates), so prepare_inputs flattens them -- see _flatten_group_rows.
        self._task_type = task_type
        #: Persistent RNG for the reranker subsampling, seeded to match legacy's RandomState(42).
        self._reranker_rng = None

    def _get_packed_seq_params(self, position_ids):
        packed = super()._get_packed_seq_params(position_ids)
        if hasattr(packed, 'cp_partition_mode'):
            packed.cp_partition_mode = self._cp_partition_mode
        return packed

    # swift-only bookkeeping fields that must not reach the collate stage (everything else passes
    # through). A blacklist, not a whitelist; label semantics are owned by the Template's encode.
    _DROP_KEYS = frozenset({'lengths', 'length', '_labels_shifted'})

    def prepare_inputs(self, inputs, **kwargs):
        """Drop swift-only bookkeeping fields, flatten group rows, then delegate to twinkle.

        A packed batch arrives as list[list[dict]] (PackingDataset + identity collate); flatten HERE,
        in ray mode the driver's slice_dp splits the batch element-wise, so flattening earlier would
        scatter one packed group over DP ranks. position_ids are injected per row -- the
        multiple-zero-reset form twinkle's collate detects as packed under padding_free.
        """
        if isinstance(inputs, dict):
            inputs = [inputs]
        # GROUP-shaped rows (anchor + candidates) expand into flat per-sequence rows FIRST, so the
        # packing flatten / position_ids / to_tensor below see the single-sequence layout.
        inputs = self._flatten_group_rows(inputs)
        if inputs and isinstance(inputs[0], list):
            inputs = [row for item in inputs for row in item]
        cleaned = [{k: v for k, v in feat.items() if k not in self._DROP_KEYS} for feat in inputs]
        for feat in cleaned:
            if feat.get('position_ids') is None and feat.get('input_ids') is not None:
                feat['position_ids'] = list(range(len(feat['input_ids'])))
        return super().prepare_inputs(cleaned, **kwargs)

    # legacy's encode stores an embedding / reranker example as ONE GROUP-shaped row; twinkle wants
    # flat per-sequence rows with a per-ROW mask label (1.0 at each group's anchor).
    _EMBEDDING_TASK_TYPES = frozenset({'embedding'})
    _RERANKER_TASK_TYPES = frozenset({'reranker', 'generative_reranker'})

    def _flatten_group_rows(self, inputs: List[InputFeature]) -> List[InputFeature]:
        """Expand embedding / reranker group rows into flat per-sequence rows; pass others through."""
        if self._task_type in self._EMBEDDING_TASK_TYPES:
            return self._flatten_embedding_rows(inputs)
        if self._task_type in self._RERANKER_TASK_TYPES:
            return self._flatten_reranker_rows(inputs)
        return inputs

    def _flatten_embedding_rows(self, rows: List[InputFeature]) -> List[InputFeature]:
        """Flatten legacy's ``anchor_* / positive_* / negative_*`` embedding encode into sequences.

        The flat order is anchor, positive, then each negative -- the group layout InfonceLoss
        expects (element 0 is the query, the rest are documents). Each flat row gets a scalar mask
        label: 1.0 on the anchor (group start), 0.0 on the positive and negatives.
        """
        flat: List[InputFeature] = []
        for row in rows:
            row = dict(row)
            # negative_<field> holds a list over negatives; split it into negative{i}_<field> scalars
            # so every candidate is addressable by a single prefix, mirroring legacy's collator.
            num_negatives = 0
            for key in [k for k in row if k.startswith('negative_')]:
                values = row.pop(key)
                suffix = key[len('negative_'):]
                num_negatives = len(values)
                for i, value in enumerate(values):
                    row[f'negative{i}_{suffix}'] = value
            prefixes = ['anchor_', 'positive_'] + [f'negative{i}_' for i in range(num_negatives)]
            group = []
            for prefix in prefixes:
                stripped = {k[len(prefix):]: v for k, v in row.items() if k.startswith(prefix)}
                if stripped:
                    group.append(stripped)
            for i, seq in enumerate(group):
                seq['labels'] = 1.0 if i == 0 else 0.0
                flat.append(seq)
        return flat

    def _flatten_reranker_rows(self, rows: List[InputFeature]) -> List[InputFeature]:
        """Flatten legacy's list-valued reranker encode into per-candidate rows.

        Each row stores every field as a LIST over candidates with ``labels = [1]*P + [0]*N``.
        Legacy caps each query at MAX_POSITIVE/MAX_NEGATIVE_SAMPLES, drawing which survive from a
        persistent ``RandomState(42)``; dev reproduces that exactly so both train on the same subset.
        """
        if self._reranker_rng is None:
            self._reranker_rng = np.random.RandomState(42)
        max_positive = int(os.environ.get('MAX_POSITIVE_SAMPLES', 1))
        max_negative = int(os.environ.get('MAX_NEGATIVE_SAMPLES', 7))
        flat: List[InputFeature] = []
        for row in rows:
            row = dict(row)
            labels = list(row.pop('labels', []))
            positive_num = int(sum(labels))
            negative_num = len(labels) - positive_num
            keep_positive = min(positive_num, max_positive)
            keep_negative = min(negative_num, max_negative)
            list_keys = [k for k, v in row.items() if isinstance(v, list)]

            def candidate(index, label):
                seq = {k: row[k][index] for k in list_keys if row[k][index] is not None}
                seq['labels'] = label
                return seq

            for i in self._reranker_rng.choice(positive_num, keep_positive, replace=False):
                flat.append(candidate(i, 1))
                for j in self._reranker_rng.choice(negative_num, keep_negative, replace=False):
                    flat.append(candidate(j + positive_num, 0))
        return flat

    # Optional sequence-level fields (e.g. mm_token_type_ids) absent on some rows of a mixed batch
    # are padded by twinkle's collate (_fill_optional_sequence_fields inside _collate_macro_batch).
    def collate_fn(self, inputs: List[InputFeature], **kwargs) -> List[InputFeature]:
        """Override: add template collate_mm_data hook after default collation."""
        # Priority 1: external collate_fn (explicit override)
        if self._external_collate_fn is not None:
            result = self._external_collate_fn(inputs)
            if isinstance(result, dict):
                return [result]
            return result

        # Default collation from parent
        collated = super().collate_fn(inputs, **kwargs)

        # Priority 2: template.collate_mm_data hook (model-specific mm collation)
        if self._template is not None and hasattr(self._template, 'collate_mm_data'):
            for i, batch in enumerate(collated):
                mm_override = self._template.collate_mm_data(batch)
                if mm_override is not None:
                    batch.update(mm_override)
                if hasattr(self._template, 'post_collate'):
                    collated[i] = self._template.post_collate(batch)

        return collated

    def postprocess_tensor_gather(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.device_mesh is None:
            return tensor
        sp_size = getattr(self.device_mesh, 'sp_world_size', 1)
        if sp_size > 1 and self.framework == 'transformers':
            raise NotImplementedError(f'transformers-backend sequence parallelism (sp_world_size={sp_size}) is not '
                                      'implemented in this phase: the output all-gather here has never been validated '
                                      'against a single-device reference, and a wrong gather would corrupt logits '
                                      'silently. It will be enabled together with the SP dataloader path.')
        return tensor
