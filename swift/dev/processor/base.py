from __future__ import annotations

import torch
from twinkle.processor import InputProcessor as TwinkleInputProcessor
from typing import Callable, List, Optional

from swift.dev.data_format import InputFeature


class InputProcessor(TwinkleInputProcessor):
    """Swift extension of twinkle InputProcessor.

    Adds template collate_mm_data hook for model-specific VLM collation,
    and post-forward gather helpers for the transformers framework.
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
        #: The task_type the template encoded with. embedding / reranker / generative_reranker rows are
        #: GROUP-shaped (one row holds an anchor + candidates, not a single sequence), so prepare_inputs
        #: must flatten them into per-sequence rows before twinkle's collate -- see _flatten_group_rows.
        self._task_type = task_type
        #: Persistent RNG for the reranker positive/negative subsampling, seeded to match legacy's
        #: _reranker_data_collator (np.random.RandomState(42)) so dev trains on the same candidate subset.
        self._reranker_rng = None

    def _get_packed_seq_params(self, position_ids):
        packed = super()._get_packed_seq_params(position_ids)
        if hasattr(packed, 'cp_partition_mode'):
            packed.cp_partition_mode = self._cp_partition_mode
        return packed

    # swift-only bookkeeping fields that must not reach the collate stage. Everything else passes
    # through untouched: the dev Template's _encode already constructs exactly the forward kwargs
    # each model needs (the legacy convention), so a WHITELIST would have to be re-checked against
    # twinkle's padding_map and every model's forward signature on each change. A blacklist only
    # grows when swift adds a bookkeeping field of its own -- which is under our control.
    #
    # Why these two must go:
    #   lengths          -- written by template.encode(return_length=True) for packing /
    #                       group_by_length (dataset/utils.py:122-129). twinkle's collate pads every
    #                       text key via padding_map[key] and would KeyError on it.
    #   _labels_shifted  -- the dev Template / RolloutEngine marker recording "labels already
    #                       next-token shifted" (contract 14). It guards shift idempotency at encode
    #                       time (template.py:64) and is queryable there; it is not a model input.
    #
    # NOTE: label semantics (next-token shift) are owned by the dev Template's encode
    # ("whoever encodes, shifts", matching twinkle's _roll_labels). The InputProcessor MUST
    # NOT mutate labels here — doing so previously created a fake double-shift guard.
    _DROP_KEYS = frozenset({'lengths', 'length', '_labels_shifted'})

    def prepare_inputs(self, inputs, **kwargs):
        """Drop swift-only bookkeeping fields before the collate stage.

        Only the swift bookkeeping keys in _DROP_KEYS are removed; every other field produced by
        the dev Template's encode is a deliberate model input and passes through.

        `length` is dropped along with `lengths` even though twinkle's padding_map accepts it: the
        only processor-side consumer is align_routed_experts, which now derives the sequence length
        from input_ids (a cached length would be stale after pad_cp extends the sequence anyway).

        Packing: PackingDataset yields a LIST of rows per item (packing.py:130 /
        IterablePackingDataset.__iter__), and the dataloader's identity collate passes it through
        unchanged, so a packed batch arrives here as list[list[dict]]. Flatten it first -- exactly what
        legacy does in Template.data_collator (template/base.py:1668-1669 `batch = sum(batch, start=[])`).
        Without this the dict-comprehension below hits AttributeError: 'list' has no 'items'.
        Flattening belongs HERE rather than in the dataloader: in ray mode the driver's slice_dp
        splits the batch element-wise across DP ranks, so flattening earlier would scatter one
        packed group over different DP ranks, and batch_size would silently change meaning from
        "packed sequences" to "rows".

        position_ids: the encode path does not emit them, but they are what makes packing work --
        each row gets range(len(input_ids)), and twinkle's _collate_macro_batch concatenates the
        rows under padding_free (processor/base.py:697-711), so the per-row [0,1,2] + [0,1] become
        [0,1,2,0,1]: the multiple-zero-reset form that _is_packed_position_ids detects and
        _get_packed_seq_params turns into cu_seqlens. Injected for BOTH frameworks: legacy's
        packing_row is backend-agnostic, and without position_ids a transformers-backend packed
        batch would be treated as one long sequence by flash-attn (cross-sample attention leak).
        """
        if isinstance(inputs, dict):
            inputs = [inputs]
        # embedding / reranker rows are GROUP-shaped (anchor + candidates in one row); expand them into
        # flat per-sequence rows FIRST, so the packing flatten / position_ids / to_tensor below see the
        # same single-sequence layout causal_lm and seq_cls already arrive in.
        inputs = self._flatten_group_rows(inputs)
        if inputs and isinstance(inputs[0], list):
            inputs = [row for item in inputs for row in item]
        cleaned = [{k: v for k, v in feat.items() if k not in self._DROP_KEYS} for feat in inputs]
        for feat in cleaned:
            if feat.get('position_ids') is None and feat.get('input_ids') is not None:
                feat['position_ids'] = list(range(len(feat['input_ids'])))
        self._fill_optional_sequence_fields(cleaned)
        return super().prepare_inputs(cleaned, **kwargs)

    # --- embedding / reranker group flatten -------------------------------------------------
    # The dev Template reuses legacy's encode, which stores an embedding / reranker example as ONE
    # GROUP-shaped row rather than one row per sequence. twinkle's collate (and its InfoNCE / reranker
    # losses) instead want flat per-sequence rows plus a per-row label, so the group is expanded here --
    # the dev counterpart of legacy's _embedding_data_collator / _reranker_data_collator, minus their
    # padding (twinkle pads downstream). The label FORM differs from legacy on purpose: twinkle's
    # InfonceLoss takes a per-ROW mask (1.0 at each group's anchor, len(labels) == len(sentences)) and
    # finds groups via nonzero(labels), whereas legacy emits one fewer label per group and re-inserts
    # the anchor offset inside its own _parse_multi_negative_sentences.
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

        Each row carries the anchor and positive as single prefixed fields and the hard negatives as
        LIST-valued ``negative_<field>`` (one entry per negative). The flat order is anchor, positive,
        then each negative -- the group layout InfonceLoss expects (element 0 is the query, the rest
        are documents). Each flat row gets a scalar mask label: 1.0 on the anchor (group start), 0.0 on
        the positive and negatives.
        """
        flat: List[InputFeature] = []
        for row in rows:
            row = dict(row)
            # negative_<field> holds a list over negatives; split it into negative{i}_<field> scalars so
            # every candidate is addressable by a single prefix, mirroring legacy's collator.
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

        Each row stores every field as a LIST over candidates (positives first, then negatives) with
        ``labels = [1]*P + [0]*N``. Legacy caps each query at MAX_POSITIVE_SAMPLES positives and, per
        positive, MAX_NEGATIVE_SAMPLES negatives, drawing which survive from a persistent
        ``RandomState(42)``; dev reproduces that exactly so both train on the same subset. Each selected
        candidate becomes a flat row carrying its scalar relevance label (1 relevant / 0 not).
        """
        import os

        import numpy as np
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

    # Sequence-level fields that only multimodal samples carry (e.g. mm_token_type_ids): the dev
    # Template emits them for image/video rows and omits them for pure-text rows, exactly like
    # legacy (qwen.py `if requires_mm_token_type_ids and any(mm_mask)`). twinkle's
    # _collate_macro_batch takes the UNION of all sample keys and then indexes every sample with
    # `item[key]`, so a mixed image+text batch KeyErrors on the text rows. legacy instead pads such
    # fields up to the full batch, filling absent rows with the padding value (measured: a 2-image /
    # 2-text batch collates mm_token_type_ids to shape (4, L) with the text rows all-zero). We
    # reproduce that here: fill each absent row with `padding_map[key]` repeated to input_ids length.
    #
    # Scope is derived, not hardcoded to mm_token_type_ids: any padding_map key that is NOT a
    # VLM_CONCAT_FIELD (those are concatenated on the patch axis, never padded to batch) and is
    # missing on some-but-not-all rows. pixel_values / image_grid_thw stay untouched -- twinkle's
    # VLM_CONCAT_FIELDS path already concatenates them, matching legacy _data_collator_mm_data.
    def _fill_optional_sequence_fields(self, batch: List[InputFeature]) -> None:
        if len(batch) < 2:
            return
        seq_fields = set(self.padding_map) - set(self.VLM_CONCAT_FIELDS) - self._DROP_KEYS
        present = {k for feat in batch for k in feat if k in seq_fields}
        for key in present:
            missing = [feat for feat in batch if feat.get(key) is None]
            if not missing or len(missing) == len(batch):
                continue  # all-present (no gap) or all-absent (nothing to align to) -> leave as is
            pad_value = self.padding_map[key]
            for feat in missing:
                input_ids = feat.get('input_ids')
                if input_ids is None:
                    continue
                feat[key] = torch.full((len(input_ids), ), pad_value, dtype=torch.long)

    # Override twinkle's collate_fn stage to inject template hook
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
