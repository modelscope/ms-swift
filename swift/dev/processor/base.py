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

    def __init__(self, *, collate_fn: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self._external_collate_fn = collate_fn
        self._template = None

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
        IterablePackingDataset.__iter__), and identity_collate passes it through unchanged, so a
        packed batch arrives here as list[list[dict]]. Flatten it first -- exactly what legacy does
        in Template.data_collator (template/base.py:1668-1669 `batch = sum(batch, start=[])`).
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
        if inputs and isinstance(inputs[0], list):
            inputs = [row for item in inputs for row in item]
        cleaned = [{k: v for k, v in feat.items() if k not in self._DROP_KEYS} for feat in inputs]
        for feat in cleaned:
            if feat.get('position_ids') is None and feat.get('input_ids') is not None:
                feat['position_ids'] = list(range(len(feat['input_ids'])))
        return super().prepare_inputs(cleaned, **kwargs)

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
