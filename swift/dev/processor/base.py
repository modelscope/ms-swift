from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

import torch
from twinkle import Platform
from twinkle.processor import InputProcessor as TwinkleInputProcessor

from swift.dev.data_format import VISION_DUMMY_FLAG, InputFeature
from swift.dev.template.conventions import is_bookkeeping_key, model_payload_keys

# Private per-row slot carrying a row's model-specific payload past twinkle's text stages to collate_fn.
_MEDIA_SLOT = "_media"


def processor_kwargs(template: Any) -> Dict[str, Any]:
    """The InputProcessor kwargs a model bound to ``template`` needs (``model.set_processor(..., **kw)``)."""
    kwargs: Dict[str, Any] = {}
    if template is None or not hasattr(template, "task_type"):
        return kwargs
    kwargs["template"] = template
    kwargs["task_type"] = template.task_type
    kwargs["multimodal_graph_adapter"] = template.multimodal_graph_adapter
    pad_token_id = getattr(template.tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


class InputProcessor(TwinkleInputProcessor):
    """twinkle's text processor plus the template family's existing media hook.

    twinkle owns padding, padding-free concatenation, packing, CP and backend output layout. The
    template only prepares each row (notably mRoPE positions) and batches the model-specific payload
    through ``_data_collator_mm_data``. There is no second family registry or media-rule language.
    Label semantics are never touched here (convention A: whoever encodes, shifts).
    """

    # Zero-padded text fields absent from twinkle's CP field lists (not media payloads).
    _EXTRA_SEQUENCE_FIELDS = ("replay_mask", "token_type_ids")
    _MODEL_CONTROL_FIELDS = (VISION_DUMMY_FLAG,)

    def __init__(
        self,
        *,
        template: Any = None,
        pad_token_id: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        cp_partition_mode: str = "zigzag",
        task_type: Optional[str] = None,
        multimodal_graph_adapter: Any = None,
        uniform_multimodal_graph: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.template = template
        # Gemma's token types share completion_mask's zero-padding rule. Keep upstream defaults.
        self.padding_map = dict(self.padding_map)
        self.padding_map.setdefault("token_type_ids", self.padding_map["completion_mask"])
        if pad_token_id is not None:
            # Legacy pads input_ids with the tokenizer's pad token, twinkle with 0.
            self.padding_map["input_ids"] = pad_token_id
        self._external_collate_fn = collate_fn
        self._cp_partition_mode = cp_partition_mode
        # Group-shaped embedding/reranker rows must be flattened before text or media collation.
        self._task_type = task_type if task_type is not None else (template.task_type if template is not None else None)
        self._multimodal_graph_adapter = multimodal_graph_adapter
        self._uniform_multimodal_graph = uniform_multimodal_graph
        if (
            uniform_multimodal_graph
            and template is not None
            and template.is_multimodal
            and multimodal_graph_adapter is None
        ):
            raise NotImplementedError(
                f"{type(template).__name__} has no model-layer multimodal graph adapter for this distributed strategy."
            )
        self._reranker_rng = None

    def prepare_inputs(self, inputs, **kwargs):
        """Normalise rows: flatten packed groups, drop bookkeeping, split media off, derive positions.

        Packing: PackingDataset yields a LIST of rows per item and the dataloader passes it through, so
        a packed batch arrives as ``list[list[dict]]``. It is flattened HERE rather than in the dataloader
        because ray's slice_dp splits the batch element-wise across DP ranks -- flattening earlier would
        scatter one packed group over several ranks.

        position_ids are what makes packing work: per-row positions restart at 0, twinkle concatenates
        them under padding_free, and the resets become cu_seqlens -- for BOTH backends (without them a
        transformers packed batch would attend across samples). ``None`` values are dropped (legacy
        emits ``loss_scale=None`` for binary loss scales).
        """
        if isinstance(inputs, dict):
            inputs = [inputs]
        # Packed groups and split encodings can coexist with unsplit dict rows in the same batch.
        inputs = [row for item in inputs for row in (item if isinstance(item, list) else [item])]
        # Embedding / reranker groups then expand into the same single-sequence row layout.
        inputs = self._flatten_group_rows(inputs)
        rows = []
        for feat in inputs:
            row = {k: v for k, v in feat.items() if v is not None and not is_bookkeeping_key(k)}
            if self.template is None:
                if row.get("position_ids") is None and row.get("input_ids") is not None:
                    row["position_ids"] = list(range(len(row["input_ids"])))
                media = {}
            else:
                self.template.prepare_row(row)
                media = self.template.pop_media(row)
            if media:
                row[_MEDIA_SLOT] = media
            rows.append(row)
        rows = super().prepare_inputs(rows, **kwargs)
        self._fill_optional_sequence_fields(rows)
        return rows

    # --- embedding / reranker group flatten -------------------------------------------------
    # The dev Template reuses legacy's encode, which stores an embedding / reranker example as ONE
    # GROUP-shaped row rather than one row per sequence. twinkle's collate (and its InfoNCE / reranker
    # losses) instead want flat per-sequence rows plus a per-row label, so the group is expanded here --
    # the dev counterpart of legacy's _embedding_data_collator / _reranker_data_collator, minus their
    # padding (twinkle pads downstream). The label FORM differs from legacy on purpose: twinkle's
    # InfonceLoss takes a per-ROW mask (1.0 at each group's anchor, len(labels) == len(sentences)) and
    # finds groups via nonzero(labels), whereas legacy emits one fewer label per group and re-inserts
    # the anchor offset inside its own _parse_multi_negative_sentences.
    _EMBEDDING_TASK_TYPES = frozenset({"embedding"})
    _RERANKER_TASK_TYPES = frozenset({"reranker", "generative_reranker"})

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
            for key in [k for k in row if k.startswith("negative_")]:
                values = row.pop(key)
                suffix = key[len("negative_") :]
                num_negatives = len(values)
                for i, value in enumerate(values):
                    row[f"negative{i}_{suffix}"] = value
            prefixes = ["anchor_", "positive_"] + [f"negative{i}_" for i in range(num_negatives)]
            group = []
            for prefix in prefixes:
                stripped = {k[len(prefix) :]: v for k, v in row.items() if k.startswith(prefix)}
                if stripped:
                    group.append(stripped)
            for i, seq in enumerate(group):
                seq["labels"] = 1.0 if i == 0 else 0.0
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
        max_positive = int(os.environ.get("MAX_POSITIVE_SAMPLES", 1))
        max_negative = int(os.environ.get("MAX_NEGATIVE_SAMPLES", 7))
        flat: List[InputFeature] = []
        for row in rows:
            row = dict(row)
            labels = list(row.pop("labels", []))
            positive_num = int(sum(labels))
            negative_num = len(labels) - positive_num
            keep_positive = min(positive_num, max_positive)
            keep_negative = min(negative_num, max_negative)
            list_keys = [k for k, v in row.items() if isinstance(v, list)]

            def candidate(index, label, row=row, list_keys=list_keys):
                seq = {k: row[k][index] for k in list_keys if row[k][index] is not None}
                seq["labels"] = label
                return seq

            for i in self._reranker_rng.choice(positive_num, keep_positive, replace=False):
                flat.append(candidate(i, 1))
                for j in self._reranker_rng.choice(negative_num, keep_negative, replace=False):
                    flat.append(candidate(j + positive_num, 0))
        return flat

    def collate_fn(
        self, inputs: List[InputFeature], micro_batch_size: Optional[int] = None, variable_seq_lengths=False, **kwargs
    ) -> List[InputFeature]:
        """twinkle's text collate plus each micro-batch's payload batched by the template family."""
        if self._external_collate_fn is not None:
            result = self._external_collate_fn(inputs)
            return [result] if isinstance(result, dict) else result
        media = [row.pop(_MEDIA_SLOT, None) or {} for row in inputs]
        outputs = super().collate_fn(
            inputs, micro_batch_size=micro_batch_size, variable_seq_lengths=variable_seq_lengths, **kwargs
        )
        groups = self._micro_batch_groups(len(inputs), micro_batch_size)
        if len(groups) != len(outputs):
            raise RuntimeError(
                f"collate produced {len(outputs)} micro-batches for {len(groups)} media groups; the "
                "grouping below no longer mirrors twinkle's collate_fn."
            )
        if self.template is None:
            return outputs
        device = Platform.get_local_device()
        for output, indices in zip(outputs, groups):
            batched = self.template.collate_media([media[i] for i in indices])
            for key, value in batched.items():
                output[key] = value.to(device) if isinstance(value, torch.Tensor) else value
            finalize_batch = getattr(self.template, "finalize_batch", None)
            if finalize_batch is not None:
                finalized = finalize_batch(output)
                if finalized is not output:
                    output.clear()
                    output.update(finalized)
            if self._uniform_multimodal_graph and self._multimodal_graph_adapter is not None:
                if self.device_mesh is not None and getattr(self.device_mesh, "cp_world_size", 1) > 1:
                    raise NotImplementedError("Multimodal graph protection must run before CP partitioning.")
                self._multimodal_graph_adapter.inject_missing_paths(output)
        return outputs

    def _micro_batch_groups(self, n: int, micro_batch_size: Optional[int]) -> List[List[int]]:
        """Row indices of each collated output, mirroring twinkle's ``collate_fn`` branches."""
        if n == 1 and self.framework != "megatron":
            return [[0]]
        if micro_batch_size is None:
            return [list(range(n))]
        return [list(range(i, min(i + micro_batch_size, n))) for i in range(0, n, micro_batch_size)]

    def to_transformers_dict(self, inputs: List[InputFeature], **kwargs) -> List[InputFeature]:
        """twinkle's text whitelist plus extra sequence fields and the family's media payload."""
        payloads = (
            [{key: inp[key] for key in model_payload_keys(inp)} for inp in inputs]
            if self.template is not None
            else [{} for _ in inputs]
        )
        # Family payloads may be ragged nested tensors (MiniCPM); twinkle's text conversion
        # must not coerce them into numpy arrays before we restore the family's batched result.
        text_inputs = [
            {key: value for key, value in source.items() if key not in payload}
            for source, payload in zip(inputs, payloads)
        ]
        results = super().to_transformers_dict(text_inputs, **kwargs)
        for source, result, payload in zip(inputs, results, payloads):
            for key in self._MODEL_CONTROL_FIELDS:
                if key in source:
                    result[key] = source[key]
            for key in self._EXTRA_SEQUENCE_FIELDS:
                if key in source:
                    result.setdefault(key, source[key])
            result.update(payload)
        return results

    def _process_extra_sequence_fields(self, inputs, process, **kwargs):
        """Reuse completion-mask CP padding/splitting for identically aligned zero-padded fields."""
        slots = [
            (i, key) for i, row in enumerate(inputs) for key in self._EXTRA_SEQUENCE_FIELDS if row.get(key) is not None
        ]
        masks = [
            {
                "completion_mask": inputs[i].pop(key),
                "position_ids": inputs[i].get("position_ids"),
                "packed_seq_params": inputs[i].get("packed_seq_params"),
            }
            for i, key in slots
        ]
        # Capture original positions/cu_seqlens before the full transform replaces them. Pop the
        # extra fields so each is transformed exactly once, even if upstream later handles it too.
        transformed = process(masks, **kwargs) if masks else []
        results = process(inputs, **kwargs)
        for (index, key), mask in zip(slots, transformed):
            results[index][key] = mask["completion_mask"]
        return results

    def pad_cp(self, inputs, **kwargs):
        return self._process_extra_sequence_fields(inputs, super().pad_cp, **kwargs)

    def split_cp(self, inputs, **kwargs):
        return self._process_extra_sequence_fields(inputs, super().split_cp, **kwargs)

    def postprocess_tensor_gather(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.device_mesh is None:
            return tensor
        sp_size = getattr(self.device_mesh, "sp_world_size", 1)
        if sp_size > 1 and self.framework == "transformers":
            raise NotImplementedError(
                f"transformers-backend sequence parallelism (sp_world_size={sp_size}) is not "
                "implemented in this phase: the output all-gather here has never been validated "
                "against a single-device reference, and a wrong gather would corrupt logits "
                "silently. It will be enabled together with the SP dataloader path."
            )
        return tensor
