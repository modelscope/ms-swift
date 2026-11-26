from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional

import torch
from twinkle import Platform
from twinkle.processor import InputProcessor as TwinkleInputProcessor

from swift.dev.data_format import VISION_DUMMY_FLAG, InputFeature
from swift.dev.template.conventions import is_bookkeeping_key, model_payload_keys
from swift.dev.utils.naming import mixed_precision_for, resolve_megatron_attn_backend

# Private per-row slot carrying a row's model-specific payload past twinkle's text stages to collate_fn.
_MEDIA_SLOT = "_media"


def get_sequence_padding_to(tp_size: int, sequence_parallel: bool, cp_size: int, attention_backend_name: Optional[str],
                            fp8_format=None, fp4_format=None, fp8_recipe: Optional[str] = None) -> Optional[int]:
    """Sequence-length alignment the kernels require (legacy ``get_padding_to``, args -> params).

    Single source of truth shared by the training data path (``InputProcessor``'s ``seq_align_to``)
    and the weight converter's repartition collation. TE's fp8 GEMM asserts the token dimension is
    a multiple of 8 and Megatron's fp8 alignment is 16 (delayed/tensorwise), 32 (mxfp8) or 128
    (blockwise); every micro-batch's token count must reach it.
    """
    padding_to = None
    if tp_size > 1 and sequence_parallel:
        padding_to = tp_size
    if cp_size > 1:
        padding_to = (padding_to or 1) * cp_size
    origin_padding_to = padding_to
    if fp8_recipe == 'blockwise':
        padding_to = (padding_to or 1) * 128
    elif fp8_recipe == 'mxfp8':
        # MXFP8 uses a block size of 32. Under sequence parallel, the sequence is split across
        # TP ranks, so each per-rank shard (seq_len / TP) must itself be divisible by 32. Pad the
        # total length to TP * 32 to guarantee this.
        padding_to = (padding_to or 1) * 32
    elif fp8_format is not None or fp4_format is not None:
        padding_to = (padding_to or 1) * 16
    if attention_backend_name == 'fused':
        padding_to = max(padding_to or 1, ((origin_padding_to) or 1) * 64)
    return padding_to


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


def processor_cp_size(distributed_config) -> int:
    """The policy's context-parallel width for token padding: explicit knob or parallel_spec."""
    cp_size = distributed_config.context_parallel_size
    if distributed_config.parallel_spec:
        from twinkle import DeviceMesh
        cp_size = max(cp_size, DeviceMesh.parse_spec(distributed_config.parallel_spec).get('cp_size', 1) or 1)
    return cp_size


def fp8_seq_align_option(model_config, distributed_config) -> Optional[int]:
    """The processor's fp8/fp4 token-dimension alignment (legacy ``get_padding_to`` on the data path).

    None when fp8/fp4 is off or the backend is not megatron, so bf16 runs keep twinkle's padding
    untouched.
    """
    if distributed_config.backend != 'megatron':
        return None
    if model_config.fp8_format is None and model_config.fp4_format is None:
        return None
    attn_backend = resolve_megatron_attn_backend(
        model_config.attn_impl, mixed_precision=mixed_precision_for(model_config.torch_dtype))
    seq_align_to = get_sequence_padding_to(
        distributed_config.tensor_model_parallel_size,
        distributed_config.sequence_parallel,
        processor_cp_size(distributed_config),
        attn_backend.name,
        fp8_format=model_config.fp8_format,
        fp4_format=model_config.fp4_format,
        fp8_recipe=model_config.fp8_recipe)
    return seq_align_to if seq_align_to and seq_align_to > 1 else None


def input_processor_kwargs(template, template_config, distributed_config, model_config) -> Dict[str, Any]:
    """Every ``set_processor(InputProcessor, ...)`` option of a model that trains on this process's rows.

    ``seq_align_to`` pads each micro-batch's token columns to the kernel alignment fp8/fp4 and
    sequence parallel require; every local recipe and auxiliary model goes through here so none of
    them misses it.
    """
    kwargs = processor_kwargs(template)
    kwargs["padding_free"] = template_config.padding_free
    # The collator must pad on the same side as the Template (builders/template.py forwards
    # padding_side to it), or a training-time --padding_side left reaches the Template only.
    kwargs["padding_side"] = template_config.padding_side
    kwargs["cp_partition_mode"] = distributed_config.cp_partition_mode
    kwargs["seq_align_to"] = fp8_seq_align_option(model_config, distributed_config)
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
    # Token columns right-padded to the fp8 sequence alignment: twinkle's CP field list plus the
    # dev extra zero-padded fields. Pad values come from ``padding_map`` (labels -100, loss_scale 0,
    # position_ids -1 keeps the pad inside the last cu_seqlens segment -- all loss-neutral).
    _SEQ_ALIGN_FIELDS = ("input_ids", "position_ids", "attention_mask", "labels", "completion_mask", "loss_scale",
                         "mm_token_type_ids") + _EXTRA_SEQUENCE_FIELDS

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
        seq_align_to: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.template = template
        # fp8/fp4 token-dimension alignment (legacy get_padding_to); None/1 disables the pad.
        self._seq_align_to = int(seq_align_to) if seq_align_to and seq_align_to > 1 else None
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
            metadata = set(feat.get('_trajectory_columns', ()))
            row = {k: v for k, v in feat.items() if v is not None and k not in metadata and not is_bookkeeping_key(k)}
            if self.template is None:
                if row.get("position_ids") is None and row.get("input_ids") is not None:
                    row["position_ids"] = list(range(len(row["input_ids"])))
                media = {}
            else:
                self.template.prepare_row(row)
                media = self.template.pop_media(row)
            row = super().prepare_inputs([row], **kwargs)[0]
            if media:
                row[_MEDIA_SLOT] = media
            rows.append(row)
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
            metadata = set(row.get('_trajectory_columns', ()))
            row = {key: value for key, value in row.items() if key not in metadata}
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
            metadata = set(row.get('_trajectory_columns', ()))
            row = {key: value for key, value in row.items() if key not in metadata}
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
        groups = self._collate_micro_batch_groups(inputs, micro_batch_size, kwargs)
        outputs = super().collate_fn(
            inputs, micro_batch_size=micro_batch_size, variable_seq_lengths=variable_seq_lengths, **kwargs
        )
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

    def _collate_micro_batch_groups(self, inputs: List[InputFeature], micro_batch_size: Optional[int],
                                    kwargs: Dict[str, Any]) -> List[List[int]]:
        """Row-index groups mirroring twinkle's collate split, with the embedding group-intact override.

        Contrastive groups (anchor+positive+negatives) must stay inside one micro-batch: for embedding
        ``micro_batch_size`` counts GROUPS, not the flat sequences ``prepare_inputs`` produced. Splitting
        mid-group orphans anchors/negatives, and InfoNCE's ``nonzero(labels)`` parse then builds ragged
        groups that index past the doc bank (defect124). Ride twinkle's generic ``micro_batch_plan`` seam
        so each group-intact micro-batch collates independently.
        """
        if self._task_type in self._EMBEDDING_TASK_TYPES and micro_batch_size is not None:
            groups = self._embedding_micro_batch_plan(inputs, micro_batch_size)
            kwargs['micro_batch_plan'] = groups
            return groups
        return self._micro_batch_groups(len(inputs), micro_batch_size)

    def _embedding_micro_batch_plan(self, inputs: List[InputFeature], micro_batch_size: int) -> List[List[int]]:
        """Group-intact micro-batch plan: ``micro_batch_size`` counts anchor-led contrastive groups.

        Each flat row carries the scalar anchor mask (1.0 on the anchor, 0.0 on positive/negatives);
        an anchor starts a group that runs until the next anchor. Whole groups are packed into
        micro-batches so InfoNCE never sees a torn group.
        """
        starts = [i for i, row in enumerate(inputs) if self._is_anchor_label(row.get("labels"))]
        if not starts:
            return self._micro_batch_groups(len(inputs), micro_batch_size)
        if starts[0] != 0:
            starts = [0] + starts
        bounds = starts + [len(inputs)]
        groups = [list(range(bounds[k], bounds[k + 1])) for k in range(len(starts))]
        return [sum(groups[i:i + micro_batch_size], []) for i in range(0, len(groups), micro_batch_size)]

    @staticmethod
    def _is_anchor_label(label: Any) -> bool:
        if label is None:
            return False
        if torch.is_tensor(label):
            label = label.view(-1)
            return label.numel() > 0 and float(label[0]) == 1.0
        if isinstance(label, (list, tuple)):
            return len(label) > 0 and float(label[0]) == 1.0
        return float(label) == 1.0

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

    def _seq_align_segments(self, row: InputFeature, seq_len: int) -> List[tuple]:
        """(start, end) token segments of a row: cu_seqlens segments under CP, else the whole row."""
        cp_size = self.device_mesh.cp_world_size if self.device_mesh is not None else 1
        position_ids = row.get("position_ids")
        if cp_size <= 1 or position_ids is None:
            return [(0, seq_len)]
        pos = position_ids[:1] if position_ids.dim() >= 2 and position_ids.shape[0] > 1 else position_ids
        pos_f = pos.flatten()
        indices = torch.arange(pos_f.shape[0], device=pos_f.device, dtype=torch.int32)
        cu_seqlens = torch.cat(
            [indices[pos_f == 0],
             torch.tensor(pos_f.shape, device=pos_f.device, dtype=torch.int32)])
        bounds = cu_seqlens.tolist()
        return list(zip(bounds[:-1], bounds[1:]))

    def _pad_4d_mask_to_seq_align(self, value: torch.Tensor, segments: List[tuple], divisor: int) -> torch.Tensor:
        if len(segments) > 1:
            raise ValueError("a 4D attention_mask cannot ride the packed fp8 sequence padding; "
                             "the collator should have dropped it for the packed batch.")
        length = segments[0][1] - segments[0][0]
        if length % divisor != 0:
            pad_len = divisor - length % divisor
            pad_value = self.padding_map.get("attention_mask", 0)
            value = torch.nn.functional.pad(value, (0, pad_len, 0, pad_len), value=pad_value)
        return value

    @staticmethod
    def _pad_token_column(value: torch.Tensor, segments: List[tuple], divisor: int, pad_value) -> torch.Tensor:
        pieces = []
        changed = False
        for start, end in segments:
            piece = value[..., start:end]
            length = end - start
            if length % divisor != 0:
                pad_len = divisor - length % divisor
                piece = torch.nn.functional.pad(piece, (0, pad_len), value=pad_value)
                changed = True
            pieces.append(piece)
        if not changed:
            return value
        return torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]

    def _pad_row_to_seq_align(self, row: InputFeature, divisor: int) -> InputFeature:
        ref = row.get("input_ids")
        if ref is None:
            ref = row.get("position_ids")
        if ref is None or not torch.is_tensor(ref):
            return row
        seq_len = ref.shape[-1]
        segments = self._seq_align_segments(row, seq_len)
        for key in self._SEQ_ALIGN_FIELDS:
            value = row.get(key)
            # Scalar labels (seq_cls/embedding) and media payloads are not token columns.
            if value is None or not torch.is_tensor(value) or value.shape[-1] != seq_len:
                continue
            if key == "attention_mask" and value.dim() == 4:
                row[key] = self._pad_4d_mask_to_seq_align(value, segments, divisor)
            else:
                row[key] = self._pad_token_column(value, segments, divisor, self.padding_map.get(key, 0))
        return row

    def _pad_to_seq_align(self, inputs: List[InputFeature]) -> List[InputFeature]:
        """Right-pad token columns to the fp8 sequence alignment (legacy collator ``padding_to``).

        Runs for every megatron micro-batch BEFORE twinkle derives cu_seqlens
        (``add_extra_padding_free_args``), so the packed params describe the padded layout.
        Unlike twinkle's ``pad_cp`` this also runs when ``device_mesh is None`` (single process)
        and on the non-padding_free ``[B, S]`` rows: each row's sequence dim is padded, packed
        rows are padded per cu_seqlens segment when CP > 1 so the zigzag split stays legal.
        Padded tokens carry labels -100 / loss_scale 0 and contribute no loss.
        """
        align = self._seq_align_to
        if not align or self.framework != "megatron":
            return inputs
        mesh = self.device_mesh
        cp_size = mesh.cp_world_size if mesh is not None else 1
        tp_size = mesh.tp_world_size if mesh is not None else 1
        divisor = align
        if cp_size > 1:
            # Each segment must stay divisible by 2*cp for the zigzag CP split; the total then
            # remains a multiple of ``align`` (which already carries the cp factor).
            divisor = math.lcm(divisor, 2 * cp_size)
        elif getattr(mesh, "sequence_parallel", False) and tp_size > 1:
            divisor = math.lcm(divisor, tp_size)
        if divisor <= 1:
            return inputs
        return [self._pad_row_to_seq_align(row, divisor) for row in inputs]

    def pad_cp(self, inputs, **kwargs):
        def process(rows, **kw):
            # twinkle's CP/SP padding first (per-segment 2*cp / total tp), then the fp8 alignment
            # top-up; the extra zero-padded fields ride the SAME composed transform so they stay
            # token-aligned even when twinkle's pad_cp early-returns (device_mesh is None).
            rows = TwinkleInputProcessor.pad_cp(self, rows, **kw)
            return self._pad_to_seq_align(rows)

        return self._process_extra_sequence_fields(inputs, process, **kwargs)

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
