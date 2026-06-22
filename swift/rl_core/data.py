# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import copy
import torch
from dacite import from_dict
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from swift.infer_engine.protocol import RolloutOutput
from swift.template import Messages
from swift.utils import get_logger, remove_response

if TYPE_CHECKING:
    from swift.infer_engine.protocol import RolloutInferRequest
    from swift.rlhf_trainers.gkd_loss import DataSource

from swift.dataset.preprocessor.core import _pair_keys as StandardKeys

logger = get_logger()

# Multimodal keys that a scheduler may override via ``rollout_infos``.
_MULTIMODAL_KEYS = ('images', 'videos', 'audios')


@dataclass
class OnPolicySample:
    """A single on-policy rollout trajectory (pre-collation, per-sample).

    Lifecycle of one sample::

        1. dataset row        -> messages + extra
        2. rollout            -> response_token_ids / rollout_logprobs / finish_reason
        3. rebuild messages   -> replace_assistant_response_with_ids
        4. encode             -> encoded = template.encode(self)   (per-sample dict)

    The collated model-forward inputs (input_ids[B,T], ...) are produced later
    at batch level from ``[s.encoded for s in samples]`` by
    ``collate_to_grpo_micro_batch`` (returns ``(model_inputs, grpo_batch)``), never on
    the sample.
    """
    # --- standard keys ---
    messages: List[Dict]
    images: List[Any] = field(default_factory=list)
    videos: List[Any] = field(default_factory=list)
    audios: List[Any] = field(default_factory=list)
    tools: Optional[List[Any]] = None
    objects: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)  # dataset passthrough columns (flattened for reward)

    # --- id ---
    prompt_id: str = ''
    request_id: str = ''

    # --- rollout output ---
    response_token_ids: List[List[int]] = field(default_factory=list)
    response_loss_mask: List[List[int]] = field(default_factory=list)
    rollout_logprobs: List[List[float]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    add_eos: bool = False
    rollout_infos: Dict[str, Any] = field(default_factory=dict)
    routed_experts: Optional[Any] = None  # R3 router replay (per-sample, pre-collation)

    # --- per-sample template.encode output, used for model forward kwargs ---
    encoded: Optional[Dict[str, Any]] = None

    @property
    def is_truncated(self) -> bool:
        return self.finish_reason == 'length'

    def _standard_fields(self) -> Dict[str, Any]:
        """Non-empty StandardKeys fields from this sample (replaces _multimodal_columns)."""
        return {k: getattr(self, k) for k in StandardKeys if getattr(self, k)}

    def to_reward_row(self) -> Dict[str, Any]:
        """Build the dict consumed by reward functions.

        Flattens ``extra`` to top level so dataset columns (``solution``,
        ``target``, ...) remain accessible as keyword args after
        ``RowPreprocessor.rows_to_batched``. ``encoded`` is excluded (heavy,
        model-internal). Keeps the legacy column contract intact so existing
        reward functions need no change.
        """
        row: Dict[str, Any] = {
            'messages': self.messages,
            'prompt_id': self.prompt_id,
            'request_id': self.request_id,
            'finish_reason': self.finish_reason,
            'is_truncated': self.is_truncated,
            'rollout_infos': self.rollout_infos,
            'response_token_ids': self.response_token_ids,
        }
        row.update(self._standard_fields())
        row.update(self.extra)
        return row

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> OnPolicySample:
        """Build a sample from a dataloader / rollout dict row.

        Known keys are mapped to explicit dataclass fields via field-name
        introspection (no hand-maintained key list); every other column
        (dataset passthrough like ``solution`` / ``chat_template_kwargs``) goes
        to ``extra``. ``is_truncated`` is a derived property and is dropped.

        ``row`` values are deep-copied so the sample owns its data: dataloaders
        (e.g. RepeatSampler with steps_per_generation) cache and re-yield the same
        row dict, and the rollout/encode pipeline mutates messages in place
        (remove_response / response injection). Sharing references would corrupt
        the cached dataset rows across micro-steps.
        """
        field_names = {f.name for f in fields(cls)} - {'extra', 'encoded'}
        standard: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for key, value in row.items():
            if key in field_names:
                standard[key] = copy.deepcopy(value)
            elif key == 'is_truncated':
                continue  # derived from finish_reason
            else:
                extra[key] = copy.deepcopy(value)
        return cls(extra=extra, **standard)

    def to_template_dict(self) -> Dict[str, Any]:
        """Reconstruct the dict consumed by ``template.encode``.

        messages + StandardKeys (images/videos/audios/tools/objects) +
        ``chat_template_kwargs`` (the only dataset-passthrough column encode
        consumes — drives enable_thinking / max_pixels / reasoning_effort) +
        add_eos. Other ``extra`` columns (solution/target/...) are reward-only
        and intentionally excluded from encode. Response tokens are already
        injected into ``messages`` via ``replace_assistant_response_with_ids``
        before encoding.
        """
        d = self._standard_fields()
        chat_template_kwargs = self.extra.get('chat_template_kwargs')
        if chat_template_kwargs is not None:
            d['chat_template_kwargs'] = chat_template_kwargs
        d['add_eos'] = self.add_eos
        return d

    def apply_rollout_output(self, *, rollout_output: RolloutOutput) -> None:
        """Merge one rollout output back onto this sample (used post-rollout).

        Single- vs multi-turn is inferred from ``rollout_output`` itself, not
        from external scheduler state: a multi-turn scheduler (colocate *or*
        server) always returns the full ``messages`` history, while single-turn
        leaves ``messages`` empty. ``choice.token_ids`` is the last single-shot
        generation, so it is only a safe fallback in the single-turn case.
        """
        choice = rollout_output.response.choices[0]
        return_by_scheduler = rollout_output.messages is not None

        # messages: multi-turn returns full history; single-turn appends response
        if return_by_scheduler:
            self.messages = rollout_output.messages
        else:
            remove_response(self.messages)
            self.messages.append({'role': 'assistant', 'content': choice.message.content})

        # response token ids: prefer explicit TITO / scheduler output; only fall
        # back to single-shot choice.token_ids in genuine single-turn rollout.
        if rollout_output.response_token_ids:
            self.response_token_ids = rollout_output.response_token_ids
            if rollout_output.response_loss_mask:
                self.response_loss_mask = rollout_output.response_loss_mask
        elif not return_by_scheduler and choice.token_ids:
            self.response_token_ids = choice.token_ids

        # rollout logprobs (for importance sampling); keep nested [turn][token]
        if rollout_output.rollout_logprobs:
            self.rollout_logprobs = rollout_output.rollout_logprobs
        elif choice.logprobs is not None and 'content' in choice.logprobs:
            self.rollout_logprobs = [[item['logprob'] for item in choice.logprobs['content']]]

        self.finish_reason = choice.finish_reason
        self.add_eos = False

        # R3 router replay: transfer routed_experts from the rollout choice
        routed_experts = getattr(choice, 'routed_experts', None)
        if routed_experts is not None:
            self.routed_experts = routed_experts

        # rollout_infos may carry scheduler-overridden multi-modal data
        if rollout_output.rollout_infos:
            self.rollout_infos = rollout_output.rollout_infos
            for key in _MULTIMODAL_KEYS:
                if key in rollout_output.rollout_infos:
                    setattr(self, key, rollout_output.rollout_infos[key])

    def to_infer_request(self, include_extra: bool = False) -> RolloutInferRequest:
        """Build the ``RolloutInferRequest`` consumed by the rollout engine.

        Maps messages + multimodal/standard columns (images/videos/audios/
        tools/objects) + ``uuid`` (defaults to ``request_id``). Images given as
        ``{'bytes': ...}`` / ``{'path': ...}`` dicts are normalized to base64 /
        path strings. ``tools`` given as a JSON string is parsed.

        When ``include_extra`` is True, dataset passthrough columns (``extra``)
        are forwarded via ``data_dict`` (used for server mode / multi-turn).
        """
        import base64
        import json

        from swift.infer_engine.protocol import RolloutInferRequest

        def _process_image_data(image_data):
            if isinstance(image_data, dict):
                if image_data.get('bytes'):
                    return base64.b64encode(image_data['bytes']).decode('utf-8')
                if image_data.get('path'):
                    return image_data['path']
            return image_data

        request_data: Dict[str, Any] = {'uuid': self.request_id}
        request_data.update(self._standard_fields())

        if request_data.get('images'):
            imgs = request_data['images']
            if not isinstance(imgs, list):
                imgs = [imgs]
            request_data['images'] = [_process_image_data(img) for img in imgs]

        if isinstance(request_data.get('tools'), str):
            try:
                request_data['tools'] = json.loads(request_data['tools'])
            except json.JSONDecodeError:
                pass

        if include_extra and self.extra:
            base_data_dict = self.extra.get('data_dict')
            if base_data_dict is not None and not isinstance(base_data_dict, dict):
                raise ValueError('data_dict exists but is not a dictionary')
            extra_data = {k: v for k, v in self.extra.items() if k != 'data_dict' and v is not None}
            request_data['data_dict'] = {**extra_data, **(base_data_dict or {})}

        return from_dict(RolloutInferRequest, request_data)


@dataclass
class GRPOSample(OnPolicySample):
    """On-policy sample with GRPO reward/advantage signals."""
    rewards: Optional[List[Optional[float]]] = None  # optional mirror; main path uses rewards_per_func tensor
    advantages: Optional[torch.Tensor] = None  # filled after _compute_advantages (0-dim tensor per sample)


@dataclass
class GRPOBatch:
    """Batch data for GRPO loss computation (post-collation, batch-level).

    1. ``completion_mask``, ``truncated_mask``, ``seq_lengths`` — derived from
       collated ``labels`` right after ``data_collator``.
    2. ``old_per_token_logps``, ``ref_per_token_logps`` — computed via
       model/ref forward on the collated batch.
    3. ``advantages`` — computed from gathered rewards.
    4. ``rollout_per_token_logps``, ``num_items_in_batch`` — optional,
       filled when rollout IS / DAPO is enabled.
    """
    completion_mask: torch.Tensor  # [B, T]
    truncated_mask: torch.Tensor  # [B]
    seq_lengths: torch.Tensor  # [B] or [B+n] for padding_free

    old_per_token_logps: Optional[torch.Tensor] = None  # [B, T]
    ref_per_token_logps: Optional[torch.Tensor] = None  # [B, T]
    rollout_per_token_logps: Optional[torch.Tensor] = None  # [B, T]
    advantages: Optional[torch.Tensor] = None  # [B]
    num_items_in_batch: Optional[torch.Tensor] = None  # scalar
    logits_to_keep: Optional[int] = None

    def to_device(self, device) -> 'GRPOBatch':
        """Move all tensor fields to ``device`` in place (Ray: collated on the CPU
        driver, moved to the GPU worker before forward). Returns self."""
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                setattr(self, f.name, v.to(device))
        return self


@dataclass
class GKDSample(OnPolicySample):
    """On-policy sample with GKD teacher-side per-sample signals."""
    teacher_prompt: Optional[Any] = None  # OPSD: dataset ``teacher_prompt`` column (pre-collation)
    teacher_messages: Optional[Messages] = None  # OPSD: messages with teacher_prompt replacing last user

    def build_teacher_view(self) -> bool:
        """Populate teacher-side fields for OPSD from ``teacher_prompt`` + on-policy response.

        OPSD trains the teacher to score its OWN (teacher_prompt + same on-policy response)
        sequence. Build that view by replacing the last user message with ``teacher_prompt``
        and keeping the assistant response; teacher and student share the same
        ``response_token_ids`` (identical response positions). No-op (returns False) when
        ``teacher_prompt`` is unset.
        """
        if self.teacher_messages is not None:
            return True

        if not self.teacher_prompt:
            return False

        messages = [dict(m) for m in self.messages]
        for msg in reversed(messages):
            if msg['role'] == 'user':
                msg['content'] = self.teacher_prompt
                break

        self.teacher_messages = messages
        return True

    def to_teacher_template_dict(self) -> Dict[str, Any]:
        """Reconstruct the teacher-side dict consumed by ``template.encode`` (OPSD).

        Uses ``teacher_messages`` (teacher_prompt-replaced) + the shared
        ``response_token_ids`` (same on-policy response as student).
        """
        d = self._standard_fields()
        d['messages'] = self.teacher_messages
        chat_template_kwargs = self.extra.get('chat_template_kwargs')
        if chat_template_kwargs is not None:
            d['chat_template_kwargs'] = chat_template_kwargs
        if self.response_token_ids:
            d['response_token_ids'] = self.response_token_ids
        d['add_eos'] = False
        return d


@dataclass
class GKDBatch:
    """Batch-level GKD signals (post-collation), symmetric to :class:`GRPOBatch`.

    - ``data_source``: STUDENT / TEACHER / DATASET for this micro-batch.
    - ``teacher_topk_logprobs`` / ``teacher_topk_indices``: assembled teacher
      top-k (teacher-API mode), batch tensors aligned to student tokens.
    """
    data_source: 'DataSource'
    teacher_topk_logprobs: Optional[torch.Tensor] = None
    teacher_topk_indices: Optional[torch.Tensor] = None
