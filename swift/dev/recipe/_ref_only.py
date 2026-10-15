# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ref-only execution boundary for online-RL payload assembly.

The driver keeps twinkle ``LazyCollect`` handles and small control descriptors.
Payloads are resolved only in this CPU actor, then handed to model actors through
another lazy remote call.  Versioning, admission and optimizer control remain in
the driver.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from twinkle import remote_class, remote_function
from twinkle_agentic.async_rl.streaming_driver import BufferRecord

from swift.dev.rollout import RolloutEngine, samples_from_responses


REF_ONLY_REMOTE_GROUP = 'rl_payload'


@dataclass(frozen=True)
class PreparedBatchInfo:
    """Small control-plane summary returned to the driver."""

    mini_batch_count: int
    replay_count: int
    row_count: int


@dataclass(frozen=True)
class PreparedBatch:
    """Heavy batch payload that remains behind a twinkle ``LazyCollect`` handle."""

    mini_batches: Tuple[Any, ...]
    replay_count: int
    row_count: int


class PreparedBatchHandle:
    """Driver proxy whose payload remains a remote reference."""

    def __init__(self, processor: 'RemoteBatchAssembler', payload: Any, info: PreparedBatchInfo):
        self._processor = processor
        self._payload = payload
        self.info = info

    @property
    def mini_batch_count(self) -> int:
        return self.info.mini_batch_count

    @property
    def replay_count(self) -> int:
        return self.info.replay_count

    @property
    def row_count(self) -> int:
        return self.info.row_count

    def forward_backward(self, mini_batch_index: int, kind: str = 'policy', **kwargs: Any) -> None:
        """Build kwargs and train one remote mini-batch without exposing it to the driver."""
        self._processor.forward_backward(self._payload, mini_batch_index, kind=kind, **kwargs)

    def release(self) -> None:
        """Compatibility no-op: the ObjectRef lifetime owns the prepared payload."""


@remote_class(execute='first')
class RemoteBatchAssembler:
    """Execute existing algorithm assemblers without materialising payloads in the driver."""

    def __init__(self,
                 assembler: Any,
                 *,
                 sampler: Any,
                 template: Any,
                 group_complete: Any = None):
        self.assembler = assembler
        self.assembler.rollout = RolloutEngine(sampler=sampler, template=template)
        self.group_complete_hook = group_complete
        if self.group_complete_hook is not None and hasattr(self.group_complete_hook, '_assembler'):
            self.group_complete_hook._assembler = self.assembler
        self._global_step = 0
        if hasattr(self.assembler, '_global_step_fn'):
            self.assembler._global_step_fn = self._current_step

    def _current_step(self) -> int:
        return self._global_step

    @staticmethod
    def _resolve(result: Any) -> Any:
        """Resolve a nested twinkle call in this actor, never in the driver."""
        if callable(result) and getattr(result, '_is_lazy_collect', False):
            return result()
        return result

    def _forward_kwargs(self, mini_batch: Any, kind: str) -> Dict[str, Any]:
        if kind == 'policy':
            return self.assembler.forward_kwargs(mini_batch)
        if kind == 'critic':
            return self.assembler.critic_kwargs(mini_batch)
        if kind == 'sft':
            return self.assembler.sft_forward_kwargs()
        raise ValueError(f'unknown prepared forward kind: {kind!r}')

    def _model(self, kind: str) -> Any:
        if kind == 'critic':
            return self.assembler.value_model
        return self.assembler.model

    def _prepare_batch(self, batch: Any) -> PreparedBatch:
        mini_batches = tuple(self.assembler.plan_mini_batches(batch))
        replay_count = self.assembler.num_replays() if hasattr(self.assembler, 'num_replays') else 1
        return PreparedBatch(mini_batches=mini_batches, replay_count=replay_count, row_count=len(batch))

    def _finish_sample(self, sample: Any, *, prompt: Any, prompt_idx: Any,
                       policy_version: int, version_span: int) -> Any:
        sample.prompt_id = str(prompt_idx)
        sample.policy_version = int(policy_version)
        sample.version_span = int(version_span)
        sample.prompt = prompt
        score = getattr(self.assembler, 'score_sample', None)
        if score is not None:
            scored = score(sample)
            if scored is not None:
                sample = scored
        return sample

    @remote_function(dispatch='all', collect='none', lazy_collect=True)
    def collect_sample(self,
                       responses: Any,
                       *,
                       submission_id: str,
                       prompt: Any,
                       prompt_idx: Any,
                       prompt_extras: Optional[Dict[str, Any]],
                       policy_version: int,
                       version_span: int,
                       require_logprobs: bool,
                       allow_message_only: bool) -> Any:
        samples = samples_from_responses(
            responses,
            prompt_extras=[prompt_extras] if prompt_extras else None,
            allow_message_only=allow_message_only,
            require_logprobs=require_logprobs)
        if len(samples) != 1:
            raise RuntimeError(f'streaming collect expected exactly 1 sample for submission {submission_id}, '
                               f'got {len(samples)}.')
        return self._finish_sample(
            samples[0],
            prompt=prompt,
            prompt_idx=prompt_idx,
            policy_version=policy_version,
            version_span=version_span)

    @remote_function(dispatch='all', collect='none', lazy_collect=True)
    def collect_multi_turn_sample(self,
                                  samples: Any,
                                  *,
                                  submission_id: str,
                                  prompt: Any,
                                  prompt_idx: Any,
                                  policy_version: int,
                                  version_span: int) -> Any:
        if len(samples) != 1:
            raise RuntimeError(f'streaming collect expected exactly 1 multi-turn sample for submission '
                               f'{submission_id}, got {len(samples)}.')
        return self._finish_sample(
            samples[0],
            prompt=prompt,
            prompt_idx=prompt_idx,
            policy_version=policy_version,
            version_span=version_span)

    @remote_function(dispatch='all', collect='none', lazy_collect=True)
    def prepare_records(self,
                        *samples: Any,
                        record_meta: Sequence[Tuple[int, str]],
                        global_step: int = 0) -> PreparedBatch:
        if len(samples) != len(record_meta):
            raise ValueError(f'sample/meta count mismatch: {len(samples)} != {len(record_meta)}')
        self._global_step = int(global_step)
        records = [
            BufferRecord(sample=sample, version=int(version), group_key=str(group_key))
            for sample, (version, group_key) in zip(samples, record_meta)
        ]
        return self._prepare_batch(self.assembler.assemble(records))

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def describe(self, prepared_batch: PreparedBatch) -> PreparedBatchInfo:
        return PreparedBatchInfo(
            mini_batch_count=len(prepared_batch.mini_batches),
            replay_count=prepared_batch.replay_count,
            row_count=prepared_batch.row_count,
        )

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def forward_backward(self,
                         prepared_batch: PreparedBatch,
                         mini_batch_index: int,
                         *,
                         kind: str = 'policy',
                         **kwargs: Any) -> None:
        """Build kwargs and call twinkle's existing model API inside this actor."""
        mini_batches = prepared_batch.mini_batches
        if not 0 <= mini_batch_index < len(mini_batches):
            raise IndexError(f'mini_batch_index {mini_batch_index} out of range for {len(mini_batches)} mini-batches')
        forward_kwargs = self._forward_kwargs(mini_batches[mini_batch_index], kind)
        overlap = set(forward_kwargs).intersection(kwargs)
        if overlap:
            raise ValueError(f'forward kwargs duplicate explicit controls: {sorted(overlap)}')
        self._resolve(self._model(kind).forward_backward(**forward_kwargs, **kwargs))

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def group_complete(self, group_key: str, *samples: Any, record_meta: Sequence[Tuple[int, str]]) -> Any:
        if self.group_complete_hook is None:
            return None
        records = [
            BufferRecord(sample=sample, version=int(version), group_key=str(record_group_key))
            for sample, (version, record_group_key) in zip(samples, record_meta)
        ]
        return self.group_complete_hook(group_key, records)

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def chord_mu(self, global_step: int) -> float:
        self._global_step = int(global_step)
        return float(self.assembler.chord_mu())

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def step_telemetry(self) -> Dict[str, Any]:
        method = getattr(self.assembler, 'step_telemetry', None)
        return dict(method() if method is not None else {})

def build_remote_batch_assembler(assembler: Any, *, group_complete: Any = None) -> RemoteBatchAssembler:
    """Create the CPU payload actor from a state-only copy of an existing assembler."""
    payload_assembler = copy.copy(assembler)
    rollout = payload_assembler.rollout
    payload_assembler.rollout = None
    if hasattr(payload_assembler, 'tracker'):
        payload_assembler.tracker = None
    if hasattr(payload_assembler, '_global_step_fn'):
        payload_assembler._global_step_fn = None

    payload_hook = copy.copy(group_complete) if group_complete is not None else None
    if payload_hook is not None and hasattr(payload_hook, '_assembler'):
        payload_hook._assembler = None
    return RemoteBatchAssembler(
        payload_assembler,
        sampler=rollout.sampler,
        template=rollout.template,
        group_complete=payload_hook,
        remote_group=REF_ONLY_REMOTE_GROUP,
    )


def as_prepared_batch(processor: RemoteBatchAssembler, payload: Any) -> PreparedBatchHandle:
    return PreparedBatchHandle(processor, payload, processor.describe(payload))
