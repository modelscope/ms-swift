"""Resource estimation and adaptive replica control for online RL recipes.

The trainer world is immutable. Resource planning may choose sampler and frozen-model replica counts, but
runtime changes always move one complete replica and never alter training collectives or batch geometry.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from swift.dev.utils.logger import get_logger

logger = get_logger()

_GIB = 1024**3
_ALLOWED_OVERRIDE_KEYS = frozenset({
    'trainer_model_parallel_width',
    'sampler_model_parallel_width',
    'sampler_replicas',
    'shared_sampler',
    'flex_gpus',
    'max_num_batched_tokens',
    'max_num_seqs',
})


@dataclass(frozen=True)
class RolePlacement:
    """Immutable physical placement of one complete model role."""

    role: str
    model_parallel_width: int
    replica_count: int
    ranks: Tuple[int, ...]
    gpus_per_worker: int
    shared: bool = False

    def __post_init__(self) -> None:
        if self.model_parallel_width < 1 or self.replica_count < 1 or self.gpus_per_worker < 1:
            raise ValueError(f'{self.role}: widths and replica_count must be positive.')
        expected = self.gpus_per_worker * self.replica_count
        if len(self.ranks) != expected:
            raise ValueError(f'{self.role}: {len(self.ranks)} ranks do not match '
                             f'gpus_per_worker*replica_count={expected}.')
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError(f'{self.role}: physical ranks contain duplicates.')
        if self.gpus_per_worker != self.model_parallel_width:
            raise ValueError(f'{self.role}: gpus_per_worker must equal model_parallel_width.')


@dataclass(frozen=True)
class RLResourcePlan:
    """Complete validated resource plan applied before Ray initialization."""

    trainer: RolePlacement
    sampler: RolePlacement
    auxiliaries: Tuple[RolePlacement, ...] = ()
    shared_sampler: Optional[RolePlacement] = None
    flex_ranks: Tuple[int, ...] = ()
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: Optional[int] = None
    explanation: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        non_shared = [self.trainer, self.sampler, *self.auxiliaries]
        physical = [rank for role in non_shared for rank in role.ranks]
        if len(physical) != len(set(physical)):
            raise ValueError('non-shared role ranks must not overlap.')
        if set(self.flex_ranks) & set(physical):
            raise ValueError('flex ranks must be disjoint from initialized non-shared roles.')
        if len(self.flex_ranks) != len(set(self.flex_ranks)):
            raise ValueError('flex ranks contain duplicates.')
        allocated = set(physical) | set(self.flex_ranks)
        if self.shared_sampler is not None:
            if not self.shared_sampler.shared:
                raise ValueError('shared_sampler must have shared=True.')
            if not set(self.shared_sampler.ranks).issubset(set(self.trainer.ranks)):
                raise ValueError('shared sampler ranks must be a subset of the trainer world.')
        if allocated and allocated != set(range(max(allocated) + 1)):
            raise ValueError('non-shared and flex physical ranks must form a contiguous range from zero.')

    @property
    def total_physical_gpus(self) -> int:
        ranks = set(self.flex_ranks)
        for role in (self.trainer, self.sampler, *self.auxiliaries):
            ranks.update(role.ranks)
        return len(ranks)

    @property
    def roles(self) -> Tuple[RolePlacement, ...]:
        shared = () if self.shared_sampler is None else (self.shared_sampler, )
        return (self.trainer, self.sampler, *self.auxiliaries, *shared)

    def describe(self) -> str:
        rows = [
            f'{role.role}: mp={role.model_parallel_width}, replicas={role.replica_count}, '
            f'ranks={list(role.ranks)}, shared={role.shared}' for role in self.roles
        ]
        rows.append(f'flex_ranks={list(self.flex_ranks)}, physical_gpus={self.total_physical_gpus}')
        rows.extend(self.explanation)
        return '\n'.join(rows)


@dataclass(frozen=True)
class ModelMemoryEstimate:
    parameter_count: int
    weight_bytes: int
    runtime_reserve_bytes: int
    kv_bytes_per_token: int
    graph_reserve_bytes: int


class RLResourceEstimator:
    """Pure, deterministic estimator; it never creates actors or mutates input configs."""

    def __init__(self, *, gpu_memory_gib: float, utilization: float = 0.9):
        if gpu_memory_gib <= 0:
            raise ValueError('gpu_memory_gib must be positive.')
        if not 0 < utilization <= 1:
            raise ValueError('utilization must be in (0, 1].')
        self.card_bytes = int(gpu_memory_gib * _GIB)
        self.utilization = utilization

    @classmethod
    def from_rollout_config(cls, rollout_config: Any) -> 'RLResourceEstimator':
        memory = rollout_config.resource_gpu_memory_gib
        if memory is None:
            memory = cls._cluster_min_gpu_memory_gib()
        utilization = (rollout_config.vllm_gpu_memory_utilization
                       if rollout_config.rollout_sampler == 'vllm'
                       else rollout_config.sglang_mem_fraction_static or 0.9)
        return cls(gpu_memory_gib=memory, utilization=utilization)

    @staticmethod
    def _cluster_min_gpu_memory_gib() -> float:
        try:
            import ray
            resources = ray.cluster_resources() if ray.is_initialized() else {}
            for key, value in resources.items():
                if key.startswith('accelerator_memory_gib_'):
                    return float(value)
        except Exception:
            pass
        raise ValueError('resource_gpu_memory_gib is required when the cluster does not expose per-GPU memory.')

    @staticmethod
    def prompt_length_summary(lengths: Iterable[int]) -> Dict[str, int]:
        values = sorted(int(value) for value in lengths)
        if not values or values[0] < 0:
            raise ValueError('prompt lengths must be a non-empty sequence of non-negative integers.')

        def percentile(q: float) -> int:
            return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]
        return {
            'p50': percentile(0.50),
            'p95': percentile(0.95),
            'p99': percentile(0.99),
            'max': values[-1],
        }

    @staticmethod
    def _dtype_bytes(dtype: Any, *, quantization: Optional[str] = None) -> float:
        if quantization:
            value = quantization.lower()
            if 'int4' in value or 'awq' in value or 'gptq' in value:
                return 0.5
            if 'int8' in value or '8bit' in value or 'fp8' in value:
                return 1.0
        value = str(dtype or 'float32').lower()
        if any(name in value for name in ('float16', 'bfloat16', 'half')):
            return 2.0
        if 'float64' in value:
            return 8.0
        if 'int8' in value or 'fp8' in value:
            return 1.0
        return 4.0

    @staticmethod
    def _parameter_count(config: Any) -> int:
        explicit = getattr(config, 'num_parameters', None)
        if isinstance(explicit, int) and explicit > 0:
            return explicit
        hidden = int(getattr(config, 'hidden_size', 0) or getattr(config, 'd_model', 0) or 0)
        layers = int(getattr(config, 'num_hidden_layers', 0) or getattr(config, 'num_layers', 0) or 0)
        vocab = int(getattr(config, 'vocab_size', 0) or 0)
        intermediate = int(getattr(config, 'intermediate_size', 0) or 4 * hidden)
        heads = int(getattr(config, 'num_attention_heads', 0) or 1)
        kv_heads = int(getattr(config, 'num_key_value_heads', 0) or heads)
        experts = int(getattr(config, 'num_local_experts', 0) or getattr(config, 'num_experts', 0) or 1)
        shared_experts = int(getattr(config, 'num_shared_experts', 0) or 0)
        if min(hidden, layers, vocab) <= 0:
            raise ValueError('HF config must expose hidden size, layer count, and vocab size for estimation.')
        head_dim = hidden // heads
        attention = 2 * hidden * hidden + 2 * hidden * kv_heads * head_dim
        mlp = 3 * hidden * intermediate * experts + 3 * hidden * intermediate * shared_experts
        return int(vocab * hidden + layers * (attention + mlp + 4 * hidden))

    def model_memory(self, config: Any, *, model_parallel_width: int, quantization: Optional[str] = None,
                     enforce_eager: bool = False, runtime_reserve_gib: float = 1.0) -> ModelMemoryEstimate:
        if model_parallel_width < 1:
            raise ValueError('model_parallel_width must be positive.')
        parameter_count = self._parameter_count(config)
        dtype_bytes = self._dtype_bytes(getattr(config, 'torch_dtype', None), quantization=quantization)
        weight_bytes = math.ceil(parameter_count * dtype_bytes / model_parallel_width)
        layers = int(getattr(config, 'num_hidden_layers', 0) or getattr(config, 'num_layers', 0) or 0)
        hidden = int(getattr(config, 'hidden_size', 0) or getattr(config, 'd_model', 0) or 0)
        heads = int(getattr(config, 'num_attention_heads', 0) or 1)
        kv_heads = int(getattr(config, 'num_key_value_heads', 0) or heads)
        kv_dtype_bytes = self._dtype_bytes(getattr(config, 'torch_dtype', None))
        kv_bytes = math.ceil(2 * layers * kv_heads * (hidden // heads) * kv_dtype_bytes / model_parallel_width)
        graph_reserve = 0 if enforce_eager else int(min(1.0, max(0.15, parameter_count / 1e11)) * _GIB)
        return ModelMemoryEstimate(
            parameter_count=parameter_count,
            weight_bytes=weight_bytes,
            runtime_reserve_bytes=int(runtime_reserve_gib * _GIB),
            kv_bytes_per_token=kv_bytes,
            graph_reserve_bytes=graph_reserve)

    def _minimum_width(self, config: Any, candidates: Sequence[int], *, quantization: Optional[str],
                       enforce_eager: bool) -> Tuple[int, ModelMemoryEstimate]:
        budget = int(self.card_bytes * self.utilization)
        for width in sorted(set(candidates)):
            estimate = self.model_memory(
                config, model_parallel_width=width, quantization=quantization, enforce_eager=enforce_eager)
            if estimate.weight_bytes + estimate.runtime_reserve_bytes + estimate.graph_reserve_bytes < budget:
                return width, estimate
        largest = max(candidates)
        estimate = self.model_memory(
            config, model_parallel_width=largest, quantization=quantization, enforce_eager=enforce_eager)
        deficit = estimate.weight_bytes + estimate.runtime_reserve_bytes + estimate.graph_reserve_bytes - budget
        raise ValueError(f'model does not fit at width={largest}; per-card memory deficit is '
                         f'{max(0, deficit) / _GIB:.2f} GiB.')

    def estimate(self, *, hf_config: Any, trainer_gpus: int, auxiliary_widths: Mapping[str, int],
                 rollout_config: Any, prompt_lengths: Sequence[int], max_completion_length: int,
                 trainer_parallel_spec: Optional[str] = None) -> RLResourcePlan:
        if trainer_gpus < 1:
            raise ValueError('trainer_gpus must be positive and remains fixed for the whole run.')
        overrides = dict(rollout_config.resource_overrides or {})
        unknown = sorted(set(overrides) - _ALLOWED_OVERRIDE_KEYS)
        if unknown:
            raise ValueError(f'unknown resource_overrides keys: {unknown}; allowed keys: '
                             f'{sorted(_ALLOWED_OVERRIDE_KEYS)}.')
        factors = [value for value in range(1, trainer_gpus + 1) if trainer_gpus % value == 0]
        if trainer_parallel_spec:
            from twinkle import DeviceMesh
            trainer_mesh = DeviceMesh.from_spec(trainer_parallel_spec)
            if trainer_mesh.world_size != trainer_gpus:
                raise ValueError(f'explicit parallel_spec {trainer_parallel_spec!r} has world_size='
                                 f'{trainer_mesh.world_size}, expected the fixed trainer GPU count {trainer_gpus}.')
            trainer_width = trainer_gpus // trainer_mesh.data_world_size
            overridden_width = overrides.get('trainer_model_parallel_width')
            if overridden_width is not None and int(overridden_width) != trainer_width:
                raise ValueError('trainer_model_parallel_width cannot contradict an explicit parallel_spec; '
                                 f'the spec implies width={trainer_width}, override={overridden_width}.')
        else:
            trainer_width, _ = self._minimum_width(
                hf_config, factors, quantization=None, enforce_eager=True)
        trainer_width = int(overrides.get('trainer_model_parallel_width', trainer_width))
        if trainer_width < 1 or trainer_gpus % trainer_width:
            raise ValueError('trainer_model_parallel_width must be positive and divide the fixed trainer GPU count.')
        trainer_memory = self.model_memory(hf_config, model_parallel_width=trainer_width, enforce_eager=True)
        trainer_required = trainer_memory.weight_bytes + trainer_memory.runtime_reserve_bytes
        if trainer_required >= int(self.card_bytes * self.utilization):
            deficit = trainer_required - int(self.card_bytes * self.utilization)
            raise ValueError(f'trainer_model_parallel_width={trainer_width} does not fit; per-card memory deficit '
                             f'is {deficit / _GIB:.2f} GiB before optimizer and activation reserves.')

        backend = rollout_config.rollout_sampler
        configured_width = (rollout_config.vllm_tensor_parallel_size * rollout_config.vllm_pipeline_parallel_size
                            if backend == 'vllm' else rollout_config.sglang_tp_size * rollout_config.sglang_pp_size)
        candidates = sorted(set(factors + [configured_width]))
        quantization = (rollout_config.vllm_quantization if backend == 'vllm'
                        else rollout_config.sglang_quantization)
        eager = rollout_config.vllm_enforce_eager if backend == 'vllm' else rollout_config.sglang_disable_cuda_graph
        sampler_width, memory = self._minimum_width(
            hf_config, candidates, quantization=quantization, enforce_eager=eager)
        sampler_width = int(overrides.get('sampler_model_parallel_width', sampler_width))
        if sampler_width < 1:
            raise ValueError('sampler_model_parallel_width must be positive.')
        # An override changes per-card weight/KV memory, so all downstream limits must use the overridden width.
        memory = self.model_memory(
            hf_config, model_parallel_width=sampler_width, quantization=quantization, enforce_eager=eager)
        required = memory.weight_bytes + memory.runtime_reserve_bytes + memory.graph_reserve_bytes
        if required >= int(self.card_bytes * self.utilization):
            deficit = required - int(self.card_bytes * self.utilization)
            raise ValueError(f'sampler_model_parallel_width={sampler_width} does not fit; per-card memory deficit '
                             f'is {deficit / _GIB:.2f} GiB.')
        lengths = self.prompt_length_summary(prompt_lengths)
        total_p95 = lengths['p95'] + int(max_completion_length)
        max_context = lengths['max'] + int(max_completion_length)
        engine_context = (rollout_config.vllm_max_model_len if backend == 'vllm'
                          else rollout_config.sglang_context_length)
        if engine_context is not None and max_context > engine_context:
            raise ValueError(f'dataset max total length {max_context} exceeds engine context {engine_context}.')
        budget = int(self.card_bytes * self.utilization)
        kv_budget = budget - memory.weight_bytes - memory.runtime_reserve_bytes - memory.graph_reserve_bytes
        if kv_budget <= 0:
            raise ValueError('sampler KV budget is non-positive after weights and runtime reserves.')
        max_batched = max(1, math.floor(0.95 * kv_budget / max(1, memory.kv_bytes_per_token)))
        max_batched = int(overrides.get('max_num_batched_tokens', max_batched))
        if max_batched < 1:
            raise ValueError('max_num_batched_tokens must be positive.')
        max_seqs = max(1, max_batched // max(1, total_p95))
        configured_seqs = rollout_config.vllm_max_num_seqs if backend == 'vllm' else None
        if configured_seqs is not None:
            max_seqs = min(max_seqs, configured_seqs)
        max_seqs = int(overrides.get('max_num_seqs', max_seqs))
        if max_seqs < 1:
            raise ValueError('max_num_seqs must be positive.')
        if configured_seqs is not None and max_seqs > configured_seqs:
            raise ValueError(f'max_num_seqs override {max_seqs} exceeds the configured engine limit '
                             f'{configured_seqs}.')

        sampler_replicas = int(overrides.get('sampler_replicas', 1))
        if sampler_replicas < 1:
            raise ValueError('sampler_replicas must be positive.')
        flex_gpus = int(overrides.get('flex_gpus', rollout_config.resource_flex_gpus))
        if flex_gpus < 0:
            raise ValueError('flex_gpus must be non-negative.')
        next_rank = trainer_gpus
        trainer = RolePlacement('trainer', trainer_width, trainer_gpus // trainer_width,
                                tuple(range(trainer_gpus)), trainer_width)
        sampler_ranks = tuple(range(next_rank, next_rank + sampler_width * sampler_replicas))
        sampler = RolePlacement('sampler', sampler_width, sampler_replicas, sampler_ranks, sampler_width)
        next_rank += len(sampler_ranks)
        auxiliaries = []
        for role, width in auxiliary_widths.items():
            if width < 1:
                raise ValueError(f'{role}: auxiliary width must be positive.')
            ranks = tuple(range(next_rank, next_rank + width))
            auxiliaries.append(RolePlacement(role, width, 1, ranks, width))
            next_rank += width
        flex_ranks = tuple(range(next_rank, next_rank + flex_gpus))
        next_rank += flex_gpus
        if rollout_config.resource_mode == 'adaptive' and flex_gpus and flex_gpus < sampler_width:
            raise ValueError(f'resource_flex_gpus={flex_gpus} cannot hold one complete sampler replica of '
                             f'width {sampler_width}. Increase the flex reservation or set it to zero.')
        total_cap = rollout_config.resource_total_gpus
        if total_cap is not None and next_rank > total_cap:
            raise ValueError(f'resource plan needs {next_rank} GPUs but resource_total_gpus={total_cap}.')
        enable_shared = bool(overrides.get('shared_sampler', rollout_config.resource_mode == 'adaptive'))
        shared = None
        if enable_shared:
            if trainer_gpus % sampler_width:
                raise ValueError('shared sampler model_parallel_width must divide the complete trainer world.')
            shared = RolePlacement('shared_sampler', sampler_width, trainer_gpus // sampler_width,
                                   tuple(range(trainer_gpus)), sampler_width, shared=True)
        return RLResourcePlan(
            trainer=trainer,
            sampler=sampler,
            auxiliaries=tuple(auxiliaries),
            shared_sampler=shared,
            flex_ranks=flex_ranks,
            max_num_batched_tokens=max_batched,
            max_num_seqs=max_seqs,
            explanation=(
                f'prompt p50/p95/p99/max={lengths["p50"]}/{lengths["p95"]}/{lengths["p99"]}/{lengths["max"]}',
                f'sampler KV budget={kv_budget / _GIB:.2f} GiB per card',
            ))


@dataclass
class ResourceWindow:
    step: int
    trainer_idle: float
    sampler_busy: float
    score_time: float
    handover_time: float


class RLResourceBootstrap:
    """Shared recipe seam that estimates, applies, and materializes one initial plan."""

    @staticmethod
    def _hf_config(model_config: Any) -> Any:
        from transformers import AutoConfig
        kwargs = {}
        if getattr(model_config, 'model_revision', None) is not None:
            kwargs['revision'] = model_config.model_revision
        if getattr(model_config, 'trust_remote_code', None) is not None:
            kwargs['trust_remote_code'] = model_config.trust_remote_code
        return AutoConfig.from_pretrained(model_config.model, **kwargs)

    @staticmethod
    def _prompt_lengths(template: Any, prompt_rows: Sequence[Mapping[str, Any]], limit: int = 2048) -> list[int]:
        if not prompt_rows:
            raise ValueError('resource estimation requires at least one prompt row.')
        if len(prompt_rows) > limit:
            stride = len(prompt_rows) / limit
            rows = [prompt_rows[min(len(prompt_rows) - 1, int(index * stride))] for index in range(limit)]
        else:
            rows = prompt_rows
        lengths = []
        for row in rows:
            encoded = template.encode(dict(row))
            input_ids = encoded.get('input_ids') if isinstance(encoded, dict) else getattr(encoded, 'input_ids', None)
            if input_ids is None:
                raise ValueError('template.encode did not return input_ids required by resource estimation.')
            lengths.append(len(input_ids))
        return lengths

    @classmethod
    def build(cls, *, model_config: Any, distributed_config: Any, rollout_config: Any, template: Any,
              prompt_rows: Sequence[Mapping[str, Any]], auxiliary_groups: Sequence[Tuple[str, int]],
              max_completion_length: int) -> Optional[RLResourcePlan]:
        if rollout_config.resource_mode == 'manual':
            return None
        estimator = RLResourceEstimator.from_rollout_config(rollout_config)
        plan = estimator.estimate(
            hf_config=cls._hf_config(model_config),
            trainer_gpus=distributed_config.nproc_per_node,
            auxiliary_widths=dict(auxiliary_groups),
            rollout_config=rollout_config,
            prompt_lengths=cls._prompt_lengths(template, prompt_rows),
            max_completion_length=max_completion_length,
            trainer_parallel_spec=distributed_config.parallel_spec)
        cls.apply(plan, distributed_config=distributed_config, rollout_config=rollout_config)
        return plan

    @staticmethod
    def apply(plan: RLResourcePlan, *, distributed_config: Any, rollout_config: Any) -> None:
        trainer = plan.trainer
        if distributed_config.parallel_spec is None:
            if trainer.model_parallel_width == 1:
                distributed_config.parallel_spec = f'dp{trainer.replica_count}'
            else:
                distributed_config.parallel_spec = (f'dp{trainer.replica_count}'
                                                    f'tp{trainer.model_parallel_width}')
        sampler = plan.sampler
        if rollout_config.rollout_sampler == 'vllm':
            rollout_config.vllm_tensor_parallel_size = sampler.model_parallel_width
            rollout_config.vllm_pipeline_parallel_size = 1
            rollout_config.vllm_data_parallel_size = sampler.replica_count
            rollout_config.vllm_max_num_seqs = plan.max_num_seqs
            engine_kwargs = dict(rollout_config.vllm_engine_kwargs or {})
            engine_kwargs['max_num_batched_tokens'] = plan.max_num_batched_tokens
            rollout_config.vllm_engine_kwargs = engine_kwargs
        else:
            rollout_config.sglang_tp_size = sampler.model_parallel_width
            rollout_config.sglang_pp_size = 1
            rollout_config.sglang_dp_size = sampler.replica_count

    @classmethod
    def sampler_groups(cls, plan: RLResourcePlan, *, shared: bool = False) -> list[tuple[str, list[int]]]:
        role = plan.shared_sampler if shared else plan.sampler
        if role is None:
            return []
        return [('shared_sampler' if shared else 'sampler', list(role.ranks))]

    @classmethod
    def flex_sampler_groups(cls, plan: RLResourcePlan) -> list[tuple[str, list[int]]]:
        width = plan.sampler.model_parallel_width
        count = len(plan.flex_ranks) // width
        return [(f'flex_sampler_{index}', list(plan.flex_ranks[index * width:(index + 1) * width]))
                for index in range(count)]

    @classmethod
    def device_groups(cls, plan: RLResourcePlan, *, adaptive: bool = False) -> list[tuple[str, list[int]]]:
        groups = [('model', list(plan.trainer.ranks))]
        groups.extend(cls.sampler_groups(plan))
        groups.extend((role.role, list(role.ranks)) for role in plan.auxiliaries)
        if plan.flex_ranks:
            # The base group reserves physical capacity. Candidate role groups below are overlays and create no
            # additional placement-group bundles; actors are constructed only when a slot receives that role.
            groups.append(('resource_flex', list(plan.flex_ranks)))
            if adaptive:
                groups.extend(cls.flex_sampler_groups(plan))
        if plan.shared_sampler is not None:
            groups.extend(cls.sampler_groups(plan, shared=True))
        return groups


class AdaptiveResourceController:
    """Single-change controller driven only at batch and publish boundaries."""

    def __init__(self, *, window_steps: int, min_hold_steps: int, sampler: Any,
                 auxiliary_pools: Optional[Mapping[str, Any]] = None, handover: Any = None,
                 flex_groups: Optional[Mapping[str, Sequence[int]]] = None):
        if window_steps < 1 or min_hold_steps < 1:
            raise ValueError('window_steps and min_hold_steps must be positive.')
        self.window_steps = window_steps
        self.min_hold_steps = min_hold_steps
        self.sampler = sampler
        self.auxiliary_pools = dict(auxiliary_pools or {})
        self.handover = handover
        self._flex_groups = {group: tuple(ranks) for group, ranks in (flex_groups or {}).items()}
        self._free_flex_groups = deque(self._flex_groups)
        self._flex_workers = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rl-flex-sampler')
        self._pending: Optional[Future] = None
        self._pending_group = None
        self._draining = None
        self._addition = None
        self._removal = None
        self._decision = None
        self._decision_step = 0
        self._shared_change = False
        self._closed = False
        self.last_error = None
        self._windows: deque[ResourceWindow] = deque(maxlen=window_steps)
        self._last_change_step = -min_hold_steps
        self._shared_pressure_windows = 0
        self._last_stats: Dict[str, float] = {}

    @classmethod
    def build(cls, *, model, model_config, template, rollout_config, backend, engine_args, plan,
              additional_handover_models=()):
        from twinkle import DeviceMesh
        from twinkle.checkpoint_engine import CheckpointEngineManager
        from swift.dev.builders import build_sampler
        from swift.dev.recipe._colocate import AdaptiveColocateHandover

        sampler = build_sampler(
            model_config, backend=backend, engine_args=dict(engine_args),
            device_mesh=DeviceMesh.from_sizes(dp_size=plan.sampler.replica_count),
            template=template, remote_group='sampler', instance_id='adaptive-sampler')
        dedicated_manager = shared_manager = handover = None
        try:
            sampler.configure_workers(plan.sampler.ranks, state='ACTIVE')
            dedicated_ids = sampler.worker_ids(shared=False)
            handover = AdaptiveColocateHandover(
                model, sampler, None, additional_models=additional_handover_models,
                sleep_level=rollout_config.sleep_level)
            if plan.shared_sampler is not None:
                sleep_option = 'enable_sleep_mode' if backend == 'vllm' else 'enable_memory_saver'
                shared_ids = handover.initialize_shared_workers(
                    'shared_sampler', plan.shared_sampler.ranks, engine_args={sleep_option: True})
                shared_manager = CheckpointEngineManager(
                    model=model, sampler=sampler, platform='GPU', mode='colocate',
                    channel='sampler:ipc', receiver_worker_ids=shared_ids,
                    colocate_trainer_ranks=[ranks[0] for ranks in sampler.worker_ranks(shared_ids)])
            dedicated_manager = CheckpointEngineManager(
                model=model, sampler=sampler, platform='GPU', mode='standalone',
                channel='sampler:nccl', receiver_worker_ids=dedicated_ids)
            handover.dedicated_manager = dedicated_manager
            handover.shared_manager = shared_manager
            controller = cls(
                window_steps=rollout_config.resource_window_steps,
                min_hold_steps=rollout_config.resource_min_hold_steps, sampler=sampler, handover=handover,
                flex_groups=dict(RLResourceBootstrap.flex_sampler_groups(plan)))
            handover.controller = controller
            return controller
        except BaseException:
            for manager in (dedicated_manager, shared_manager):
                if manager is not None:
                    try:
                        manager.close()
                    except Exception:
                        logger.warning('初始化失败后未能完整关闭采样同步通道。', exc_info=True)
            try:
                sampler.close_workers()
            except Exception:
                logger.warning('初始化失败后未能完整关闭 sampler actors。', exc_info=True)
            if handover is not None:
                try:
                    handover.reclaim()
                except Exception:
                    logger.warning('初始化失败后未能完整恢复训练模型。', exc_info=True)
            raise

    @property
    def change_pending(self):
        return (self._decision is not None or self._pending is not None or self._draining is not None
                or self._addition is not None or self._removal is not None)

    def register_decision(self, decision, *, step):
        self._decision_step = step
        if decision is not None and not self._closed and not self.change_pending:
            self._decision = decision

    def prepare_publish(self):
        """在 pause barrier 之后应用决策；后台构造不包含同步或准入。"""
        decision, self._decision = self._decision, None
        self._shared_change = decision in ('activate_shared', 'deactivate_shared')
        if decision == 'activate_shared':
            self.handover.set_enabled(True)
        elif decision == 'deactivate_shared':
            self.handover.set_enabled(False)
        elif decision == 'scale_up_sampler' and self._free_flex_groups:
            group = self._free_flex_groups.popleft()
            self._pending_group = group
            self._pending = self._executor.submit(self.sampler.add_workers, group, self._flex_groups[group])
        elif decision == 'scale_down_sampler' and self._flex_workers:
            group, worker_ids = next(reversed(self._flex_workers.items()))
            self.sampler.drain_workers(worker_ids)
            self._draining = (group, worker_ids)
        if decision is not None:
            self._shared_pressure_windows = 0
        if self._pending is not None and self._pending.done():
            future, group = self._pending, self._pending_group
            self._pending = self._pending_group = None
            try:
                self._addition = (group, future.result())
            except Exception as error:
                self.last_error = f'{type(error).__name__}: {error}'
                logger.warning('Failed to construct adaptive sampler actors in %s: %s', group, error, exc_info=True)
                self._free_flex_groups.appendleft(group)
        if self._draining is not None and self.sampler.workers_idle(self._draining[1]):
            self._removal = self._draining
        receivers = self.sampler.worker_ids(shared=False, states=('ACTIVE', 'DRAINING'))
        if self._addition is not None:
            receivers += self._addition[1]
        removed = () if self._removal is None else self._removal[1]
        return tuple(worker_id for worker_id in receivers if worker_id not in removed)

    def finish_publish(self):
        changed = self._shared_change or self._removal is not None or self._addition is not None
        if self._removal is not None:
            group, worker_ids = self._removal
            if not self.sampler.remove_workers(worker_ids):
                raise RuntimeError('A drained sampler actor received new work during its publication barrier.')
            self._flex_workers.pop(group)
            self._free_flex_groups.append(group)
            self._removal = self._draining = None
        if self._addition is not None:
            group, worker_ids = self._addition
            self.sampler.activate_workers(worker_ids)
            self._flex_workers[group] = worker_ids
            self._addition = None
            self.last_error = None
        if changed:
            self._last_change_step = self._decision_step
        self._shared_change = False

    def fail_publish(self):
        if self._addition is None:
            return
        group, worker_ids = self._addition
        try:
            self.handover.dedicated_manager.close()
        finally:
            self.handover.dedicated_manager.set_receiver_workers(
                self.sampler.worker_ids(shared=False, states=('ACTIVE', 'DRAINING')))
            try:
                removed = self.sampler.remove_workers(worker_ids)
            finally:
                if not set(worker_ids) & set(self.sampler.worker_ids()):
                    self._free_flex_groups.appendleft(group)
                    self._addition = None
        if not removed:
            raise RuntimeError('Failed publication left a sampler actor with unfinished initialization work.')

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self.sampler.close_admission()
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
            if self._pending is not None and not self._pending.cancelled():
                try:
                    self._pending.result()
                except Exception as error:
                    logger.warning('Adaptive sampler construction failed during shutdown: %s', error, exc_info=True)
            try:
                self.sampler.cancel_all_generations()
            finally:
                self.handover.reclaim()
        finally:
            try:
                self.handover.close_channels()
            finally:
                try:
                    self.sampler.close_workers()
                finally:
                    self.handover.reclaim()

    def observe(self, *, step: int, driver_stats: Any, score_time: float = 0.0) -> ResourceWindow:
        current = {
            'train': float(driver_stats.train_active_time),
            'idle': float(driver_stats.idle_time),
            'score': float(score_time),
            'sampler': float(self.sampler.service_time),
            'handover': float(getattr(self.handover, 'total_handover_time', 0.0)) if self.handover else 0.0,
        }
        delta = {key: value - self._last_stats.get(key, value) for key, value in current.items()}
        self._last_stats = current
        total = max(1e-9, delta['train'] + delta['idle'])
        window = ResourceWindow(step, delta['idle'] / total, delta['sampler'], delta['score'], delta['handover'])
        self._windows.append(window)
        return window

    def decide(self, *, step: int, publish_boundary: bool) -> Optional[str]:
        if (not publish_boundary or self.change_pending or step - self._last_change_step < self.min_hold_steps
                or len(self._windows) < self.window_steps):
            return None
        idle = statistics.fmean(window.trainer_idle for window in self._windows)
        handover = statistics.fmean(window.handover_time for window in self._windows)
        loaded = self.sampler.inflight_count > 0
        profitable = loaded and idle > 0 and idle > 1.25 * handover
        self._shared_pressure_windows = self._shared_pressure_windows + 1 if profitable else 0
        shared_enabled = bool(self.handover is not None and self.handover.enabled)
        if self._shared_pressure_windows >= 2:
            if not shared_enabled and self.handover.shared_manager is not None:
                return 'activate_shared'
            if self._free_flex_groups:
                return 'scale_up_sampler'
        if idle < 0.75 * handover:
            if self._flex_workers:
                return 'scale_down_sampler'
            if shared_enabled:
                return 'deactivate_shared'
        return None

