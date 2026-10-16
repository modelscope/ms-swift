"""Placement and construction of the frozen auxiliary models an RL run scores with.

An online-RL / offline-preference run may forward on several frozen models besides the trainable policy --
a KL reference, a distillation teacher, K domain teachers -- and each is a separate weight copy that must be
placed on GPUs. This module is the single home for that placement, so the four shapes those models used to
take (a ``'disable_lora'`` string sentinel, five ``_teacher`` wrapper classes, and two near-duplicate build
helpers) converge on ONE description and ONE builder:

* :class:`AuxiliarySpec` -- a frozen value object describing one auxiliary (role, model id, DeviceGroup name,
  parallel layout, adapters, offload). BOTH the DeviceGroup planning (:meth:`AuxiliarySpec.group_request`)
  and the build (:func:`build_auxiliary`) read this one object, so they cannot drift -- the invariant the
  recipes used to guard with a mirrored ``_frozen_model_needs_group`` predicate.
* :class:`AuxiliaryModel` -- the structural scoring contract (``forward_only`` + ``placement_key``). Two
  concrete shapes: :class:`PolicyAuxiliary` (forwards on the POLICY actor: an adapter-disabled base or the
  student's own current weights, ``placement_key is None``) and :class:`FrozenAuxiliary` (a separate model,
  optionally CPU-offloaded, on its own DeviceGroup). :class:`AuxiliaryBundle` holds K of them plus mixture
  weights (MOPD's domain teachers).
* :func:`plan_rl_device_groups` -- the pure DeviceGroup planner (moved here from ``run_grpo`` so the offline
  family can share it), and the group-name vocabulary every recipe plans and builds under.

Scale safety is the point: a frozen auxiliary's rank count comes from its OWN ``parallel_spec`` (a wide
reference shards across as many ranks as it asks for), never a hard-coded single GPU.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple

# --- DeviceGroup name vocabulary --------------------------------------------------------------------
#: The trainer is always ``'model'`` (``build_model`` targets it); the sampler is a separate ``'sampler'``
#: group when disaggregated, or shares ``'model'`` when colocated.
MODEL_GROUP = 'model'
SAMPLER_GROUP = 'sampler'
#: A separate frozen teacher (GKD/OPSD/MOPD, or a GRPO RLSD/SDAR teacher) is its own Ray DeviceGroup, on GPUs
#: apart from the trainer/sampler. A ``disable_adapter`` / dynamic-self teacher reuses the student's actor and
#: plans no group. This is the ``name`` such a teacher is built with and planned under.
TEACHER_GROUP = 'teacher'
#: A FULL-PARAMETER frozen reference is an independent weight copy, so it is heterogeneous: its OWN Ray
#: DeviceGroup. A LoRA / ``disable_adapter`` reference reuses the policy's base and plans no group.
REF_GROUP = 'ref'


def reward_group(index: int) -> str:
    """The DeviceGroup name for the ``index``-th frozen reward model (each full-parameter RM gets its own)."""
    return f'reward_{index}'


def prm_group(index: int) -> str:
    """The DeviceGroup name for the ``index``-th frozen PRM model (each full-parameter PRM gets its own).

    Distinct from :func:`reward_group` so an ``orm`` reward model and a ``prm`` process-reward model that
    happen to be the same checkpoint still land on their own actors -- they are scored at different points
    (outcome vs per-step prefix) and must not share one forward.
    """
    return f'prm_{index}'


def teacher_group(index: int) -> str:
    """The DeviceGroup name for the ``index``-th frozen teacher when MOPD SPREADS its K teachers apart.

    A spread placement gives each of MOPD's K domain teachers its own group (and thus a parallel
    ``forward_only``); a packed placement instead builds all K into the single shared :data:`TEACHER_GROUP`
    and scores them serially. Distinct from ``TEACHER_GROUP`` so the spread groups never collide with it.
    """
    return f'teacher_{index}'


def spec_world_size(parallel_spec: Optional[str], default: int = 1) -> int:
    """The DeviceGroup rank count a frozen auxiliary's ``parallel_spec`` needs (``default`` when it has none).

    Online RL sizes a reference / teacher / reward / PRM model's group from that model's OWN parallel spec
    (the product of its mesh dims), so a wide frozen model is sharded across as many ranks as it asks for
    instead of the historical single GPU. No spec keeps the card-frugal ``default`` of one rank. The value
    MUST equal what :func:`plan_rl_device_groups` allocated to the group, since ``build_model`` reads the
    frozen config's ``nproc_per_node`` as the group size.
    """
    if not parallel_spec:
        return default
    from twinkle import DeviceMesh
    return DeviceMesh.from_spec(parallel_spec).world_size


def teacher_group_world_size(rl_config: Any) -> int:
    """GPU count for a separate frozen teacher's OWN DeviceGroup: ``teacher_parallel_spec``'s world size, else 1.

    ``teacher_parallel_spec = None`` is the card-frugal default -- one teacher rank, serial ``forward_only``;
    a spec (``dp2``/``fsdp2``/...) sizes the group for a big or concurrency-friendly teacher. A
    ``disable_adapter`` / dynamic-self teacher builds no separate actor and needs no group (caller passes 0).
    """
    return spec_world_size(getattr(rl_config, 'teacher_parallel_spec', None))


def plan_rl_device_groups(nproc_per_node: int, sampler_mode: Optional[str], sampler_world_size: int,
                          auxiliary_groups: Sequence[Tuple[str, int]] = ()) -> Tuple[List[Tuple[str, List[int]]], str,
                                                                                    bool]:
    """Plan the twinkle DeviceGroups for an RL run. Pure function (no twinkle import).

    Args:
        nproc_per_node: the TRAINER's GPU count (== ``DistributedConfig.nproc_per_node``, which sizes the
            model's data-parallel mesh in ``build_model``).
        sampler_mode: ``'colocate'`` shares the trainer's GPUs; anything else (None/``'disaggregated'``, or
            its legacy alias ``'server'``) disaggregates.
        sampler_world_size: the sampler's GPU count, computed per backend by ``RolloutSamplerSpec.world_size``
            (vLLM tp*dp; SGLang tp*pp*dp).
        auxiliary_groups: ``(name, world_size)`` for each FULL-PARAMETER frozen auxiliary that needs its own
            GPUs -- a reference, a teacher, each seq_cls reward model, a frozen PRM. Callers derive these from
            their :class:`AuxiliarySpec`\\ s via :meth:`AuxiliarySpec.group_request` (plus the reward/PRM groups
            the shared scorer builds), so the plan and the build read one source. One disjoint group is
            appended per entry with ``world_size > 0``; ``name`` is the ``remote_group`` the caller builds that
            model with. An auxiliary that reuses the policy actor (a ``disable_adapter`` reference/teacher)
            requests no group and is simply absent here.

    Returns:
        ``(groups, sampler_remote_group, colocate)`` where ``groups`` is a list of ``(name, ranks)`` to hand
        ``twinkle.initialize``, ``sampler_remote_group`` is the DeviceGroup the sampler is placed in, and
        ``colocate`` is the flag for ``CheckpointEngineManager``. Every ``auxiliary_groups`` entry is appended
        last, disjoint, in argument order.

    Colocate puts trainer+sampler in ONE group over ``nproc_per_node`` GPUs (independent rank spaces);
    disaggregated appends a disjoint ``sampler`` group after the trainer's ranks. Every frozen auxiliary group
    is always disjoint (its resident weights never contend with the trainer/sampler), appended after them, so
    total GPUs = nproc_per_node + (sampler_world_size if disaggregated) + sum(each auxiliary group's size).
    """
    if nproc_per_node is None or nproc_per_node < 1:
        raise ValueError(f'nproc_per_node must be >= 1 (the trainer GPU count), got {nproc_per_node!r}.')
    if sampler_world_size < 1:
        raise ValueError(f'sampler_world_size must be >= 1, got {sampler_world_size}.')

    if sampler_mode == 'colocate':
        if sampler_world_size > nproc_per_node:
            raise ValueError(f'colocate needs the sampler ({sampler_world_size} GPUs) to fit within the trainer '
                             f'GPUs ({nproc_per_node}); it shares them. Use sampler_mode="disaggregated" to '
                             'place the sampler on its own GPUs.')
        # The overlay has its own worker-rank space. Reusing MODEL_GROUP would inherit the trainer worker
        # count and DeviceMesh, which is wrong as soon as sampler TP/PP is wider than one GPU.
        groups = [
            (MODEL_GROUP, list(range(nproc_per_node))),
            (SAMPLER_GROUP, list(range(sampler_world_size))),
        ]
        sampler_remote_group, colocate, next_rank = SAMPLER_GROUP, True, nproc_per_node
    else:
        total = nproc_per_node + sampler_world_size
        groups = [
            (MODEL_GROUP, list(range(nproc_per_node))),
            (SAMPLER_GROUP, list(range(nproc_per_node, total))),
        ]
        sampler_remote_group, colocate, next_rank = SAMPLER_GROUP, False, total

    # Append one disjoint group per frozen auxiliary model, in argument order. An auxiliary that reuses the
    # policy actor (disable_adapter) requests no group, so it is never listed here.
    reserved = {MODEL_GROUP, sampler_remote_group}
    appended: List[str] = []
    for name, world_size in auxiliary_groups:
        if world_size < 0:
            raise ValueError(f'auxiliary DeviceGroup {name!r} world_size must be >= 0, got {world_size}.')
        if world_size == 0:
            continue
        if name in reserved or name in appended:
            raise ValueError(f'auxiliary DeviceGroup name {name!r} collides with a reserved or already-planned '
                             f'group ({sorted(reserved | set(appended))}); each frozen model needs a unique name.')
        groups.append((name, list(range(next_rank, next_rank + world_size))))
        appended.append(name)
        next_rank += world_size
    return groups, sampler_remote_group, colocate


# --- the auxiliary description ----------------------------------------------------------------------


@dataclass(frozen=True)
class AuxiliarySpec:
    """The single description of one frozen forward-only auxiliary (a KL reference or a distillation teacher).

    A frozen value object: the recipe decides WHAT the auxiliary is (which model, adapter-disabled or a
    separate copy, how wide) and packs it here once; :meth:`group_request` feeds the DeviceGroup planner and
    :func:`build_auxiliary` builds the model, both off this one object, so the two cannot drift.

    Fields:
        role: ``'reference'`` or ``'teacher'`` -- the adapter role name ``configure_frozen_adapter`` applies
            and the label in error messages.
        model_id: the frozen model's hub id / local dir. ``None`` with ``disable_adapter=False`` means "no such
            auxiliary" (:func:`build_auxiliary` returns ``None`` and no group is planned).
        name: the DeviceGroup name to build/plan under (``REF_GROUP`` / ``TEACHER_GROUP`` / ``teacher_group(i)``).
            Only read when a separate frozen actor is built.
        parallel_spec: the auxiliary's OWN parallel layout (``'dp2'``/``'tp2'``/...); ``None`` keeps one rank.
        adapters: adapter ids to configure on the frozen copy (an explicit reference/teacher adapter).
        model_type / model_revision: architecture overrides so an auxiliary may differ from the policy;
            ``None`` inherits the base config's value (see :func:`build_auxiliary`).
        deepspeed: an auxiliary-specific DeepSpeed config (a teacher may need one the reference does not).
        disable_adapter: score on the POLICY actor with the LoRA adapter disabled (a LoRA student's own frozen
            base). Reuses the policy's weights -- no second copy, no DeviceGroup -- so it is a
            :class:`PolicyAuxiliary`, not a :class:`FrozenAuxiliary`.
        offload: keep the frozen model on CPU between forwards (GKD's full-vocab-logits memory hook).
        base_model_config: whether :func:`build_auxiliary` starts from a COPY of the policy's ``ModelConfig``
            (``True`` -- a reference, usually the same checkpoint as the policy, inheriting its processor
            knobs) or from a bare ``ModelConfig(model_id)`` (``False`` -- a teacher that is a distinct, often
            larger, model). Preserves the two recipes' historical build bases exactly.
        instance_id: disambiguates several auxiliaries built into ONE group (MOPD's K packed teachers), whose
            Ray actor names (group+class+caller+rank) would otherwise collide.
    """
    role: str
    model_id: Optional[str] = None
    name: Optional[str] = None
    parallel_spec: Optional[str] = None
    adapters: Tuple[str, ...] = ()
    model_type: Optional[str] = None
    model_revision: Optional[str] = None
    deepspeed: Optional[str] = None
    disable_adapter: bool = False
    offload: bool = False
    base_model_config: bool = True
    instance_id: Optional[str] = None

    @property
    def world_size(self) -> int:
        """Ranks this auxiliary's DeviceGroup needs -- its ``parallel_spec``'s world size, else one."""
        return spec_world_size(self.parallel_spec)

    @property
    def needs_group(self) -> bool:
        """Whether a separate frozen actor is built (so a DeviceGroup must be planned for it).

        Mirrors :func:`build_auxiliary`'s branch exactly: an adapter-disabled auxiliary reuses the policy
        actor (no group) and no ``model_id`` means no auxiliary at all (no group); only a distinct
        full-parameter frozen copy is a separate Ray actor on its own GPUs.
        """
        return bool(self.model_id) and not self.disable_adapter

    def group_request(self) -> Optional[Tuple[str, int]]:
        """The ``(name, world_size)`` this spec contributes to :func:`plan_rl_device_groups`, or ``None``.

        ``None`` when the auxiliary reuses the policy actor (so the planner lists no group for it); otherwise
        its own disjoint group, sized by :attr:`world_size`.
        """
        if not self.needs_group:
            return None
        return (self.name, self.world_size)


# --- the auxiliary model wrappers -------------------------------------------------------------------


class AuxiliaryModel(Protocol):
    """A scoring owner: anything that can ``forward_only`` a batch of features, plus its placement identity.

    Structural on purpose -- the student model itself already has ``forward_only``, so a loop may score on the
    raw student (recomputing old log-probs) or on one of the wrappers below without ever branching on the type.

    Every owner also exposes a ``placement_key``: the identity of the DeviceGroup it forwards on, used by
    :func:`~swift.dev.recipe.common.scoring.overlap_by_placement` to tell which scorers may run concurrently.
    ``None`` is the sentinel for "the policy actor itself", so all policy-actor scorers share one key and
    serialize; a separate frozen model carries its group name, so scorers on disjoint groups overlap. The raw
    student has no wrapper, so a loop resolves its key with ``getattr(owner, 'placement_key', None)``.
    """

    placement_key: Any

    def forward_only(self, inputs: List[dict], **kwargs: Any) -> Any:
        ...


class PolicyAuxiliary:
    """An auxiliary that forwards on the POLICY actor itself -- no separate model, no DeviceGroup.

    Two roles that were once distinct wrapper types collapse here, because they differ only by one flag:
    ``disable_adapter=True`` scores on the student's adapter-DISABLED frozen base (a LoRA student's own
    reference/teacher, ``forward_only(disable_lora=True)``), and ``disable_adapter=False`` scores on the
    student's CURRENT weights (OPSD's dynamic self-distillation, a plain forward). Both share the policy's
    placement key (``None``), so they serialize with the policy actor rather than overlapping on a disjoint
    group.
    """

    def __init__(self, student: Any, *, disable_adapter: bool = False):
        self.student = student
        self.disable_adapter = disable_adapter
        #: Forwards on the policy actor, so it shares the policy's placement key.
        self.placement_key = None

    def forward_only(self, inputs: List[dict], **kwargs: Any) -> Any:
        if self.disable_adapter:
            return self.student.forward_only(inputs=inputs, disable_lora=True, **kwargs)
        return self.student.forward_only(inputs=inputs, **kwargs)


class FrozenAuxiliary:
    """A separate frozen model -- a local copy or a Ray actor on its own DeviceGroup.

    ``offload=True`` keeps the model on CPU between forwards and reloads it only to score, the memory hook for
    GKD's full-vocab logits. The reload/offload wrap lives here so a loop's scoring code is identical whether
    or not the auxiliary offloads. ``model`` is exposed so reference synchronization (which mutates a distinct
    mutable reference) can reach the underlying model.
    """

    def __init__(self, model: Any, *, offload: bool = False, placement_key: Any = None):
        self.model = model
        self.offload = offload
        #: The DeviceGroup this frozen model occupies -- its ``remote_group``. Two frozen models built into the
        #: SAME group (MOPD's packed K teachers) share one key and serialize; distinct groups (spread) get
        #: distinct keys and overlap. A model with no group name falls back to a per-instance token so it is
        #: never mistaken for the policy actor (``None``) or for another frozen model.
        self.placement_key = placement_key if placement_key is not None else ('frozen', id(model))
        if offload:
            model.offload_to_cpu()

    def forward_only(self, inputs: List[dict], **kwargs: Any) -> Any:
        if not self.offload:
            return self.model.forward_only(inputs=inputs, **kwargs)
        self.model.reload_to_gpu()
        try:
            return self.model.forward_only(inputs=inputs, **kwargs)
        finally:
            self.model.offload_to_cpu()


class AuxiliaryReplicaState(str, Enum):
    WARMING = 'warming'
    ACTIVE = 'active'
    DRAINING = 'draining'
    FAILED = 'failed'
    CLOSED = 'closed'


@dataclass
class AuxiliaryReplica:
    replica_id: str
    model: AuxiliaryModel
    state: AuxiliaryReplicaState = AuxiliaryReplicaState.WARMING
    inflight: int = 0
    requests: int = 0
    service_time: float = 0.0


class AuxiliaryReplicaPool:
    """Load-balanced complete auxiliary replicas with stable output ordering."""

    def __init__(self, role: str, replicas: Sequence[AuxiliaryReplica] = ()):
        self.role = role
        self.placement_key = ('auxiliary_pool', role)
        self._replicas: Dict[str, AuxiliaryReplica] = {}
        self._lock = threading.RLock()
        for replica in replicas:
            self.add_replica(replica, activate=True)

    @property
    def replicas(self) -> Tuple[AuxiliaryReplica, ...]:
        with self._lock:
            return tuple(self._replicas.values())

    def add_replica(self, replica: AuxiliaryReplica, *, activate: bool = False, warmup: Any = None) -> None:
        with self._lock:
            if not replica.replica_id or replica.replica_id in self._replicas:
                raise ValueError(f'invalid or duplicate auxiliary replica id: {replica.replica_id!r}.')
            self._replicas[replica.replica_id] = replica
        try:
            if warmup is not None:
                warmup(replica.model)
        except Exception:
            replica.state = AuxiliaryReplicaState.FAILED
            raise
        replica.state = AuxiliaryReplicaState.ACTIVE if activate or warmup is not None else AuxiliaryReplicaState.WARMING

    def activate_replica(self, replica_id: str) -> None:
        with self._lock:
            replica = self._replicas[replica_id]
            if replica.state is not AuxiliaryReplicaState.WARMING:
                raise RuntimeError(f'{replica_id} is not WARMING.')
            replica.state = AuxiliaryReplicaState.ACTIVE

    def request_drain(self, replica_id: str) -> None:
        with self._lock:
            replica = self._replicas[replica_id]
            if replica.state is AuxiliaryReplicaState.ACTIVE:
                replica.state = AuxiliaryReplicaState.DRAINING

    def retire_replica(self, replica_id: str) -> bool:
        with self._lock:
            replica = self._replicas[replica_id]
            if replica.state is not AuxiliaryReplicaState.DRAINING:
                raise RuntimeError(f'{replica_id} must be DRAINING before retirement.')
            if replica.inflight:
                return False
            replica.state = AuxiliaryReplicaState.CLOSED
            self._replicas.pop(replica_id)
        model = getattr(replica.model, 'model', replica.model)
        shutdown = getattr(model, 'shutdown', None)
        if shutdown is not None:
            shutdown()
        return True

    @staticmethod
    def _merge_outputs(outputs: Sequence[Any]) -> Any:
        if not outputs:
            return []
        first = outputs[0]
        if isinstance(first, dict):
            keys = set(first)
            if any(set(output) != keys for output in outputs):
                raise ValueError('auxiliary replica outputs have inconsistent fields.')
            return {key: AuxiliaryReplicaPool._merge_outputs([output[key] for output in outputs]) for key in first}
        if isinstance(first, tuple):
            if any(len(output) != len(first) for output in outputs):
                raise ValueError('auxiliary replica tuple outputs have inconsistent lengths.')
            return tuple(AuxiliaryReplicaPool._merge_outputs([output[index] for output in outputs])
                         for index in range(len(first)))
        if isinstance(first, list):
            merged = []
            for output in outputs:
                merged.extend(output)
            return merged
        try:
            import torch
            if torch.is_tensor(first):
                if any(not torch.is_tensor(output) or output.shape[1:] != first.shape[1:] for output in outputs):
                    raise ValueError('auxiliary replica tensor outputs have inconsistent trailing shapes.')
                return torch.cat(list(outputs), dim=0)
        except ImportError:
            pass
        raise TypeError(f'cannot merge auxiliary output type {type(first).__name__}.')

    def forward_only(self, inputs: List[dict], **kwargs: Any) -> Any:
        if not inputs:
            return []
        with self._lock:
            active = sorted((replica for replica in self._replicas.values()
                             if replica.state is AuxiliaryReplicaState.ACTIVE), key=lambda item: item.replica_id)
        if not active:
            raise RuntimeError(f'no ACTIVE {self.role} auxiliary replica is available.')
        size, remainder = divmod(len(inputs), len(active))
        jobs = []
        offset = 0
        for index, replica in enumerate(active):
            count = size + (1 if index < remainder else 0)
            if count:
                jobs.append((replica, inputs[offset:offset + count]))
                offset += count

        def invoke(job: Tuple[AuxiliaryReplica, List[dict]]) -> Any:
            replica, shard = job
            started = time.perf_counter()
            with self._lock:
                if replica.state is not AuxiliaryReplicaState.ACTIVE:
                    raise RuntimeError(f'auxiliary replica left ACTIVE state before dispatch: {replica.replica_id}.')
                replica.inflight += 1
                replica.requests += 1
            try:
                return replica.model.forward_only(shard, **kwargs)
            finally:
                with self._lock:
                    replica.inflight -= 1
                    replica.service_time += time.perf_counter() - started

        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            outputs = list(executor.map(invoke, jobs))
        return self._merge_outputs(outputs)


class AuxiliaryBundle:
    """K frozen auxiliaries (MOPD's domain teachers) plus their mixture weights.

    Not an :class:`AuxiliaryModel` itself -- it does not forward a single batch -- but a container the loop
    iterates, scoring each member over the student's own features to produce one signal channel each, which
    ``MOPDLoss`` fuses by ``weights``. Members carry their own ``placement_key``, so a spread bundle (disjoint
    groups) overlaps its forwards while a packed one (one shared group) serializes -- the loop drives that
    through :func:`~swift.dev.recipe.common.scoring.overlap_by_placement`.
    """

    def __init__(self, auxiliaries: Sequence[AuxiliaryModel], weights: Optional[List[float]] = None):
        self.auxiliaries = list(auxiliaries)
        self.weights = weights

    def __iter__(self) -> Iterator[AuxiliaryModel]:
        return iter(self.auxiliaries)

    def __len__(self) -> int:
        return len(self.auxiliaries)


def build_auxiliary(spec: AuxiliarySpec,
                    *,
                    template: Any,
                    student: Any,
                    distributed_config: Any,
                    model_config: Optional[Any] = None) -> Optional[AuxiliaryModel]:
    """Build one frozen auxiliary from its spec -> a :class:`PolicyAuxiliary` / :class:`FrozenAuxiliary` / ``None``.

    ``disable_adapter`` scores on the student actor with the LoRA adapter disabled (a :class:`PolicyAuxiliary`,
    reusing the policy's own weights -- no separate model, no DeviceGroup); no ``model_id`` means "no such
    auxiliary" (``None``, skipped by the loop); otherwise a distinct full-parameter frozen copy is built as a
    separate Ray actor on the spec's ``name`` DeviceGroup and wrapped in a :class:`FrozenAuxiliary`.

    The frozen copy inherits the run's backend and is built ``mode='ray'`` via
    :func:`~swift.dev.builders.frozen_auxiliary_distributed_config` (a forward-only auxiliary is identical on
    megatron and transformers). ``spec.world_size`` MUST equal the rank count :func:`plan_rl_device_groups`
    allocated to ``spec.name`` (``build_model`` reads it as the group size), and ``spec.parallel_spec`` lays
    out a sharded auxiliary. ``base_model_config`` picks the build base: a COPY of the policy's ``model_config``
    (a reference, usually the same checkpoint) or a bare ``ModelConfig(model_id)`` (a distinct teacher) -- so
    the two historical build bases are preserved exactly. ``model_type`` / ``model_revision`` override the base
    only when set, so an auxiliary may differ in architecture from the student.
    """
    if spec.disable_adapter:
        return PolicyAuxiliary(student, disable_adapter=True)
    if not spec.model_id:
        return None
    from copy import copy

    from swift.dev.builders import build_model, frozen_auxiliary_distributed_config
    from swift.dev.config import ModelConfig
    from swift.dev.recipe.common.assembly import configure_frozen_adapter

    if spec.base_model_config:
        if model_config is None:
            raise ValueError(f'building the {spec.role!r} auxiliary from the policy ModelConfig requires '
                             'model_config, but none was passed.')
        config = copy(model_config)
        config.model = spec.model_id
    else:
        config = ModelConfig(model=spec.model_id)
    if spec.model_type is not None:
        config.model_type = spec.model_type
    if spec.model_revision is not None:
        config.model_revision = spec.model_revision
    aux_dist = frozen_auxiliary_distributed_config(
        distributed_config, spec.world_size, parallel_spec=spec.parallel_spec, deepspeed=spec.deepspeed)
    # device_mesh drives the transformers placement; the megatron path ignores it and derives its own mesh from
    # aux_dist.parallel_spec, so setting both keeps the auxiliary correct on either backend.
    device_mesh = None
    if spec.parallel_spec:
        from twinkle import DeviceMesh
        device_mesh = DeviceMesh.from_spec(spec.parallel_spec)
    frozen = build_model(
        config, aux_dist, device_mesh=device_mesh, remote_group=spec.name, instance_id=spec.instance_id)
    configured = configure_frozen_adapter(frozen, template, list(spec.adapters), role=spec.role)
    return FrozenAuxiliary(configured, offload=spec.offload, placement_key=spec.name)
