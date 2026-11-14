# Copyright (c) ModelScope Contributors. All rights reserved.
# ruff: noqa: C901
# Distillation's assembly: teacher group requests, the dataset rows, and the explicit cluster build.
from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from swift.dev.recipe.distill.worker import DistillWorker
from swift.dev.recipe.utils import TEACHER_GROUP, auxiliary_mesh, build_rl_sampler, teacher_group
from swift.dev.template.conventions import PROMPT_INPUT_KEYS
from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        RLConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = get_logger()

#: Each signal form's teacher count and GKD-only knobs, keyed by the recipe's rl_type.
DISTILL_SIGNALS = {"gkd": "logits", "opsd": "logps", "mopd": "multi_logps"}


def distill_dataset_rows(template: Any, raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The off-policy training rows: dataset rows that carry their own assistant completion.

    Each is encoded once with the student's template (its labels mark the completion's trainable
    positions) and keeps the conversation plus the raw prompt inputs the teacher view re-encodes --
    the same keys a rollout trajectory carries, so an update never branches on its source.
    """
    dataset_rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        messages = row.get("messages") if isinstance(row, dict) else None
        if not messages or messages[-1].get("role") != "assistant":
            continue
        encoded = dict(template.encode(row))
        # The privileged teacher view RE-ENCODES the conversation, so it needs the raw prompt inputs
        # (media references, tools, render kwargs), not the student feature's tensors. Nothing else
        # from the dataset row may ride along: this feature is what both models collate, and a column
        # outside the field contract is rejected there as model payload no collator can batch.
        for key in PROMPT_INPUT_KEYS:
            if row.get(key) is not None:
                encoded.setdefault(key, copy.deepcopy(row[key]))
        encoded["messages"] = copy.deepcopy(messages)
        dataset_rows.append(encoded)
    return dataset_rows


def distill_sampling_params(rl_config: Any, generation_config: Any) -> Dict[str, Any]:
    """SamplingParams dict for the on-policy distillation rollout (shared by gkd/opsd/mopd).

    The rollout uses ``sampling_temperature``, NOT the distillation ``distill_temperature``: those
    are separate roles. ``sampling_temperature=None`` falls back to ``distill_temperature`` so
    pre-split runs are byte-identical; setting it decouples how hot the student samples from how
    softly the divergence scores.
    """
    sampling_temperature = rl_config.sampling_temperature
    if sampling_temperature is None:
        sampling_temperature = rl_config.distill_temperature
    params: Dict[str, Any] = {
        "max_tokens": rl_config.max_completion_length,
        "temperature": sampling_temperature,
    }
    if generation_config is not None:
        if generation_config.top_p is not None:
            params["top_p"] = generation_config.top_p
        if generation_config.top_k is not None:
            params["top_k"] = generation_config.top_k
    return params


def distill_group_requests(rl_config: "RLConfig") -> List[Tuple[str, Optional[str]]]:
    """The frozen-teacher DeviceGroups this run needs -> ``[(name, teacher_parallel_spec)]``.

    Mirrors the teacher builds in :func:`initialize_distill_cluster` exactly: a disable_adapter /
    dynamic-self teacher reuses the policy actor and plans NO group; MOPD's ``spread`` placement
    gives each of its K teachers its own ``teacher_<i>`` group, ``pack`` (the default) shares ONE.
    """
    if rl_config.rl_type == "mopd":
        spread = rl_config.auxiliary_placement == "spread"
        return (
            [
                (teacher_group(index), rl_config.teacher_parallel_spec)
                for index in range(len(rl_config.teacher_model or []))
            ]
            if spread
            else [(TEACHER_GROUP, rl_config.teacher_parallel_spec)]
        )
    from swift.dev.recipe.utils import single_teacher_id

    separate = (
        not rl_config._teacher_use_disable_adapter
        and single_teacher_id(rl_config.teacher_model, algo="distill") is not None
    )
    return [(TEACHER_GROUP, rl_config.teacher_parallel_spec)] if separate else []


class Cluster(NamedTuple):
    """Every actor and handle the distillation loop drives (a plain handle bundle, not a layer)."""

    model: Any
    sampler: Any
    rollout: Any
    engine: Any
    adapter: Any
    teachers: List[Any]
    worker: Any
    sampling_params: Dict[str, Any]
    colocate: bool


def initialize_distill_cluster(
    *,
    model_config: "ModelConfig",
    template_config: "TemplateConfig",
    template: Any,
    train_config: "TrainConfig",
    distributed_config: "DistributedConfig",
    checkpoint_config: Optional["CheckpointConfig"],
    rollout_config: "RolloutConfig",
    rl_config: "RLConfig",
    tuner_config: Optional["TunerConfig"],
    max_steps: int,
    signal: str,
    train_batch_size: int,
    batch_size: int,
    quantize_config: Optional["QuantizeConfig"] = None,
    megatron_config: Optional["MegatronConfig"] = None,
    moe_config: Optional["MoEConfig"] = None,
    generation_config: Optional[Any] = None,
    resource_plan: Any = None,
    auxiliary_meshes: Optional[Dict[str, Any]] = None,
) -> Cluster:
    """Build every actor distillation trains with, explicitly: policy, sampler, teachers, worker.

    Nothing is hidden behind a shared assembly class (deep_refactor's initialize_cluster note).
    ``max_steps`` is the resolved step budget derived by the caller BEFORE this stage, so the
    optimizer's LR scheduler is fully configured here; ``batch_size`` is the rows of one driver
    batch (it bounds the episodes in flight).
    """
    from swift.dev.builders import apply_full_param_freeze, build_dp_mesh, build_model
    from swift.dev.builders.adapter import apply_tuner
    from swift.dev.builders.optimizer import configure_optimizer
    from swift.dev.builders.streaming import (
        CONTROL_GROUP,
        build_checkpoint_manager,
        build_cluster_adapter,
        build_rollout,
        inflight_rows,
    )
    from swift.dev.loss import configure_rl_loss
    from swift.dev.processor import InputProcessor, processor_kwargs

    sp = template_config.sequence_parallel_size
    # mode='remote' runs ONE client process: the student AND every teacher are served-model handles.
    remote_run = distributed_config.mode == 'remote'
    sp_mesh = None if remote_run else (build_dp_mesh(distributed_config, sp) if sp > 1 else None)
    resume_dir = checkpoint_config.resume_from_checkpoint if checkpoint_config is not None else None
    policy_config = model_config
    if resume_dir and tuner_config is None and not remote_run:
        # A full-parameter resume loads weights from the ckpt dir instead of the original model id;
        # a tuned policy keeps the original base and reloads its adapter via resume (run_distill).
        policy_config = copy.deepcopy(model_config)
        policy_config.model = resume_dir
    if remote_run:
        from swift.dev.builders.remote import (
            apply_remote_tuner,
            build_remote_model,
            configure_remote_processor,
        )

        model = build_remote_model(model_config, distributed_config)
        if tuner_config is not None:
            apply_remote_tuner(
                model,
                tuner_config,
                gradient_accumulation_steps=1,
                task_type=model_config.task_type,
                model_config=model_config,
            )
        configure_remote_processor(model, template_config, model_config, distributed_config)
    else:
        model = build_model(
            policy_config,
            distributed_config,
            train_config,
            tuner_config,
            device_mesh=sp_mesh,
            quantize_config=quantize_config,
            megatron_config=megatron_config,
            moe_config=moe_config,
            torch_dtype="float32",
        )
        if tuner_config is not None:
            apply_tuner(
                model,
                tuner_config,
                gradient_accumulation_steps=1,
                task_type=model_config.task_type,
            )
        else:
            apply_full_param_freeze(model, train_config)
        model.set_processor(
            InputProcessor,
            padding_free=template_config.padding_free,
            padding_side=template_config.padding_side,
            cp_partition_mode=distributed_config.cp_partition_mode,
            **processor_kwargs(template),
        )
        model.set_template(template)
    # The optimizer reads the tuner's param groups and sizes its LR scheduler from the step budget;
    # the policy metric appends onto the optimizer's metric list (twinkle add_metric contract).
    # The loss rides set_loss by CLASS NAME + kwargs -- one wire form for every transport
    # (local model, Ray proxy, remote served-model handle): see configure_rl_loss.
    configure_rl_loss(model, rl_config)
    configure_optimizer(
        model,
        train_config,
        num_training_steps=max_steps,
        distributed_config=distributed_config,
        tuner_config=tuner_config,
        plugin_config=None,
    )

    colocate = rollout_config.sampler_mode == "colocate"
    sampler = build_rl_sampler(
        model_config=model_config,
        rollout_config=rollout_config,
        template=template,
        tuner_config=tuner_config,
        resource_plan=resource_plan,
        colocate=colocate,
        enable_sampling_replay=rl_config.enable_sampling_replay,
        distributed_config=distributed_config,
        template_config=template_config,
    )
    if remote_run:
        from swift.dev.builders.remote import RemoteWeightSync

        engine = RemoteWeightSync(model, sampler)
    else:
        engine = build_checkpoint_manager(model, sampler, rollout_config, remote=True)
    adapter = None
    if resource_plan is not None and rollout_config.resource_mode == "adaptive":
        adapter = build_cluster_adapter(
            resource_plan, rollout_config, model=model, sampler=sampler, checkpoint_manager=engine)
    params = distill_sampling_params(rl_config, generation_config)
    # One rollout for every on-policy generation (distillation rolls out single-turn); under ray
    # it is an actor on the CPU rollout group, remotely a plain in-process object over the
    # client sampler handle.
    rollout = build_rollout(
        sampler, template, rollout_config, sampling_params=params, remote=not remote_run,
        inflight=inflight_rows(rollout_config, batch_size, max(1, rl_config.num_generations)))

    teachers, teacher_options = _build_teachers(
        model=model,
        model_config=model_config,
        template=template,
        template_config=template_config,
        rl_config=rl_config,
        tuner_config=tuner_config,
        distributed_config=distributed_config,
        signal=signal,
        auxiliary_meshes=auxiliary_meshes,
    )
    worker = DistillWorker(
        teachers=teachers,
        signal=signal,
        template=template,
        gkd_logits_topk=rl_config.gkd_logits_topk,
        sft_alpha=rl_config.sft_alpha,
        train_batch_size=train_batch_size,
        remote_group=None if remote_run else CONTROL_GROUP,
        **teacher_options,
    )
    return Cluster(
        model=model,
        sampler=sampler,
        rollout=rollout,
        engine=engine,
        adapter=adapter,
        teachers=teachers,
        worker=worker,
        sampling_params=params,
        colocate=colocate,
    )


def _build_teachers(
    *,
    model: Any,
    model_config: "ModelConfig",
    template: Any,
    template_config: Optional["TemplateConfig"] = None,
    rl_config: "RLConfig",
    tuner_config: Optional["TunerConfig"],
    distributed_config: "DistributedConfig",
    signal: str,
    auxiliary_meshes: Optional[Dict[str, Any]],
) -> Tuple[List[Any], Dict[str, Any]]:
    """Resolve the bare frozen teachers plus the plain worker options that describe them.

    GKD/OPSD build ONE teacher: the LoRA student's adapter-disabled base (the policy itself, so the
    worker forwards it with ``disable_lora``), a separate frozen model (``teacher_model``), or --
    OPSD's default -- the student's CURRENT weights (dynamic self-distillation). MOPD builds K.
    """
    from swift.dev.builders import build_ref_model
    from swift.dev.recipe.utils import single_teacher_id

    if signal == "multi_logps":
        if rl_config._teacher_use_disable_adapter:
            raise ValueError(
                "MOPD distils from K separate domain teachers; _teacher_use_disable_adapter "
                "(a single adapter-disabled base) is an OPSD/GKD self-distillation mode and "
                "does not apply here."
            )
        if rl_config.teacher_adapters:
            raise ValueError(
                "MOPD builds each teacher as a distinct full model and does not apply "
                "per-teacher adapters; teacher_adapters would be ambiguous. Remove it, or use "
                "OPSD for a single adapter-based teacher."
            )
        if rl_config.offload_teacher_model:
            raise ValueError(
                "offload_teacher_model is not supported for MOPD: it keeps K teachers resident "
                "on their own DeviceGroup to score every trajectory. Leave it off."
            )
        teacher_models = rl_config.teacher_model
        if not teacher_models:
            raise ValueError(
                "MOPD needs at least one teacher: set RLConfig.teacher_model to a list of "
                "model ids (one per domain expert)."
            )
        spread = rl_config.auxiliary_placement == "spread"
        spec = rl_config.teacher_parallel_spec
        teachers = [
            build_ref_model(
                name=(teacher_group(index) if spread else TEACHER_GROUP),
                model_id=model_id,
                model_config=model_config,
                template=template,
                template_config=template_config,
                distributed_config=distributed_config,
                device_mesh=auxiliary_mesh((teacher_group(index) if spread else TEACHER_GROUP), spec, auxiliary_meshes),
                parallel_spec=spec,
                policy_model=model,
                model_type=rl_config.teacher_model_type,
                model_revision=rl_config.teacher_model_revision,
                instance_id=f"teacher{index}",
                deepspeed=rl_config.teacher_deepspeed,
            )
            for index, model_id in enumerate(teacher_models)
        ]
        weights = None if rl_config.teacher_weights is None else list(rl_config.teacher_weights)
        if weights is not None and len(weights) != len(teachers):
            raise ValueError(
                f"teacher_weights has {len(weights)} entries but {len(teachers)} teachers were "
                "built; they must match, or leave teacher_weights unset for uniform weights."
            )
        # pack shares the ONE teacher group (serial forwards); spread gives each its own (parallel).
        placements = [(teacher_group(index) if spread else TEACHER_GROUP) for index in range(len(teachers))]
        return teachers, {"placements": placements, "teacher_weights": weights}

    # GKD / OPSD: one teacher, three shapes.
    if signal == "logits" and rl_config.offload_teacher_model and rl_config._teacher_use_disable_adapter:
        raise ValueError(
            "offload_teacher_model requires a distinct frozen teacher model, not the "
            "adapter-disabled base of the student (which has no separate weights to offload)."
        )
    if signal == "logps" and rl_config.offload_teacher_model:
        raise ValueError(
            "offload_teacher_model is GKD-only: its full-vocab teacher forward wraps "
            "reload/offload, while OPSD scores response-only teacher_logps via a chunked "
            "forward_only with no offload hook. Leave it off."
        )
    if rl_config._teacher_use_disable_adapter:
        if tuner_config is None:
            raise ValueError(
                "rl_config._teacher_use_disable_adapter=True requires a LoRA student "
                "(tuner_config), since it distils from the adapter-disabled base of that "
                "same model."
            )
        # the teacher IS the policy actor with its adapter off -- the worker forwards it disable_lora
        return [model], {"disable_lora": True}
    teacher_model = single_teacher_id(rl_config.teacher_model, algo="distill")
    if teacher_model is None:
        if signal == "logits":
            raise ValueError(
                "GKD needs a teacher: set RLConfig.teacher_model, or use a LoRA student "
                "with _teacher_use_disable_adapter=True to distil from its own frozen base."
            )
        # OPSD's default: dynamic self-distillation -- the student's CURRENT weights score the
        # privileged view on the policy actor (no separate group, adapters ON).
        return [model], {}
    offload = bool(signal == "logits" and rl_config.offload_teacher_model)
    if offload and distributed_config.mode == 'remote':
        # The CPU-offload hook toggles the LOCAL teacher's residency; a served teacher's memory
        # is the server's, so the knob has no consumer remotely.
        raise ValueError(
            "--mode remote does not support --offload_teacher_model (the teacher's CPU offload is "
            'a local memory hook on the in-process teacher). Drop it, or run mode=ray.')
    teacher = build_ref_model(
        name=TEACHER_GROUP,
        model_id=teacher_model,
        model_config=model_config,
        template=template,
        template_config=template_config,
        distributed_config=distributed_config,
        device_mesh=auxiliary_mesh(TEACHER_GROUP, rl_config.teacher_parallel_spec, auxiliary_meshes),
        parallel_spec=rl_config.teacher_parallel_spec,
        adapters=tuple(rl_config.teacher_adapters or ()),
        policy_model=model,
        model_type=rl_config.teacher_model_type,
        model_revision=rl_config.teacher_model_revision,
        deepspeed=rl_config.teacher_deepspeed,
    )
    if offload:
        # GKD's full-vocab memory hook: keep the teacher on CPU between forwards.
        teacher.offload_to_cpu()
    return [teacher], {"offload": offload}


def build_distill_driver(*, cluster: Cluster, rl_config: 'RLConfig', rollout_config: 'RolloutConfig',
                         prompt_dataloader: Any, batch_size: int, num_train_epochs: float,
                         max_steps: Optional[int]) -> Any:
    """Assemble twinkle's StreamingDriver over the cluster for the on-policy updates.

    Distillation scores no outcome channel and has no group verdict: the driver only streams the
    student's rollouts; the teacher scoring happens in the DistillWorker per update.
    """
    from swift.dev.builders.streaming import build_streaming_driver

    return build_streaming_driver(
        dataloader=prompt_dataloader,
        rollout=cluster.rollout,
        rollout_config=rollout_config,
        batch_size=batch_size,
        num_generations=max(1, rl_config.num_generations),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        checkpoint_manager=cluster.engine,
        cluster_adapter=cluster.adapter,
        remote=True,
    )
