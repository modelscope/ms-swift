from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        RLConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

#: rl_types that consult a frozen reference model (the rest build their own baseline in the loss).
_REF_TYPES = frozenset({'dpo', 'kto'})
#: rl_types this recipe handles (ppo is online + needs a critic; grpo/gkd have their own recipes).
_OFFLINE_TYPES = frozenset({'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'rm'})
#: The DeviceGroup name for a full-parameter frozen reference. Defined locally rather than imported from
#: the online run_grpo base because the offline family plans its OWN groups -- it has no sampler, so it does
#: not reuse plan_rl_device_groups -- and the name only has to agree between this module's initialize_twinkle
#: planning and its build_model(remote_group=...) below.
_REF_GROUP = 'ref'


def _reference_needs_group(rl_config: RLConfig, tuner_config: Optional[TunerConfig]) -> bool:
    """Whether _build_reference builds a separate frozen reference actor needing its own DeviceGroup.

    Mirrors ``_build_reference``'s branch exactly so the group PLANNING and the BUILD cannot drift: the
    reference-free types (cpo/orpo/simpo/rm), an explicit ``reference_free`` dpo/kto, and a LoRA run without
    a reference adapter (which reuses the adapter-disabled policy base, 'disable_lora') all need NO group;
    only a full-parameter frozen copy -- full fine-tuning, or an explicit reference adapter -- is a separate
    Ray actor on its own GPUs (RL_PLAN §3.5: full-parameter auxiliaries are heterogeneous, LoRA ones share).
    """
    if rl_config.rl_type not in _REF_TYPES or rl_config.reference_free:
        return False
    return not (tuner_config is not None and not rl_config.ref_adapters)


def run_dpo(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    rl_config: RLConfig,
    tuner_config: Optional[TunerConfig] = None,
    logging_config: Optional[LoggingConfig] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    megatron_config: Optional[MegatronConfig] = None,
    moe_config: Optional[MoEConfig] = None,
    *,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run an offline preference optimisation from atomic Configs. Returns loss history.

    Dispatches on ``rl_config.rl_type`` (one of dpo/kto/cpo/orpo/simpo/rm). The build order is
    :class:`~swift.dev.recipe.common.assembly.TrainAssembly`'s, shared with every other recipe; what this one owns is
    the preference data pipeline (raw rows -> chosen/rejected -> interleaved features) and the optional
    reference-logps forward that dpo/kto add, packaged as a :class:`PreferenceBatchEncoder` the flat offline
    layer calls as its ``micro_step_kwargs`` seam. Because the ASSEMBLY differs (``encode=False`` to keep the
    raw preference columns, the rlhf loss/metric, the reference build), the stages are driven one by one
    instead of through ``build_objects``; the LOOP half then rides the same flat ``run_offline`` every
    other offline recipe uses.
    """
    from swift.dev.loss import configure_rl_loss, configure_rl_metrics
    from swift.dev.builders.optimizer import configure_optimizer
    from swift.dev.recipe.common.assembly import TrainAssembly
    from swift.dev.recipe.common.offline_loop import run_offline
    from swift.dev.recipe.preference.encoder import PreferenceBatchEncoder, preference_log_line

    rl_type = rl_config.rl_type
    if rl_type not in _OFFLINE_TYPES:
        raise ValueError(f'run_dpo handles the offline preference types {sorted(_OFFLINE_TYPES)}, got '
                         f'rl_type={rl_type!r}. Use run_grpo (online RL), run_gkd (distillation) or '
                         'run_ppo instead.')
    assembly = TrainAssembly(
        'run_dpo',
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        rl_config=rl_config,
        output_dir=output_dir,
        logging_config=logging_config,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config)
    assembly.prepare()
    # A FULL-PARAMETER frozen reference is a separate Ray actor on its own 'ref' DeviceGroup (RL_PLAN
    # §3.5); a LoRA / reference-free run plans none. The name/world size here match what _build_reference
    # builds below, decided by the SAME predicate (_reference_needs_group) so plan and build cannot drift.
    ref_remote_group: Optional[str] = None
    auxiliary_groups: List[Tuple[str, int]] = []
    if _reference_needs_group(rl_config, tuner_config):
        ref_remote_group = _REF_GROUP
        auxiliary_groups.append((_REF_GROUP, 1))
    TrainAssembly.initialize_twinkle(
        distributed_config,
        seed=train_config.seed,
        full_determinism=train_config.full_determinism,
        tf32=train_config.tf32,
        auxiliary_groups=auxiliary_groups)

    assembly.build_template()
    # Encode with the preference template mode: 'kto' for kto (allows a missing rejected), else
    # 'rlhf'. RM additionally rides task_type='seq_cls', which makes encode drop the labels.
    assembly.template.set_mode('kto' if rl_type == 'kto' else 'rlhf')

    # Raw (un-encoded) rows: build_dataset(encode=False) keeps the preference columns intact and hands
    # back list[row] batches; PreferenceBatchEncoder encodes + interleaves each batch itself (the SFT encode
    # path only knows single-sequence causal_lm, not chosen/rejected pairs).
    assembly.build_dataset(encode=False)
    assembly.plan_steps()
    assembly.build_model()
    configure_rl_loss(assembly.model, rl_config)
    configure_optimizer(
        assembly.model,
        train_config,
        num_training_steps=assembly.total_opt_steps,
        distributed_config=distributed_config,
        tuner_config=assembly.tuner_config,
        plugin_config=assembly.plugin_config)
    # Register the preference metric (DPOMetric for dpo/cpo/orpo/simpo; a no-op for the unpaired kto and the
    # seq_cls rm) AFTER the optimizer group exists. twinkle accumulates it inside forward_backward from the
    # ref_logps the loop forwards, so rewards/margins + accuracies ride calculate_metric into every record.
    configure_rl_metrics(assembly.model, rl_config)

    # The reference the loop probes for ref-logps: either a separate frozen model or the 'disable_lora'
    # sentinel (the policy with its adapter disabled). Both are read with an empty-loss forward_only
    # (compute_loss=False), so on Megatron neither routes a probe batch through the configured loss.
    reference = _build_reference(
        assembly.template, rl_config, tuner_config, distributed_config, remote_group=ref_remote_group)
    calculate_kl = bool(rl_config.calculate_KL is not False)
    # The preference batch encoder is the one leaf-specific piece the flat offline layer asks for: how a raw
    # preference row becomes forward_backward/forward_only arguments (encode + interleave, the optional
    # reference-logps forward, the KTO layout, the RM seq_cls task). Everything else -- periodic val-loss eval,
    # best-model tracking, callbacks, hub push, save/resume -- is the SAME flat StepRunner / Evaluator /
    # CheckpointSaver cadence the other offline recipes ride, wired from offline_component_kwargs(). The
    # encoder's console line (DPO margin/accuracy) is injected as the recorder's log_formatter so the
    # hook-free StepRunner still surfaces it.
    encoder = PreferenceBatchEncoder(
        assembly.model,
        assembly.template,
        rl_type=rl_type,
        reference=reference,
        calculate_kl=calculate_kl,
        gradient_accumulation_steps=assembly.ga)
    return run_offline(
        assembly,
        save_final=_save_final,
        micro_step_kwargs=encoder,
        log_formatter=preference_log_line)


def _build_reference(template: Any,
                     rl_config: RLConfig,
                     tuner_config: Optional[TunerConfig],
                     distributed_config: DistributedConfig,
                     *,
                     remote_group: Optional[str]) -> Optional[Any]:
    """The reference the loop consults for ref-logps, or None for the reference-free types.

    Returns one of:
      - None: cpo/orpo/simpo/rm (reference-free), dpo/kto with ``reference_free=True``, OR dpo/kto under
        LoRA -- LoRA needs no object because the reference is the SAME model with the adapter disabled,
        which the loop reaches via ``forward_only(disable_lora=True)`` (signalled by the sentinel string
        'disable_lora').
      - 'disable_lora': dpo/kto + LoRA (adapter-disabled base is the reference).
      - a frozen TrainableModel: dpo/kto + full fine-tuning (a separate ref_model copy), a Ray actor on the
        ``remote_group`` DeviceGroup the caller planned for it.
    """
    if rl_config.rl_type not in _REF_TYPES:
        return None
    if rl_config.reference_free:
        # Reference-free DPO/KTO: DPOLoss.reference_free anchors on the policy alone, so no reference is
        # built and the loop passes no ref-logps. This is wired in lockstep with configure forwarding
        # ``reference_free`` to the loss -- with a None reference but reference_free=False, DPOLoss would
        # hit its zero-loss fallthrough and silently train on nothing.
        return None
    if tuner_config is not None and not rl_config.ref_adapters:
        # LoRA without an explicit reference adapter: the frozen base is the reference. It reuses the
        # policy's already-loaded base (no second copy), so it needs no group and no remote_group.
        return 'disable_lora'
    # Full fine-tuning, or an explicit reference adapter: load a separate frozen reference.
    return _load_frozen_reference(template, rl_config, distributed_config, remote_group=remote_group)


def _load_frozen_reference(template: Any,
                           rl_config: RLConfig,
                           distributed_config: DistributedConfig,
                           *,
                           remote_group: Optional[str]) -> Any:
    """Build a frozen reference model from rl_config.ref_model, sharing the policy's processor/template.

    Reuses the policy's InputProcessor + template so both models encode a batch identically (same
    padding, same shift), which is what lets the loop feed one interleaved feature list to both and
    line up the per-token logps. No optimizer/tuner: the reference is only ever forward_only'd.

    The full-parameter reference is an independent weight copy, so it is heterogeneous (RL_PLAN §3.5): a
    separate Ray actor on its own ``remote_group`` DeviceGroup, built with the run's backend (basic
    principle 1 -- ``forward_only`` is identical for megatron and transformers).
    """
    from swift.dev.builders import build_model, frozen_auxiliary_distributed_config
    from swift.dev.config import ModelConfig
    from swift.dev.recipe.common.assembly import configure_frozen_adapter

    if rl_config.ref_model is None:
        raise ValueError('dpo/kto full fine-tuning needs a reference model, but RLConfig.ref_model is None. '
                         'It is normally defaulted to the policy model by process.py::_derive_rlhf_ref_model; '
                         'pass --ref_model explicitly if you bypassed config processing.')
    if remote_group is None:
        raise ValueError('A full-parameter frozen reference must be built on its own DeviceGroup, but '
                         'remote_group is None. _reference_needs_group planned no group for it, which '
                         'contradicts reaching _load_frozen_reference -- the plan and the build have drifted.')
    # A bare ModelConfig pointing at ref_model, no tuner and no optimizer: the reference is only ever
    # forward_only'd. Its DistributedConfig inherits the run backend but is forced to ray on world_size=1.
    ref_cfg = ModelConfig(model=rl_config.ref_model)
    ref_cfg.model_type = rl_config.ref_model_type
    ref_cfg.model_revision = rl_config.ref_model_revision
    ref = build_model(
        ref_cfg, frozen_auxiliary_distributed_config(distributed_config, 1), remote_group=remote_group)
    return configure_frozen_adapter(ref, template, rl_config.ref_adapters, role='ref')
