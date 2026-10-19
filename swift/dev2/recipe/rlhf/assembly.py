# Copyright (c) ModelScope Contributors. All rights reserved.
# The offline RLHF family's assembly: the preference config -> object stages + the reference.
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

from swift.dev.utils.logger import get_logger  # TODO(dev2): bridge until the utils batch

from swift.dev2.recipe.utils import (
    REF_GROUP, initialize_supervised_twinkle, plan_supervised_steps, process_and_validate,
    resolve_step_intervals, write_ckpt_args_json)

if TYPE_CHECKING:
    from swift.dev2.config import (
        CheckpointConfig, DatasetConfig, DistributedConfig, LoggingConfig, MegatronConfig, ModelConfig,
        MoEConfig, QuantizeConfig, RLConfig, RewardConfig, TemplateConfig, TrainConfig, TunerConfig)

logger = get_logger()

#: rl_types that consult a frozen reference model (the rest build their own baseline in the loss).
_REF_TYPES = frozenset({'dpo', 'kto'})


def _reference_needs_group(rl_config: 'RLConfig', tuner_config: Optional['TunerConfig']) -> bool:
    """Whether the reference build needs its own DeviceGroup (plan and build share this predicate).

    The reference-free types (cpo/orpo/simpo/rm), an explicit ``reference_free`` dpo/kto, and a LoRA
    run without a reference adapter (the shared-weights view) need NO group; only a full-parameter
    frozen copy -- full fine-tuning, or an explicit reference adapter -- is a separate Ray actor.
    """
    if rl_config.rl_type not in _REF_TYPES or rl_config.reference_free:
        return False
    return not (tuner_config is not None and not rl_config.ref_adapters)


def _build_reference(template: Any, model_config: 'ModelConfig', rl_config: 'RLConfig',
                     tuner_config: Optional['TunerConfig'],
                     distributed_config: 'DistributedConfig', model: Any,
                     *, remote_group: Optional[str]) -> Optional[Any]:
    """The frozen reference the encoder probes for ref-logps, or None for the reference-free types.

    LoRA without a reference adapter returns the POLICY itself -- the encoder derives
    ``disable_lora`` from that identity (the same shared-weights view the online recipes use). Full
    fine-tuning, or an explicit reference adapter, builds a separate frozen copy on the
    ``remote_group`` DeviceGroup the caller planned via the SAME predicate, so the two cannot drift.
    """
    if rl_config.rl_type not in _REF_TYPES or rl_config.reference_free:
        # Reference-free dpo/kto: DPOLoss.reference_free anchors on the policy alone; wired in
        # lockstep with configure_rl_loss forwarding ``reference_free`` to the loss.
        return None
    from swift.dev2.builders import build_ref_model

    if tuner_config is not None and not rl_config.ref_adapters:
        return model
    if rl_config.ref_model is None:
        raise ValueError(
            'dpo/kto full fine-tuning needs a reference model, but RLConfig.ref_model is None. It is '
            'normally defaulted to the policy model by process.py::_derive_rlhf_ref_model; pass '
            '--ref_model explicitly if you bypassed config processing.')
    if remote_group is None:
        raise ValueError('A full-parameter frozen reference must be built on its own DeviceGroup, but '
                         'remote_group is None: _reference_needs_group planned no group for it -- the '
                         'plan and the build have drifted.')
    return build_ref_model(
        name=remote_group, model_id=rl_config.ref_model, model_config=model_config,
        template=template, distributed_config=distributed_config,
        adapters=tuple(rl_config.ref_adapters or ()), model_type=rl_config.ref_model_type,
        model_revision=rl_config.ref_model_revision)


class PreferenceCluster(NamedTuple):
    """The supervised cluster's handles plus the one preference-specific field: the reference.

    ``reference`` is a separate frozen model, the POLICY itself (the shared-weights LoRA view), or
    None for the reference-free types -- the encoder switches on that shape.
    """

    model: Any
    processor: Any
    template: Any
    dataloader: Any
    eval_dataloader: Any
    total_opt_steps: int
    task_type: Optional[str]
    forward_task: str
    reference: Optional[Any]


def build_preference_objects(*, recipe: str, model_config: 'ModelConfig',
                             template_config: 'TemplateConfig', dataset_config: 'DatasetConfig',
                             train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
                             checkpoint_config: Optional['CheckpointConfig'],
                             rl_config: 'RLConfig', reward_config: Optional['RewardConfig'],
                             tuner_config: Optional['TunerConfig'],
                             logging_config: Optional['LoggingConfig'],
                             quantize_config: Optional['QuantizeConfig'],
                             megatron_config: Optional['MegatronConfig'],
                             moe_config: Optional['MoEConfig'],
                             output_dir: str = 'output') -> PreferenceCluster:
    """Run the preference config -> object stages in the locked order, up to (not including) the loop.

    Same stage order as the supervised family, with the preference differences inline: the full-
    parameter reference gets its own DeviceGroup at twinkle init, the template rides the preference
    encode mode, the dataset stays RAW (``encode=False``; the encoder encodes each batch), and the
    loss / metric are the rlhf ones. The reference build shares the group-planning predicate.
    """
    from swift.dev2.builders import (
        apply_full_param_freeze, build_dataset, build_dp_mesh, build_model, build_template,
        load_model_processor)
    from swift.dev2.builders.adapter import apply_tuner
    from swift.dev2.builders.optimizer import configure_optimizer
    from swift.dev2.loss import configure_rl_loss, configure_rl_metrics
    from swift.dev2.processor import InputProcessor

    rl_type = rl_config.rl_type
    ref_remote_group = REF_GROUP if _reference_needs_group(rl_config, tuner_config) else None
    initialize_supervised_twinkle(
        distributed_config, seed=train_config.seed, full_determinism=train_config.full_determinism,
        tf32=train_config.tf32,
        auxiliary_groups=([] if ref_remote_group is None else [(ref_remote_group, 1)]))
    process_and_validate(
        model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, logging_config, quantize_config, megatron_config,
        moe_config, rl_config=rl_config, reward_config=reward_config)
    sp = template_config.sequence_parallel_size
    sp_mesh = build_dp_mesh(distributed_config, sp) if sp > 1 else None
    # rm rides task_type='seq_cls' (process.py set it before the loss defaults read it), which also
    # threads the score head into LoRA's modules_to_save; the other types keep the Config's value.
    task_type = model_config.task_type
    _, processor = load_model_processor(model_config)
    kwargs = {'task_type': task_type} if task_type else {}
    template = build_template(template_config, processor, **kwargs)
    # The preference encode mode: 'kto' allows a missing rejected turn, 'rlhf' pairs chosen/rejected.
    template.set_mode('kto' if rl_type == 'kto' else 'rlhf')
    write_ckpt_args_json(
        output_dir, processor, model_config, template_config, tuner_config, quantize_config,
        task_type=task_type)
    # RAW rows: encode=False keeps the chosen/rejected columns intact; the encoder encodes each
    # batch itself (the SFT encode path only knows single-sequence causal_lm).
    dataloader, eval_dataloader = build_dataset(
        dataset_config, template, train_config, distributed_config,
        template_config=template_config, device_mesh=sp_mesh, encode=False)
    total_opt_steps = plan_supervised_steps(recipe, train_config, dataloader)
    resolve_step_intervals(train_config, checkpoint_config, total_opt_steps)
    resume_dir = checkpoint_config.resume_from_checkpoint if checkpoint_config is not None else None
    build_config = model_config
    if resume_dir and tuner_config is None:
        build_config = copy.copy(model_config)
        build_config.model = resume_dir
    model = build_model(
        build_config, distributed_config, train_config, tuner_config, device_mesh=sp_mesh,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config)
    if tuner_config is not None:
        apply_tuner(
            model, tuner_config, gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            task_type=task_type)
    else:
        apply_full_param_freeze(model, train_config)
    model.set_processor(
        InputProcessor, padding_free=template_config.padding_free, padding_side=template_config.padding_side,
        cp_partition_mode=distributed_config.cp_partition_mode, task_type=task_type)
    model.set_template(template)
    configure_rl_loss(model, rl_config)
    configure_optimizer(
        model, train_config, num_training_steps=total_opt_steps, distributed_config=distributed_config,
        tuner_config=tuner_config, plugin_config=None)
    # AFTER the optimizer group exists (twinkle's add_metric contract): the DPO metric accumulates
    # inside forward_backward from the ref_logps the loop forwards, so margins/accuracies ride out.
    configure_rl_metrics(model, rl_config)
    reference = _build_reference(
        template, model_config, rl_config, tuner_config, distributed_config, model,
        remote_group=ref_remote_group)
    return PreferenceCluster(
        model=model, processor=processor, template=template, dataloader=dataloader,
        eval_dataloader=eval_dataloader, total_opt_steps=total_opt_steps, task_type=task_type,
        forward_task='causal_lm', reference=reference)
