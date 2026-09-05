"""SFT assembly: run_sft orchestration.

Entry point shared by CLI / cookbook / server-less (all call run_sft or copy it). Takes
atomic Configs (composition over inheritance) rather than one aggregate god-config, so each
form assembles only the Configs it needs.

The orchestration itself -- validate/plugins -> template -> dataloaders -> step budget -> model ->
loss/optimizer -> loop -> final checkpoint -- is :class:`~swift.dev.recipe.assembly.TrainAssembly`,
shared with every other training recipe; SFT's own contribution is one line, the objective. The
config -> object construction glue lives in ``swift.dev.builders`` (a recipe is a complete,
cookbook-copyable loop; the Config->kwargs translation it delegates to is a separate, lower layer).
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )


def run_sft(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    tuner_config: Optional[TunerConfig] = None,
    *,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run a minimal SFT from atomic Configs. Returns loss/grad_norm history.

    Orchestration (backend/mode-agnostic; SFT has no RL policy hooks). Resume follows the
    twinkle-locked order (configure_optimizer BEFORE resume_from_checkpoint, since the latter
    restores optimizer/scheduler state into an already-built optimizer):
        1. plugins + cross-config validation
        2. build_template + build_dataset(resumable=True)
        3. step budget + the zero-optimizer-steps fail-fast (BEFORE build_model, so a too-small
           dataset fails before any weight load)
        4. build_model            (ModelConfig -> TransformersModel; for FULL-PARAM resume the
                                   model_id points at the ckpt dir so weights load from it)
        5. apply_tuner (LoRA) -> set_processor/set_template -> configure_loss + configure_optimizer
           (order is forced: the tuner creates the optimizer group the rest target)
        6. if resume: model.resume_from_checkpoint(ckpt) -> restores optim/sched/RNG/cur_step
        7.            loop.resume(state) -> seed global_step/micro_step + dataloader skip
        8. SFTLoop.fit

    _save_final (internal): write a final checkpoint after training (default True -- production
    always persists). The underscore marks it test-oriented: the only reason to pass False is to
    get the loss/grad_norm trajectory WITHOUT a checkpoint -- the legacy-vs-dev loss comparison and
    resume bit tests -- which also sidesteps the Megatron mode='local' distributed-save gap. Not a
    knob production callers need.
    """
    from swift.dev.recipe.assembly import TrainAssembly

    TrainAssembly.initialize_twinkle(distributed_config)
    return _run_sft_body(
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        output_dir=output_dir,
        _save_final=_save_final)


def _run_sft_body(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    tuner_config: Optional[TunerConfig] = None,
    *,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """The backend-agnostic SFT orchestration body (see run_sft for the step-by-step contract).

    Separate from ``run_sft`` because twinkle initialization must NOT be repeated: the Megatron Ray
    path initializes once and then drives this body inside the workers.
    """
    from swift.dev.builders import is_megatron_backend
    from swift.dev.loss import configure_loss
    from swift.dev.recipe.assembly import TrainAssembly

    def sft_loss(model) -> None:
        # Megatron computes CE internally (vocab-parallel) inside forward_backward, so set_loss is a
        # no-op there; skip it and let the default group's loss stand.
        if not is_megatron_backend(distributed_config):
            configure_loss(model)

    return TrainAssembly(
        'run_sft',
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        output_dir=output_dir,
    ).fit(sft_loss, save_final=_save_final)
