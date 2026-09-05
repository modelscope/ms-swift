"""SFT assembly: run_sft orchestration.

Entry point shared by CLI / cookbook / server-less (all call run_sft or copy it). Takes
atomic Configs (composition over inheritance) rather than one aggregate god-config, so each
form assembles only the Configs it needs.

run_sft(...) orchestrates: build_model -> build_template -> build_dataset ->
configure_loss/optimizer -> SFTLoop.fit. The config -> object construction glue lives in
``swift.dev.builders`` (a recipe is a complete, cookbook-copyable loop; the Config->kwargs
translation it delegates to is a separate, lower layer).
"""
from __future__ import annotations
import logging
import math
import os
from typing import TYPE_CHECKING, Any, List, Optional

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

logger = logging.getLogger(__name__)


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
        1. build_template + build_dataset(resumable=True)
        2. step budget + the zero-optimizer-steps fail-fast (BEFORE build_model, so a too-small
           dataset fails before any weight load)
        3. build_model            (ModelConfig -> TransformersModel; for FULL-PARAM resume the
                                   model_id points at the ckpt dir so weights load from it)
        4. apply_tuner (LoRA) -> set_processor -> configure_loss + configure_optimizer
           (order is forced: the tuner creates the optimizer group the rest target)
        5. if resume: model.resume_from_checkpoint(ckpt) -> restores optim/sched/RNG/cur_step
        6.            loop.resume(state) -> seed global_step/micro_step + dataloader skip
        7. SFTLoop.fit

    _save_final (internal): write a final checkpoint after training (default True -- production
    always persists). The underscore marks it test-oriented: the only reason to pass False is to
    get the loss/grad_norm trajectory WITHOUT a checkpoint -- the legacy-vs-dev loss comparison and
    resume bit tests -- which also sidesteps the Megatron mode='local' distributed-save gap. Not a
    knob production callers need.
    """
    _initialize_twinkle(distributed_config)
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


def _initialize_twinkle(distributed_config: DistributedConfig) -> None:
    """Initialize twinkle for EVERY backend -- required for a DeviceMesh to reach the model.

    Megatron + mode='ray' builds the 'model' DeviceGroup the workers hold; everything else
    (including Megatron under torchrun) initializes in 'local' mode. hf + mode='ray' is not
    rejected here -- twinkle supports it, it simply has no dev CLI path.

    See doc.md 'run_sft twinkle 初始化' for why this is load-bearing on the transformers backend
    and why there is no teardown counterpart.
    """
    import twinkle
    from swift.dev.builders import is_megatron_backend
    from twinkle import DeviceGroup

    if is_megatron_backend(distributed_config) and distributed_config.mode == 'ray':
        # The DeviceGroup named 'model' is what build_model's MegatronModel(remote_group='model')
        # targets; the driver orchestrates and is not part of the model process group.
        nproc = distributed_config.nproc_per_node
        if nproc is None:
            raise ValueError("DistributedConfig.nproc_per_node is required for the Megatron backend in mode='ray' "
                             '(it sizes the Ray DeviceGroup). Pass it explicitly -- there is no default.')
        twinkle.initialize(
            mode='ray',
            nproc_per_node=nproc,
            groups=[DeviceGroup(name='model', ranks=list(range(nproc)), device_type='GPU', gpus_per_worker=1)])
    else:
        twinkle.initialize(mode='local')


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
    """The backend-agnostic SFT orchestration body (see run_sft for the step-by-step contract)."""
    from swift.dev.adapter import apply_tuner
    from swift.dev.builders import build_dataset, build_model, build_template, is_megatron_backend
    from swift.dev.config import validate_configs
    from swift.dev.loss import configure_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.processor import InputProcessor
    from swift.dev.recipe.train_loop import SFTLoop, num_optimizer_steps
    from swift.model import get_model_processor

    # Cross-config validation (the single call site -- CLI, cookbook and tests all funnel through
    # run_sft). Runs FIRST so an illegal combination fails in milliseconds, before any dataset
    # encode or weight load. Rules that need a runtime quantity (the zero-optimizer-steps check
    # below, which depends on len(dataloader)) intentionally stay at their call site.
    validate_configs(model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
                     tuner_config)

    # gradient_accumulation_steps: dev defaults to 1 (TrainConfig field default) and does NOT
    # replicate legacy's implicit derivation -- legacy swift.trainers.arguments (arguments.py:252-255)
    # derives `max(1, ceil(16 / per_device_train_batch_size / world_size))` when it is left unset,
    # targeting a global batch of ~16. dev requires the user to set it explicitly (explicit over
    # implicit), so the same argv yields a DIFFERENT effective batch than legacy on multi-GPU /
    # small per-device batch. Documented as a known behavioral difference (see doc P2 contract list).
    ga = train_config.gradient_accumulation_steps
    is_megatron = is_megatron_backend(distributed_config)

    resume_dir = checkpoint_config.resume_from_checkpoint
    resume_only_model = checkpoint_config.resume_only_model

    # Full-param resume loads weights from the ckpt dir instead of the original model id. This
    # mirrors legacy swift.arguments.base_args.BaseArguments._init_ckpt_dir (base_args.py:236-244),
    # which resolves `model` -> `ckpt_dir` via get_ckpt_dir so training continues from the ckpt.
    # LoRA (tuner_config) and resume_only_model keep the original model id: the base weights are
    # unchanged and only the adapter / optimizer state is restored later by resume_from_checkpoint.
    redirect_to_ckpt = bool(resume_dir) and (not resume_only_model) and (tuner_config is None)

    # TODO: refactor to get only processor
    _, processor = get_model_processor(model_config.model, model_type=model_config.model_type, load_model=False)
    template = build_template(template_config, processor)

    # 3. dataloader (list[InputFeature]; twinkle processor collates). One call loads train + val
    # (split-off or separate val_dataset) with a single load_dataset; either loader may be None.
    # template_config is passed so DatasetConfig.cached_dataset (pre-encoded splits written by
    # `swift export --to_cached_dataset`) gets legacy's max_length / truncation_strategy='delete'
    # length filter applied on load.
    dataloader, eval_dataloader = build_dataset(
        dataset_config, template, train_config, distributed_config=distributed_config, template_config=template_config)

    if train_config.max_steps and train_config.max_steps > 0:
        total_opt_steps = train_config.max_steps
    else:
        try:
            micro_per_epoch = len(dataloader)  # micro-batches (already /batch_size) per epoch
        except TypeError:
            micro_per_epoch = 0  # IterableDataset: caller must set max_steps
        epochs = train_config.num_train_epochs
        total_micro = math.ceil(micro_per_epoch * epochs)
        total_opt_steps = num_optimizer_steps(total_micro, ga)

    # Contract 5 (fail-fast): N <= ga micro-batches yield 0 optimizer steps (the GA gate lags one
    # step, so a dataset that never fills a full lagged window would train forward/backward but
    # NEVER update the model). Rather than silently run a no-op training, fail fast with an
    # actionable message so the user fixes data size / max_steps instead of getting a "green" run
    # that changed nothing. Checked BEFORE build_model so we fail before loading heavy weights.
    if total_opt_steps <= 0:
        raise ValueError(f'run_sft: computed {total_opt_steps} optimizer steps -- the dataloader is too small '
                         f'for gradient_accumulation_steps={ga}, or it is a streaming/iterable dataset with no '
                         f'max_steps. Set TrainConfig.max_steps explicitly, or provide enough data.')

    if redirect_to_ckpt:
        import copy as _copy
        model_config_for_build = _copy.copy(model_config)
        model_config_for_build.model = resume_dir
        model = build_model(model_config_for_build, distributed_config, train_config, tuner_config)
    else:
        model = build_model(model_config, distributed_config, train_config, tuner_config)

    # 3.5 tuner (LoRA): MUST run before configure_loss/optimizer so they target the adapter's
    #     optimizer group (twinkle add_adapter_to_model creates+activates it). Skipped for
    #     full-param (tuner_config is None).
    #     On LoRA resume: apply_tuner still runs FIRST to make the model a PeftModel, THEN
    #     resume_from_checkpoint's has_adapter branch loads the saved adapter weights into it
    #     (twinkle load() requires an existing PeftModel, else NotImplementedError).
    if tuner_config is not None:
        apply_tuner(model, tuner_config, gradient_accumulation_steps=ga)

    # Install the dev InputProcessor (drops swift bookkeeping fields like `lengths` before
    # twinkle's collate) on the ACTIVE optimizer group. Must run AFTER apply_tuner, because
    # add_adapter_to_model creates a fresh adapter group with twinkle's default processor;
    # set_processor targets _get_default_group() (the adapter group once a tuner is active).
    #
    # Pass the CLASS on both backends (not an instance): twinkle's set_processor injects
    # device_mesh (+ framework) into the constructor via construct_class, but construct_class
    # returns an *instance* unchanged and silently drops those kwargs. An instance would thus
    # lose device_mesh (needed for CP/SP split) on the transformers path. Passing padding_free
    # from the template config so padding-free training reaches twinkle's packing/patch logic
    # (Megatron additionally requires variable_seq_lengths=True, else set_processor raises).
    model.set_processor(InputProcessor, padding_free=template_config.padding_free)

    # Install the SAME template the dataset encodes with, so ONE implementation (swift's, carrying
    # every TemplateConfig field) produces the training tokens. The model encodes each batch itself
    # whenever the dataloader yields raw messages -- which is the default eager path, where
    # AddLengthPreprocessor keeps the raw row and only adds `lengths`. Without this the model falls
    # back to the bare twinkle Template that _construct_default_optimizer_group builds from model_id
    # alone, which knows no TemplateConfig at all: `--system` was silently ignored and every sample
    # differed from legacy by the system segment.
    #
    # Must run AFTER apply_tuner for the same reason as set_processor (the adapter group is fresh and
    # carries twinkle's defaults). An INSTANCE is passed on purpose -- construct_class returns a
    # Template instance unchanged, so the configured template survives; the processor takes the class
    # instead because it needs device_mesh injected.
    model.set_template(template)

    # 4/5. loss + optimizer/scheduler. Megatron computes CE internally (vocab-parallel) inside
    # forward_backward, so set_loss is a no-op there; skip it and let the default group's loss
    # stand. configure_optimizer is Megatron-aware (routes to the Megatron distributed optimizer).
    if not is_megatron:
        configure_loss(model)
    configure_optimizer(model, train_config, num_training_steps=total_opt_steps)

    # 6. loop (truncate by the same step budget; SFTLoop counts integer optimizer steps, matching
    #    save_steps/eval_steps which are int optimizer-step intervals on the Configs).
    loop = SFTLoop(
        model,
        dataloader,
        max_steps=total_opt_steps,
        num_train_epochs=train_config.num_train_epochs,
        gradient_accumulation_steps=ga,
        max_grad_norm=resolve_max_grad_norm(train_config),
        output_dir=output_dir,
        eval_dataloader=eval_dataloader,
        eval_steps=train_config.eval_steps,
        save_steps=checkpoint_config.save_steps,
    )

    # 4/5(resume). Restore optim/sched/RNG/cur_step (twinkle order: optimizer already built),
    #   then seed loop counters + dataloader skip position. Full weights already loaded in step 1.
    #   LoRA: pass adapter_name so twinkle's has_adapter load() targets the right peft_config
    #   (twinkle defaults adapter_name='' but apply_tuner created 'default').
    if resume_dir:
        resume_kwargs = {'resume_only_model': resume_only_model}
        if tuner_config is not None:
            resume_kwargs['adapter_name'] = 'default'
        state = model.resume_from_checkpoint(resume_dir, **resume_kwargs)
        loop.resume(state)

    history = loop.fit()
    # Persist a final checkpoint unless the caller opted out (_save_final=False, test-only).
    # Periodic saves are governed by SFTLoop.save_steps. _save_final=False is used when only the
    # loss trajectory is wanted (legacy-vs-dev loss comparison / resume bit tests) and writing a
    # checkpoint is unnecessary -- notably the Megatron distributed-optimizer save does not yet
    # work under mode='local' (torchrun). SFTLoop.save writes to os.path.join(output_dir, name);
    # recompute that path locally rather than trust the return value -- in Ray (Megatron) mode
    # save() returns a deferred handle, not a path string.
    if _save_final:
        loop.save('checkpoint-final')
        ckpt_dir = os.path.join(output_dir, 'checkpoint-final')
        # Write a minimal args.json so `swift infer <ckpt>` is self-describing (no manual
        # --model_type/--template). Without it swift falls back to config-based matching, which is
        # ambiguous for qwen2 (qwen2 vs qwen2_gte) / qwen2_5 (many). Aligns with legacy sft output.
        # Master-only: loop.save creates checkpoint-final only on the master rank
        # (twinkle save_pretrained guards on Platform.is_master()), so the other ranks have no such
        # directory and an unguarded open() there raised FileNotFoundError under multi-GPU DDP.
        _write_ckpt_args_json(ckpt_dir, processor, model_config, template_config, tuner_config)
    return history


def _write_ckpt_args_json(ckpt_dir: str,
                          processor: Any,
                          model_config: ModelConfig,
                          template_config: TemplateConfig,
                          tuner_config: Optional['TunerConfig'] = None) -> None:
    """Write the self-describing args.json swift infer reads back from the ckpt.

    Legacy save_args (base_args.py:303-310) dumps the FULL argument dict; infer's read side
    load_args_from_ckpt (base_args.py:246-301) only consumes two lists:
      - force_load_keys (always applied): tuner_type, task_type, bnb_4bit_* -- a MISSING key here
        silently leaves infer on its default (task_type='causal_lm', no adapter), degrading
        seq_cls / reranker / LoRA checkpoints.
      - load_keys (applied only when the current value is None/empty): model, model_type,
        model_revision, torch_dtype, attn_impl, template, system, truncation_strategy, ...
    We write that consumed subset (not the full dict): the two force_load keys that SFT can set
    (tuner_type/task_type; bnb_* is quant, not wired) plus the load_keys dev already knows.
    """
    import os

    import json
    import torch.distributed as dist

    # Only the master rank's checkpoint-final exists (twinkle saves there alone), so writing on any
    # other rank hits a missing directory. Guard, and still makedirs so a lone master run is robust.
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    os.makedirs(ckpt_dir, exist_ok=True)

    model_meta = getattr(processor, 'model_meta', None)
    model_type = getattr(model_meta, 'model_type', None)
    template = template_config.template or getattr(model_meta, 'template', None)
    from swift.dev.version import __version__ as swift_version
    args = {
        # swift_version gates model_type loading in BaseArguments.load_args_from_ckpt
        # (model_type is only honored when swift_version >= 4.0.0.dev); must be present.
        'swift_version': swift_version,
        # force_load_keys: infer applies these regardless of its current value.
        'task_type': model_config.task_type,
        'tuner_type': tuner_config.tuner_type if tuner_config is not None else None,
        # load_keys: infer applies these only when its own value is None/empty.
        'model': model_config.model,
        'model_type': model_type,
        'model_revision': model_config.model_revision,
        'torch_dtype': model_config.torch_dtype,
        'attn_impl': model_config.attn_impl,
        'template': template,
        'system': template_config.system,
        'truncation_strategy': template_config.truncation_strategy,
        'max_length': template_config.max_length,
    }
    args = {k: v for k, v in args.items() if v is not None}
    with open(os.path.join(ckpt_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(args, f, ensure_ascii=False, indent=2)
