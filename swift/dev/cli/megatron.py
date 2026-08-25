"""CLI entry mapping legacy ``MegatronSftArguments`` onto dev's atomic Configs.

Separate from ``cli/sft.py`` because ``MegatronSftArguments`` is an INDEPENDENT hierarchy
(MegatronSftArguments -> MegatronBaseArguments -> MegatronArguments + BaseArguments), not an
``SftArguments`` subclass. A name-based copy alone -- what ``cli/sft.py`` does -- leaves 34 of
TrainConfig's 58 fields, 8 DistributedConfig fields (including ``backend``/``mode``/
``nproc_per_node``) and 7 CheckpointConfig fields at their dev defaults on this surface, because
Megatron names the same hyperparameters differently (``lr``/``train_iters``/``micro_batch_size``/
``lr_decay_style``/``adam_eps``) and has no notion of dev's launch knobs. ``cli/sft.py`` therefore
refuses Megatron args via a ``train_iters`` sentinel and this module owns the Megatron mapping.

The load-bearing deliverable is ``audit_coverage`` (and the test that runs it): every dev Config
field must be accounted for as exactly one of NAME_HIT / RENAMED / DERIVED / ABSENT, so adding a
Config field later cannot silently go unmapped -- which is the failure mode that motivated the
sentinel in the first place.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                   TrainConfig, TunerConfig)
    from swift.megatron.arguments import MegatronSftArguments

# --- Explicit renames -------------------------------------
RENAMES: Dict[str, Dict[str, str]] = {
    'ModelConfig': {
        'attn_impl': 'attention_backend',
    },
    'TrainConfig': {
        # Without this LoRA silently trains at TrainConfig's 1e-5 default instead of legacy's 1e-4
        # (break-change list #1: legacy resolves lr per tuner_type in megatron_args._set_default).
        'learning_rate': 'lr',
        'max_steps': 'train_iters',
        'per_device_train_batch_size': 'micro_batch_size',
        'adam_epsilon': 'adam_eps',
        # legacy priority: lr_warmup_fraction wins over lr_warmup_iters when set
        # (megatron_lm_utils.py). Both are mapped; warmup_budget() reproduces that priority.
        'warmup_ratio': 'lr_warmup_fraction',
        'warmup_steps': 'lr_warmup_iters',
        'lr_scheduler_type': 'lr_decay_style',  # value also translated (see _decay_style_to_swift)
        # Same knob under two legacy names: Megatron clips from OptimizerConfig.clip_grad, HF from
        # max_grad_norm. dev merged them under max_grad_norm (legacy's Megatron path never read
        # max_grad_norm, so setting it there was silently dropped -- that drop was the defect).
        # Mapping it here is what makes `--clip_grad` on Megatron argv keep working; the dev field
        # `clip_grad` remains as a deprecated alias for programmatic callers.
        'max_grad_norm': 'clip_grad',
    },
}

# --- Derived fields: computed, not copied. --------------------------------------------------------
DERIVED: Dict[str, Tuple[str, ...]] = {
    'TrainConfig': ('gradient_accumulation_steps', ),
    'DistributedConfig': ('backend', 'mode', 'nproc_per_node'),
}

# --- Fields with a Megatron counterpart that dev deliberately routes elsewhere. -------------------
# Distinct from ABSENT: the Megatron surface DOES have an equivalent flag, but dev carries the
# setting on a DIFFERENT Config field, so mapping the pair would create two names for one knob.
# Value is the Megatron arg that supersedes the dev field, for the audit report.
SUPERSEDED: Dict[str, Dict[str, str]] = {
    'TrainConfig': {
        # Megatron's `optimizer` is the optimizer TYPE (adam/sgd/muon); dev's `optim` is an HF
        # optimizer name (adamw_torch_fused/...). Not synonyms, so not a rename.
        # _reject_unmappable refuses `--optimizer` other than 'adam'.
        'optim': 'optimizer',
        # Megatron drives recompute off DistributedConfig.recompute_granularity/method/num_layers,
        # not a boolean; validate_configs._check_megatron_recompute cross-validates the two.
        'gradient_checkpointing': 'recompute_granularity',
    },
}

# --- Fields with no counterpart on the Megatron surface: dev default stands. ----------------------
# Registered explicitly so the audit can tell "deliberately absent" from "forgotten". Each entry is
# a field the Megatron CLI cannot set; the dev default applies. Grouped by reason in the comments.
ABSENT: Dict[str, Tuple[str, ...]] = {
    'TrainConfig': (
        'optim_args',
        # HF-only trainer knobs (validate_configs._HF_ONLY rejects them on the Megatron backend).
        'gradient_checkpointing_kwargs',
        'full_determinism',
        'use_liger_kernel',
        'neftune_noise_alpha',
        'router_aux_loss_coef',
        'use_logits_to_keep',
        'predict_with_generate',
        'eval_use_evalscope',
        # No Megatron equivalent / not wired on this surface.
        'lr_scheduler_kwargs',
        'per_device_eval_batch_size',
        'max_epochs',
        'average_tokens_across_devices',
        'acc_strategy',
        'ds3_gather_for_generation',
        'eval_strategy',
        'eval_on_start',
        'eval_metric',
        'eval_dataset',
        'eval_dataset_args',
        'eval_limit',
        'eval_generation_config',
        'extra_eval_args',
        'early_stop_interval',
    ),
    'DistributedConfig': (
        # Mutually exclusive with the Megatron backend by construction.
        'deepspeed',
        'zero_hpz_partition_size',
        'deepspeed_autotp_size',
        'fsdp',
        'ddp_find_unused_parameters',
    ),
    'CheckpointConfig': (
        # Megatron resume is `load` + `finetune` + `no_load_optim`, not resume_from_checkpoint;
        # designing that mapping is a separate task (odoc注4 tail), so it stays unset here.
        'resume_from_checkpoint',
        'resume_only_model',
        'ignore_data_skip',
        # HF-serialization knobs with no Megatron dist-checkpoint counterpart.
        'safe_serialization',
        'save_on_each_node',
        'save_only_model',
        'use_flash_ckpt',
    ),
    'TunerConfig': (
        # Megatron-SWIFT supports LoRA only; the other PEFT families and the HF-side optimizer
        # plugins (galore/lisa) have no Megatron surface at all.
        'target_parameters',
        'lorap_lr_ratio',
        'lorap_emb_lr',
        'use_dora',
        'init_weights',
        'trainable_token_indices',
        'lora_ga_batch_size',
        'lora_ga_iters',
        'lora_ga_max_length',
        'lora_ga_direction',
        'lora_ga_scale',
        'lora_ga_stable_gamma',
        'fourier_n_frequency',
        'fourier_scaling',
        'boft_block_size',
        'boft_block_num',
        'boft_dropout',
        'vera_rank',
        'vera_projection_prng_key',
        'vera_dropout',
        'vera_d_initial',
        'adalora_target_r',
        'adalora_init_r',
        'adalora_tinit',
        'adalora_tfinal',
        'adalora_deltaT',
        'adalora_beta1',
        'adalora_beta2',
        'adalora_orth_reg_weight',
        'llamapro_num_new_blocks',
        'llamapro_num_groups',
        'reft_layers',
        'reft_rank',
        'reft_intervention_type',
        'reft_args',
        'use_galore',
        'galore_target_modules',
        'galore_rank',
        'galore_update_proj_gap',
        'galore_scale',
        'galore_proj_type',
        'galore_optim_per_parameter',
        'galore_with_embedding',
        'galore_quantization',
        'galore_proj_quant',
        'galore_proj_bits',
        'galore_proj_group_size',
        'galore_cos_threshold',
        'galore_gamma_proj',
        'galore_queue_size',
        'lisa_activated_layers',
        'lisa_step_interval',
    ),
}

_CONFIG_ORDER = ('ModelConfig', 'TemplateConfig', 'DatasetConfig', 'TrainConfig', 'DistributedConfig',
                 'CheckpointConfig', 'TunerConfig')


def _config_classes() -> Dict[str, type]:
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                   TrainConfig, TunerConfig)
    return {
        'ModelConfig': ModelConfig,
        'TemplateConfig': TemplateConfig,
        'DatasetConfig': DatasetConfig,
        'TrainConfig': TrainConfig,
        'DistributedConfig': DistributedConfig,
        'CheckpointConfig': CheckpointConfig,
        'TunerConfig': TunerConfig,
    }


def audit_coverage(arg_names: Optional[set] = None) -> Dict[str, Dict[str, List[str]]]:
    """Classify every dev Config field against the Megatron arg surface.

    Returns ``{ConfigName: {'name_hit': [...], 'renamed': [...], 'derived': [...],
    'superseded': [...], 'absent': [...], 'unaccounted': [...]}}``.
    ``unaccounted`` MUST be empty: a field there is one this mapping would silently leave at its dev
    default while the user thinks their CLI flag took effect. The guard test asserts exactly that, so
    a newly added Config field fails loudly here instead of during a training run.

    ``superseded`` is checked BEFORE ``name_hit``: the field exists on both surfaces but dev routes
    the setting through another Config field, so the same-name copy would be misleading.

    arg_names defaults to the real MegatronSftArguments field set; tests may inject a set.
    """
    if arg_names is None:
        from swift.megatron.arguments import MegatronSftArguments
        arg_names = {f.name for f in dataclasses.fields(MegatronSftArguments)}

    report: Dict[str, Dict[str, List[str]]] = {}
    for cfg_name, cls in _config_classes().items():
        renames = RENAMES.get(cfg_name, {})
        derived = set(DERIVED.get(cfg_name, ()))
        superseded = set(SUPERSEDED.get(cfg_name, {}))
        absent = set(ABSENT.get(cfg_name, ()))
        buckets: Dict[str, List[str]] = {
            'name_hit': [],
            'renamed': [],
            'derived': [],
            'superseded': [],
            'absent': [],
            'unaccounted': []
        }
        for f in dataclasses.fields(cls):
            if f.name in renames:
                buckets['renamed'].append(f.name)
            elif f.name in derived:
                buckets['derived'].append(f.name)
            elif f.name in superseded:
                buckets['superseded'].append(f.name)
            elif f.name in arg_names:
                # Same name on both surfaces -> the plain copy in _fill_from_args handles it.
                buckets['name_hit'].append(f.name)
            elif f.name in absent:
                buckets['absent'].append(f.name)
            else:
                buckets['unaccounted'].append(f.name)
        report[cfg_name] = buckets
    return report


def _decay_style_to_swift(decay_style: str) -> str:
    """Megatron lr_decay_style -> dev TrainConfig.lr_scheduler_type (reverse of naming's map).

    The forward map is NOT injective: both 'constant'/'constant_with_warmup' produce 'constant' and
    both 'cosine'/'cosine_with_min_lr' produce 'cosine'. Rule: prefer the swift name IDENTICAL to
    the Megatron style, since that is the round-trip-stable choice; if no identical name exists the
    reverse is genuinely ambiguous and we refuse rather than pick one (choosing e.g.
    'cosine_with_min_lr' would additionally trip validate_configs' min_lr requirement).
    """
    from swift.dev.naming import _MEGATRON_DECAY_STYLE_MAP

    candidates = [swift for swift, mega in _MEGATRON_DECAY_STYLE_MAP.items() if mega == decay_style]
    if not candidates:
        raise NotImplementedError(f'--lr_decay_style {decay_style!r} has no dev lr_scheduler_type. '
                                  f'Supported Megatron styles: {sorted(set(_MEGATRON_DECAY_STYLE_MAP.values()))}.')
    if decay_style in candidates:
        return decay_style
    raise NotImplementedError(f'--lr_decay_style {decay_style!r} maps ambiguously back to dev lr_scheduler_type '
                              f'(candidates: {sorted(candidates)}) and none matches the style name itself. '
                              f'Set TrainConfig.lr_scheduler_type explicitly through the programmatic API instead.')


def _derive_gradient_accumulation_steps(args: 'MegatronSftArguments', world_size: int) -> int:
    """GA = global_batch_size / (micro_batch_size * dp), dp = world / (tp*pp*cp).

    Megatron expresses the step budget as a global batch size; dev expresses it as GA. Fail-fast on
    a non-integral result: silently flooring would train on a different global batch than the flag
    asked for.
    """
    tp = args.tensor_model_parallel_size
    pp = args.pipeline_model_parallel_size
    cp = args.context_parallel_size
    model_parallel = tp * pp * cp
    if world_size % model_parallel != 0:
        raise ValueError(f'world_size={world_size} is not divisible by tp*pp*cp={model_parallel} '
                         f'(tp={tp}, pp={pp}, cp={cp}).')
    dp = world_size // model_parallel
    denom = args.micro_batch_size * dp
    gbs = args.global_batch_size
    if gbs % denom != 0:
        raise ValueError(f'global_batch_size={gbs} is not divisible by micro_batch_size*dp={denom} '
                         f'(micro_batch_size={args.micro_batch_size}, dp={dp}); dev expresses the budget as '
                         f'gradient_accumulation_steps, so the division must be exact.')
    return gbs // denom


def _fill_from_args(config, args: 'MegatronSftArguments'):
    """Same-name copy; None means "leave the Config default" (defaults live in the Config)."""
    for f in dataclasses.fields(config):
        if not hasattr(args, f.name):
            continue
        value = getattr(args, f.name)
        if value is not None:
            setattr(config, f.name, value)
    return config


def _attn_backend_name(value: object) -> Optional[str]:
    """legacy --attention_backend -> the string dev's ModelConfig.attn_impl carries.

    legacy's post-init converts the flag to a mcore AttnBackend enum
    (megatron_args._init_attention_backend), so this may receive either the raw string or the enum
    depending on how far post-init has run. Normalised back to the plain name because dev's Config is
    a serialisable description of the run -- the enum is produced later, at build time, by
    naming.resolve_megatron_attn_backend.

    Note the version-pinning asymmetry: legacy accepts --attention_backend flash_2|flash_3|flash_4 and
    collapses them to 'flash' after mutating transformer_engine globals. If post-init already ran, the
    pin is therefore invisible here (it arrives as AttnBackend.flash) and dev will run whichever FA
    version TE picks; if it has not, the raw 'flash_N' reaches resolve_megatron_attn_backend and is
    rejected. Either way dev does not silently claim to honour the pin.
    """
    if value is None:
        return None
    return getattr(value, 'name', None) or str(value)


def _dtype_name(value: object) -> Optional[str]:
    if value is None:
        return None
    name = str(value)
    return name.split('.')[-1] if name.startswith('torch.') else name


def _reject_unmappable(args: 'MegatronSftArguments') -> None:
    """Refuse Megatron flags dev cannot honour, instead of dropping them on the floor."""
    # Megatron's `optimizer` is the optimizer TYPE. dev/twinkle only wires Adam on this path
    optimizer = getattr(args, 'optimizer', None)
    if optimizer is not None and optimizer != 'adam':
        raise NotImplementedError(
            f'--optimizer {optimizer!r} is not supported by the dev Megatron pipeline yet (only '
            "'adam'). This is a KNOWN GAP vs legacy, which supports adam/sgd/muon/dist_muon plus 10 "
            'muon_* knobs. The blocker is upstream in twinkle, not here: MegatronModel.set_optimizer '
            "accepts only 'MegatronOptimizer'/'default'/'Adam', and _create_megatron_optimizer "
            "hardcodes optimizer='adam' inside OptimizerConfig(...) while also forwarding **kwargs, "
            'so an optimizer= kwarg raises a duplicate-keyword TypeError. TODO: add an '
            'optimizer-type surface to twinkle, then map --optimizer + muon_* here.')


def megatron_args_to_configs(
    args: 'MegatronSftArguments',
    *,
    world_size: Optional[int] = None,
) -> Tuple['ModelConfig', 'TemplateConfig', 'DatasetConfig', 'TrainConfig', 'DistributedConfig', 'CheckpointConfig',
           Optional['TunerConfig']]:
    """MegatronSftArguments -> the same 7 Configs ``cli/sft.py::args_to_configs`` returns.

    world_size defaults to the launcher's WORLD_SIZE (needed only to derive GA from
    global_batch_size); pass it explicitly in tests.
    """
    import os

    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                   TrainConfig, TunerConfig)

    if not hasattr(args, 'train_iters'):
        raise ValueError('megatron_args_to_configs expects MegatronSftArguments (no train_iters found). Use '
                         'swift.dev.cli.sft.args_to_configs for the HF-surface SftArguments.')
    _reject_unmappable(args)

    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))

    # Model/Template/Dataset are name-compatible on this surface apart from the attention kernel,
    # which legacy exposes under a different flag on each backend (see RENAMES['ModelConfig']).
    model_config = _fill_from_args(ModelConfig(), args)
    model_config.torch_dtype = _dtype_name(args.torch_dtype)
    # Overwrite the same-name copy of attn_impl: on this surface the attention kernel comes from
    # --attention_backend (see RENAMES['ModelConfig']). Done after _fill_from_args because that copied
    # the transformers-surface attn_impl, which legacy's Megatron path ignores.
    for cfg_field, arg_name in RENAMES['ModelConfig'].items():
        setattr(model_config, cfg_field, _attn_backend_name(getattr(args, arg_name, None)))
    template_config = _fill_from_args(TemplateConfig(), args)
    dataset_config = _fill_from_args(DatasetConfig(), args)

    train_config = _fill_from_args(TrainConfig(), args)
    if getattr(args, 'weight_decay_incr_style', 'constant') == 'constant':
        train_config.start_weight_decay = None
        train_config.end_weight_decay = None
    for cfg_field, arg_name in RENAMES['TrainConfig'].items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(train_config, cfg_field, value)
    # `clip_grad` is the ONE Megatron arg that also name-hits a dev field, so _fill_from_args above
    # already copied it into the deprecated `clip_grad` alias while the rename put it on
    # `max_grad_norm`. Leaving both populated would make resolve_max_grad_norm report "both set" on
    # every run that passes --clip_grad -- a warning about a conflict the user never created. The
    # rename is authoritative, so clear the alias and keep one field carrying the value.
    train_config.clip_grad = None
    train_config.lr_scheduler_type = _decay_style_to_swift(args.lr_decay_style)
    # train_iters=None means unset on the Megatron surface; max_steps uses -1 for that.
    train_config.max_steps = -1 if args.train_iters is None else int(args.train_iters)
    train_config.gradient_accumulation_steps = _derive_gradient_accumulation_steps(args, world_size)

    distributed_config = _fill_from_args(DistributedConfig(), args)
    # dev-only launch knobs: this entry point IS the Megatron backend, and the CLI is torchrun-based
    # (legacy's launcher), which is dev's 'local' mode. Copying them from args is impossible -- the
    # Megatron surface has no such fields.
    distributed_config.backend = 'megatron'
    distributed_config.mode = 'local'
    distributed_config.nproc_per_node = world_size

    checkpoint_config = _fill_from_args(CheckpointConfig(), args)
    checkpoint_config.save_steps = int(checkpoint_config.save_steps)

    tuner_type = args.tuner_type
    if tuner_type == 'full':
        tuner_config = None
    elif tuner_type == 'lora':
        tuner_config = _fill_from_args(TunerConfig(), args)
        tuner_config.tuner_type = 'lora'
    else:
        raise NotImplementedError(f'dev Megatron CLI supports tuner_type in {{full, lora}}, got {tuner_type!r}.')

    return (model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
            tuner_config)


def megatron_sft_main(args: Optional[List[str]] = None) -> List[dict]:
    """dev Megatron SFT entry. Same name as legacy ``swift.megatron.megatron_sft_main`` on purpose
    (drop-in argv compatibility); import one of them under an alias when using both."""
    from swift.dev.recipe import run_sft
    from swift.megatron.arguments import MegatronSftArguments
    from swift.utils import parse_args

    if isinstance(args, MegatronSftArguments):
        megatron_args: Any = args
    else:
        # skip_megatron_init: legacy MegatronSftArguments.__post_init__ eagerly runs
        # initialize_megatron (mpu init + output-dir setup); dev instead inits mpu during
        # build_model (MegatronStrategy.__init__), so without this the parse-time init and the
        # build-time init collide with "data parallel group is already initialized". This is the
        # same flag swift/ray/megatron/driver_utils.py:179 sets for the same reason -- the dev
        # entry point owns Megatron initialization, so legacy's must be suppressed at parse time.
        argv = list(args) if args is not None else []
        if '--skip_megatron_init' not in argv:
            argv = ['--skip_megatron_init', 'true'] + argv
        megatron_args, remaining = parse_args(MegatronSftArguments, argv)
        if remaining:
            raise ValueError(f'Unrecognized arguments: {remaining}')

    (model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
     tuner_config) = megatron_args_to_configs(megatron_args)

    return run_sft(
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config=distributed_config,
        checkpoint_config=checkpoint_config,
        tuner_config=tuner_config,
        output_dir=checkpoint_config.output_dir,
    )


if __name__ == '__main__':
    megatron_sft_main()
