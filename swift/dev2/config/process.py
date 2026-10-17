# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import dataclasses
import os
from typing import TYPE_CHECKING, Optional

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev2.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        MegatronConfig,
        ModelConfig,
        PluginConfig,
        QuantizeConfig,
        RLConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = get_logger()


def process_configs(
    model_config: 'ModelConfig',
    template_config: 'TemplateConfig',
    dataset_config: 'DatasetConfig',
    train_config: 'TrainConfig',
    distributed_config: 'DistributedConfig',
    checkpoint_config: Optional['CheckpointConfig'] = None,
    tuner_config: Optional['TunerConfig'] = None,
    rl_config: Optional['RLConfig'] = None,
    rollout_config: Optional['RolloutConfig'] = None,
    megatron_config: Optional['MegatronConfig'] = None,
    quantize_config: Optional['QuantizeConfig'] = None,
    plugin_config: Optional['PluginConfig'] = None,
) -> None:
    """Load configured plugins and resolve every cross-Config derived value in place.

    Idempotent: running it twice leaves the same result, so callers may repeat the lifecycle.
    """
    # Bridge enable_npu_model_patch to the env var BEFORE the first swift.model import (the patch
    # applies once at import and reads this var); create_workers copies os.environ to every worker.
    os.environ['SWIFT_ENABLE_NPU_MODEL_PATCH'] = str(bool(model_config.enable_npu_model_patch))

    from swift.dev.builders.model import is_megatron_backend
    from swift.dev.utils.plugin import PluginRegistry

    # Plugins must be registered before validation resolves their configured names.
    PluginRegistry.load_configured(plugin_config)
    is_megatron = is_megatron_backend(distributed_config)

    _derive_gradient_accumulation_steps(train_config, distributed_config, is_megatron)
    # Order matters: the eval schedule reads split_dataset_ratio after the val-dataset rule zeroed
    # it, and task_type must settle (rm -> seq_cls) before the per-token-loss default reads it.
    _derive_vit_gradient_checkpointing(train_config, tuner_config)
    _derive_packing(dataset_config, template_config)
    _derive_split_dataset_ratio(dataset_config)
    _derive_eval_schedule(train_config, dataset_config, checkpoint_config)
    _derive_streaming_dataloader_workers(dataset_config)
    _derive_bnb_compute_dtype(model_config, quantize_config)
    _derive_rlhf_task_type(model_config, rl_config)
    _derive_rlhf_beta(rl_config)
    _derive_rlhf_loss_scale(template_config, rl_config, rollout_config)
    _derive_rlhf_ref_model(model_config, tuner_config, rl_config)
    _derive_rlhf_teacher(model_config, tuner_config, rl_config)
    _derive_grpo_reward_defaults(rl_config)
    _derive_best_model_metric(train_config, rl_config)
    _normalize_recompute_granularity(distributed_config)
    _derive_lr_decay_style(train_config, is_megatron)
    _derive_virtual_pipeline(distributed_config, is_megatron)
    _derive_grad_accum_dtype(model_config, train_config, is_megatron)
    _derive_per_token_loss(model_config, train_config, rl_config, is_megatron)


def process_and_validate_configs(
    configs: dict,
    *,
    add_version: Optional[bool] = None,
    create_output_dir: bool = True,
    resolve_model: bool = True,
) -> None:
    """Process, validate, and initialize a parsed Config mapping."""
    from .distributed_config import DistributedConfig
    from .train_config import TrainConfig
    from .validate import validate_configs

    train_config = configs.get('train_config') or TrainConfig()
    distributed_config = configs.get('distributed_config') or DistributedConfig()
    process_configs(
        configs['model_config'],
        configs['template_config'],
        configs['dataset_config'],
        train_config,
        distributed_config,
        configs.get('checkpoint_config'),
        configs.get('tuner_config'),
        rl_config=configs.get('rl_config'),
        rollout_config=configs.get('rollout_config'),
        megatron_config=configs.get('megatron_config'),
        quantize_config=configs.get('quantize_config'),
        plugin_config=configs.get('plugin_config'),
    )
    validate_configs(
        configs['model_config'],
        configs['template_config'],
        configs['dataset_config'],
        train_config,
        distributed_config,
        configs.get('checkpoint_config'),
        configs.get('tuner_config'),
        configs.get('rl_config'),
        reward_config=configs.get('reward_config'),
        rollout_config=configs.get('rollout_config'),
        logging_config=configs.get('logging_config'),
        quantize_config=configs.get('quantize_config'),
        megatron_config=configs.get('megatron_config'),
        moe_config=configs.get('moe_config'),
        training='train_config' in configs,
    )
    rollout_config = configs.get('rollout_config')
    if rollout_config is not None:
        from .validate import validate_multi_turn_config
        validate_multi_turn_config(rollout_config)
    if rollout_config is not None:
        from .validate import validate_rollout_config
        # Fold the legacy ``async_generate`` bool into ``async_mode`` BEFORE validation reads it, so
        # ``async_mode`` is the one source of truth downstream (process modifies, validate only reads).
        _derive_async_mode(rollout_config)
        # A full-parameter or colocated run's only sound publication is in_place; derive from the
        # defaults BEFORE validation reads them, so a legacy colocated script needs no extra flags.
        _derive_weight_sync_strategy(rollout_config, configs.get('rl_config'), configs.get('tuner_config'))
        # max_turns lives on the RolloutConfig itself; rl_config/tuner_config tell _check_async_mode and
        # _check_rollout_tools the training/tuner surface (None = sampling).
        validate_rollout_config(rollout_config, configs.get('rl_config'), configs.get('tuner_config'))
    infer_config = configs.get('infer_config')
    if infer_config is not None:
        from .validate import validate_infer_config
        validate_infer_config(infer_config)
    convert_config = configs.get('convert_config')
    if convert_config is not None:
        from .validate import validate_convert_config
        validate_convert_config(convert_config)
    seed_config = configs.get('train_config') or configs.get('runtime_config')
    bootstrap_run(
        configs['model_config'],
        configs['checkpoint_config'],
        configs.get('dataset_config'),
        configs.get('tuner_config'),
        seed=getattr(seed_config, 'seed', None),
        add_version=add_version,
        create_output_dir=create_output_dir,
        resolve_model=resolve_model,
    )


def bootstrap_run(
    model_config: 'ModelConfig',
    checkpoint_config: 'CheckpointConfig',
    dataset_config: Optional['DatasetConfig'] = None,
    tuner_config: Optional['TunerConfig'] = None,
    *,
    seed: Optional[int] = None,
    add_version: Optional[bool] = None,
    create_output_dir: bool = True,
    resolve_model: bool = True,
) -> None:
    """Apply CLI runtime side effects after processing and validation."""
    if seed is not None:
        from swift.utils import seed_everything
        rank = max(int(os.environ.get('RANK', '-1')), 0)
        seed_everything(seed + rank)
    if dataset_config is not None and dataset_config.use_hf:
        os.environ['USE_HF'] = '1'
    _prepare_output_dir(checkpoint_config, add_version=add_version, create_output_dir=create_output_dir)
    _export_model_kwargs(model_config)
    _login_hub(dataset_config)
    if resolve_model:
        _resolve_model(model_config, dataset_config)
    _resolve_adapters(tuner_config, dataset_config)


def _prepare_output_dir(checkpoint_config: 'CheckpointConfig', *, add_version: Optional[bool],
                        create_output_dir: bool) -> None:
    from swift.utils import add_version_to_work_dir

    checkpoint_config.output_dir = os.path.abspath(os.path.expanduser(checkpoint_config.output_dir))
    use_version = checkpoint_config.add_version if add_version is None else add_version
    if use_version:
        checkpoint_config.output_dir = add_version_to_work_dir(checkpoint_config.output_dir)
    if create_output_dir:
        os.makedirs(checkpoint_config.output_dir, exist_ok=True)

    if checkpoint_config.resume_from_checkpoint:
        resume = os.path.abspath(os.path.expanduser(checkpoint_config.resume_from_checkpoint))
        if not os.path.exists(resume):
            raise ValueError(f'resume_from_checkpoint does not exist: {resume}')
        checkpoint_config.resume_from_checkpoint = resume


def _export_model_kwargs(model_config: 'ModelConfig') -> None:
    for key, value in (model_config.model_kwargs or {}).items():
        os.environ[key.upper()] = str(value)


def _resolve_model(model_config: 'ModelConfig', dataset_config: Optional['DatasetConfig']) -> None:
    if not model_config.model:
        return
    from swift.dev.utils.hub import safe_snapshot_download

    use_hf = dataset_config.use_hf if dataset_config is not None else None
    hub_token = dataset_config.hub_token if dataset_config is not None else None
    # Preserve the user-passed id before overwriting with the local dir: args.json records the
    # portable id (`swift infer <ckpt>` re-resolves it anywhere); transient attr, not a CLI field.
    model_config._model_id_or_path = model_config.model
    model_config.model = safe_snapshot_download(
        model_config.model, revision=model_config.model_revision, use_hf=use_hf, hub_token=hub_token)


def split_served_name(adapter: str) -> tuple:
    """``name=path`` -> ``(name, path)``; a bare path/hub id -> ``(None, adapter)``.

    Splits only when the segment before ``=`` is a bare name, so a path containing ``=`` survives.
    """
    head, sep, tail = adapter.partition('=')
    if sep and head and '/' not in head and '\\' not in head:
        return head, tail
    return None, adapter


def _resolve_adapters(tuner_config: Optional['TunerConfig'], dataset_config: Optional['DatasetConfig']) -> None:
    if tuner_config is None or not tuner_config.adapters:
        return
    from swift.dev.utils.hub import safe_snapshot_download

    use_hf = dataset_config.use_hf if dataset_config is not None else None
    hub_token = dataset_config.hub_token if dataset_config is not None else None
    resolved = []
    for adapter in tuner_config.adapters:
        # Resolve the path portion, then re-attach the served name so deploy's routing label survives.
        name, path = split_served_name(adapter)
        local = safe_snapshot_download(path, use_hf=use_hf, hub_token=hub_token)
        resolved.append(f'{name}={local}' if name else local)
    tuner_config.adapters = resolved


def _login_hub(dataset_config: Optional['DatasetConfig']) -> None:
    if dataset_config is None or not dataset_config.hub_token:
        return
    from swift.dev.utils.hub import get_hub
    get_hub(dataset_config.use_hf).try_login(dataset_config.hub_token)


def _derive_gradient_accumulation_steps(train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
                                        is_megatron: bool) -> None:
    """Derive gradient accumulation from Megatron's global batch size."""
    if train_config.global_batch_size is None:
        return
    if not is_megatron:
        raise ValueError('global_batch_size is a Megatron-only derived setting. Use '
                         'gradient_accumulation_steps with the transformers backend.')

    import os
    world_size = int(os.environ.get('WORLD_SIZE') or distributed_config.nproc_per_node or 1)
    tp = distributed_config.tensor_model_parallel_size
    pp = distributed_config.pipeline_model_parallel_size
    cp = distributed_config.context_parallel_size
    model_parallel_size = tp * pp * cp
    if model_parallel_size < 1 or world_size % model_parallel_size:
        raise ValueError(
            f'world_size={world_size} must be divisible by tp*pp*cp={model_parallel_size} '
            f'(tp={tp}, pp={pp}, cp={cp}).')
    data_parallel_size = world_size // model_parallel_size
    micro_batch_size = train_config.per_device_train_batch_size
    denominator = micro_batch_size * data_parallel_size
    if train_config.global_batch_size % denominator:
        raise ValueError(
            f'global_batch_size={train_config.global_batch_size} must be divisible by '
            f'micro_batch_size*data_parallel_size={denominator}.')
    derived = train_config.global_batch_size // denominator
    explicit = 'gradient_accumulation_steps' in getattr(train_config, '_explicit_fields', set())
    default = next(f.default for f in dataclasses.fields(train_config) if f.name == 'gradient_accumulation_steps')
    if ((explicit or train_config.gradient_accumulation_steps != default)
            and train_config.gradient_accumulation_steps != derived):
        raise ValueError(
            f'gradient_accumulation_steps={train_config.gradient_accumulation_steps} conflicts with '
            f'global_batch_size={train_config.global_batch_size}, which derives {derived}.')
    train_config.gradient_accumulation_steps = derived


def _derive_lr_decay_style(train_config: 'TrainConfig', is_megatron: bool) -> None:
    """Make `lr_decay_style` the single value the Megatron scheduler reads.

    An explicit style stands, else it is derived from `lr_scheduler` ('WSD' has no equivalent).
    """
    if not is_megatron:
        return
    explicit = getattr(train_config, '_explicit_fields', set())
    defaults = {f.name: f.default for f in dataclasses.fields(train_config)}
    style_explicit = ('lr_decay_style' in explicit
                      or train_config.lr_decay_style != defaults['lr_decay_style'])
    scheduler_explicit = ('lr_scheduler' in explicit
                          or train_config.lr_scheduler != defaults['lr_scheduler'])
    derived = _try_megatron_decay_style(train_config.lr_scheduler)

    if style_explicit:
        if scheduler_explicit and derived is not None and derived != train_config.lr_decay_style:
            raise ValueError(
                f'lr_decay_style={train_config.lr_decay_style!r} conflicts with '
                f'lr_scheduler={train_config.lr_scheduler!r}, which maps to {derived!r}. '
                'Use one spelling or select equivalent schedules.')
        return
    if derived is not None:
        train_config.lr_decay_style = derived


def _try_megatron_decay_style(lr_scheduler: str) -> Optional[str]:
    """The Megatron style for an `lr_scheduler`, or None when there is no mapping.

    Errors are swallowed only here: the fast-fail report belongs to the scheduler build site.
    """
    from swift.dev.utils.naming import resolve_megatron_decay_style
    try:
        return resolve_megatron_decay_style(lr_scheduler)
    except Exception:
        return None


def _derive_virtual_pipeline(distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """Resolve the virtual-pipeline width, and the two overlaps that only exist when it is set.

    Mirrors legacy _init_vpp_size: a layout implies the stage count; 1 normalises to None.
    """
    if not is_megatron:
        return
    if distributed_config.pipeline_model_parallel_layout is not None:
        num_stages = _layout_num_stages(distributed_config.pipeline_model_parallel_layout)
        if num_stages is not None:
            pp = distributed_config.pipeline_model_parallel_size
            if num_stages % pp:
                raise ValueError(
                    f'pipeline_model_parallel_layout describes {num_stages} stages, which is not divisible by '
                    f'pipeline_model_parallel_size={pp}. Each pipeline rank has to hold a whole number of '
                    'virtual stages.')
            distributed_config.virtual_pipeline_model_parallel_size = num_stages // pp
    if distributed_config.virtual_pipeline_model_parallel_size == 1:
        distributed_config.virtual_pipeline_model_parallel_size = None
    if distributed_config.virtual_pipeline_model_parallel_size is None:
        distributed_config.overlap_p2p_comm = False
        distributed_config.align_param_gather = False
    if distributed_config.batch_p2p_comm is None:
        # Batching the pair and overlapping them are alternatives.
        distributed_config.batch_p2p_comm = not distributed_config.overlap_p2p_comm


def _layout_num_stages(layout: str) -> Optional[int]:
    """Stage count encoded in a pipeline layout string, or None when megatron cannot be asked.

    None on ImportError keeps a config-only test / dry run going; Megatron validates at build time.
    """
    try:
        from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
    except ImportError:
        return None
    return PipelineParallelLayerLayout.get_num_stages_from_str(layout)


def _derive_grad_accum_dtype(model_config: 'ModelConfig', train_config: 'TrainConfig', is_megatron: bool) -> None:
    """Accumulate gradients in fp32 when training in bf16 and keeping fp32 master gradients.

    Mirrors legacy _map_dtype: a bf16 reduction would round every accumulation. Only ever turns it on.
    """
    explicit = getattr(train_config, '_explicit_fields', set())
    if (not is_megatron or train_config.accumulate_allreduce_grads_in_fp32
            or 'accumulate_allreduce_grads_in_fp32' in explicit):
        return
    if model_config.torch_dtype == 'bfloat16' and train_config.main_grads_dtype == 'fp32':
        train_config.accumulate_allreduce_grads_in_fp32 = True


def _derive_per_token_loss(model_config: 'ModelConfig', train_config: 'TrainConfig',
                           rl_config: Optional['RLConfig'], is_megatron: bool) -> None:
    """Average the loss per token for plain causal-LM training, per batch otherwise.

    Mirrors legacy _set_default; only fills in None.
    """
    if not is_megatron or train_config.calculate_per_token_loss is not None:
        return
    is_causal_lm = model_config.task_type in (None, 'causal_lm')
    is_rlhf = rl_config is not None and getattr(rl_config, 'rl_type', None) is not None
    train_config.calculate_per_token_loss = is_causal_lm and not is_rlhf


def _derive_vit_gradient_checkpointing(train_config: 'TrainConfig',
                                       tuner_config: Optional['TunerConfig']) -> None:
    """Checkpoint the vision tower unless it is frozen, matching the LLM tower (legacy sft_args:211).

    Only fills in None; no TunerConfig (full-param) leaves the builder default.
    """
    if train_config.vit_gradient_checkpointing is not None or tuner_config is None:
        return
    freeze_vit = getattr(tuner_config, 'freeze_vit', None)
    if freeze_vit is None:
        return
    train_config.vit_gradient_checkpointing = not freeze_vit


def _derive_packing(dataset_config: 'DatasetConfig', template_config: 'TemplateConfig') -> None:
    """Enable the padding-free representation required by packing and derive its default length."""
    if not dataset_config.packing:
        return
    if not template_config.padding_free:
        logger.info('Setting padding_free=True because packing is enabled.')
        template_config.padding_free = True
    if dataset_config.packing_length is None and template_config.max_length is not None:
        dataset_config.packing_length = template_config.max_length


def _derive_split_dataset_ratio(dataset_config: 'DatasetConfig') -> None:
    """Stop carving a validation split off the train set once a real one is available.

    Mirrors legacy data_args:110-116; only lowers the ratio to 0, never raises it.
    """
    if dataset_config.split_dataset_ratio <= 0:
        return
    has_val = bool(dataset_config.val_dataset) or bool(dataset_config.cached_val_dataset)
    if has_val or dataset_config.streaming:
        reason = 'a val_dataset is set' if has_val else 'streaming is enabled'
        logger.info('Setting split_dataset_ratio=0.0 because %s.', reason)
        dataset_config.split_dataset_ratio = 0.0


def _derive_eval_schedule(train_config: 'TrainConfig', dataset_config: 'DatasetConfig',
                          checkpoint_config: Optional['CheckpointConfig']) -> None:
    """Make the evaluation cadence follow the save cadence, and turn it off when nothing validates it.

    Mirrors legacy _init_eval_strategy: no eval data forces 'no'; eval_steps inherits save_steps.
    """
    if train_config.predict_with_generate:
        has_eval_data = bool(train_config.eval_dataset)
    else:
        has_eval_data = (bool(dataset_config.val_dataset) or bool(dataset_config.cached_val_dataset)
                         or (bool(dataset_config.dataset) and dataset_config.split_dataset_ratio > 0))
    if not has_eval_data:
        train_config.eval_strategy = 'no'
    elif checkpoint_config is not None and train_config.eval_strategy is None:
        train_config.eval_strategy = checkpoint_config.save_strategy
    if train_config.eval_strategy == 'no':
        train_config.eval_steps = None
        if dataset_config.split_dataset_ratio > 0:
            logger.info('Setting split_dataset_ratio=0.0 because eval_strategy is "no".')
            dataset_config.split_dataset_ratio = 0.0
        return
    if checkpoint_config is None:
        return
    if train_config.eval_strategy == 'steps' and train_config.eval_steps is None:
        train_config.eval_steps = checkpoint_config.save_steps


def _derive_streaming_dataloader_workers(dataset_config: 'DatasetConfig') -> None:
    """Read a streaming dataset from a single worker (legacy megatron_base_args:54-57).

    An IterableDataset has no random access; several workers duplicate or race. Only lowers.
    """
    if not dataset_config.streaming:
        return
    if dataset_config.dataloader_num_workers is not None and dataset_config.dataloader_num_workers > 1:
        logger.info('Setting dataloader_num_workers=1 because the dataset is streaming.')
        dataset_config.dataloader_num_workers = 1


# torch_dtype -> the fp compute dtype bitsandbytes should dequantize its 4-bit weights into. fp16 and
# fp32 both compute in fp32 (a 4-bit path gains nothing from an fp16 accumulate and loses range).
_BNB_COMPUTE_DTYPE = {'float16': 'float32', 'float32': 'float32', 'bfloat16': 'bfloat16'}


def _derive_bnb_compute_dtype(model_config: 'ModelConfig', quantize_config: Optional['QuantizeConfig']) -> None:
    """Default the bnb 4-bit compute dtype from the model's torch_dtype (legacy quant_args:116-122).

    When unset it tracks the model's dtype, so a bf16 run does not compute in fp32.
    """
    if quantize_config is None or quantize_config.bnb_4bit_compute_dtype is not None:
        return
    derived = _BNB_COMPUTE_DTYPE.get(model_config.torch_dtype)
    if derived is not None:
        quantize_config.bnb_4bit_compute_dtype = derived


def _derive_async_mode(rollout_config: 'RolloutConfig') -> None:
    """Fold the legacy ``async_generate`` bool into ``async_mode`` (the one source of truth).

    True meant a one-batch lookahead (successor 'one_step_off'); an explicit async_mode wins. Idempotent.
    """
    if rollout_config.async_generate and rollout_config.async_mode == 'none':
        rollout_config.async_mode = 'one_step_off'


def _derive_weight_sync_strategy(rollout_config: 'RolloutConfig', rl_config: Optional['RLConfig'],
                                 tuner_config: Optional['TunerConfig']) -> None:
    """Default a run that rides the streaming driver to the only publication its placement/tuner allow.

    A full-param or colocated run derives in_place; an overlapping regime adds partial rollout.
    """
    from .validate import _changed_fields, _streams_under_none
    mode = rollout_config.async_mode
    # Under 'none' the publication knob is read only if the config actually rides the streaming
    # driver; an overlapping mode always does. Skip the sync-only / non-online fallback.
    if mode == 'none' and not _streams_under_none(rl_config):
        return
    changed = _changed_fields(rollout_config)
    if 'weight_sync_strategy' in changed:
        # An explicit choice is left alone, so an adapter_snapshot on a full-param/colocated run
        # still reaches validate._check_streaming_publication and is rejected there.
        return
    tuner = getattr(tuner_config, 'tuner', 'full') if tuner_config is not None else 'full'
    colocate = rollout_config.sampler_mode == 'colocate'
    if not (colocate or tuner == 'full'):
        return  # a disaggregated LoRA run pins each version by adapter path (the default) -- sound as-is.
    rollout_config.weight_sync_strategy = 'in_place'
    if mode != 'none' and 'allow_partial_rollout' not in changed:
        rollout_config.allow_partial_rollout = True


def _derive_rlhf_task_type(model_config: 'ModelConfig', rl_config: Optional['RLConfig']) -> None:
    """A reward model is a single-logit sequence classifier (legacy rlhf_args.py::_init_rm).

    `rl_type=rm` trains a seq_cls head with num_labels=1; only fills in defaults.
    """
    if rl_config is None or getattr(rl_config, 'rl_type', None) != 'rm':
        return
    if model_config.task_type is None:
        model_config.task_type = 'seq_cls'
    if model_config.num_labels is None:
        model_config.num_labels = 1


# rl_type -> its default `beta` (KL / deviation-from-reference weight). opsd/mopd are self-distillation
# (no reference-KL term by default) and rft is plain rejection-sampling SFT, so all three take 0.
_RLHF_BETA_DEFAULTS = {'grpo': 0.04, 'gkd': 0.5, 'simpo': 2.0, 'opsd': 0.0, 'mopd': 0.0, 'rft': 0.0}


def _derive_rlhf_beta(rl_config: Optional['RLConfig']) -> None:
    """Fill the reference-deviation weight with the chosen algorithm's default.

    The scattered legacy defaults collected into one table; only fills None (`--beta 0` preserved).
    """
    if rl_config is None or rl_config.beta is not None:
        return
    rl_config.beta = _RLHF_BETA_DEFAULTS.get(getattr(rl_config, 'rl_type', None), 0.1)


def _derive_rlhf_loss_scale(template_config: 'TemplateConfig', rl_config: Optional['RLConfig'],
                            rollout_config: Optional['RolloutConfig'] = None) -> None:
    """Default the loss-mask strategy to the algorithm's sensible one (legacy _set_loss_scale).

    'last_round' everywhere except GRPO multi-turn (-> 'default'); coincide on single-turn data.
    """
    if rl_config is None or getattr(rl_config, 'rl_type', None) is None:
        return
    if 'loss_scale' in getattr(template_config, '_explicit_fields', set()):
        return
    # dev's multi-turn rollout is GRPO-only (validate rejects max_turns elsewhere), so every other
    # rl_type always takes 'last_round' here.
    if rl_config.rl_type == 'grpo' and rollout_config is not None and rollout_config.max_turns is not None:
        template_config.loss_scale = 'default'
    else:
        template_config.loss_scale = 'last_round'


def _derive_rlhf_ref_model(model_config: 'ModelConfig', tuner_config: Optional['TunerConfig'],
                           rl_config: Optional['RLConfig']) -> None:
    """Point the reference model at the policy model when the algorithm needs one and none was given.

    Full-param DPO/KTO/PPO/GRPO default ref_model to `model` (legacy :289-297); GRPO beta=0 needs none.
    """
    if rl_config is None:
        return
    if isinstance(rl_config.ref_adapters, str):
        rl_config.ref_adapters = [rl_config.ref_adapters]
    rl_type = getattr(rl_config, 'rl_type', None)
    tuner = getattr(tuner_config, 'tuner', 'full') if tuner_config is not None else 'full'
    if rl_type == 'grpo' and rl_config.beta == 0.0:
        rl_config.ref_model = None
    elif rl_type in ('dpo', 'kto', 'ppo', 'grpo') and (tuner == 'full' or rl_config.ref_adapters):
        rl_config.ref_model = rl_config.ref_model or model_config.model
        rl_config.ref_model_type = rl_config.ref_model_type or model_config.model_type
        rl_config.ref_model_revision = rl_config.ref_model_revision or model_config.model_revision


def _derive_rlhf_teacher(model_config: 'ModelConfig', tuner_config: Optional['TunerConfig'],
                         rl_config: Optional['RLConfig']) -> None:
    """Resolve self-distillation defaults without loading a teacher."""
    if rl_config is None:
        return
    if isinstance(rl_config.teacher_adapters, str):
        rl_config.teacher_adapters = [rl_config.teacher_adapters]
    for field in ('teacher_model', ):
        value = getattr(rl_config, field)
        if isinstance(value, str):
            setattr(rl_config, field, [value])
    # teacher_model is a list (shared with MOPD); self-distillation is when the one teacher IS the student.
    if (rl_config.teacher_model == [model_config.model] and tuner_config is not None
            and not rl_config.teacher_adapters):
        rl_config._teacher_use_disable_adapter = True
        rl_config.teacher_model = None


def _derive_grpo_reward_defaults(rl_config: Optional['RLConfig']) -> None:
    """Choose the reward-normalisation defaults that match the GRPO advantage estimator.

    Mirrors legacy _init_grpo; only fills None.
    """
    if rl_config is None or getattr(rl_config, 'rl_type', None) != 'grpo':
        return
    estimator = rl_config.advantage_estimator
    if rl_config.kl_in_reward is None:
        rl_config.kl_in_reward = estimator in ('rloo', 'reinforce_plus_plus', 'reinforce_plus_plus_baseline')
    if rl_config.scale_rewards is None:
        rl_config.scale_rewards = {
            'grpo': 'group',
            'rloo': 'none',
            'reinforce_plus_plus': 'batch',
            'reinforce_plus_plus_baseline': 'batch',
            'remax': 'none',
            'opo': 'none',
            'gpg': 'none',
            'grpo_passk': 'group',
        }.get(estimator)


def _derive_best_model_metric(train_config: 'TrainConfig', rl_config: Optional['RLConfig']) -> None:
    """Pick the metric that selects the best checkpoint, and which direction counts as better.

    GRPO -> reward, generation -> ROUGE, else loss; PPO stays unset (loss not monotone in quality).
    """
    rl_type = getattr(rl_config, 'rl_type', None) if rl_config is not None else None
    if train_config.metric_for_best_model is None:
        if rl_type == 'grpo':
            train_config.metric_for_best_model = 'reward'
        elif rl_type != 'ppo':
            train_config.metric_for_best_model = 'rouge-l' if train_config.predict_with_generate else 'loss'
    if train_config.greater_is_better is None and train_config.metric_for_best_model is not None:
        train_config.greater_is_better = 'loss' not in train_config.metric_for_best_model


def _normalize_recompute_granularity(distributed_config: 'DistributedConfig') -> None:
    """Treat the string 'none' as no recomputation (legacy megatron_args:797-798).

    Every downstream check tests the field for None, so "off" gets a single representation.
    """
    if distributed_config.recompute_granularity == 'none':
        distributed_config.recompute_granularity = None
