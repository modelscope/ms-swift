"""Config processing and run initialization.

Everything that changes Config or runtime state lives here. ``validate.py`` only reads the resolved
state and rejects invalid combinations. Programmatic callers use :func:`process_configs`; CLI entry
points use :func:`process_and_validate_configs` for the complete run lifecycle.
"""
from __future__ import annotations
import dataclasses
import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        MegatronConfig,
        ModelConfig,
        QuantizeConfig,
        RLHFConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = logging.getLogger(__name__)


def process_configs(
    model_config: 'ModelConfig',
    template_config: 'TemplateConfig',
    dataset_config: 'DatasetConfig',
    train_config: 'TrainConfig',
    distributed_config: 'DistributedConfig',
    checkpoint_config: Optional['CheckpointConfig'] = None,
    tuner_config: Optional['TunerConfig'] = None,
    rlhf_config: Optional['RLHFConfig'] = None,
    megatron_config: Optional['MegatronConfig'] = None,
    quantize_config: Optional['QuantizeConfig'] = None,
) -> None:
    """Load configured plugins and resolve every cross-Config derived value in place.

    Idempotent: running it twice leaves the same result, so callers may safely repeat the lifecycle.
    """
    from swift.dev.builders.model import is_megatron_backend
    from swift.dev.plugin import PluginRegistry

    # Plugins must be registered before validation resolves their configured names.
    PluginRegistry.load_configured(model_config)
    is_megatron = is_megatron_backend(distributed_config)

    _fold_megatron_aliases(train_config)
    _derive_gradient_accumulation_steps(train_config, distributed_config, is_megatron)
    _derive_finetune_resume(train_config, checkpoint_config, is_megatron)
    # Order matters below: the eval schedule reads split_dataset_ratio after the val-dataset rule has
    # zeroed it, and task_type must be settled (rm -> seq_cls) before the per-token-loss default and
    # the best-model metric read it.
    _derive_vit_gradient_checkpointing(train_config, tuner_config)
    _derive_packing(dataset_config, template_config)
    _derive_split_dataset_ratio(dataset_config)
    _derive_eval_schedule(train_config, dataset_config, checkpoint_config)
    _derive_streaming_dataloader_workers(dataset_config)
    _derive_bnb_compute_dtype(model_config, quantize_config)
    _derive_rlhf_task_type(model_config, rlhf_config)
    _derive_rlhf_beta(rlhf_config)
    _derive_rlhf_ref_model(model_config, tuner_config, rlhf_config)
    _derive_rlhf_teacher(model_config, tuner_config, rlhf_config)
    _derive_grpo_reward_defaults(rlhf_config)
    _derive_best_model_metric(train_config, rlhf_config)
    _normalize_recompute_granularity(distributed_config)
    _derive_lr_decay_style(train_config, is_megatron)
    _derive_virtual_pipeline(distributed_config, is_megatron)
    _derive_grad_accum_dtype(model_config, train_config, is_megatron)
    _derive_per_token_loss(model_config, train_config, rlhf_config, is_megatron)


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
        rlhf_config=configs.get('rlhf_config'),
        megatron_config=configs.get('megatron_config'),
        quantize_config=configs.get('quantize_config'),
    )
    validate_configs(
        configs['model_config'],
        configs['template_config'],
        configs['dataset_config'],
        train_config,
        distributed_config,
        configs.get('checkpoint_config'),
        configs.get('tuner_config'),
        configs.get('rlhf_config'),
        configs.get('logging_config'),
        quantize_config=configs.get('quantize_config'),
        megatron_config=configs.get('megatron_config'),
        moe_config=configs.get('moe_config'),
        training='train_config' in configs,
    )
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
    model_config.model = safe_snapshot_download(
        model_config.model, revision=model_config.model_revision, use_hf=use_hf, hub_token=hub_token)


def _resolve_adapters(tuner_config: Optional['TunerConfig'], dataset_config: Optional['DatasetConfig']) -> None:
    if tuner_config is None or not tuner_config.adapters:
        return
    from swift.dev.utils.hub import safe_snapshot_download

    use_hf = dataset_config.use_hf if dataset_config is not None else None
    hub_token = dataset_config.hub_token if dataset_config is not None else None
    tuner_config.adapters = [
        safe_snapshot_download(adapter, use_hf=use_hf, hub_token=hub_token) for adapter in tuner_config.adapters
    ]


def _login_hub(dataset_config: Optional['DatasetConfig']) -> None:
    if dataset_config is None or not dataset_config.hub_token:
        return
    from swift.dev.utils.hub import get_hub
    get_hub(dataset_config.use_hf).try_login(dataset_config.hub_token)


#: (megatron spelling, HF spelling) for the aliases that are one knob under two names. Only exact 1:1
#: pairs are here: `global_batch_size` is absent because it is a PRODUCT of the HF fields rather than
#: an alias of one, and `lr_decay_style` is absent because 'WSD' has no HF value to fold into (both are
#: handled separately below / at build time).
_MEGATRON_ALIASES = (
    ('lr', 'learning_rate'),
    ('train_iters', 'max_steps'),
    ('micro_batch_size', 'per_device_train_batch_size'),
    ('lr_warmup_fraction', 'warmup_ratio'),
    ('lr_warmup_iters', 'warmup_steps'),
    ('adam_eps', 'adam_epsilon'),
)


def _fold_megatron_aliases(train_config: 'TrainConfig') -> None:
    """Copy each Megatron-spelled value onto the HF field that everything already reads.

    Megatron and HF name the same quantities differently, and legacy never had to reconcile them: its
    Megatron path parsed `lr`/`train_iters`/... and its HF path parsed `learning_rate`/`max_steps`/...,
    in separate argument classes that never met. dev has one TrainConfig for both backends, so the
    Megatron spellings would otherwise be accepted and then read by nobody -- a `--lr 2e-5` that
    silently trains at the default.

    Folding here rather than at each point of use is what makes them work everywhere at once, including
    in code written before these fields existed. The alternative, a resolve_lr() called wherever the
    learning rate is read, is the pattern resolve_max_grad_norm follows -- appropriate there because
    `clip_grad` is deprecated and has exactly one consumer, but these five are current spellings with
    many.

    When both names are set, equal values are accepted and different values fail. CLI aliases are
    normally folded before dataclass construction; this path preserves the same rule for programmatic
    Config construction.
    """
    defaults = {f.name: f.default for f in dataclasses.fields(train_config)}
    for mg_name, hf_name in _MEGATRON_ALIASES:
        mg_value = getattr(train_config, mg_name)
        if mg_value is None:
            continue
        hf_value = getattr(train_config, hf_name)
        if hf_value != defaults[hf_name] and hf_value != mg_value:
            raise ValueError(
                f'{mg_name}={mg_value!r} conflicts with its canonical spelling {hf_name}={hf_value!r}. '
                'Use one spelling or pass the same value.')
        setattr(train_config, hf_name, mg_value)


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


def _derive_finetune_resume(train_config: 'TrainConfig', checkpoint_config: Optional['CheckpointConfig'],
                            is_megatron: bool) -> None:
    """Translate Megatron's ``finetune`` intent onto the dev checkpoint contract.

    A fresh dev run already loads the HF model weights and starts optimizer state at step zero, which
    is exactly legacy ``finetune=True``. For a dev checkpoint, ``finetune=True`` means weights-only
    restore and therefore derives ``resume_only_model=True``; ``finetune=False`` means a full resume.
    Native mcore checkpoint directories are rejected earlier by the CLI contract because their layout
    is not interchangeable with Twinkle's HF-weights-plus-mcore-state format.
    """
    if not is_megatron or 'finetune' not in getattr(train_config, '_explicit_fields', set()):
        return
    if checkpoint_config is None or not checkpoint_config.resume_from_checkpoint:
        if not train_config.finetune:
            raise ValueError(
                'Megatron --finetune false requires --resume_from_checkpoint pointing to a checkpoint written by '
                'the dev runtime. Native mcore checkpoints must first be converted with `swift megatron export '
                '--to_hf true`.')
        return

    derived = bool(train_config.finetune)
    resume_explicit = 'resume_only_model' in getattr(checkpoint_config, '_explicit_fields', set())
    if resume_explicit and checkpoint_config.resume_only_model != derived:
        raise ValueError(
            f'finetune={train_config.finetune!r} conflicts with resume_only_model='
            f'{checkpoint_config.resume_only_model!r}. Megatron finetune=true means a weights-only new run; '
            'finetune=false means a full resume.')
    checkpoint_config.resume_only_model = derived


def _derive_lr_decay_style(train_config: 'TrainConfig', is_megatron: bool) -> None:
    """Make `lr_decay_style` the single value the Megatron scheduler reads.

    These two are not a plain alias pair. `lr_scheduler_type` is dev's canonical field and already maps
    onto Megatron's styles through resolve_megatron_decay_style; `lr_decay_style` is Megatron's own
    spelling, and it carries one value -- 'WSD' -- that has no `lr_scheduler_type` equivalent, so it
    cannot simply be folded away.

    So the rule is by precedence rather than by copying: an explicitly chosen `lr_decay_style` stands,
    otherwise it is derived from `lr_scheduler_type`. Either way exactly one field holds the answer
    afterwards, which is the point -- the previous state had two fields and no rule.
    """
    if not is_megatron:
        return
    explicit = getattr(train_config, '_explicit_fields', set())
    defaults = {f.name: f.default for f in dataclasses.fields(train_config)}
    style_explicit = ('lr_decay_style' in explicit
                      or train_config.lr_decay_style != defaults['lr_decay_style'])
    scheduler_explicit = ('lr_scheduler_type' in explicit
                          or train_config.lr_scheduler_type != defaults['lr_scheduler_type'])
    derived = _try_megatron_decay_style(train_config.lr_scheduler_type)

    if style_explicit:
        if scheduler_explicit and derived is not None and derived != train_config.lr_decay_style:
            raise ValueError(
                f'lr_decay_style={train_config.lr_decay_style!r} conflicts with '
                f'lr_scheduler_type={train_config.lr_scheduler_type!r}, which maps to {derived!r}. '
                'Use one spelling or select equivalent schedules.')
        return
    if derived is not None:
        train_config.lr_decay_style = derived


def _try_megatron_decay_style(lr_scheduler_type: str) -> Optional[str]:
    """The Megatron style for an `lr_scheduler_type`, or None when there is no mapping.

    Swallowing the error is right here and only here: resolve_megatron_decay_style fails fast so an
    unsupported schedule cannot silently become cosine, and that report belongs to the scheduler build,
    where it names the supported values. Raising it from a derivation pass would report the same problem
    twice, from a place that was not asked about schedules.
    """
    from swift.dev.naming import resolve_megatron_decay_style
    try:
        return resolve_megatron_decay_style(lr_scheduler_type)
    except Exception:
        return None


def _derive_virtual_pipeline(distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """Resolve the virtual-pipeline width, and the two overlaps that only exist when it is set.

    Mirrors legacy megatron_args.py::_init_vpp_size. Three separate derivations, each of which turns a
    combination that would fail deep inside Megatron into one that is consistent up front:

    - A layout string implies the number of virtual stages, so it wins over an explicit size; giving
      both a layout and a mismatched size is the one case this refuses outright.
    - A width of 1 is not interleaving, it is the ordinary pipeline. Normalising it to None means the
      checks below (and Megatron's own) have a single representation of "off" to test.
    - Without interleaving there are no alternating stages to overlap, so `overlap_p2p_comm` and
      `align_param_gather` cannot do anything; leaving them on would report a configuration the run does
      not have. `batch_p2p_comm` then defaults to the opposite of the overlap, since batching the pair
      and overlapping them are alternatives.
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
        distributed_config.batch_p2p_comm = not distributed_config.overlap_p2p_comm


def _layout_num_stages(layout: str) -> Optional[int]:
    """Stage count encoded in a pipeline layout string, or None when megatron cannot be asked.

    The parsing lives in Megatron, and importing it here follows what _check_megatron_attn_backend
    already does on this side. None when the import fails, so a driver without megatron installed --
    a config-only test, or a dry run -- still gets through the rest of the pass; Megatron itself
    validates the layout properly when the model is built.
    """
    try:
        from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
    except ImportError:
        return None
    return PipelineParallelLayerLayout.get_num_stages_from_str(layout)


def _derive_grad_accum_dtype(model_config: 'ModelConfig', train_config: 'TrainConfig', is_megatron: bool) -> None:
    """Accumulate gradients in fp32 when training in bf16 and keeping fp32 master gradients.

    Mirrors legacy megatron_args.py::_map_dtype, which set this for the bf16 + main_grads_dtype='fp32'
    pair. The reason is that the two settings only agree that way: master gradients declared fp32 while
    the reduction runs in bf16 would round every accumulation, which is precisely what the fp32 master
    copy exists to avoid.

    Only ever turns it on. A user who explicitly wants bf16 accumulation can say so, and legacy's
    version could not tell that apart from the default.
    """
    explicit = getattr(train_config, '_explicit_fields', set())
    if (not is_megatron or train_config.accumulate_allreduce_grads_in_fp32
            or 'accumulate_allreduce_grads_in_fp32' in explicit):
        return
    if model_config.torch_dtype == 'bfloat16' and train_config.main_grads_dtype == 'fp32':
        train_config.accumulate_allreduce_grads_in_fp32 = True


def _derive_per_token_loss(model_config: 'ModelConfig', train_config: 'TrainConfig',
                           rlhf_config: Optional['RLHFConfig'], is_megatron: bool) -> None:
    """Average the loss per token for plain causal-LM training, per batch otherwise.

    Mirrors legacy megatron_args.py::_set_default. Per-token is the right average for causal LM, where
    samples have unequal lengths and the per-micro-batch mean would weight a short sample as heavily as
    a long one. It is left off for the other objectives because their losses are already defined per
    sequence -- an RLHF advantage or a classification logit is not a sum over tokens to be divided.

    Only fills in None, so an explicit choice is never overridden.
    """
    if not is_megatron or train_config.calculate_per_token_loss is not None:
        return
    is_causal_lm = model_config.task_type in (None, 'causal_lm')
    is_rlhf = rlhf_config is not None and getattr(rlhf_config, 'rlhf_type', None) is not None
    train_config.calculate_per_token_loss = is_causal_lm and not is_rlhf


def _derive_vit_gradient_checkpointing(train_config: 'TrainConfig', tuner_config: Optional['TunerConfig']) -> None:
    """Checkpoint the vision tower unless it is frozen, matching whichever way the LLM tower is trained.

    Mirrors legacy sft_args.py:211-212 and megatron_args.py:806-807 (`vit_gradient_checkpointing =
    not freeze_vit`). Recomputing a tower whose parameters are frozen saves the activation memory it
    would cost but buys nothing back -- there is no gradient to recompute for -- so the default follows
    freeze_vit: recompute when training the tower, skip it when frozen.

    Only fills in None, so an explicit choice is kept. When there is no TunerConfig (full-parameter
    training) freeze_vit is not available and nothing is trained frozen, so the field is left for the
    builder's own default.
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

    Mirrors legacy data_args.py:110-116. `split_dataset_ratio` reserves a slice of the training data
    for validation; a supplied `val_dataset` already provides that slice, and a streaming train set has
    no random access to carve one from, so in both cases the ratio would either double up or fail.
    Setting it to 0 is what makes `--val_dataset` alone do the obvious thing.

    Only lowers the ratio to 0, never raises it, so it cannot manufacture a split the user did not ask
    for.
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

    Mirrors legacy sft_args.py::_init_eval_strategy plus the guard at sft_args.py:231-232. Three
    couplings, each removing a knob the user would otherwise have to keep in sync by hand:

    - No validation data (no val_dataset, no split, no cached val) means there is nothing to evaluate,
      so the strategy is forced to 'no'; leaving it on would evaluate an empty set every period.
    - `eval_strategy` defaults to `save_strategy`, so a run that saves every N steps also evaluates
      every N steps without being told twice.
    - When evaluating by steps without an explicit `eval_steps`, it inherits `save_steps` for the same
      reason.

    Only fills in unset fields.
    """
    has_val = (bool(dataset_config.val_dataset) or bool(dataset_config.cached_val_dataset)
               or (bool(dataset_config.dataset) and dataset_config.split_dataset_ratio > 0))
    if not has_val:
        train_config.eval_strategy = 'no'
        train_config.eval_steps = None
        return
    if checkpoint_config is None:
        return
    if train_config.eval_strategy is None:
        train_config.eval_strategy = checkpoint_config.save_strategy
    if train_config.eval_strategy == 'steps' and train_config.eval_steps is None:
        train_config.eval_steps = checkpoint_config.save_steps


def _derive_streaming_dataloader_workers(dataset_config: 'DatasetConfig') -> None:
    """Read a streaming dataset from a single worker.

    Mirrors legacy megatron_base_args.py:54-57. An IterableDataset has no random access, so several
    workers cannot each take a disjoint slice; more than one either duplicates samples or races on the
    same iterator. legacy clamps to 1 with a log, and so does this.

    Only lowers the count; a None (auto) or already-<=1 value is left for the loader to interpret.
    """
    if not dataset_config.streaming:
        return
    if dataset_config.dataloader_num_workers is not None and dataset_config.dataloader_num_workers > 1:
        logger.info('Setting dataloader_num_workers=1 because the dataset is streaming.')
        dataset_config.dataloader_num_workers = 1


#: torch_dtype -> the fp compute dtype bitsandbytes should dequantize its 4-bit weights into. fp16 and
#: fp32 both compute in fp32 (a 4-bit path gains nothing from an fp16 accumulate and loses range);
#: bf16 keeps bf16 so the compute dtype matches the rest of the model.
_BNB_COMPUTE_DTYPE = {'float16': 'float32', 'float32': 'float32', 'bfloat16': 'bfloat16'}


def _derive_bnb_compute_dtype(model_config: 'ModelConfig', quantize_config: Optional['QuantizeConfig']) -> None:
    """Default the bnb 4-bit compute dtype from the model's torch_dtype.

    Mirrors legacy quant_args.py:116-122. `bnb_4bit_compute_dtype` is the dtype the dequantized weights
    are matmul'd in; when unset it should track the model's own dtype rather than a fixed default, so a
    bf16 run does not silently compute its quantized layers in fp32.

    Only fills in None, and only when torch_dtype is itself known.
    """
    if quantize_config is None or quantize_config.bnb_4bit_compute_dtype is not None:
        return
    derived = _BNB_COMPUTE_DTYPE.get(model_config.torch_dtype)
    if derived is not None:
        quantize_config.bnb_4bit_compute_dtype = derived


def _derive_rlhf_task_type(model_config: 'ModelConfig', rlhf_config: Optional['RLHFConfig']) -> None:
    """A reward model is a single-logit sequence classifier.

    Mirrors legacy rlhf_args.py::_init_rm (and the same block in megatron rlhf_args.py). `rlhf_type=rm`
    trains a scalar reward head, which is a `seq_cls` task with `num_labels=1`; deriving the two here
    means the user picks the algorithm and the model shape follows, instead of having to state both and
    keep them consistent.

    Only fills in defaults: an explicit task_type/num_labels is left alone.
    """
    if rlhf_config is None or getattr(rlhf_config, 'rlhf_type', None) != 'rm':
        return
    if model_config.task_type is None:
        model_config.task_type = 'seq_cls'
    if model_config.num_labels is None:
        model_config.num_labels = 1


#: rlhf_type -> its default `beta` (KL / deviation-from-reference weight). Absent types (dpo, cpo, kto,
#: ...) share the 0.1 fallback below; the three here are the ones legacy singles out.
_RLHF_BETA_DEFAULTS = {'grpo': 0.04, 'gkd': 0.5, 'simpo': 2.0}


def _derive_rlhf_beta(rlhf_config: Optional['RLHFConfig']) -> None:
    """Fill the reference-deviation weight with the chosen algorithm's default.

    Mirrors the scattered legacy defaults (rlhf_args.py::_set_default 0.1/0.5, _init_grpo 0.04,
    _init_simpo 2.0). `beta` weights how far the policy may drift from the reference, and the sensible
    starting value differs by algorithm; collecting the defaults in one table keeps them from
    disagreeing across the code paths that used to each set their own.

    Only fills in None, so an explicit `--beta` -- including `--beta 0` to disable the reference model
    entirely -- is preserved.
    """
    if rlhf_config is None or rlhf_config.beta is not None:
        return
    rlhf_config.beta = _RLHF_BETA_DEFAULTS.get(getattr(rlhf_config, 'rlhf_type', None), 0.1)


def _derive_rlhf_ref_model(model_config: 'ModelConfig', tuner_config: Optional['TunerConfig'],
                           rlhf_config: Optional['RLHFConfig']) -> None:
    """Point the reference model at the policy model when the algorithm needs one and none was given.

    Mirrors legacy rlhf_args.py:289-297. DPO/KTO/PPO/GRPO under full-parameter training compare the
    policy against a frozen copy of the starting weights, so an unset `ref_model` defaults to `model`
    (and likewise its type/revision). GRPO with `beta=0` drops the KL term, hence needs no reference at
    all, so the field is cleared. Adapter training has no separate reference -- the base model with the
    adapter disabled is the reference -- so nothing is derived there.

    The mirror is the derivation only; the rejection of a ref_model passed to CPO/ORPO/LoRA (legacy's
    trailing `elif ... raise`) lives in validate.py, so this module still never refuses.
    """
    if rlhf_config is None:
        return
    if isinstance(rlhf_config.ref_adapters, str):
        rlhf_config.ref_adapters = [rlhf_config.ref_adapters]
    rlhf_type = getattr(rlhf_config, 'rlhf_type', None)
    tuner_type = getattr(tuner_config, 'tuner_type', 'full') if tuner_config is not None else 'full'
    if rlhf_type == 'grpo' and rlhf_config.beta == 0.0:
        rlhf_config.ref_model = None
    elif rlhf_type in ('dpo', 'kto', 'ppo', 'grpo') and (tuner_type == 'full' or rlhf_config.ref_adapters):
        rlhf_config.ref_model = rlhf_config.ref_model or model_config.model
        rlhf_config.ref_model_type = rlhf_config.ref_model_type or model_config.model_type
        rlhf_config.ref_model_revision = rlhf_config.ref_model_revision or model_config.model_revision


def _derive_rlhf_teacher(model_config: 'ModelConfig', tuner_config: Optional['TunerConfig'],
                         rlhf_config: Optional['RLHFConfig']) -> None:
    """Resolve self-distillation and gym defaults without loading a teacher or environment."""
    if rlhf_config is None:
        return
    if isinstance(rlhf_config.teacher_adapters, str):
        rlhf_config.teacher_adapters = [rlhf_config.teacher_adapters]
    for field in ('reward_model', 'reward_adapters', 'reward_model_type', 'reward_model_revision',
                  'reward_model_plugin', 'reward_template'):
        value = getattr(rlhf_config, field)
        if isinstance(value, str):
            setattr(rlhf_config, field, [value])
    if rlhf_config.use_gym_env is None and (
            rlhf_config.gym_env is not None
            or rlhf_config.multi_turn_scheduler in ('gym_scheduler', 'openenv_scheduler')):
        rlhf_config.use_gym_env = True
    if rlhf_config.use_gym_env and rlhf_config.multi_turn_scheduler is None:
        rlhf_config.multi_turn_scheduler = 'gym_scheduler'
    if (rlhf_config.teacher_model == model_config.model and tuner_config is not None
            and not rlhf_config.teacher_adapters):
        rlhf_config._teacher_use_disable_adapter = True
        rlhf_config.teacher_model = None


def _derive_grpo_reward_defaults(rlhf_config: Optional['RLHFConfig']) -> None:
    """Choose the reward-normalisation defaults that match the GRPO advantage estimator.

    Mirrors legacy rlhf_args.py::_init_grpo. Each estimator implies how its rewards should be scaled
    and whether the KL belongs in the reward: plain 'grpo' normalises per group and keeps KL as a
    separate loss term, 'rloo' does neither, 'reinforce_plus_plus' scales per batch and folds KL into
    the reward. Deriving both from `advantage_estimator` keeps a user from pairing an estimator with a
    scaling that contradicts it.

    Only fills in None. The estimator is a closed Literal, so the value always resolves.
    """
    if rlhf_config is None or getattr(rlhf_config, 'rlhf_type', None) != 'grpo':
        return
    estimator = rlhf_config.advantage_estimator
    if rlhf_config.kl_in_reward is None:
        rlhf_config.kl_in_reward = estimator in ('rloo', 'reinforce_plus_plus')
    if rlhf_config.scale_rewards is None:
        rlhf_config.scale_rewards = {'grpo': 'group', 'rloo': 'none', 'reinforce_plus_plus': 'batch'}.get(estimator)


def _derive_best_model_metric(train_config: 'TrainConfig', rlhf_config: Optional['RLHFConfig']) -> None:
    """Pick the metric that selects the best checkpoint, and which direction counts as better.

    Mirrors legacy sft_args.py::_init_metric_for_best_model, its GRPO override in rlhf_args.py, and the
    megatron block at megatron_args.py:862-865. The best-checkpoint metric depends on what the run
    produces: a generation run is judged by ROUGE, a GRPO run by its reward, everything else by the
    loss. `greater_is_better` then follows the metric -- loss is minimised, a reward or ROUGE score is
    maximised -- so the two cannot be set to disagree.

    Only fills in unset fields.
    """
    is_grpo = rlhf_config is not None and getattr(rlhf_config, 'rlhf_type', None) == 'grpo'
    if train_config.metric_for_best_model is None:
        if is_grpo:
            train_config.metric_for_best_model = 'reward'
        else:
            train_config.metric_for_best_model = 'rouge-l' if train_config.predict_with_generate else 'loss'
    if train_config.greater_is_better is None and train_config.metric_for_best_model is not None:
        train_config.greater_is_better = 'loss' not in train_config.metric_for_best_model


def _normalize_recompute_granularity(distributed_config: 'DistributedConfig') -> None:
    """Treat the string 'none' as no recomputation.

    Mirrors legacy megatron_args.py:797-798. The CLI can only spell "off" as the word 'none', but every
    downstream check tests `recompute_granularity` for None; collapsing the two here gives "off" a
    single representation, so a `--recompute_granularity none` is not read as an enabled mode named
    'none'.
    """
    if distributed_config.recompute_granularity == 'none':
        distributed_config.recompute_granularity = None
