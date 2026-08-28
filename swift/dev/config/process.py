# Cross-config derivation: the one place where a Config's value is computed from other Configs.
#
# Split from validate.py on purpose. That module only reads and refuses; this one writes. Keeping the
# two apart means a reader can trust that validation never quietly changes a run, and that every
# derived value has exactly one origin. Call process_configs() FIRST, then validate_configs(): the
# checks are written against resolved values, so validating first would test the un-derived state.
#
# What is deliberately NOT here: anything that touches the world. legacy's __post_init__ also set
# environment variables, initialised process groups, downloaded checkpoints and imported plugin
# modules (base_args.py:172-203, sft_args.py:198-238, megatron_args.py:790-902). Those need a real
# runtime and belong where the model is built, not in a pass over dataclasses -- and on the Ray path
# the driver running this is not even the process that trains.

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, MegatronConfig, ModelConfig,
                                  QuantizeConfig, RLHFConfig, TemplateConfig, TrainConfig, TunerConfig)

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
    """Resolve every value that is derived from another Config, in place.

    Idempotent: running it twice leaves the same result, so a caller that cannot easily tell whether an
    earlier stage already ran it does not have to find out.
    """
    from swift.dev.builders.model import is_megatron_backend
    is_megatron = is_megatron_backend(distributed_config)

    _parse_json_fields(model_config, dataset_config, train_config, distributed_config, megatron_config)
    _coerce_mrl_dims(train_config)
    _fold_megatron_aliases(train_config)
    # Order matters below: the eval schedule reads split_dataset_ratio after the val-dataset rule has
    # zeroed it, and task_type must be settled (rm -> seq_cls) before the per-token-loss default and
    # the best-model metric read it.
    _derive_vit_gradient_checkpointing(train_config, tuner_config)
    _derive_packing_length(dataset_config, template_config)
    _derive_split_dataset_ratio(dataset_config)
    _derive_eval_schedule(train_config, dataset_config, checkpoint_config)
    _derive_streaming_dataloader_workers(dataset_config)
    _derive_bnb_compute_dtype(model_config, quantize_config)
    _derive_rlhf_task_type(model_config, rlhf_config)
    _derive_rlhf_beta(rlhf_config)
    _derive_rlhf_ref_model(model_config, tuner_config, rlhf_config)
    _derive_grpo_reward_defaults(rlhf_config)
    _derive_best_model_metric(train_config, rlhf_config)
    _normalize_recompute_granularity(distributed_config)
    _derive_lr_decay_style(train_config, is_megatron)
    _derive_virtual_pipeline(distributed_config, is_megatron)
    _derive_grad_accum_dtype(model_config, train_config, is_megatron)
    _derive_per_token_loss(model_config, train_config, rlhf_config, is_megatron)


#: (config attribute, field, strict) for every field that accepts a JSON string as well as its parsed
#: form. ``strict=False`` is for the fields where a bare word is also legal -- ``device_map='auto'`` is
#: not JSON, and repairing it into something else would be worse than leaving it alone.
_JSON_FIELDS = (
    ('model_config', 'model_kwargs', True),
    ('model_config', 'max_memory', True),
    ('model_config', 'device_map', False),
    ('dataset_config', 'columns', True),
    ('train_config', 'lr_scheduler_kwargs', True),
    ('train_config', 'gradient_checkpointing_kwargs', True),
    ('train_config', 'vit_gradient_checkpointing_kwargs', True),
    ('train_config', 'accelerator_config', True),
    ('train_config', 'mrl_dims', True),
    ('distributed_config', 'fsdp_config', True),
    ('megatron_config', 'megatron_extra_kwargs', True),
)


def _parse_json_fields(model_config, dataset_config, train_config, distributed_config, megatron_config) -> None:
    """Turn JSON strings into dicts once, here, instead of at each point of use.

    A command line can only carry text, so every nested config arrives as a string; every consumer
    would otherwise have to remember to parse it, and the one that forgets sees a str where it expects
    a mapping. legacy did this per field inside __post_init__ (e.g. base_args.py:205 for model_kwargs);
    this is the same conversion in one table.

    ``None`` is left as ``None`` rather than becoming ``{}``, which is what ``json_parse_to_dict`` does
    on its own: for most of these fields None means "let the library choose its own defaults" and an
    empty dict does not say that -- ``fsdp_config={}`` would claim FSDP was configured with nothing.
    """
    from swift.dev.utils import json_parse_to_dict

    holders = {
        'model_config': model_config,
        'dataset_config': dataset_config,
        'train_config': train_config,
        'distributed_config': distributed_config,
        'megatron_config': megatron_config,
    }
    for holder_name, attr, strict in _JSON_FIELDS:
        holder = holders[holder_name]
        if holder is None:
            continue
        value = getattr(holder, attr)
        if value is None or not isinstance(value, str):
            continue
        setattr(holder, attr, json_parse_to_dict(value, strict=strict))


def _coerce_mrl_dims(train_config: 'TrainConfig') -> None:
    """Give `mrl_dims` the {int: float} shape the Matryoshka aggregation indexes with.

    JSON object keys are always strings, so `{"768": 1.0}` parses to a str key that no lookup by
    dimension finds; the loss then silently aggregates nothing at that dimension. legacy coerces the
    same pair right after parsing (megatron_args.py:842-844).

    Kept separate from _JSON_FIELDS because that table only decides *whether* a field is JSON; this is
    the one field whose parsed form still needs its key/value types fixed.
    """
    if not isinstance(train_config.mrl_dims, dict):
        return
    train_config.mrl_dims = {int(k): float(v) for k, v in train_config.mrl_dims.items()}


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

    When both names are set the HF one wins and the conflict is reported, matching resolve_max_grad_norm
    so the two aliases do not disagree about precedence.
    """
    defaults = {f.name: f.default for f in dataclasses.fields(train_config)}
    for mg_name, hf_name in _MEGATRON_ALIASES:
        mg_value = getattr(train_config, mg_name)
        if mg_value is None:
            continue
        hf_value = getattr(train_config, hf_name)
        if hf_value != defaults[hf_name]:
            logger.warning(
                'Both %s=%r and %s=%r are set; they are the same knob under two spellings, so %s is '
                'ignored and %r is used.', hf_name, hf_value, mg_name, mg_value, mg_name, hf_value)
            continue
        setattr(train_config, hf_name, mg_value)


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
    default = next(f.default for f in dataclasses.fields(train_config) if f.name == 'lr_decay_style')
    if train_config.lr_decay_style != default:
        # Explicit, and possibly 'WSD'. Nothing to derive; only report if the other name disagrees.
        derived = _try_megatron_decay_style(train_config.lr_scheduler_type)
        if derived is not None and derived != train_config.lr_decay_style:
            logger.warning(
                'lr_decay_style=%r and lr_scheduler_type=%r describe different schedules; the Megatron '
                'backend follows lr_decay_style. Drop one of the two.', train_config.lr_decay_style,
                train_config.lr_scheduler_type)
        return
    derived = _try_megatron_decay_style(train_config.lr_scheduler_type)
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
    if not is_megatron or train_config.accumulate_allreduce_grads_in_fp32:
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


def _derive_packing_length(dataset_config: 'DatasetConfig', template_config: 'TemplateConfig') -> None:
    """Pack to the sequence length unless a shorter packing window was asked for.

    Mirrors legacy base_args.py:198-199. packing bin-packs samples up to `packing_length`; when it is
    unset the natural target is the model's `max_length`, so a `--packing true` with no explicit window
    packs to the same length the samples are truncated to rather than to some library default.

    Only acts when packing is on and the window is unset, and only when `max_length` is itself known --
    `max_length` is otherwise derived from the model's own limit at build time, which this pass does
    not touch.
    """
    if not dataset_config.packing or dataset_config.packing_length is not None:
        return
    if template_config.max_length is not None:
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
    elif rlhf_type in ('dpo', 'kto', 'ppo', 'grpo') and tuner_type == 'full':
        rlhf_config.ref_model = rlhf_config.ref_model or model_config.model
        rlhf_config.ref_model_type = rlhf_config.ref_model_type or model_config.model_type
        rlhf_config.ref_model_revision = rlhf_config.ref_model_revision or model_config.model_revision


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
