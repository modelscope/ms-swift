# Copyright (c) ModelScope Contributors. All rights reserved.
"""S2a -- every cross-Config derivation in config/process.py, one behaviour per test (ut_plan S2-01..S2-26).

Each derivation mirrors a legacy rule (the docstrings name them), so the oracle is the legacy
behaviour stated in the plan: value in -> derived value out, refusal on the ambiguous cases. All
tests are CPU-fast Config-only -- no model, no hub (the two resolution steps monkeypatch
``safe_snapshot_download``). Test ids in docstrings.
"""
from __future__ import annotations

import dataclasses
import os

import pytest

from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    ModelConfig,
    QuantizeConfig,
    RLConfig,
    RolloutConfig,
    TemplateConfig,
    TrainConfig,
    TunerConfig,
)
from swift.dev.config.process import (
    _derive_async_mode,
    _derive_best_model_metric,
    _derive_bnb_compute_dtype,
    _derive_eval_schedule,
    _derive_gradient_accumulation_steps,
    _derive_grpo_reward_defaults,
    _derive_grad_accum_dtype,
    _derive_lr_decay_style,
    _derive_packing,
    _derive_per_token_loss,
    _derive_rlhf_beta,
    _derive_rlhf_loss_scale,
    _derive_rlhf_ref_model,
    _derive_rlhf_task_type,
    _derive_rlhf_teacher,
    _derive_split_dataset_ratio,
    _derive_streaming_dataloader_workers,
    _derive_virtual_pipeline,
    _derive_vit_gradient_checkpointing,
    _derive_weight_sync_strategy,
    _export_model_kwargs,
    _login_hub,
    _normalize_recompute_granularity,
    _prepare_output_dir,
    _resolve_adapters,
    _resolve_model,
    split_served_name,
)


def _explicit(config, *names):
    """Mark fields as CLI-explicit, the way parse_configs does (derivations branch on this)."""
    config._explicit_fields = set(getattr(config, '_explicit_fields', set())) | set(names)
    return config


# --- batch size / steps ---------------------------------------------------------------


@pytest.mark.parametrize('world, tp, pp, cp, micro, gbs, expected', [
    (None, 1, 1, 1, 2, 8, 4),  # no WORLD_SIZE: nproc default 1
    (None, 1, 1, 1, 4, 8, 2),
    ('2', 1, 1, 1, 2, 8, 2),  # WORLD_SIZE wins over nproc
    ('4', 2, 1, 1, 2, 16, 4),  # tp=2 halves the data-parallel size
    ('4', 1, 2, 1, 2, 16, 4),  # pp=2 likewise
    ('2', 1, 1, 2, 2, 8, 4),  # cp=2 likewise (cp is model-parallel too)
])
def test_ga_derivation_from_global_batch_size(monkeypatch, world, tp, pp, cp, micro, gbs, expected):
    """S2-01: GA = global_batch_size / (micro_batch * data_parallel); WORLD_SIZE takes precedence."""
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    if world is not None:
        monkeypatch.setenv('WORLD_SIZE', world)
    train = TrainConfig(global_batch_size=gbs, per_device_train_batch_size=micro)
    distributed = DistributedConfig(backend='megatron', tensor_model_parallel_size=tp,
                                    pipeline_model_parallel_size=pp, context_parallel_size=cp)
    _derive_gradient_accumulation_steps(train, distributed, True)
    assert train.gradient_accumulation_steps == expected


def test_ga_derivation_refusals(monkeypatch):
    """S2-01 (refusals): transformers backend rejects global_batch_size outright; a non-divisible
    model-parallel split and a non-divisible batch both fail loudly; an explicit conflicting GA
    is an error while the matching value passes."""
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    train = TrainConfig(global_batch_size=8, per_device_train_batch_size=2)
    with pytest.raises(ValueError, match='Megatron-only'):
        _derive_gradient_accumulation_steps(train, DistributedConfig(backend='hf'), False)

    train = TrainConfig(global_batch_size=8, per_device_train_batch_size=2)
    distributed = DistributedConfig(backend='megatron', tensor_model_parallel_size=3)
    with pytest.raises(ValueError, match='divisible by tp'):
        _derive_gradient_accumulation_steps(train, distributed, True)

    train = TrainConfig(global_batch_size=7, per_device_train_batch_size=2)
    with pytest.raises(ValueError, match='divisible by'):
        _derive_gradient_accumulation_steps(train, DistributedConfig(backend='megatron'), True)

    conflicting = _explicit(TrainConfig(global_batch_size=8, per_device_train_batch_size=2,
                                        gradient_accumulation_steps=3), 'gradient_accumulation_steps')
    with pytest.raises(ValueError, match='conflicts with'):
        _derive_gradient_accumulation_steps(conflicting, DistributedConfig(backend='megatron'), True)

    agreeing = _explicit(TrainConfig(global_batch_size=8, per_device_train_batch_size=2,
                                     gradient_accumulation_steps=4), 'gradient_accumulation_steps')
    _derive_gradient_accumulation_steps(agreeing, DistributedConfig(backend='megatron'), True)
    assert agreeing.gradient_accumulation_steps == 4


def test_eval_schedule_derivation():
    """S2-02: no eval data -> 'no' (and eval_steps dropped, split ratio zeroed); eval_strategy
    inherits save_strategy; 'steps' inherits save_steps; an explicit strategy is kept."""
    # no eval data at all
    train, dataset, checkpoint = TrainConfig(), DatasetConfig(dataset=['d']), CheckpointConfig()
    _derive_eval_schedule(train, dataset, checkpoint)
    assert train.eval_strategy == 'no' and train.eval_steps is None

    # a val dataset makes save_strategy the default eval strategy
    train, dataset = TrainConfig(), DatasetConfig(dataset=['d'], val_dataset=['v'])
    checkpoint = CheckpointConfig(save_strategy='steps', save_steps=500)
    _derive_eval_schedule(train, dataset, checkpoint)
    assert train.eval_strategy == 'steps' and train.eval_steps == 500

    # an explicit eval_strategy survives
    train = TrainConfig(eval_strategy='epoch')
    _derive_eval_schedule(train, DatasetConfig(dataset=['d'], val_dataset=['v']),
                          CheckpointConfig(save_strategy='steps'))
    assert train.eval_strategy == 'epoch'

    # an explicit eval_strategy='no' zeroes a positive split ratio (nothing would validate it);
    # a dataset with split_ratio>0 alone would have carved a val split, so eval stays on there
    train = _explicit(TrainConfig(eval_strategy='no'), 'eval_strategy')
    dataset = DatasetConfig(dataset=['d'], split_dataset_ratio=0.05)
    _derive_eval_schedule(train, dataset, CheckpointConfig())
    assert dataset.split_dataset_ratio == 0.0

    # predict_with_generate looks at train_config.eval_dataset instead
    train, dataset = TrainConfig(predict_with_generate=True, eval_dataset=['e']), DatasetConfig(dataset=['d'])
    _derive_eval_schedule(train, dataset, None)
    assert train.eval_strategy != 'no'


def test_streaming_dataloader_workers_derivation():
    """S2-03: a streaming dataset forces dataloader_num_workers down to 1 (never raises it); a
    non-streaming dataset is untouched."""
    dataset = DatasetConfig(streaming=True, dataloader_num_workers=4)
    _derive_streaming_dataloader_workers(dataset)
    assert dataset.dataloader_num_workers == 1
    dataset = DatasetConfig(streaming=True, dataloader_num_workers=None)  # None stays None
    _derive_streaming_dataloader_workers(dataset)
    assert dataset.dataloader_num_workers is None
    dataset = DatasetConfig(streaming=False, dataloader_num_workers=8)
    _derive_streaming_dataloader_workers(dataset)
    assert dataset.dataloader_num_workers == 8


# --- megatron scheduler / pipeline -----------------------------------------------------


@pytest.mark.parametrize('scheduler, expected', [
    ('cosine', 'cosine'),
    ('linear', 'linear'),
    ('constant', 'constant'),
    ('inverse_sqrt', 'inverse-square-root'),  # the mcore spelling of the same curve
])
def test_lr_decay_style_mapping(scheduler, expected):
    """S2-04: an unset lr_decay_style is derived from lr_scheduler through the mcore style map."""
    train = TrainConfig(lr_scheduler=scheduler)
    _derive_lr_decay_style(train, True)
    assert train.lr_decay_style == expected


@pytest.mark.parametrize('train_scheduler', ['cosine', 'linear'])
def test_lr_decay_style_explicit_and_conflict(train_scheduler):
    """S2-04 (rest): an explicit style stands; an explicit style conflicting with an explicit
    scheduler is refused; a scheduler with no mcore mapping leaves the style alone."""
    train = _explicit(TrainConfig(lr_decay_style='linear'), 'lr_decay_style')
    _derive_lr_decay_style(train, True)
    assert train.lr_decay_style == 'linear'

    train = _explicit(TrainConfig(lr_scheduler='cosine', lr_decay_style='linear'), 'lr_scheduler',
                      'lr_decay_style')
    with pytest.raises(ValueError, match='conflicts with'):
        _derive_lr_decay_style(train, True)

    train = _explicit(TrainConfig(lr_scheduler='wsd'), 'lr_scheduler')  # 'WSD' has no mcore mapping
    _derive_lr_decay_style(train, True)
    assert train.lr_decay_style == 'cosine'  # TrainConfig's own default stays

    # the transformers backend never derives
    train = TrainConfig(lr_scheduler='linear')
    _derive_lr_decay_style(train, False)
    assert train.lr_decay_style == 'cosine'


def test_virtual_pipeline_derivation(monkeypatch):
    """S2-05: a layout implies the vpp width (stages / pp); 1 normalises to None; no vpp pins the
    two overlaps off and defaults batch_p2p_comm to its complement; a layout not divisible by pp
    is refused."""
    monkeypatch.setattr('swift.dev.config.process._layout_num_stages', lambda layout: {
        'dummy-circle-4': 4, 'dummy-circle-8': 8}.get(layout))
    distributed = DistributedConfig(backend='megatron', pipeline_model_parallel_size=2,
                                    pipeline_model_parallel_layout='dummy-circle-4')
    _derive_virtual_pipeline(distributed, True)
    assert distributed.virtual_pipeline_model_parallel_size == 2
    assert distributed.batch_p2p_comm is (not distributed.overlap_p2p_comm)

    # vpp=1 normalises to None, and the overlaps go off
    distributed = DistributedConfig(backend='megatron', virtual_pipeline_model_parallel_size=1)
    _derive_virtual_pipeline(distributed, True)
    assert distributed.virtual_pipeline_model_parallel_size is None
    assert distributed.overlap_p2p_comm is False and distributed.align_param_gather is False
    assert distributed.batch_p2p_comm is True

    # a layout whose stage count is not divisible by pp is refused
    distributed = DistributedConfig(backend='megatron', pipeline_model_parallel_size=3,
                                    pipeline_model_parallel_layout='dummy-circle-8')
    with pytest.raises(ValueError, match='divisible by'):
        _derive_virtual_pipeline(distributed, True)


def test_grad_accum_dtype_derivation():
    """S2-06: a megatron bf16 run with fp32 master grads accumulates in fp32; explicit or
    non-bf16 runs are untouched."""
    train = TrainConfig(main_grads_dtype='fp32')
    _derive_grad_accum_dtype(ModelConfig(torch_dtype='bfloat16'), train, True)
    assert train.accumulate_allreduce_grads_in_fp32 is True

    train = _explicit(TrainConfig(main_grads_dtype='fp32'), 'accumulate_allreduce_grads_in_fp32')
    _derive_grad_accum_dtype(ModelConfig(torch_dtype='bfloat16'), train, True)
    assert train.accumulate_allreduce_grads_in_fp32 is False  # explicit stays

    train = TrainConfig(main_grads_dtype='fp32')
    _derive_grad_accum_dtype(ModelConfig(torch_dtype='float16'), train, True)
    assert train.accumulate_allreduce_grads_in_fp32 is False  # fp16 not auto-upgraded

    train = TrainConfig(main_grads_dtype='fp32')
    _derive_grad_accum_dtype(ModelConfig(torch_dtype='bfloat16'), train, False)
    assert train.accumulate_allreduce_grads_in_fp32 is False  # transformers untouched


def test_recompute_granularity_none_normalisation():
    """S2-07: the string 'none' becomes the None every downstream check tests against."""
    distributed = DistributedConfig(recompute_granularity='none')
    _normalize_recompute_granularity(distributed)
    assert distributed.recompute_granularity is None
    distributed = DistributedConfig(recompute_granularity='full')
    _normalize_recompute_granularity(distributed)
    assert distributed.recompute_granularity == 'full'


# --- multimodal / quantization ---------------------------------------------------------


def test_vit_gradient_checkpointing_derivation():
    """S2-08: the vision tower checkpoints unless frozen; full-param (no tuner) and an explicit
    value are untouched."""
    train = TrainConfig()
    _derive_vit_gradient_checkpointing(train, TunerConfig(tuner='lora', freeze_vit=False))
    assert train.vit_gradient_checkpointing is True

    train = TrainConfig()
    _derive_vit_gradient_checkpointing(train, TunerConfig(tuner='lora', freeze_vit=True))
    assert train.vit_gradient_checkpointing is False

    train = TrainConfig(vit_gradient_checkpointing=False)
    _derive_vit_gradient_checkpointing(train, TunerConfig(tuner='lora', freeze_vit=False))
    assert train.vit_gradient_checkpointing is False  # explicit stays

    train = TrainConfig()
    _derive_vit_gradient_checkpointing(train, None)
    assert train.vit_gradient_checkpointing is None  # full-param: builder default


@pytest.mark.parametrize('torch_dtype, expected', [('float16', 'float32'), ('float32', 'float32'),
                                                   ('bfloat16', 'bfloat16')])
def test_bnb_compute_dtype_mapping(torch_dtype, expected):
    """S2-09: bnb's dequant compute dtype tracks the model dtype (fp16 computes in fp32 -- range)."""
    quantize = QuantizeConfig(quant_method='bnb')
    _derive_bnb_compute_dtype(ModelConfig(torch_dtype=torch_dtype), quantize)
    assert quantize.bnb_4bit_compute_dtype == expected
    # an explicit value and a non-bnb run are untouched
    quantize = QuantizeConfig(quant_method='bnb', bnb_4bit_compute_dtype='float16')
    _derive_bnb_compute_dtype(ModelConfig(torch_dtype='bfloat16'), quantize)
    assert quantize.bnb_4bit_compute_dtype == 'float16'
    # the derivation keys on the unset compute dtype only -- even a quant_method=None Config
    # carries the tracking default (anchoring actual behaviour: nothing branches on quant_method)
    quantize = QuantizeConfig()
    _derive_bnb_compute_dtype(ModelConfig(torch_dtype='bfloat16'), quantize)
    assert quantize.bnb_4bit_compute_dtype == 'bfloat16'
    quantize = QuantizeConfig()
    _derive_bnb_compute_dtype(ModelConfig(torch_dtype='awq'), quantize)  # unmapped dtype: no-op
    assert quantize.bnb_4bit_compute_dtype is None
    quantize = QuantizeConfig()
    _derive_bnb_compute_dtype(ModelConfig(torch_dtype='bfloat16'), None)  # no Config: no-op
    assert quantize.bnb_4bit_compute_dtype is None


# --- RL derivations ---------------------------------------------------------------------


def test_async_mode_folding():
    """S2-10: legacy async_generate=True folds to one_step_off; an explicit async_mode wins."""
    rollout = RolloutConfig(async_generate=True, async_mode='none')
    _derive_async_mode(rollout)
    assert rollout.async_mode == 'one_step_off'
    rollout = RolloutConfig(async_generate=True, async_mode='fully_async')
    _derive_async_mode(rollout)
    assert rollout.async_mode == 'fully_async'
    rollout = RolloutConfig(async_generate=False, async_mode='none')
    _derive_async_mode(rollout)
    assert rollout.async_mode == 'none'


@pytest.mark.parametrize('tuner, sampler_mode, expected_strategy, expected_partial', [
    (None, 'colocate', 'in_place', None),  # full + colocate under sync: no partial
    (None, 'colocate', 'in_place', None),
    (None, 'disaggregated', 'in_place', None),  # full + disaggregated: in_place, no partial under sync
])
def test_weight_sync_strategy_derivation(tuner, sampler_mode, expected_strategy, expected_partial):
    """S2-11: a full-param or colocated streaming run derives the in_place publication (its only
    sound one); an explicit choice is left for validate to police; a disaggregated LoRA run keeps
    the adapter-pin default."""
    # sync mode ('none') but an online type that streams: derive happens
    rollout = RolloutConfig(async_mode='none', sampler_mode=sampler_mode)
    rl = RLConfig(rl_type='grpo')
    _derive_weight_sync_strategy(rollout, rl, None if tuner is None else TunerConfig(tuner=tuner))
    assert rollout.weight_sync_strategy == expected_strategy
    if expected_partial is None:
        assert rollout.allow_partial_rollout is False  # untouched default under sync

    # explicit choice survives (validate's rejection is tested in S2b)
    rollout = _explicit(RolloutConfig(sampler_mode='colocate'), 'weight_sync_strategy')
    rollout.weight_sync_strategy = 'adapter_snapshot'
    _derive_weight_sync_strategy(rollout, RLConfig(rl_type='grpo'), None)
    assert rollout.weight_sync_strategy == 'adapter_snapshot'

    # disaggregated LoRA keeps the default pin-by-adapter ('adapter_snapshot' is the field default)
    rollout = RolloutConfig(sampler_mode='disaggregated')
    _derive_weight_sync_strategy(rollout, RLConfig(rl_type='grpo'), TunerConfig(tuner='lora'))
    assert rollout.weight_sync_strategy == 'adapter_snapshot'


def test_rm_task_type_derivation():
    """S2-12: rl_type=rm trains a seq_cls head with num_labels=1; other rl_types and explicit
    task_types are untouched."""
    model = ModelConfig(model='m')
    _derive_rlhf_task_type(model, RLConfig(rl_type='rm'))
    assert model.task_type == 'seq_cls' and model.num_labels == 1

    model = ModelConfig(model='m', task_type='causal_lm', num_labels=3)
    _derive_rlhf_task_type(model, RLConfig(rl_type='rm'))
    assert model.task_type == 'causal_lm' and model.num_labels == 3  # only fills defaults

    model = ModelConfig(model='m')
    _derive_rlhf_task_type(model, RLConfig(rl_type='dpo'))
    assert model.task_type is None


@pytest.mark.parametrize('rl_type, expected', [
    ('grpo', 0.04), ('gkd', 0.5), ('simpo', 2.0), ('opsd', 0.0), ('mopd', 0.0), ('rft', 0.0),
    ('dpo', 0.1),  # not in the table -> the generic 0.1
])
def test_rlhf_beta_defaults(rl_type, expected):
    """S2-13: each algorithm's default beta (the collected legacy table); explicit wins, and
    --beta 0 is preserved rather than overwritten."""
    rl = RLConfig(rl_type=rl_type)
    _derive_rlhf_beta(rl)
    assert rl.beta == expected
    rl = RLConfig(rl_type=rl_type, beta=0.0)
    _derive_rlhf_beta(rl)
    assert rl.beta == 0.0


def test_rlhf_loss_scale_derivation():
    """S2-14: every rl_type defaults loss_scale to last_round except grpo-with-max_turns (default);
    an explicit loss_scale survives."""
    template = TemplateConfig()
    _derive_rlhf_loss_scale(template, RLConfig(rl_type='dpo'))
    assert template.loss_scale == 'last_round'

    template = TemplateConfig()
    _derive_rlhf_loss_scale(template, RLConfig(rl_type='rm'))
    assert template.loss_scale == 'last_round'

    template = TemplateConfig()
    _derive_rlhf_loss_scale(template, RLConfig(rl_type='grpo'), RolloutConfig(max_turns=4))
    assert template.loss_scale == 'default'

    template = TemplateConfig()
    _derive_rlhf_loss_scale(template, RLConfig(rl_type='grpo'))  # single-turn grpo stays last_round
    assert template.loss_scale == 'last_round'

    template = _explicit(TemplateConfig(loss_scale='all'), 'loss_scale')
    _derive_rlhf_loss_scale(template, RLConfig(rl_type='dpo'))
    assert template.loss_scale == 'all'


def test_rlhf_ref_model_derivation():
    """S2-15: full-param dpo/kto/ppo/grpo default ref_model to the policy model; grpo beta=0
    drops the reference; lora without ref_adapters needs an explicit one; a str ref_adapters
    is listified."""
    model = ModelConfig(model='m', model_type='qwen2')
    rl = RLConfig(rl_type='dpo')
    _derive_rlhf_ref_model(model, None, rl)
    assert rl.ref_model == 'm' and rl.ref_model_type == 'qwen2'

    rl = RLConfig(rl_type='grpo', beta=0.0, ref_model='someone')
    _derive_rlhf_ref_model(model, None, rl)
    assert rl.ref_model is None  # reference-free grpo needs none

    rl = RLConfig(rl_type='dpo')  # lora student: no silent adapter-based ref
    _derive_rlhf_ref_model(model, TunerConfig(tuner='lora'), rl)
    assert rl.ref_model is None

    rl = RLConfig(rl_type='dpo', ref_adapters='a')
    _derive_rlhf_ref_model(model, TunerConfig(tuner='lora'), rl)
    assert rl.ref_adapters == ['a'] and rl.ref_model == 'm'


def test_grpo_reward_defaults():
    """S2-16: kl_in_reward only for the reinforce-family estimators; scale_rewards follows the
    per-estimator table."""
    rl = RLConfig(rl_type='grpo', advantage_estimator='rloo')
    _derive_grpo_reward_defaults(rl)
    assert rl.kl_in_reward is True and rl.scale_rewards == 'none'

    rl = RLConfig(rl_type='grpo')  # default estimator 'grpo'
    _derive_grpo_reward_defaults(rl)
    assert rl.kl_in_reward is False and rl.scale_rewards == 'group'

    rl = RLConfig(rl_type='grpo', advantage_estimator='reinforce_plus_plus')
    _derive_grpo_reward_defaults(rl)
    assert rl.kl_in_reward is True and rl.scale_rewards == 'batch'

    rl = RLConfig(rl_type='dpo')  # non-grpo untouched
    _derive_grpo_reward_defaults(rl)
    assert rl.kl_in_reward is None and rl.scale_rewards is None


@pytest.mark.parametrize('rl_type, predict_with_generate, metric', [
    ('grpo', False, 'reward'),
    ('ppo', False, None),  # ppo stays unset: loss is not monotone in quality
    (None, True, 'rouge-l'),
    (None, False, 'loss'),
    ('dpo', False, 'loss'),
])
def test_best_model_metric_derivation(rl_type, predict_with_generate, metric):
    """S2-17: grpo selects reward, generation eval selects rouge-l, plain training selects loss,
    ppo leaves it unset; greater_is_better flips for loss."""
    train = TrainConfig(predict_with_generate=predict_with_generate)
    _derive_best_model_metric(train, RLConfig(rl_type=rl_type) if rl_type else None)
    assert train.metric_for_best_model == metric
    if metric is not None:
        assert train.greater_is_better == ('loss' not in metric)


# --- dataset-side derivations -----------------------------------------------------------


def test_packing_and_split_ratio_derivation():
    """S2-18: packing forces the padding-free representation and inherits max_length as the
    packing length; split_dataset_ratio is zeroed once a real val set (or streaming) exists."""
    dataset, template = DatasetConfig(packing=True), TemplateConfig(max_length=512)
    _derive_packing(dataset, template)
    assert template.padding_free is True and dataset.packing_length == 512

    dataset = DatasetConfig(dataset=['d'], val_dataset=['v'], split_dataset_ratio=0.05)
    _derive_split_dataset_ratio(dataset)
    assert dataset.split_dataset_ratio == 0.0

    dataset = DatasetConfig(dataset=['d'], streaming=True, split_dataset_ratio=0.05)
    _derive_split_dataset_ratio(dataset)
    assert dataset.split_dataset_ratio == 0.0

    dataset = DatasetConfig(dataset=['d'], split_dataset_ratio=0.05)
    _derive_split_dataset_ratio(dataset)
    assert dataset.split_dataset_ratio == 0.05  # no val / no streaming: keep carving


def test_per_token_loss_derivation():
    """S2-19: plain megatron causal-LM training averages per token; an RL run averages per batch;
    an explicit value and the transformers backend are untouched."""
    train = TrainConfig()
    _derive_per_token_loss(ModelConfig(task_type='causal_lm'), train, None, True)
    assert train.calculate_per_token_loss is True

    train = TrainConfig()
    _derive_per_token_loss(ModelConfig(task_type='causal_lm'), train, RLConfig(rl_type='grpo'), True)
    assert train.calculate_per_token_loss is False

    train = TrainConfig(calculate_per_token_loss=False)
    _derive_per_token_loss(ModelConfig(task_type='causal_lm'), train, None, True)
    assert train.calculate_per_token_loss is False

    train = TrainConfig()
    _derive_per_token_loss(ModelConfig(task_type='causal_lm'), train, None, False)
    assert train.calculate_per_token_loss is None


# --- resolution steps (hub monkeypatched) ------------------------------------------------


def test_resolve_model_snapshots_and_keeps_id(monkeypatch):
    """S2-20: bootstrap resolves the model id to its local snapshot while preserving the portable
    id on a transient attribute (args.json records it)."""
    calls = []

    def fake_download(model, *, revision=None, use_hf=None, hub_token=None):
        calls.append((model, revision))
        return '/cache/models/m'

    monkeypatch.setattr('swift.dev.utils.hub.safe_snapshot_download', fake_download)
    model = ModelConfig(model='m', model_revision='v2')
    _resolve_model(model, DatasetConfig())
    assert model.model == '/cache/models/m'
    assert model._model_id_or_path == 'm'
    assert calls == [('m', 'v2')]
    # an empty model is a no-op
    model = ModelConfig()
    _resolve_model(model, None)
    assert model.model is None


def test_resolve_adapters_keeps_served_names(monkeypatch):
    """S2-21: each adapter path is resolved to its local snapshot with the served name re-attached,
    so deploy's routing label survives the resolution."""
    def fake_download(path, *, revision=None, use_hf=None, hub_token=None):
        return f'/cache/{path}'

    monkeypatch.setattr('swift.dev.utils.hub.safe_snapshot_download', fake_download)
    tuner = TunerConfig(adapters=['pol=a', 'b'])
    _resolve_adapters(tuner, None)
    assert tuner.adapters == ['pol=/cache/a', '/cache/b']


def test_split_served_name():
    """S2-22: ``name=path`` splits only when the head is a bare name; paths containing ``=`` and
    windows-style heads survive; a bare path reads as (None, path)."""
    assert split_served_name('a=/tmp/l1') == ('a', '/tmp/l1')
    assert split_served_name('/tmp/x=1') == (None, '/tmp/x=1')  # '/' in head: no split
    assert split_served_name('rel/path') == (None, 'rel/path')
    assert split_served_name('name') == (None, 'name')
    assert split_served_name('C:\\x') == (None, 'C:\\x')  # '\\' in head: no split
    assert split_served_name('a=C:\\x') == ('a', 'C:\\x')  # head 'a' is a bare name: splits


# --- bootstrap side effects ---------------------------------------------------------------


def test_prepare_output_dir_versioning(tmp_path):
    """S2-23: the output dir is absolutised; add_version stamps a version subdir; create_output_dir
    off leaves the filesystem untouched; a missing resume path is refused."""
    checkpoint = CheckpointConfig(output_dir=str(tmp_path / 'run'))
    _prepare_output_dir(checkpoint, add_version=True, create_output_dir=True)
    assert checkpoint.output_dir.startswith(str((tmp_path / 'run').resolve()))
    assert 'v' in os.path.basename(checkpoint.output_dir)  # the version stamp
    assert os.path.isdir(checkpoint.output_dir)

    checkpoint = CheckpointConfig(output_dir=str(tmp_path / 'plain'))
    _prepare_output_dir(checkpoint, add_version=False, create_output_dir=False)
    assert checkpoint.output_dir == str((tmp_path / 'plain').resolve())
    assert not os.path.exists(checkpoint.output_dir)

    checkpoint = CheckpointConfig(resume_from_checkpoint=str(tmp_path / 'missing'))
    with pytest.raises(ValueError, match='does not exist'):
        _prepare_output_dir(checkpoint, add_version=False, create_output_dir=False)


def test_export_model_kwargs_uppercases_into_environ(monkeypatch):
    """S2-24: model_kwargs entries are exported as UPPERCASE env vars (the twinkle pre-import knobs)."""
    for key in ('CLI_RUNTIME_TEST',):
        monkeypatch.delenv(key, raising=False)
    _export_model_kwargs(ModelConfig(model_kwargs={'cli_runtime_test': 'yes'}))
    assert os.environ['CLI_RUNTIME_TEST'] == 'yes'
    _export_model_kwargs(ModelConfig())  # None is a no-op


def test_login_hub_only_with_token(monkeypatch):
    """S2-25: a hub_token triggers the hub login; without one the login is never attempted."""
    logins = []
    monkeypatch.setattr('swift.dev.utils.hub.get_hub',
                        lambda use_hf: type('H', (), {'try_login': staticmethod(
                            lambda token: logins.append(token))})())
    _login_hub(DatasetConfig(hub_token='tok'))
    assert logins == ['tok']
    _login_hub(DatasetConfig())
    _login_hub(None)
    assert logins == ['tok']


def test_mtp_compat_fix(monkeypatch):
    """S2-26: the megatron shim derives enable_mtp_training from mtp_num_layers, and clears the
    loss scaling factor when there are no MTP layers."""
    from swift.dev.cli._megatron_compat import _fix_mtp

    monkeypatch.delenv('WORLD_SIZE', raising=False)
    model = ModelConfig(mtp_num_layers=None, mtp_loss_scaling_factor=0.1)
    _fix_mtp(model)
    assert model.mtp_loss_scaling_factor is None and model.enable_mtp_training is False

    model = ModelConfig(mtp_num_layers=1)
    _fix_mtp(model)
    assert model.enable_mtp_training is True


def test_rlhf_teacher_self_distillation_detection():
    """S2-15 companion (teacher half): a teacher that IS the student with no teacher_adapters is
    self-distillation -- the flag is set and the teacher list drops to None; a str teacher_model
    is listified; a distinct teacher keeps the list."""
    model = ModelConfig(model='m')
    rl = RLConfig(rl_type='opsd', teacher_model='m')
    _derive_rlhf_teacher(model, TunerConfig(tuner='lora'), rl)
    assert rl._teacher_use_disable_adapter is True and rl.teacher_model is None

    rl = RLConfig(rl_type='opsd', teacher_model='other')
    _derive_rlhf_teacher(model, None, rl)
    assert rl.teacher_model == ['other'] and not getattr(rl, '_teacher_use_disable_adapter', False)

    rl = RLConfig(rl_type='opsd', teacher_model='m', teacher_adapters='t')
    _derive_rlhf_teacher(model, None, rl)
    assert rl.teacher_model == ['m'] and rl.teacher_adapters == ['t']  # adapters: not self-distill
