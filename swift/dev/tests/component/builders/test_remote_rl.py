# Copyright (c) ModelScope Contributors. All rights reserved.
"""mode='remote' + RL: the wire contracts of the remote object sources.

Every recipe algorithm runs remotely with the SAME loop code, so what must be pinned is what
crosses HTTP: the weight-sync hand-off (save -> adapter_uri), the loss / metric registrations
BY CLASS NAME with the SAME kwargs the local configure_* family forwards (a drift trains a
different objective server-side than the local reference run), and the assembly guards that
fail loudly instead of silently dropping remote-unsupported knobs. The policy / critic /
reference / teacher / reward handles themselves are exercised by the remote e2e chain (F9);
these tests pin the client-side face that produces the wire calls.
"""
from __future__ import annotations
from typing import Any, List, Optional, Tuple

import pytest

from swift.dev.builders.model import build_ref_model
from swift.dev.builders.remote import RemoteWeightSync
from swift.dev.builders.reward import _build_frozen_reward_model
from swift.dev.config import DistributedConfig, ModelConfig, RLConfig, RolloutConfig, TemplateConfig
from swift.dev.loss.configure import (
    configure_ppo_value_loss,
    configure_ppo_value_metric,
    configure_rl_loss,
    configure_rl_metrics,
)
from swift.dev.recipe.utils import build_rl_sampler
from swift.dev.rewards.scorer import RewardSpec


class _RecordingModel:
    """The client-side handle face: names + scalar kwargs only (what crosses HTTP)."""

    def __init__(self) -> None:
        self.losses: List[Tuple[Any, dict]] = []
        self.metrics: List[Tuple[Any, dict]] = []
        self.adapter_calls: List[Tuple[str, dict, dict]] = []
        self.load_calls: List[Tuple[str, dict]] = []

    def set_loss(self, loss: Any, **kwargs: Any) -> None:
        self.losses.append((loss, kwargs))

    def add_metric(self, metric: Any, **kwargs: Any) -> None:
        self.metrics.append((metric, kwargs))

    def add_adapter_to_model(self, adapter_name: str, config: Any = None, **kwargs: Any) -> None:
        self.adapter_calls.append((adapter_name, config, kwargs))

    def load(self, name: str, **kwargs: Any) -> None:
        self.load_calls.append((name, kwargs))


class _SaveModel:
    """The model face RemoteWeightSync drives: save() returns the server's save response."""

    def __init__(self, twinkle_path: Optional[str]) -> None:
        self._path = twinkle_path
        self.save_calls: List[Tuple[tuple, dict]] = []

    def save(self, *args: Any, **kwargs: Any) -> Any:
        import types

        self.save_calls.append((args, kwargs))
        return types.SimpleNamespace(twinkle_path=self._path)


# --- RemoteWeightSync: the save -> adapter_uri weight channel -----------------------------


def test_remote_weight_sync_saves_a_sampler_checkpoint_and_stamps_the_uri():
    save_model = _SaveModel('twinkle://ckpts/step-1')
    sampler = type('Sampler', (), {'adapter_uri': None})()
    engine = RemoteWeightSync(save_model, sampler)
    engine.sync_weights()
    # exactly the sampler-serving checkpoint face: no optimizer state, newest-only server-side
    assert save_model.save_calls == [
        (('remote-sampler-weights',), {'save_optimizer': False, 'is_sampler': True})]
    assert sampler.adapter_uri == 'twinkle://ckpts/step-1'
    assert engine.version() == 1


def test_remote_weight_sync_replaces_the_uri_and_counts_one_version_per_sync():
    save_model = _SaveModel('twinkle://ckpts/step-2')
    sampler = type('Sampler', (), {'adapter_uri': 'twinkle://ckpts/step-1'})()
    engine = RemoteWeightSync(save_model, sampler)
    engine.sync_weights()
    engine.sync_weights()
    assert sampler.adapter_uri == 'twinkle://ckpts/step-2'  # a new sync REPLACES the stamped URI
    assert engine.version() == 2


def test_remote_weight_sync_fails_loudly_without_a_twinkle_path():
    save_model = _SaveModel(None)
    sampler = type('Sampler', (), {'adapter_uri': 'twinkle://old'})()
    engine = RemoteWeightSync(save_model, sampler)
    with pytest.raises(ValueError, match='twinkle://'):
        engine.sync_weights()
    assert engine.version() == 0  # a failed sync must not advance the loop's version() gate
    assert sampler.adapter_uri == 'twinkle://old'  # and must not restamp a stale URI
    assert engine.close() is None  # no channel to tear down


# --- configure_rl_loss: the loss crosses HTTP as CLASS NAME + kwargs ---------------


def test_remote_rl_loss_grpo_forwards_the_clip_and_kl_kwargs():
    model = _RecordingModel()
    configure_rl_loss(model, RLConfig(rl_type='grpo', epsilon=0.2, epsilon_high=0.28, beta=0.1))
    name, kwargs = model.losses[0]  # the rest of the kwarg set is _rl_loss_kwargs' shared base
    assert name == 'GRPOLoss'
    assert kwargs['beta'] == 0.1 and kwargs['epsilon'] == 0.2 and kwargs['epsilon_high'] == 0.28


def test_remote_rl_loss_folds_the_grpo_variants():
    model = _RecordingModel()
    configure_rl_loss(model, RLConfig(rl_type='grpo', loss_type=['dapo'], epsilon=0.2))
    assert model.losses[0][0] == 'BNPOLoss'  # dapo reuses BNPO's aggregation
    model = _RecordingModel()
    configure_rl_loss(model, RLConfig(rl_type='grpo', loss_type=['fipo'], epsilon=0.2))
    assert model.losses[0][0] == 'GRPOLoss'  # fipo rides the plain surrogate


def test_remote_rl_loss_ppo_uses_the_shared_surrogate_with_cliprange():
    model = _RecordingModel()
    configure_rl_loss(model, RLConfig(rl_type='ppo', cliprange=0.15))
    assert model.losses == [('GRPOLoss', {'epsilon': 0.15})]


def test_remote_rl_loss_kto_gkd_opsd_mopd_rm_forward_their_own_knobs():
    model = _RecordingModel()
    configure_rl_loss(
        model, RLConfig(rl_type='kto', beta=0.2, desirable_weight=1.4, undesirable_weight=0.8))
    assert model.losses == [('KTOLoss', {'beta': 0.2, 'desirable_weight': 1.4, 'undesirable_weight': 0.8})]

    model = _RecordingModel()
    configure_rl_loss(
        model, RLConfig(rl_type='gkd', beta=0.1, distill_temperature=2.0, sft_alpha=0.3))
    assert model.losses == [('GKDLoss', {'beta': 0.1, 'temperature': 2.0, 'sft_alpha': 0.3})]

    model = _RecordingModel()
    configure_rl_loss(model, RLConfig(rl_type='opsd', opsd_reverse=False))
    assert model.losses == [('OPSDLoss', {'reverse': False})]

    model = _RecordingModel()
    configure_rl_loss(model, RLConfig(rl_type='mopd'))
    assert model.losses == [('MOPDLoss', {'reverse': True})]

    model = _RecordingModel()
    configure_rl_loss(model, RLConfig(rl_type='rm'))
    assert model.losses[0][0] == 'RewardLoss'  # the score head's plain regression loss


def test_remote_rl_loss_dpo_family_subset_of_the_pairwise_kwargs():
    model = _RecordingModel()
    configure_rl_loss(model, RLConfig(rl_type='dpo', beta=0.1))
    name, kwargs = model.losses[0]
    assert name == 'DPOLoss'
    assert kwargs['beta'] == 0.1 and kwargs['loss_type'] == ['sigmoid'] and 'reference_free' in kwargs


def test_remote_rl_loss_refuses_rft_and_unknown_types():
    model = _RecordingModel()
    with pytest.raises(ValueError, match='Unknown rl_type'):  # rft trains on configure_remote_loss
        configure_rl_loss(model, RLConfig(rl_type='rft'))
    with pytest.raises(ValueError, match='Unknown rl_type'):
        configure_rl_loss(model, RLConfig(rl_type='bogus'))


def test_remote_rl_loss_registers_the_chord_mixture_as_two_named_losses():
    """CHORD mixes the policy gradient with a supervised CE term on dataset rounds: the server
    model needs BOTH registered under their loss_keys, or the mixture silently degrades to
    plain GRPO (the exact gap this pins -- the local branch registers the pair)."""
    model = _RecordingModel()
    configure_rl_loss(
        model, RLConfig(rl_type='grpo', epsilon=0.2, chord_sft_dataset=['d'], chord_enable_phi_function=True))
    assert [name for name, _ in model.losses] == ['GRPOLoss', 'CrossEntropyLoss']
    assert model.losses[0][1]['loss_key'] == 'grpo' and model.losses[0][1]['epsilon'] == 0.2
    assert model.losses[1][1]['loss_key'] == 'sft' and model.losses[1][1]['reduction'] == 'sum'
    assert model.losses[1][1]['probability_weight'] == 'p_one_minus_p'
    model = _RecordingModel()
    configure_rl_loss(model, RLConfig(rl_type='grpo', chord_sft_dataset=['d']))
    assert model.losses[1][1]['probability_weight'] is None  # phi off: no probability weighting


def test_remote_rl_loss_names_resolve_on_the_server_face():
    """set_loss resolves a class NAME through twinkle.loss (construct_class): every name the
    remote path can emit must construct there, or the server-side run crashes on an unknown
    loss."""
    import twinkle.loss
    from twinkle.loss import Loss
    from twinkle.utils import construct_class

    for rl_type in ('grpo', 'ppo', 'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'gkd', 'opsd', 'mopd', 'rm'):
        model = _RecordingModel()
        configure_rl_loss(model, RLConfig(rl_type=rl_type))
        for name, _ in model.losses:
            assert isinstance(construct_class(name, Loss, twinkle.loss), Loss), (rl_type, name)
    model = _RecordingModel()
    configure_rl_loss(model, RLConfig(rl_type='grpo', chord_sft_dataset=['d']))
    for name, _ in model.losses:
        assert isinstance(construct_class(name, Loss, twinkle.loss), Loss), name


# --- configure_rl_metrics / PPO value loss+metric: class NAME + same kwargs ----------------


def test_remote_rl_metrics_pass_class_names_with_the_same_kwargs():
    model = _RecordingModel()
    configure_rl_metrics(model, RLConfig(rl_type='grpo', epsilon=0.2), 0.7)
    assert model.metrics == [('GRPOMetric', {'is_training': True, 'epsilon': 0.2, 'temperature': 0.7})]

    model = _RecordingModel()
    configure_rl_metrics(model, RLConfig(rl_type='grpo', loss_type=['gspo'], epsilon=0.2))
    assert model.metrics[0][0] == 'GSPOMetric'  # the clip statistic must match the loss's variant
    model = _RecordingModel()
    configure_rl_metrics(model, RLConfig(rl_type='grpo', loss_type=['cispo'], epsilon=0.2))
    assert model.metrics[0][0] == 'CISPOMetric'

    model = _RecordingModel()
    configure_rl_metrics(model, RLConfig(rl_type='grpo', epsilon=0.2, chord_sft_dataset=['d']))
    assert model.metrics[0][1]['loss_key'] == 'grpo'  # CHORD keys the metric on the policy-gradient loss


def test_remote_rl_metrics_for_ppo_and_the_preference_family():
    model = _RecordingModel()
    configure_rl_metrics(model, RLConfig(rl_type='ppo', cliprange=0.25), 0.5)
    assert model.metrics == [('PPOMetric', {'is_training': True, 'epsilon': 0.25, 'temperature': 0.5})]

    model = _RecordingModel()
    configure_rl_metrics(model, RLConfig(rl_type='dpo', beta=0.1))
    assert model.metrics == [('DPOMetric', {'is_training': True, 'beta': 0.1})]

    model = _RecordingModel()
    configure_rl_metrics(model, RLConfig(rl_type='orpo'))  # beta unset: kwarg not forwarded
    assert model.metrics == [('DPOMetric', {'is_training': True})]

    for rl_type in ('kto', 'rm', 'gkd', 'opsd', 'mopd'):  # no algorithm-specific component
        model = _RecordingModel()
        configure_rl_metrics(model, RLConfig(rl_type=rl_type))
        assert model.metrics == [], rl_type


def test_rl_metrics_use_the_same_name_form_on_every_face():
    """One wire form, no remote fork: the CLASS NAME + kwargs is what a local TrainableModel's
    construct_class and the served-model client face BOTH resolve -- configure_rl_metrics emits
    it unconditionally."""
    model = _RecordingModel()
    configure_rl_metrics(model, RLConfig(rl_type='grpo', epsilon=0.2))
    assert model.metrics == [('GRPOMetric', {'is_training': True, 'epsilon': 0.2, 'temperature': 1.0})]
    model = _RecordingModel()
    configure_rl_metrics(model, RLConfig(rl_type='dpo', beta=0.1))
    assert model.metrics == [('DPOMetric', {'is_training': True, 'beta': 0.1})]


def test_ppo_value_loss_and_metric_ride_the_name_form():
    """One wire form, no remote fork: the critic's loss and metric cross every transport as the
    CLASS NAME + the same kwargs the local construct_class would build from."""
    model = _RecordingModel()
    rl_config = RLConfig(rl_type='ppo', cliprange_value=0.12, vf_coef=0.5)
    configure_ppo_value_loss(model, rl_config)
    assert model.losses == [('PPOValueLoss', {'cliprange_value': 0.12, 'vf_coef': 0.5})]
    configure_ppo_value_metric(model, rl_config)
    assert model.metrics == [('PPOValueMetric', {'is_training': True, 'epsilon': 0.12})]


# --- the assembly guards: fail loudly instead of silently dropping remote-unsupported knobs --


def _remote_dist() -> DistributedConfig:
    return DistributedConfig(mode='remote', server_base_url='http://127.0.0.1:9999')


def test_build_ref_model_remote_guards():
    """The frozen auxiliary needs its serialized template spec; a frozen ADAPTER preload is the
    server deployment's own configuration, refused instead of silently not applying."""
    with pytest.raises(ValueError, match='template_config'):
        build_ref_model(
            name='ref', model_id='ms://other/model', model_config=ModelConfig(model='m'), template=None,
            distributed_config=_remote_dist(), template_config=None)
    with pytest.raises(ValueError, match='frozen adapter'):
        build_ref_model(
            name='ref', model_id='ms://other/model', model_config=ModelConfig(model='m'), template=None,
            distributed_config=_remote_dist(), template_config=TemplateConfig(), adapters=('a',))


def test_build_ref_model_remote_disable_adapter_returns_the_policy_itself():
    """The LoRA KL anchor keeps its shared-weights semantics remotely: disable_adapter shorts
    to the policy handle BEFORE any client call (no network, no second served model)."""
    policy = object()
    out = build_ref_model(
        name='ref', model_id='ms://other/model', model_config=ModelConfig(model='m'), template=policy,
        distributed_config=_remote_dist(), policy_model=policy, disable_adapter=True)
    assert out is policy


def test_build_rl_sampler_remote_needs_vllm_and_a_template_config():
    with pytest.raises(ValueError, match="rollout_sampler='vllm'"):
        build_rl_sampler(
            model_config=ModelConfig(model='m'), rollout_config=RolloutConfig(rollout_sampler='sglang'),
            template=None, tuner_config=None, distributed_config=_remote_dist(),
            template_config=TemplateConfig())
    with pytest.raises(ValueError, match='template_config'):
        build_rl_sampler(
            model_config=ModelConfig(model='m'), rollout_config=RolloutConfig(), template=None,
            tuner_config=None, distributed_config=_remote_dist(), template_config=None)


def test_frozen_reward_model_remote_loads_a_pretrained_lora_head(monkeypatch):
    """A pre-trained scalar RM (LoRA + seq_cls head -- e.g. an earlier remote seq_cls
    training's checkpoint) rides the same channels that training used: the served-model
    handle gets the adapter created with the checkpoint's OWN config plus the head kwargs,
    then load() pours the LoRA weights and the head's task_head.safetensors back in."""
    import types

    import peft
    import swift.dev.builders as builders
    import swift.dev.builders.remote as remote_mod
    import swift.dev.model.loader as model_loader

    monkeypatch.setattr(
        model_loader, 'resolve_model_info',
        lambda *a, **k: types.SimpleNamespace(model_type='qwen2', task_type='seq_cls', num_labels=1))
    monkeypatch.setattr(builders, 'load_model_processor', lambda *a, **k: (None, None))
    monkeypatch.setattr(builders, 'build_template', lambda *a, **k: types.SimpleNamespace())
    handle = _RecordingModel()
    monkeypatch.setattr(remote_mod, 'build_remote_model', lambda *a, **k: handle)
    monkeypatch.setattr(peft.PeftConfig, 'from_pretrained', staticmethod(lambda path: peft.LoraConfig(r=8)))

    spec = RewardSpec(kind='scalar', model_id='ms://rm/model', adapter='/shared/rm-ckpt')
    frozen, _ = _build_frozen_reward_model(
        spec, ModelConfig(model='m'), TemplateConfig(), distributed_config=_remote_dist())
    assert frozen is handle
    # the adapter + head first (the checkpoint's OWN config; the seq_cls head kwargs), then the
    # weight pour with NO optimizer state -- the RM is frozen, only its forward is driven.
    assert len(handle.adapter_calls) == 1
    adapter_name, adapter_config, head_kwargs = handle.adapter_calls[0]
    assert adapter_name == 'reward'
    assert isinstance(adapter_config, peft.LoraConfig) and adapter_config.r == 8
    assert head_kwargs == {'task': 'seq_cls', 'num_labels': 1}
    assert handle.load_calls == [('/shared/rm-ckpt', {'load_optimizer': False})]


def test_frozen_reward_model_remote_loads_from_the_server_store(monkeypatch):
    """A twinkle:// store URI is SERVER-LOCAL -- the natural stage-1 -> stage-2 same-server flow.
    The client cannot read its adapter_config, so the adapter is created with the dev DEFAULT LoRA
    shape and load() pours the weights from the store; a rank/shape mismatch fails loudly at load
    time instead of being pre-refused here."""
    import types

    import peft
    import swift.dev.builders as builders
    import swift.dev.builders.remote as remote_mod
    import swift.dev.model.loader as model_loader

    monkeypatch.setattr(
        model_loader, 'resolve_model_info',
        lambda *a, **k: types.SimpleNamespace(model_type='qwen2', task_type='seq_cls', num_labels=1))
    monkeypatch.setattr(builders, 'load_model_processor', lambda *a, **k: (None, None))
    monkeypatch.setattr(builders, 'build_template', lambda *a, **k: types.SimpleNamespace())
    handle = _RecordingModel()
    monkeypatch.setattr(remote_mod, 'build_remote_model', lambda *a, **k: handle)

    spec = RewardSpec(kind='scalar', model_id='ms://rm/model', adapter='twinkle://runs/rm/ckpts/1')
    frozen, _ = _build_frozen_reward_model(
        spec, ModelConfig(model='m'), TemplateConfig(), distributed_config=_remote_dist())
    assert frozen is handle
    assert len(handle.adapter_calls) == 1
    adapter_name, adapter_config, head_kwargs = handle.adapter_calls[0]
    assert adapter_name == 'reward'
    # the dev default LoRA shape (the same one an unconfigured --tuner lora run gets)
    assert isinstance(adapter_config, peft.LoraConfig)
    assert adapter_config.r == 8 and adapter_config.target_modules == 'all-linear'
    assert head_kwargs == {'task': 'seq_cls', 'num_labels': 1}
    assert handle.load_calls == [('twinkle://runs/rm/ckpts/1', {'load_optimizer': False})]


def test_frozen_reward_model_remote_refuses_an_unreadable_adapter(monkeypatch):
    """A frozen adapter the client cannot READ (a bad path, a hub id that does not resolve)
    fails loudly -- a hand-built config guessing the checkpoint's rank would either crash
    at load time or silently train a re-initialized RM."""
    import types

    import peft
    import swift.dev.builders as builders
    import swift.dev.builders.remote as remote_mod
    import swift.dev.model.loader as model_loader

    monkeypatch.setattr(
        model_loader, 'resolve_model_info',
        lambda *a, **k: types.SimpleNamespace(model_type='qwen2', task_type='seq_cls', num_labels=1))
    monkeypatch.setattr(builders, 'load_model_processor', lambda *a, **k: (None, None))
    monkeypatch.setattr(builders, 'build_template', lambda *a, **k: types.SimpleNamespace())
    handle = _RecordingModel()
    monkeypatch.setattr(remote_mod, 'build_remote_model', lambda *a, **k: handle)

    def _boom(path):
        raise OSError(f'cannot open {path}')

    monkeypatch.setattr(peft.PeftConfig, 'from_pretrained', staticmethod(_boom))
    spec = RewardSpec(kind='scalar', model_id='ms://rm/model', adapter='/nowhere/rm-ckpt')
    with pytest.raises(ValueError, match='cannot read the frozen adapter'):
        _build_frozen_reward_model(
            spec, ModelConfig(model='m'), TemplateConfig(), distributed_config=_remote_dist())
    assert handle.adapter_calls == [] and handle.load_calls == []
