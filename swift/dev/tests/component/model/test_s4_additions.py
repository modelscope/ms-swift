# Copyright (c) Modelscope Contributors. All rights reserved.
"""S4 additions -- the plan's newly-derived rows the v5-port files do not cover.

Covers: S4-01 registry reachability, S4-02 arch ambiguity, S4-04 reward family, S4-05 embedding
family, S4-06 mamba requires, S4-07 telechat token copy, S4-08 trust_remote_code families,
S4-12 megatron attn backend, S4-13 _set_load_kwarg, S4-14 rope_scaling, S4-22 protocol,
S4-27 mixed precision, S4-29 all-linear folding, S4-30/31 init_weights, S4-33 trainable_tokens.

Deferred (annotation in ut_plan): S4-11 (special-token add path lands with the processor suite),
S4-20 (multi_lora ST build), S4-24 (precision three-state build), S4-34/35 (frozen/multi adapter
shims live in the feature/rl and megatron suites, S13/S14).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from swift.dev.model.loader.base import (
    MODEL_ALIASES,
    MODEL_MAPPING,
    get_model_loader,
    match_model_types_by_architectures,
)


# --- S4-01: the registry is fully reachable ---------------------------------------------------


def test_every_registered_family_and_alias_resolves():
    """Each of the loader files' families is live in MODEL_MAPPING and resolves; each alias lands
    on the family it declares. A dead key here is a family silently unreachable from the CLI."""
    assert len(MODEL_MAPPING) > 100, f'registry unexpectedly small: {len(MODEL_MAPPING)}'
    for model_type in MODEL_MAPPING:
        assert get_model_loader(model_type) is MODEL_MAPPING[model_type], model_type
    for alias, target in MODEL_ALIASES.items():
        assert get_model_loader(alias) is get_model_loader(target), alias
    with pytest.raises(ValueError, match='not registered'):
        get_model_loader('no_such_family_xyz')


def test_ambiguous_architecture_returns_every_candidate():
    """S4-02: qwen2 and qwen2_gte declare the same architecture, so the reverse lookup must return
    BOTH -- the caller then demands an explicit model_type instead of silently picking one."""
    assert get_model_loader('qwen2').architectures == ['Qwen2ForCausalLM']
    assert get_model_loader('qwen2_gte').architectures == ['Qwen2ForCausalLM']
    assert match_model_types_by_architectures(['Qwen2ForCausalLM']) == ['qwen2', 'qwen2_gte']


# --- family declarations (S4-04 / S4-05 / S4-06 / S4-08) ---------------------------------------


def test_reward_families_declare_the_seq_cls_head():
    """S4-04: a reward family is a seq_cls head with num_labels=1 -- declared on the family so the
    head is built 1-wide without the user passing num_labels."""
    for model_type in ('qwen2_reward', 'internlm2_reward', 'qwen2_5_prm'):
        assert get_model_loader(model_type).is_reward is True, model_type
    assert get_model_loader('qwen2').is_reward is False


def test_bert_bge_families_declare_their_tasks():
    """S4-05: the encoder families declare their task shape -- embedding for the gte pooler,
    reranker for the score head, plain bert carrying no default task."""
    assert get_model_loader('modern_bert_gte').task_type == 'embedding'
    assert get_model_loader('bge_reranker').task_type == 'reranker'
    assert get_model_loader('bert').task_type is None


def test_mamba_declares_its_version_gate():
    """S4-06: the mamba family carries its hard requirement (transformers>=4.39.0 -- the version
    that ships MambaForCausalLM) so a too-old install fails the gate, not the forward."""
    assert get_model_loader('mamba').requires == ['transformers>=4.39.0']


@pytest.mark.parametrize('model_type', ['deepseek', 'chatglm4'])
def test_remote_code_families_declare_trust_remote_code(model_type):
    """S4-08: checkpoints whose code lives beside the config declare trust_remote_code; plain
    families stay False so a user's --trust_remote_code decision is never silently overridden."""
    assert get_model_loader(model_type).trust_remote_code is True
    assert get_model_loader('qwen2').trust_remote_code is False


# --- S4-07: telechat copies special-token ids onto the processor ------------------------------


def test_telechat_copies_special_token_ids_from_generation_config(monkeypatch):
    """The TeleChat processor does not expose the chat control ids; the loader copies
    bos/eos/pad/user/bot off generation_config onto the processor after the base build."""
    import swift.dev.model.loader.telechat as telechat_module
    from swift.dev.model.loader.base import LoaderModelInfo, ModelLoader

    token_ids = {'bos_token_id': 1, 'eos_token_id': 2, 'pad_token_id': 3,
                 'user_token_id': 100001, 'bot_token_id': 100002}
    monkeypatch.setattr(
        ModelLoader, 'build_processor',
        lambda self, model_dir, config, **kw: SimpleNamespace())
    monkeypatch.setattr(
        telechat_module, 'GenerationConfig',
        SimpleNamespace(from_pretrained=lambda model_dir: SimpleNamespace(**token_ids)))
    loader = get_model_loader('telechat')(LoaderModelInfo(model_type='telechat', model_dir='m'))
    processor = loader.build_processor('m', config=SimpleNamespace())
    for key, value in token_ids.items():
        assert getattr(processor, key) == value, key


# --- S4-12 / S4-13: attn backend + load-kwarg seam --------------------------------------------


def test_resolve_megatron_attn_backend_maps_known_impls():
    """S4-12: megatron attention backends resolve by name; an unknown name is refused with the
    supported set instead of reaching mcore as a silently-wrong string."""
    from swift.dev.utils.naming import resolve_megatron_attn_backend
    assert resolve_megatron_attn_backend('flash').name == 'flash'
    assert resolve_megatron_attn_backend(None).name == 'flash'  # the unset default IS flash
    with pytest.raises(NotImplementedError, match='[Uu]nknown attn_impl'):
        resolve_megatron_attn_backend('bogus')


def test_set_load_kwarg_rejects_disagreement():
    """S4-13: the load-kwargs seam -- a set value lands, None is a no-op (the default never
    overwrites a user's model_kwargs), and a DISAGREEMENT with model_kwargs is refused rather than
    silently last-wins (the seam max_memory/attn_implementation/torch_dtype all flow through)."""
    from swift.dev.builders.model import _set_load_kwarg

    kwargs: dict = {}
    _set_load_kwarg(kwargs, 'max_memory', {0: '10GiB'})
    assert kwargs['max_memory'] == {0: '10GiB'}
    _set_load_kwarg(kwargs, 'max_memory', None)
    assert kwargs['max_memory'] == {0: '10GiB'}, 'None must be a no-op, not an unset'
    with pytest.raises(ValueError, match='conflicts'):
        _set_load_kwarg(kwargs, 'max_memory', {0: '20GiB'})


# --- S4-14: rope_scaling dual form -------------------------------------------------------------


def _fake_loader_returning(config):
    """A ModelLoader stand-in whose build hooks hand back a real HF config object."""
    loader = SimpleNamespace(
        build_config=lambda model, **kw: config,
        process_config=lambda cfg: cfg)
    return loader


def test_rope_scaling_name_maps_to_type_dict():
    """S4-14: a bare strategy name maps to {'type': name}; a dict passes through; an unknown name
    is refused with the supported set (both forms end on the config)."""
    from transformers import Qwen2Config

    from swift.dev.builders.model import _build_hf_config
    from swift.dev.config import ModelConfig

    base = Qwen2Config(max_position_embeddings=512)

    model_config = ModelConfig(model='m', model_type='qwen2', rope_scaling='linear', max_model_len=1024)
    config = _build_hf_config(model_config, model_loader=_fake_loader_returning(base))
    assert config.rope_scaling['type'] == 'linear'

    model_config = ModelConfig(model='m', model_type='qwen2',
                               rope_scaling={'type': 'yarn', 'factor': 2.0}, max_model_len=1024)
    config = _build_hf_config(model_config, model_loader=_fake_loader_returning(base))
    assert config.rope_scaling['type'] == 'yarn'
    # the origin is DERIVED from the checkpoint when the dict omits it (the budget math needs it)
    assert config.rope_scaling['original_max_position_embeddings'] == 512

    model_config = ModelConfig(model='m', model_type='qwen2', rope_scaling='bogus', max_model_len=1024)
    with pytest.raises(ValueError, match='linear, dynamic, yarn'):
        _build_hf_config(model_config, model_loader=_fake_loader_returning(base))


# --- S4-22: the TrainableModel protocol across backends ---------------------------------------


def test_transformers_backend_satisfies_the_trainable_protocol():
    """S4-22 (transformers half): every backend model is a TrainableModel with a forward; the
    unsloth / sentence-transformers halves are pinned in their own env-guarded tests."""
    from swift.dev.model import TransformersModel
    from swift.dev.model.base import TrainableModel

    assert issubclass(TransformersModel, TrainableModel)
    assert callable(TransformersModel.forward)


@pytest.mark.requires_unsloth
def test_unsloth_backend_satisfies_the_trainable_protocol():
    from swift.dev.model import UnslothModel
    from swift.dev.model.base import TrainableModel

    assert issubclass(UnslothModel, TrainableModel)


@pytest.mark.requires_st
def test_sentence_transformers_backend_satisfies_the_trainable_protocol():
    from swift.dev.model import SentenceTransformerModel
    from swift.dev.model.base import TrainableModel

    assert issubclass(SentenceTransformerModel, TrainableModel)


# --- S4-27: mixed precision derivation --------------------------------------------------------


def test_mixed_precision_follows_torch_dtype():
    """S4-27: _mixed_precision_for is the single source both backends read -- bf16 stays bf16,
    fp16 stays fp16, and an unset dtype derives the autocast default bf16."""
    from swift.dev.builders.model import _mixed_precision_for

    assert _mixed_precision_for('bfloat16') == 'bf16'
    assert _mixed_precision_for('float16') == 'fp16'
    assert _mixed_precision_for(None) == 'bf16'  # the unset default


# --- adapter init / targeting (S4-29 / S4-30 / S4-31 / S4-33) ----------------------------------


def test_all_linear_single_element_folds_to_the_bare_string():
    """S4-29: ['all-linear'] folds to 'all-linear' -- peft accepts the bare string but a 1-element
    list is a different (regex-list) meaning, so the fold is semantic, not cosmetic."""
    from swift.dev.builders.adapter import _resolve_target_modules

    assert _resolve_target_modules(__import__('swift.dev.config', fromlist=['TunerConfig']).TunerConfig(
        target_modules=['all-linear'])) == 'all-linear'


def test_init_weights_string_parsing():
    """S4-30: 'true'/'false' become peft's bool; the named strategies pass through verbatim."""
    from swift.dev.builders.adapter import _resolve_init_weights
    from swift.dev.config import TunerConfig

    assert _resolve_init_weights(TunerConfig(init_weights='true')) is True
    assert _resolve_init_weights(TunerConfig(init_weights='false')) is False
    assert _resolve_init_weights(TunerConfig(init_weights='pissa')) == 'pissa'
    assert _resolve_init_weights(TunerConfig(init_weights='olora')) == 'olora'


@pytest.mark.parametrize('name, expected', [
    ('pissa', True), ('pissa_niter_4', True), ('corda', True), ('olora', True),
    ('lora_ga', True), ('loftq', True), ('gaussian', False), (True, False), (None, False),
])
def test_is_base_modifying_init_table(name, expected):
    """S4-31: exactly the strategies that read/rewrite the base weight at injection -- they make
    the adapter fit only the RESIDUAL base it produced, so merge/save must route them apart."""
    from swift.dev.builders.adapter import is_base_modifying_init

    assert is_base_modifying_init(name) is expected, name


def test_trainable_tokens_requires_indices():
    """S4-33: trainable_tokens without indices is refused (peft would silently target nothing);
    with indices the config carries them and infers the embedding module."""
    from swift.dev.builders.adapter import _build_adapter_config
    from swift.dev.config import TunerConfig

    with pytest.raises(ValueError, match='requires trainable_token_indices'):
        _build_adapter_config(TunerConfig(tuner='trainable_tokens'))
    config = _build_adapter_config(TunerConfig(tuner='trainable_tokens', trainable_token_indices=[1, 2, 3]))
    assert config.token_indices == [1, 2, 3]
