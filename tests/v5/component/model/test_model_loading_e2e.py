# Copyright (c) ModelScope Contributors. All rights reserved.
"""Family loading smoke: real checkpoints, tiny and hermetic, driven end to end through the loader.

``test_model.py`` pins the *contract* (which methods exist, what the data formats are) and never loads
a model -- it says so itself: "Constructing a real twinkle Model needs a downloadable model + process
group, so those paths are covered by skip-guarded integration tests". This file is that missing loading
smoke, made hermetic so it needs neither:

- every checkpoint is BUILT here from a reduced-layer transformers config and saved to ``tmp_path``, so
  there is nothing to download and no network;
- ``TransformersModel`` constructs config/processor/model through the family ``ModelLoader``'s six hooks
  WITHOUT a process group -- ``_try_init_process_group`` is a no-op at world size 1 -- so no GPU and no
  ``torchrun``;
- the models are a few thousand parameters, so construction and even a forward pass stay in the fast lane.

Each representative family proves a different hook actually fires, by an observable effect rather than a
mock: qwen2_5 loads and runs a forward; llama's ``process_config`` forces ``pretraining_tp`` off;
gemma3_text's ``build_model`` defaults attention to eager; qwen3_moe builds its expert layers;
qwen2_5_vl's ``build_processor`` selects the multimodal processor and ``process_model`` installs the
vision keep-alive. One test instruments a loader to pin that all six hooks fire, in order.

The Megatron branch shares the exact same loader config path (``_build_hf_config`` == ``build_config`` +
``process_config``), so it is smoked at the config level; constructing a ``MegatronModel`` needs mcore +
Ray + a device mesh, which is out of scope for a loading smoke.
"""
import importlib.util
import os
import sys

import pytest
import torch

# ---- tiny checkpoint builders: one per representative family ------------------------------------
#
# Each writes a real (weights + tokenizer) checkpoint reduced to 2 layers / hidden 16 so it loads on
# CPU in well under a second, while still exercising the family's real transformers classes and hooks.


def _save_tokenizer(directory):
    """A minimal fast tokenizer so ``build_processor`` has something real to load."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    tokenizer = Tokenizer(WordLevel({f'w{i}': i for i in range(64)}, unk_token='w0'))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token='w0', pad_token='w0', bos_token_id=1, eos_token_id=2
    ).save_pretrained(directory)


def _build_qwen2_5(directory):
    import transformers
    config = transformers.Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=True)
    transformers.Qwen2ForCausalLM(config).save_pretrained(directory)
    _save_tokenizer(directory)
    return str(directory)


def _build_llama(directory):
    import transformers
    # pretraining_tp=2 is the tensor-parallel training artifact llama's process_config exists to undo.
    config = transformers.LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=64, pretraining_tp=2, tie_word_embeddings=True)
    transformers.LlamaForCausalLM(config).save_pretrained(directory)
    _save_tokenizer(directory)
    return str(directory)


def _build_gemma3_text(directory):
    import transformers
    config = transformers.Gemma3TextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=64, head_dim=8, tie_word_embeddings=True)
    transformers.Gemma3ForCausalLM(config).save_pretrained(directory)
    _save_tokenizer(directory)
    return str(directory)


def _build_qwen3_moe(directory):
    import transformers
    config = transformers.Qwen3MoeConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, moe_intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64, num_experts=4,
        num_experts_per_tok=2, head_dim=8, tie_word_embeddings=True, bos_token_id=1, eos_token_id=2)
    transformers.Qwen3MoeForCausalLM(config).save_pretrained(directory)
    _save_tokenizer(directory)
    return str(directory)


def _build_qwen2_5_vl(directory):
    import transformers
    text = dict(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64, head_dim=8, tie_word_embeddings=True,
        bos_token_id=1, eos_token_id=2)
    vision = dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=4, out_hidden_size=16,
                  patch_size=2, temporal_patch_size=2, spatial_merge_size=2, in_channels=3)
    config = transformers.Qwen2_5_VLConfig(text_config=text, vision_config=vision)
    transformers.Qwen2_5_VLForConditionalGeneration(config).save_pretrained(directory)
    _save_tokenizer(directory)
    # A multimodal checkpoint ships a (pre)processor config, which is what makes build_processor take
    # the AutoProcessor branch; the VL processor needs an image and a video processor alongside the
    # tokenizer.
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(str(directory), 'tokenizer.json'), unk_token='w0', pad_token='w0')
    video_cls = getattr(transformers, 'Qwen2_5_VLVideoProcessor', None) or transformers.Qwen2VLVideoProcessor
    transformers.Qwen2_5_VLProcessor(
        image_processor=transformers.Qwen2VLImageProcessor(), tokenizer=tokenizer,
        video_processor=video_cls()).save_pretrained(directory)
    return str(directory)


def _load(model_type, directory, **config_kwargs):
    """Resolve the family loader exactly as dev does, then construct through it. Returns (loader, model)."""
    from swift.dev.builder.model import _resolve_model_loader
    from swift.dev.config import ModelConfig
    from swift.dev.model import TransformersModel
    loader = _resolve_model_loader(ModelConfig(model=directory, model_type=model_type, **config_kwargs))
    model = TransformersModel(model_id=directory, model_loader=loader, mixed_precision='no')
    return loader, model


# ---- loader resolution: no construction, no download --------------------------------------------


def test_resolve_model_loader_honours_explicit_type_and_infers_from_id():
    """``model_type`` is authoritative; without it the family is inferred from the checkpoint name."""
    from swift.dev.builder.model import _resolve_model_loader
    from swift.dev.config import ModelConfig
    from swift.dev.model.loader import MODEL_MAPPING

    # An explicit model_type resolves even for a path that does not exist -- nothing is read off disk.
    loader = _resolve_model_loader(ModelConfig(model='/nonexistent/dir', model_type='qwen2_5'))
    assert type(loader) is MODEL_MAPPING['qwen2_5']
    # With no explicit type, a registered checkpoint id matches by basename.
    loader = _resolve_model_loader(ModelConfig(model='Qwen/Qwen2.5-7B-Instruct'))
    assert type(loader) is MODEL_MAPPING['qwen2_5']
    # An id no family claims resolves to None, so the caller falls back to the generic AutoModel path.
    assert _resolve_model_loader(ModelConfig(model='some-org/some-unknown-model')) is None


# ---- the six hooks, as a contract and by observable effect ---------------------------------------


def test_all_six_loader_hooks_fire_in_order(tmp_path):
    """``TransformersModel`` drives the loader through exactly its six construction hooks, in order."""
    from swift.dev.builder.model import _resolve_model_loader
    from swift.dev.config import ModelConfig
    from swift.dev.model import TransformersModel

    directory = _build_qwen2_5(tmp_path)
    base = _resolve_model_loader(ModelConfig(model=directory, model_type='qwen2_5'))
    calls = []

    class Recording(type(base)):
        """Same family loader, but each hook records that it ran before delegating."""

        def build_config(self, *args, **kwargs):
            calls.append('build_config')
            return super().build_config(*args, **kwargs)

        def process_config(self, *args, **kwargs):
            calls.append('process_config')
            return super().process_config(*args, **kwargs)

        def build_processor(self, *args, **kwargs):
            calls.append('build_processor')
            return super().build_processor(*args, **kwargs)

        def process_tokenizer(self, *args, **kwargs):
            calls.append('process_tokenizer')
            return super().process_tokenizer(*args, **kwargs)

        def build_model(self, *args, **kwargs):
            calls.append('build_model')
            return super().build_model(*args, **kwargs)

        def process_model(self, *args, **kwargs):
            calls.append('process_model')
            return super().process_model(*args, **kwargs)

    TransformersModel(model_id=directory, model_loader=Recording(base.model_info), mixed_precision='no')
    assert calls == [
        'build_config', 'process_config', 'build_processor', 'process_tokenizer', 'build_model',
        'process_model'
    ]


def test_qwen2_5_family_loads_and_runs_a_forward(tmp_path):
    """The plain-LLM happy path: resolve -> load -> a real forward with finite logits."""
    from swift.dev.model.loader import MODEL_MAPPING

    directory = _build_qwen2_5(tmp_path)
    loader, model = _load('qwen2_5', directory)
    assert type(loader) is MODEL_MAPPING['qwen2_5']
    assert model.model.__class__.__name__ == 'Qwen2ForCausalLM'
    assert model._default_tokenizer is not None
    with torch.no_grad():
        outputs = model.model(input_ids=torch.tensor([[1, 2, 3, 4]]))
    assert outputs.logits.shape == (1, 4, 64)
    assert torch.isfinite(outputs.logits).all()


def test_llama_process_config_forces_pretraining_tp_off(tmp_path):
    """``process_config`` undoes the tensor-parallel artifact the checkpoint ships with."""
    from transformers import LlamaConfig

    directory = _build_llama(tmp_path)
    assert LlamaConfig.from_pretrained(directory).pretraining_tp == 2  # present on disk
    _, model = _load('llama', directory)
    assert model.model.config.pretraining_tp == 1  # forced off by the loader hook


def test_gemma3_text_build_model_defaults_to_eager_attention(tmp_path):
    """Gemma-3 trains best with eager attention; ``build_model`` sets it unless the user picked one."""
    directory = _build_gemma3_text(tmp_path)
    _, model = _load('gemma3_text', directory)
    assert model.model.__class__.__name__ == 'Gemma3ForCausalLM'
    assert model.model.config._attn_implementation == 'eager'


def test_qwen3_moe_builds_expert_layers(tmp_path):
    """The MoE family loads its sparse expert blocks -- one per decoder layer."""
    directory = _build_qwen3_moe(tmp_path)
    loader, model = _load('qwen3_moe', directory)
    assert getattr(type(loader), 'is_moe', False) is True
    assert model.model.__class__.__name__ == 'Qwen3MoeForCausalLM'
    assert model.model.config.num_experts == 4
    expert_blocks = [name for name, _ in model.model.named_modules() if name.endswith('experts')]
    assert len(expert_blocks) == model.model.config.num_hidden_layers


def test_qwen2_5_vl_loads_multimodal_processor_and_installs_keep_alive(tmp_path):
    """The multimodal family: AutoProcessor branch for the processor, vision keep-alive on the model."""
    pytest.importorskip('qwen_vl_utils')  # build_processor/setup_environment hard-require it

    directory = _build_qwen2_5_vl(tmp_path)
    loader, model = _load('qwen2_5_vl', directory)
    assert type(loader).is_multimodal is True
    assert model.model.__class__.__name__ == 'Qwen2_5_VLForConditionalGeneration'
    # A (pre)processor config on disk makes build_processor pick AutoProcessor -> the VL processor.
    assert model._default_tokenizer.__class__.__name__ == 'Qwen2_5_VLProcessor'
    # The ZeRO-3 vision keep-alive now lives on the data side: the family loader builds a
    # MultimodalGraphAdapter (InputProcessor consumes it at collate time), driven by
    # model_arch.vision_tower / aligner.
    adapter = loader.build_multimodal_graph_adapter(model._default_tokenizer, model.model.config)
    assert adapter is not None
    assert loader.model_arch.vision_tower == ['model.visual']
    assert loader.model_arch.aligner == ['model.visual.merger']


# ---- Megatron: the same loader, exercised on its config path only --------------------------------


def test_megatron_config_path_runs_loader_build_and_process_config(tmp_path):
    """``_build_megatron_model`` builds its HF config through the very same loader hooks.

    Constructing a MegatronModel needs mcore + Ray + a device mesh, so the smoke stops where the
    loader is actually used on that branch: ``_build_hf_config`` == ``build_config`` + ``process_config``
    (+ the rope/max-length overrides the caller threads in).
    """
    from swift.dev.builder.model import _build_hf_config, _resolve_model_loader
    from swift.dev.config import ModelConfig
    from swift.dev.utils import HfConfigFactory

    directory = _build_llama(tmp_path)
    model_config = ModelConfig(model=directory, model_type='llama', max_model_len=4096, rope_scaling='linear')
    loader = _resolve_model_loader(model_config)
    config = _build_hf_config(model_config, loader)

    assert config.__class__.__name__ == 'LlamaConfig'
    assert config.pretraining_tp == 1  # process_config ran on the megatron path too
    assert HfConfigFactory.get_max_model_len(config) == 4096
    rope_scaling = HfConfigFactory.get_config_attr(config, 'rope_scaling')
    assert rope_scaling['type'] == 'linear'
    assert rope_scaling['factor'] == 64.0  # ceil(4096 / 64 declared max positions)


# ---- declared class candidates and the ``prepare`` seam (20-D-1 / B-3) --------------------------
# PLAN (case | input | expected | failure it kills):
#  P1 ordered candidates | (missing module, in-tree V3 config) | V3 config class | a tuple declaration
#     aborting on the first failed import (the old try/except-at-import it replaces).
#  P2 nothing resolves | (missing, missing) | ImportError naming BOTH specs and ``Owner.attr`` | a
#     silent Auto* fallback hiding a class the environment lacks.
#  P3 deepseek_v32 | build_config on a real dir | first candidate's class + trust_remote_code=True |
#     regression of the removed override (class attr must drive the kwarg).
#  P4 prepare | tiny qwen2_5, dtype+attn given | hooks in order, config carries dtype/attn, no
#     trust_remote_code kwarg when None | sampler-worker path skipping dtype/attn; engine's False
#     forced on a remote-code family.
#  P5 ASR without qwen_asr | native transformers, build_model/build_model_from_config | ImportError
#     from resolve_model_cls BEFORE any from_pretrained | the G47-d hole: native class loading the
#     thinker-layout checkpoint with every weight unused.
#  P6 ASR source order | fake ``qwen_asr`` package on sys.path, native registered | package imported
#     before the config is built | the old CONFIG_MAPPING_NAMES probe never importing it.
#  P7 training builder | load_model_processor with rope_scaling | rope + dtype on the prepared config |
#     ``_apply_rope_scaling`` dropped from the prepare branch.


def test_ordered_class_candidates_resolve_first_available_and_name_every_refusal():
    from swift.dev.model.loader.base import ModelLoader, _import_candidates

    cls = _import_candidates(
        ('no_such_pkg:X', 'transformers.models.deepseek_v3:DeepseekV3Config'), ModelLoader, 'config_cls')
    assert cls.__name__ == 'DeepseekV3Config'
    with pytest.raises(ImportError) as excinfo:
        _import_candidates(('no_such_pkg:X', 'transformers:NoSuchClass'), ModelLoader, 'config_cls')
    message = str(excinfo.value)
    assert 'ModelLoader.config_cls' in message
    assert 'no_such_pkg:X' in message and 'transformers:NoSuchClass' in message


def test_deepseek_v32_declares_its_config_fallback(tmp_path, monkeypatch):
    from transformers.models.deepseek_v32 import DeepseekV32Config

    from swift.dev.model.loader import MODEL_MAPPING

    directory = _build_llama(tmp_path)
    seen = {}
    origin = DeepseekV32Config.from_pretrained

    def spy(cls, *args, **kwargs):
        seen.update(kwargs)
        return origin(*args, **kwargs)

    monkeypatch.setattr(DeepseekV32Config, 'from_pretrained', classmethod(spy))
    loader = MODEL_MAPPING['deepseek_v32'].from_model_dir(directory)
    config = loader.build_config(directory)
    assert type(config) is DeepseekV32Config  # first candidate wins where it exists
    assert seen['trust_remote_code'] is True  # the family declaration, not an override, supplies it


def test_prepare_runs_the_config_then_processor_hooks_with_dtype_and_attn(tmp_path):
    from swift.dev.model.loader import MODEL_MAPPING

    directory = _build_qwen2_5(tmp_path)
    base = MODEL_MAPPING['qwen2_5']
    calls, config_kwargs = [], {}

    class Recording(base):

        def build_config(self, *args, **kwargs):
            calls.append('build_config')
            config_kwargs.update(kwargs)
            return super().build_config(*args, **kwargs)

        def process_config(self, *args, **kwargs):
            calls.append('process_config')
            return super().process_config(*args, **kwargs)

        def build_processor(self, *args, **kwargs):
            calls.append('build_processor')
            return super().build_processor(*args, **kwargs)

        def process_tokenizer(self, *args, **kwargs):
            calls.append('process_tokenizer')
            return super().process_tokenizer(*args, **kwargs)

    config, processor = Recording.from_model_dir(directory).prepare(
        directory, torch_dtype=torch.bfloat16, attn_implementation='flash_attn')
    assert calls == ['build_config', 'process_config', 'build_processor', 'process_tokenizer']
    assert 'trust_remote_code' not in config_kwargs  # None = let the family declaration decide
    assert config.torch_dtype == torch.bfloat16
    assert config._attn_implementation == 'flash_attention_2'  # dev's alias resolved for transformers
    assert 'past_key_values' in config.keys_to_ignore_at_inference
    from transformers import PreTrainedTokenizerBase

    assert isinstance(processor, PreTrainedTokenizerBase)  # the family's own tokenizer class

    config_kwargs.clear()
    Recording.from_model_dir(directory).prepare(directory, trust_remote_code=True)
    assert config_kwargs['trust_remote_code'] is True


_ASR_DIR = '/mnt/workspace/.cache/modelscope/hub/models/Qwen--Qwen3-ASR-1.7B/snapshots/master'


def _asr_loader_directory(tmp_path):
    """The cached real checkpoint when present, else a config-only stand-in with the same model_type."""
    if os.path.isdir(_ASR_DIR):
        return _ASR_DIR
    import json

    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_asr'}))
    return str(tmp_path)


def test_asr_refuses_hf_loading_before_any_weight_read_without_its_package(tmp_path, monkeypatch):
    """Main env: transformers has a native ``qwen3_asr`` registration, ``qwen_asr`` is not installed."""
    pytest.importorskip('transformers.models.qwen3_asr')  # the native fallback under test
    assert importlib.util.find_spec('qwen_asr') is None

    from swift.dev.model.loader import MODEL_MAPPING

    directory = _asr_loader_directory(tmp_path)
    loader = MODEL_MAPPING['qwen3_asr'].from_model_dir(directory)
    config = loader.build_config(directory)  # sampler setup keeps working off the native registration
    assert type(config).__module__.startswith('transformers.')

    def no_weight_read(*args, **kwargs):
        raise AssertionError('from_pretrained must not be reached: the native class cannot load this layout')

    monkeypatch.setattr('transformers.PreTrainedModel.from_pretrained', no_weight_read)
    for build in (lambda: loader.build_model(directory, config, None),
                  lambda: loader.build_model_from_config(config, None)):
        with pytest.raises(ImportError, match=r'Qwen3ASRLoader\.model_cls.*qwen_asr'):
            build()


def test_asr_build_config_tries_the_package_before_the_native_registration(tmp_path, monkeypatch):
    """A both-present environment imports ``qwen_asr`` first, so config and model_cls share one source."""
    pytest.importorskip('transformers.models.qwen3_asr')
    package = tmp_path / 'qwen_asr'
    package.mkdir()
    marker = tmp_path / 'imported'
    (package / '__init__.py').write_text(f'import pathlib; pathlib.Path({str(marker)!r}).touch()\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, 'qwen_asr', raising=False)

    from swift.dev.model.loader import MODEL_MAPPING

    directory = _asr_loader_directory(tmp_path / 'ckpt')
    loader = MODEL_MAPPING['qwen3_asr'].from_model_dir(directory)
    config = loader.build_config(directory)
    monkeypatch.delitem(sys.modules, 'qwen_asr', raising=False)  # the stand-in must not leak to other tests
    assert marker.exists(), 'the package must be imported before the config is resolved'
    assert type(config).__module__.startswith('transformers.')  # a package that registers nothing:
    # the native class is what transformers resolves -- the point is the ORDER, not the winner.


def test_builder_applies_rope_and_dtype_on_the_prepared_loader_config(tmp_path):
    from swift.dev.builder.model import load_model_processor
    from swift.dev.config import ModelConfig
    from swift.dev.utils import HfConfigFactory

    directory = _build_qwen2_5(tmp_path)
    model_config = ModelConfig(
        model=directory, model_type='qwen2_5', torch_dtype='bfloat16', max_model_len=128, rope_scaling='linear')
    model, processor = load_model_processor(model_config, load_model=False)
    assert model is None
    config = processor.model_info.config
    assert config.torch_dtype == torch.bfloat16
    assert HfConfigFactory.get_config_attr(config, 'rope_scaling')['type'] == 'linear'
    assert HfConfigFactory.get_config_attr(config, 'rope_scaling')['factor'] == 2.0  # ceil(128 / 64)
