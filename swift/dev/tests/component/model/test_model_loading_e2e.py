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
import os

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
    from swift.dev.builders.model import _resolve_model_loader
    from swift.dev.config import ModelConfig
    from swift.dev.model import TransformersModel
    loader = _resolve_model_loader(ModelConfig(model=directory, model_type=model_type, **config_kwargs))
    model = TransformersModel(model_id=directory, model_loader=loader, mixed_precision='no')
    return loader, model


# ---- loader resolution: no construction, no download --------------------------------------------


def test_resolve_model_loader_honours_explicit_type_and_infers_from_id():
    """``model_type`` is authoritative; without it the family is inferred from the checkpoint name."""
    from swift.dev.builders.model import _resolve_model_loader
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
    from swift.dev.builders.model import _resolve_model_loader
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
    # process_model installed the ZeRO-3 vision keep-alive, driven by model_arch.aligner.
    assert getattr(model.model, '_vision_keep_alive', None) is not None
    assert loader.model_arch.vision_tower == ['model.visual']
    assert loader.model_arch.aligner == ['model.visual.merger']


# ---- Megatron: the same loader, exercised on its config path only --------------------------------


def test_megatron_config_path_runs_loader_build_and_process_config(tmp_path):
    """``_build_megatron_model`` builds its HF config through the very same loader hooks.

    Constructing a MegatronModel needs mcore + Ray + a device mesh, so the smoke stops where the
    loader is actually used on that branch: ``_build_hf_config`` == ``build_config`` + ``process_config``
    (+ the rope/max-length overrides the caller threads in).
    """
    from swift.dev.builders.model import _build_hf_config, _resolve_model_loader
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
