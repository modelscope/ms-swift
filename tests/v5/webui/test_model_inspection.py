# Copyright (c) ModelScope Contributors. All rights reserved.
import dataclasses
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from swift.dev.config import ModelConfig, TrainConfig, TunerConfig
from swift.dev.webui.model_inspection import (
    StructureNode,
    _apply_config_training,
    _clear_cache,
    _meta_structure,
    _part_for_path,
    inspect_model,
)
from swift.dev.model.loader import ModelArch


def _save_tokenizer(directory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    tokenizer = Tokenizer(WordLevel({f'w{i}': i for i in range(64)}, unk_token='w0'))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token='w0', pad_token='w0', bos_token_id=1, eos_token_id=2
    ).save_pretrained(directory)


def _build_text(directory, layers=2):
    import transformers
    config = transformers.Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=True)
    transformers.Qwen2ForCausalLM(config).save_pretrained(directory)
    _save_tokenizer(directory)
    return str(directory)


def _build_vl(directory):
    import transformers
    text = {
        'vocab_size': 64, 'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 2,
        'num_attention_heads': 4, 'num_key_value_heads': 2, 'max_position_embeddings': 64,
        'head_dim': 8, 'tie_word_embeddings': True, 'bos_token_id': 1, 'eos_token_id': 2,
    }
    vision = {
        'depth': 2, 'hidden_size': 16, 'intermediate_size': 32, 'num_heads': 4, 'out_hidden_size': 16,
        'patch_size': 2, 'temporal_patch_size': 2, 'spatial_merge_size': 2, 'in_channels': 3,
    }
    config = transformers.Qwen2_5_VLConfig(text_config=text, vision_config=vision)
    transformers.Qwen2_5_VLForConditionalGeneration(config).save_pretrained(directory)
    _save_tokenizer(directory)
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(str(directory), 'tokenizer.json'), unk_token='w0', pad_token='w0')
    video_cls = getattr(transformers, 'Qwen2_5_VLVideoProcessor', None) or transformers.Qwen2VLVideoProcessor
    transformers.Qwen2_5_VLProcessor(
        image_processor=transformers.Qwen2VLImageProcessor(), tokenizer=tokenizer,
        video_processor=video_cls()).save_pretrained(directory)
    return str(directory)


def _node(result, node_id):
    return next(node for node in result['nodes'] if node['id'] == node_id)


def _meta(path, model_type='qwen2_5', **kwargs):
    return _meta_structure(ModelConfig(model=path, model_type=model_type), **kwargs)


def _trainable_paths(result):
    """Trainable parameter paths, with folded repeated groups expanded back to every member."""
    paths = set()
    for node in result['nodes']:
        training = node['training']
        members = node['module_paths'] or [node['id']]
        for parameter in training['parameters']:
            if parameter['status'] not in ('adapter_trainable', 'trainable_copy', 'full_trainable', 'trainable_base'):
                continue
            for member in members:
                paths.add(parameter['name'] if member == 'root' else f"{member}.{parameter['name']}")
    return paths


@pytest.fixture(autouse=True)
def _clean_cache():
    _clear_cache()
    yield
    _clear_cache()


def test_config_uses_checkpoint_dimensions_without_meta(tmp_path, monkeypatch):
    path = _build_text(tmp_path, layers=3)
    monkeypatch.setattr('swift.dev.webui.model_inspection._run_meta_worker',
                        lambda *args, **kwargs: pytest.fail('config mode started the meta worker'))
    result = inspect_model(ModelConfig(model=path, model_type='qwen2_5'))
    assert result['detail'] == 'config'
    assert result['summary']['total_params'] > 0
    assert result['summary']['total_params_source'] == 'safetensors'
    assert _node(result, 'llm.decoder')['repeat'] == 3
    assert 'llm.decoder' in _node(result, 'llm')['children']
    assert all(node['source'] == 'config' for node in result['nodes'])


def test_config_mode_does_not_import_training_stack(tmp_path):
    path = _build_text(tmp_path)
    code = """
import sys
from swift.dev.config import ModelConfig
from swift.dev.webui.model_inspection import inspect_model
result = inspect_model(ModelConfig(model=sys.argv[1], model_type='qwen2_5'))
assert result['detail'] == 'config'
assert 'swift.dev.model.loader' not in sys.modules
assert 'transformers' not in sys.modules
assert 'torch' not in sys.modules
"""
    subprocess.run(
        [sys.executable, '-c', code, path], check=True, env=os.environ.copy(), capture_output=True, text=True)


def test_megatron_static_view_carries_parallel_and_lora_rules(tmp_path):
    path = _build_text(tmp_path)
    result = inspect_model(
        ModelConfig(model=path, model_type='qwen2_5'),
        training=True,
        backend='megatron',
        parallel={'tp': 2, 'pp': 2, 'sp': True},
        tuner_config=TunerConfig(target_modules=['all-linear']))
    assert result['model']['backend'] == 'megatron'
    assert result['model']['parallel'] == {'tp': 2, 'pp': 2, 'sp': True}
    assert result['warnings'] == []


def test_megatron_all_linear_leaves_router_frozen():
    node = StructureNode('llm.decoder.router', 'llm.decoder', 'Router', '', 'router', part='llm')
    tuner = SimpleNamespace(
        tuner='lora',
        target_regex=None,
        target_modules=['all-linear'],
        freeze_llm=False,
        freeze_vit=True,
        freeze_aligner=True)
    _apply_config_training([node], tuner, None, False, backend='megatron')
    assert node.training['status'] == 'frozen_base'


def test_megatron_meta_request_keeps_static_view(tmp_path, monkeypatch):
    path = _build_text(tmp_path)
    monkeypatch.setattr('swift.dev.webui.model_inspection._run_meta_worker',
                        lambda *args, **kwargs: pytest.fail('Megatron meta started the HF worker'))
    result = inspect_model(ModelConfig(model=path, model_type='qwen2_5'), detail='meta', backend='megatron')
    assert result['detail'] == 'config'
    assert result['requested_detail'] == 'meta'
    assert result['warnings'][-1]['code'] == 'megatron_meta_unsupported'


def test_resolve_reuses_local_hub_cache(tmp_path, monkeypatch):
    import transformers
    snapshot = tmp_path / 'models' / 'Org--Tiny' / 'snapshots' / 'master'
    snapshot.mkdir(parents=True)
    transformers.Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2).save_pretrained(snapshot)
    monkeypatch.setenv('MODELSCOPE_CACHE', str(tmp_path))
    result = inspect_model(ModelConfig(model='Org/Tiny', model_type='qwen2_5'))
    assert result['model']['model_dir'] == str(snapshot)


def test_config_estimates_params_when_weights_are_absent(tmp_path):
    import transformers
    config = transformers.Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2)
    config.save_pretrained(tmp_path)
    _save_tokenizer(tmp_path)
    result = inspect_model(ModelConfig(model=str(tmp_path), model_type='qwen2_5'))
    assert result['summary']['total_params'] == 64 * 16 + 3 * (12 * 16**2 + 3 * 16 * 32)
    assert result['summary']['total_params_source'] == 'config-estimate'


def test_config_multimodal_parts_and_estimated_training(tmp_path):
    path = _build_vl(tmp_path)
    result = inspect_model(
        ModelConfig(model=path, model_type='qwen2_5_vl'), training=True, tuner_config=TunerConfig())
    assert _node(result, 'llm.decoder')['repeat'] == 2
    assert _node(result, 'vit')['training']['status'] == 'frozen_base'
    assert _node(result, 'aligner')['training']['status'] == 'frozen_base'
    assert _node(result, 'aligner')['module_path'] == 'model.visual.merger'
    assert _node(result, 'llm')['training']['status'] == 'adapter_estimated'
    assert not result['summary']['training']['verified']


def test_public_meta_structure_and_layer_folding(tmp_path):
    path = _build_text(tmp_path)
    config = ModelConfig(model=path, model_type='qwen2_5')
    collapsed = inspect_model(config, detail='meta', timeout_s=120)
    assert collapsed['detail'] == 'meta', collapsed['warnings']
    group = _node(collapsed, 'model.layers.0')
    assert group['kind'] == 'repeated'
    assert group['repeat'] == 2
    assert group['module_paths'] == ['model.layers.0', 'model.layers.1']
    assert _node(collapsed, 'model.layers.0.self_attn.q_proj')['class_name'] == 'Linear'


def test_meta_timeout_falls_back_to_config(tmp_path, monkeypatch):
    path = _build_text(tmp_path)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('inspector', 1)

    monkeypatch.setattr('swift.dev.webui.model_inspection._run_meta_worker', timeout)
    result = inspect_model(ModelConfig(model=path, model_type='qwen2_5'), detail='meta')
    assert result['detail'] == 'config'
    assert result['requested_detail'] == 'meta'
    assert result['warnings'][0]['code'] == 'meta_timeout'


def test_meta_cache_and_training_config_invalidation(tmp_path, monkeypatch):
    path = _build_text(tmp_path)
    calls = []

    def worker(model_config, tuner_config, train_config, **kwargs):
        calls.append(tuner_config.target_modules)
        return {
            'detail': 'meta', 'requested_detail': 'meta', 'nodes': [], 'warnings': [],
            'config_fingerprint': 'test', 'summary': {},
        }

    monkeypatch.setattr('swift.dev.webui.model_inspection._run_meta_worker', worker)
    config = ModelConfig(model=path, model_type='qwen2_5')
    tuner = TunerConfig(target_modules=['q_proj'])
    inspect_model(config, detail='meta', training=True, tuner_config=tuner)
    inspect_model(config, detail='meta', training=True, tuner_config=tuner)
    assert len(calls) == 1
    inspect_model(
        config, detail='meta', training=True,
        tuner_config=TunerConfig(target_modules=['v_proj']))
    assert len(calls) == 2


def test_explicit_lora_matches_peft_trainable_paths(tmp_path):
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    from swift.dev.builders.adapter import _build_adapter_config

    path = _build_text(tmp_path)
    tuner = TunerConfig(target_modules=['q_proj', 'v_proj'])
    # Unfold the layer group: this test audits per-path trainable status against PEFT.
    result = _meta(
        path, expand='model.layers', training=True, tuner_config=tuner, train_config=None, frozen=False)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path))
        model = get_peft_model(model, _build_adapter_config(tuner))
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert _trainable_paths(result) == expected
    assert result['summary']['training']['trainable_params'] == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    q_proj = _node(result, 'base_model.model.model.layers.0.self_attn.q_proj.base_layer')
    assert q_proj['training']['status'] == 'frozen_base'
    assert result['summary']['training']['adapter_params'] > 0


def test_multimodal_all_linear_respects_freeze_parts(tmp_path):
    path = _build_vl(tmp_path)
    cases = [
        ({}, {'llm'}),
        ({'freeze_vit': False}, {'llm', 'vit'}),
        ({'freeze_aligner': False}, {'llm', 'aligner'}),
    ]
    for overrides, expected_parts in cases:
        result = _meta(
            path, model_type='qwen2_5_vl', training=True,
            tuner_config=TunerConfig(**overrides), train_config=None, frozen=False)
        adapter_parts = {
            node['part'] for node in result['nodes']
            if node['kind'] == 'adapter' and node['training']['trainable_params']
        }
        assert adapter_parts == expected_parts
        assert result['summary']['training']['verified']


def test_modules_to_save_and_frozen_model_are_verified(tmp_path):
    path = _build_text(tmp_path)
    result = _meta(
        path, training=True,
        tuner_config=TunerConfig(target_modules=['q_proj'], modules_to_save=['all-norm']),
        train_config=None, frozen=False)
    assert result['summary']['training']['modules_to_save_params'] > 0
    assert any(parameter['status'] == 'trainable_copy'
               for node in result['nodes'] for parameter in node['training']['parameters'])
    frozen = _meta(path, training=True, tuner_config=TunerConfig(), train_config=None, frozen=True)
    assert frozen['summary']['training']['trainable_params'] == 0
    assert frozen['summary']['training']['status'] == 'frozen_base'
    assert any(node['kind'] == 'adapter' for node in frozen['nodes'])


def test_full_freeze_rules_and_part_priority(tmp_path):
    path = _build_text(tmp_path)
    result = _meta(
        path, training=True, tuner_config=None,
        train_config=TrainConfig(freeze_parameters_ratio=1.0), frozen=False)
    assert result['summary']['training']['trainable_params'] == 0
    arch = ModelArch(language_model='model.language_model', vision_tower='model.visual',
                     aligner='model.visual.merger')
    assert _part_for_path('model.visual.merger.proj', arch) == 'aligner'
    assert _part_for_path('model.visual.blocks.0', arch) == 'vit'


def test_eighty_layers_are_one_folded_group():
    from swift.dev.webui.model_inspection import StructureNode, _collapse_repeated
    nodes = {
        'root': StructureNode('root', None, 'root', '', 'model'),
        'layers': StructureNode('layers', 'root', 'layers', 'layers', 'module'),
    }
    nodes['root'].children = ['layers']
    for index in range(80):
        node_id = f'layers.{index}'
        nodes[node_id] = StructureNode(node_id, 'layers', str(index), node_id, 'module', class_name='DecoderLayer')
        nodes['layers'].children.append(node_id)
    assert _collapse_repeated(nodes) is None
    assert len(nodes) == 3
    assert nodes['layers.0'].repeat == 80


def test_trainable_tokens_marks_only_the_embedding(tmp_path):
    path = _build_text(tmp_path)
    result = inspect_model(
        ModelConfig(model=path, model_type='qwen2_5'),
        training=True,
        tuner_config=TunerConfig(tuner='trainable_tokens', trainable_token_indices=[1, 2]))
    assert _node(result, 'llm.embedding')['training']['status'] == 'trainable_base'
    assert _node(result, 'llm.decoder')['training']['status'] == 'frozen_base'
    assert result['summary']['training']['status'] == 'mixed'


def test_config_arch_matches_loader_arch():
    """The dependency-free table must mirror ``loader/*.py`` for the families it covers."""
    from swift.dev.webui.model_inspection import _config_arch
    from swift.dev.model.loader import MODEL_MAPPING

    covered = 0
    for model_type, loader_cls in MODEL_MAPPING.items():
        arch = dataclasses.asdict(loader_cls(None).model_arch)
        if not any(arch.get(part) for part in ('language_model', 'vision_tower', 'aligner', 'generator')):
            continue  # text-only family: the table is only consulted for multimodal configs
        table = _config_arch(model_type, [])
        if not table:
            continue
        covered += 1
        for part in ('language_model', 'vision_tower', 'aligner', 'generator'):
            assert list(table.get(part) or []) == list(arch.get(part) or []), (model_type, part)
    assert covered >= 10
