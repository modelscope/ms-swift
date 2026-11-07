# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import pytest
import torch
from transformers import AutoConfig, LlamaConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from swift.dev.utils import HfConfigFactory


class OverriddenLlamaConfig(PretrainedConfig):
    model_type = 'llama'

    def __init__(self, hidden_size=64, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / 'config.json').write_text(
        json.dumps({'model_type': 'llama', 'hidden_size': 128, 'num_hidden_layers': 2}), encoding='utf-8')
    origin = CONFIG_MAPPING._extra_content.get('llama')
    CONFIG_MAPPING.register('llama', OverriddenLlamaConfig, exist_ok=True)
    yield tmp_path
    if origin is None:
        CONFIG_MAPPING._extra_content.pop('llama', None)
    else:
        CONFIG_MAPPING._extra_content['llama'] = origin


def _write_config(model_dir, config):
    (model_dir / 'config.json').write_text(json.dumps(config), encoding='utf-8')


def test_load_hf_config_bypasses_auto_config_override(model_dir):
    assert type(AutoConfig.from_pretrained(model_dir)) is OverriddenLlamaConfig
    config = HfConfigFactory.load_hf_config(str(model_dir))
    assert type(config) is LlamaConfig
    assert config.hidden_size == 128
    assert config.num_hidden_layers == 2


def test_load_hf_config_without_override(model_dir):
    CONFIG_MAPPING._extra_content.pop('llama', None)
    assert type(HfConfigFactory.load_hf_config(str(model_dir))) is LlamaConfig


def test_load_hf_config_falls_back_for_remote_code(model_dir):
    _write_config(model_dir, {'model_type': 'llama', 'auto_map': {'AutoConfig': 'configuration_x.X'}})
    assert HfConfigFactory.load_hf_config(str(model_dir)) is None


def test_load_hf_config_falls_back_for_unknown_model_type(model_dir):
    _write_config(model_dir, {'model_type': 'swift_unregistered_model_type'})
    assert HfConfigFactory.load_hf_config(str(model_dir)) is None


def test_load_hf_config_forwards_revision(model_dir, monkeypatch):
    revisions = {}

    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        revisions['read'] = kwargs['revision']
        return {'model_type': 'llama'}, None

    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        revisions['load'] = kwargs['revision']
        return cls()

    monkeypatch.setattr(PretrainedConfig, 'get_config_dict', classmethod(get_config_dict))
    monkeypatch.setattr(LlamaConfig, 'from_pretrained', classmethod(from_pretrained))
    config = HfConfigFactory.load_hf_config(str(model_dir), revision='test-revision')
    assert type(config) is LlamaConfig
    assert revisions == {'read': 'test-revision', 'load': 'test-revision'}


def test_generic_builder_bypasses_auto_config_override(model_dir):
    from swift.dev.builders.model import _build_hf_config
    from swift.dev.config import ModelConfig

    config = _build_hf_config(ModelConfig(model=str(model_dir)))
    assert type(config) is LlamaConfig
    assert config.hidden_size == 128


def test_model_loader_bypasses_auto_config_override(model_dir):
    from swift.dev.model.loader import LoaderModelInfo, ModelLoader

    loader = ModelLoader(LoaderModelInfo(model_dir=str(model_dir)))
    config = loader.build_config(str(model_dir), revision='test-revision')
    assert type(config) is LlamaConfig
    assert config.hidden_size == 128


def test_resolve_model_info_bypasses_auto_config_override(model_dir):
    from swift.dev.model.loader import resolve_model_info

    info = resolve_model_info(str(model_dir), model_type='llama', torch_dtype=torch.float32)
    assert type(info.config) is LlamaConfig
    assert info.config.hidden_size == 128
