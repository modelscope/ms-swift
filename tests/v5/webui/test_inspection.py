# Copyright (c) ModelScope Contributors. All rights reserved.
import io

import json
import pytest

from swift.dev.config import ModelConfig, TrainConfig, TunerConfig
from swift.dev.webui import inspection
from swift.dev.webui.component import ComponentError
from swift.dev.webui.inspection import inspect_model_node


def test_model_node_builds_typed_configs(monkeypatch):
    captured = {}

    def inspect(model_config, **kwargs):
        captured['model'] = model_config
        captured.update(kwargs)
        return {'detail': kwargs['detail']}

    monkeypatch.setattr('swift.dev.webui.model_inspection.inspect_model', inspect)
    result = inspect_model_node('model-node', {
        'model': '/tiny/model',
        'model_config': {'model_type': 'qwen2_5'},
        'train_type': 'lora',
        'tuner_config': {'target_modules': ['q_proj'], 'freeze_vit': False},
        'train_config': {'learning_rate': 1e-4},
    }, detail='meta')
    assert result['detail'] == 'meta'
    assert isinstance(captured['model'], ModelConfig)
    assert isinstance(captured['tuner_config'], TunerConfig)
    assert isinstance(captured['train_config'], TrainConfig)
    assert captured['model'].model_type == 'qwen2_5'
    assert captured['tuner_config'].target_modules == ['q_proj']
    assert not captured['tuner_config'].freeze_vit
    assert captured['training']


def test_full_and_frozen_model_node(monkeypatch):
    captured = {}

    def inspect(model_config, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr('swift.dev.webui.model_inspection.inspect_model', inspect)
    inspect_model_node('model-node', {'model': '/tiny/model', 'train_type': 'full', 'freeze': True})
    assert captured['tuner_config'].tuner == 'full'
    assert captured['frozen']


def test_full_train_type_wins_over_stale_lora_fields(monkeypatch):
    captured = {}

    def inspect(model_config, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr('swift.dev.webui.model_inspection.inspect_model', inspect)
    inspect_model_node('model-node', {
        'model': '/tiny/model',
        'train_type': 'full',
        'tuner_config': {'lora_rank': 8},
    })
    assert captured['tuner_config'].tuner == 'full'


def test_model_descriptor_exposes_inspection_factory():
    from swift.dev.webui.export import validate_factories
    from swift.dev.webui.registry import REGISTRY
    assert REGISTRY['model'].client_factory == 'swift.dev.webui.inspection:inspect_model_node'
    validate_factories()


def test_model_node_rejects_unknown_config_fields():
    with pytest.raises(ComponentError, match='not_a_real_field'):
        inspect_model_node('model-node', {
            'model': '/tiny/model',
            'model_config': {'not_a_real_field': True},
        })


def test_inspection_jsonl_protocol(monkeypatch, capsys):
    monkeypatch.setattr(inspection, 'inspect_model_node', lambda *args, **kwargs: {'detail': kwargs['detail']})
    monkeypatch.setattr(inspection.sys, 'stdin', io.StringIO(
        json.dumps({'request_id': 'req-1', 'node_id': 'model', 'params': {}, 'detail': 'config'}) + '\n'))
    assert inspection.main(warmup=False) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith('[model-inspection] ')
    reply = json.loads(line.removeprefix('[model-inspection] '))
    assert reply == {'request_id': 'req-1', 'ok': True, 'result': {'detail': 'config'}}
