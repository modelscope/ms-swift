# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import pytest
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from types import SimpleNamespace


@pytest.mark.parametrize('quant_method', ['gptq', 'gptq_v2'])
@pytest.mark.parametrize('format_key', ['checkpoint_format', 'format'])
@pytest.mark.parametrize('converted', [False, True])
def test_gptq_export_saves_final_quantization_config(tmp_path, monkeypatch, quant_method, format_key, converted):
    optimum_quantizer = pytest.importorskip('optimum.gptq.quantizer')
    from swift.pipelines.export import quant

    class TinyModel(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.proj = nn.Linear(4, 4)
            self.model_meta = SimpleNamespace(additional_saved_files=[])

    class Quantizer(optimum_quantizer.GPTQQuantizer):

        def __init__(self):
            # Avoid kernel initialization while exercising the real Optimum save/to_dict methods.
            self.bits = 4
            self.sym = True
            self.format = 'gptq_v2'
            self.quant_method = 'gptq'
            self.group_size = 128
            self.quant_linear = nn.Linear
            self.dataset = 'calibration-data'
            self.meta = {'quantizer': ['test']}
            self.block_name_to_quantize = 'model.layers'
            self.modules_in_block_to_quantize = [['mlp.gate_proj'], ['mlp.down_proj']]
            self.serialization_keys = [
                'bits', 'sym', format_key, 'quant_method', 'group_size', 'dataset', 'meta', 'block_name_to_quantize',
                'modules_in_block_to_quantize'
            ]

        @property
        def format(self):
            return self._format

        @format.setter
        def format(self, value):
            self._format = value

        # Support either installed Optimum API while testing both serialized field names.
        checkpoint_format = format

    monkeypatch.setattr(optimum_quantizer, 'is_gptqmodel_available', lambda: True)
    monkeypatch.setattr(
        optimum_quantizer, 'hf_convert_gptq_v2_to_v1_format', lambda model, *args: (model, converted), raising=False)
    quantizer = Quantizer()
    initial_config = quantizer.to_dict()
    initial_config.pop('dataset')
    model = TinyModel(PretrainedConfig(quantization_config=initial_config))
    save_calls = []
    original_save = model.save_pretrained

    def save_pretrained(*args, **kwargs):
        save_calls.append(args[0])
        return original_save(*args, **kwargs)

    monkeypatch.setattr(model, 'save_pretrained', save_pretrained)
    engine = quant.QuantEngine.__new__(quant.QuantEngine)
    engine.args = SimpleNamespace(
        quant_method=quant_method,
        quant_bits=4,
        output_dir=str(tmp_path),
        safe_serialization=True,
        max_shard_size='10MB',
        model_dir=str(tmp_path))
    engine.model = model
    engine.template = SimpleNamespace()
    engine.processor = None
    monkeypatch.setattr(engine, 'gptq_model_quantize', lambda **kwargs: quantizer)
    monkeypatch.setattr(quant, 'save_checkpoint', lambda *args, **kwargs: None)

    engine.quantize()

    config = json.loads((tmp_path / 'config.json').read_text())['quantization_config']
    standalone_config = json.loads((tmp_path / 'quantize_config.json').read_text())
    expected_format = 'gptq' if converted else 'gptq_v2'
    assert standalone_config[format_key] == expected_format
    assert config[format_key] == expected_format
    assert model.config.quantization_config == config
    expected_config = standalone_config.copy()
    expected_config.pop('dataset')
    assert config == expected_config
    assert quantizer.dataset == 'calibration-data'
    assert save_calls == [str(tmp_path)]
    assert (tmp_path / 'model.safetensors').is_file()
