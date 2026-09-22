from types import SimpleNamespace

import pytest

from swift.dev.builders.model import _apply_unsloth_kwargs
from swift.dev.builders.quantization import build_load_quantization_config, modules_to_not_convert, quantizer_kwargs
from swift.dev.builders.sampler import build_sampler
from swift.dev.config import (
    DatasetConfig,
    DistributedConfig,
    ModelConfig,
    QuantizeConfig,
    TemplateConfig,
    TrainConfig,
    TunerConfig,
    process_configs,
    validate_configs,
)


@pytest.mark.parametrize(
    ('config', 'expected'),
    [
        (QuantizeConfig(
            quant_method='bnb',
            quant_bits=4,
            bnb_4bit_compute_dtype='bfloat16',
            bnb_4bit_quant_type='fp4',
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_storage='bfloat16'), {
                'quant_bits': 4,
                'bnb_4bit_compute_dtype': 'bfloat16',
                'bnb_4bit_quant_type': 'fp4',
                'bnb_4bit_use_double_quant': False,
                'bnb_4bit_quant_storage': 'bfloat16',
                'llm_int8_skip_modules': ['lm_head'],
            }),
        (QuantizeConfig(quant_method='hqq', quant_bits=3, hqq_axis=0, group_size=64), {
            'quant_bits': 3,
            'axis': 0,
            'group_size': 64,
            'skip_modules': ['lm_head'],
        }),
        (QuantizeConfig(quant_method='eetq', quant_bits=8), {
            'quant_bits': 8,
            'modules_to_not_convert': ['lm_head'],
        }),
        (QuantizeConfig(quant_method='quanto', quant_bits='float8'), {
            'quant_bits': 'float8',
            'modules_to_not_convert': ['lm_head'],
        }),
        (QuantizeConfig(quant_method='fp8'), {
            'modules_to_not_convert': ['lm_head'],
        }),
    ],
)
def test_load_quantizer_kwargs_are_complete(config, expected):
    assert quantizer_kwargs(config, modules_to_not_convert=['lm_head']) == expected


def test_modules_to_not_convert_matches_legacy_architecture_rules():
    loader = SimpleNamespace(
        model_info=SimpleNamespace(is_moe_model=True),
        model_arch=SimpleNamespace(
            vision_tower=['visual'], aligner=['multi_modal_projector'], lm_head_name='output'))
    assert modules_to_not_convert(loader) == [
        'mlp.gate', 'mlp.shared_expert_gate', 'visual', 'multi_modal_projector', 'output'
    ]
    plain = SimpleNamespace(
        model_info=SimpleNamespace(is_moe_model=False),
        model_arch=SimpleNamespace(vision_tower=None, aligner=None, lm_head_name='lm_head'))
    assert modules_to_not_convert(plain) is None


def test_build_load_quantization_config_uses_shared_twinkle_builder(monkeypatch):
    captured = {}

    class Quantizer:

        def get_quantization_config(self):
            return {'built': True}

    def fake_get_quantizer(method, **kwargs):
        captured.update(method=method, kwargs=kwargs)
        return Quantizer()

    monkeypatch.setattr('twinkle.quantizer.get_quantizer', fake_get_quantizer)
    result = build_load_quantization_config(QuantizeConfig(quant_method='hqq', quant_bits=4, hqq_axis=1))
    assert result == {'built': True}
    assert captured == {'method': 'hqq', 'kwargs': {'quant_bits': 4, 'axis': 1, 'group_size': 128}}


def test_calibration_quantizers_are_not_accepted_as_load_time_configs():
    with pytest.raises(ValueError, match='calibration/export method'):
        build_load_quantization_config(QuantizeConfig(quant_method='awq', quant_bits=4))


def test_unsloth_maps_only_bnb_4bit_and_8bit():
    model = ModelConfig(model='m')
    tuner = TunerConfig(tuner_backend='unsloth', tuner_type='lora')
    kwargs = {}
    _apply_unsloth_kwargs(kwargs, model, tuner, TrainConfig(), QuantizeConfig(quant_method='bnb', quant_bits=4))
    assert kwargs['load_in_4bit'] is True and kwargs['load_in_8bit'] is False
    assert 'quantization_config' not in kwargs

    kwargs = {}
    _apply_unsloth_kwargs(kwargs, model, tuner, TrainConfig(), QuantizeConfig(quant_method='bnb', quant_bits=8))
    assert kwargs['load_in_4bit'] is False and kwargs['load_in_8bit'] is True
    with pytest.raises(NotImplementedError, match='only supports BNB'):
        _apply_unsloth_kwargs(kwargs, model, tuner, TrainConfig(), QuantizeConfig(quant_method='hqq', quant_bits=4))


def test_transformers_sampler_receives_quantization_config(monkeypatch):
    captured = {}

    class Sampler:

        def __init__(self, model, **kwargs):
            captured.update(model=model, **kwargs)

    monkeypatch.setattr('twinkle.sampler.TransformersSampler', Sampler)
    monkeypatch.setattr(
        'swift.dev.builders.quantization.build_load_quantization_config', lambda config: {'method': config.quant_method})
    build_sampler(ModelConfig(model='m'), backend='transformers', quantize_config=QuantizeConfig(quant_method='bnb'))
    assert captured['engine_args']['quantization_config'] == {'method': 'bnb'}


@pytest.mark.parametrize(('backend', 'option'), [('vllm', 'vllm_quantization'), ('sglang', 'sglang_quantization')])
def test_engine_backends_reject_transformers_load_quantization(backend, option):
    with pytest.raises(ValueError, match=option):
        build_sampler(ModelConfig(model='m'), backend=backend, quantize_config=QuantizeConfig(quant_method='bnb'))


def test_training_quantization_requires_adapter_and_rejects_megatron():
    common = (ModelConfig(model='m'), TemplateConfig(), DatasetConfig(dataset=['d']), TrainConfig())
    quant = QuantizeConfig(quant_method='bnb', quant_bits=4)
    with pytest.raises(ValueError, match='full-parameter training'):
        validate_configs(*common, DistributedConfig(), quantize_config=quant)
    with pytest.raises(ValueError, match='not supported by the Megatron backend'):
        validate_configs(
            *common, DistributedConfig(backend='megatron'), TunerConfig(tuner_type='lora'), quantize_config=quant)


def test_bnb_compute_dtype_is_derived_from_model_dtype():
    model = ModelConfig(model='m', torch_dtype='bfloat16')
    quant = QuantizeConfig(quant_method='bnb', quant_bits=4)
    process_configs(model, TemplateConfig(), DatasetConfig(), TrainConfig(), DistributedConfig(),
                    TunerConfig(tuner_type='lora'), quantize_config=quant)
    assert quant.bnb_4bit_compute_dtype == 'bfloat16'
