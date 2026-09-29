# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fast tier: ``run_eval``'s pure helpers and the ``Evaluator``/adapter construction guards.

No GPU, no model, no EvalScope ``run_task``: these assert the config-to-task translation and the fail-loudly
guards that protect the sampler-only, Native-only contract. The real end-to-end run lives in
``test_evaluator_e2e.py``; the real command in the slow tier.
"""
import pytest

from swift.dev.config import EvalConfig, ModelConfig, TemplateConfig
from swift.dev.recipe.run_eval import (_build_task_config, _guard_backend, _model_name, _validate_eval_datasets,
                                       run_eval)


# --------------------------------------------------------------------------- _model_name
def test_model_name_is_last_path_segment():
    assert _model_name(ModelConfig(model='Qwen/Qwen2.5-0.5B-Instruct')) == 'Qwen2.5-0.5B-Instruct'


def test_model_name_strips_trailing_slash():
    assert _model_name(ModelConfig(model='/data/checkpoints/lora-merged/')) == 'lora-merged'


def test_model_name_falls_back_when_unset():
    assert _model_name(ModelConfig(model=None)) == 'model'


# --------------------------------------------------------------------------- _guard_backend
@pytest.mark.parametrize('backend', ['client', 'no'])
def test_guard_backend_rejects_message_only_backends(backend):
    """eval builds and scores a LOCAL model, so the backends that load none must be refused with a fix."""
    with pytest.raises(ValueError) as exc:
        _guard_backend(backend)
    message = str(exc.value)
    assert 'builds a local sampler' in message
    assert 'local backend' in message


@pytest.mark.parametrize('backend', ['vllm', 'sglang', 'transformers'])
def test_guard_backend_allows_local_backends(backend):
    assert _guard_backend(backend) is None


# --------------------------------------------------------------------------- _validate_eval_datasets
def test_validate_eval_datasets_normalizes_case_in_place():
    eval_config = EvalConfig(eval_dataset=['GSM8K', 'MmLu'])
    _validate_eval_datasets(eval_config)
    assert eval_config.eval_dataset == ['gsm8k', 'mmlu']


def test_validate_eval_datasets_rejects_unknown_benchmark():
    eval_config = EvalConfig(eval_dataset=['gsm8k', 'not_a_real_benchmark'])
    with pytest.raises(ValueError) as exc:
        _validate_eval_datasets(eval_config)
    message = str(exc.value)
    assert 'not_a_real_benchmark' in message
    # the message must point at what IS supported, not just say "no".
    assert 'supported datasets' in message
    assert 'gsm8k' in message


def test_run_eval_requires_at_least_one_dataset():
    with pytest.raises(ValueError, match='At least one --eval_dataset'):
        run_eval(ModelConfig(model='m'), TemplateConfig(), EvalConfig(eval_dataset=[]))


# --------------------------------------------------------------------------- _build_task_config
def test_build_task_config_maps_eval_config_onto_evalscope_fields():
    eval_config = EvalConfig(
        eval_dataset=['gsm8k'],
        eval_limit=7,
        eval_num_proc=4,
        eval_dataset_args={'gsm8k': {'few_shot_num': 2}},
        eval_generation_config={'max_tokens': 32, 'temperature': 0.0},
        eval_output_dir='/tmp/eval_out',
    )
    task_config = _build_task_config(eval_config)
    assert task_config['work_dir'] == '/tmp/eval_out'
    assert task_config['limit'] == 7
    # eval_num_proc is EvalScope's request concurrency AND the non-continuous sampler micro-batch width.
    assert task_config['eval_batch_size'] == 4
    assert task_config['dataset_args'] == {'gsm8k': {'few_shot_num': 2}}
    assert task_config['generation_config'] == {'max_tokens': 32, 'temperature': 0.0}


def test_build_task_config_merges_extra_eval_args_last():
    """``extra_eval_args`` is the escape hatch for any other EvalScope field and wins over the mapped ones."""
    eval_config = EvalConfig(eval_limit=7, extra_eval_args={'limit': 99, 'use_cache': False})
    task_config = _build_task_config(eval_config)
    assert task_config['limit'] == 99
    assert task_config['use_cache'] is False


def test_build_task_config_never_carries_owned_keys():
    """The Evaluator pins model/datasets/eval_type/eval_backend/model_task; the recipe must not set them."""
    from twinkle_agentic.evaluator.evaluator import _OWNED_TASK_KEYS
    task_config = _build_task_config(EvalConfig(eval_dataset=['gsm8k']))
    assert _OWNED_TASK_KEYS.isdisjoint(task_config)


# --------------------------------------------------------------------------- Evaluator guards
def test_evaluator_requires_exactly_one_of_sampler_or_api():
    from twinkle_agentic.evaluator import Evaluator, EvaluatorConfigError
    with pytest.raises(EvaluatorConfigError, match='exactly one of sampler or api'):
        Evaluator(datasets=['gsm8k'], model_id='m')
    with pytest.raises(EvaluatorConfigError, match='exactly one of sampler or api'):
        Evaluator(datasets=['gsm8k'], sampler=object(), api=object(), model_id='m')


def test_evaluator_rejects_owned_key_smuggled_through_task_config():
    """An owned key in ``task_config`` (e.g. via ``extra_eval_args``) is refused, not silently dropped."""
    from twinkle_agentic.evaluator import Evaluator, EvaluatorConfigError
    with pytest.raises(EvaluatorConfigError, match='Twinkle-owned field'):
        Evaluator(
            datasets=['gsm8k'],
            sampler=object(),
            model_id='m',
            task_config={'eval_backend': 'opencompass'},
        )


def test_evaluator_requires_resolvable_model_id():
    from twinkle_agentic.evaluator import Evaluator, EvaluatorConfigError
    with pytest.raises(EvaluatorConfigError, match='model_id is required'):
        Evaluator(datasets=['gsm8k'], sampler=object())


# --------------------------------------------------------------------------- adapter guards
def _sampler_model_api(*, continuous: bool, explicit_generation_keys):
    from twinkle_agentic.evaluator.evalscope_adapter import SamplerModelAPI
    from .conftest import make_scripted_sampler
    return SamplerModelAPI(
        make_scripted_sampler(continuous=continuous),
        'scripted',
        None,
        set(explicit_generation_keys),
        batch_size=2,
        batch_wait_ms=5.0,
        sampler_kwargs={},
    )


def test_continuous_sampler_bypasses_batcher():
    """A continuous-work engine (vLLM/SGLang) batches inside itself, so the adapter drives it per trajectory."""
    api = _sampler_model_api(continuous=True, explicit_generation_keys=())
    assert api.continuous is True
    assert api.batcher is None


def test_non_continuous_sampler_uses_batcher():
    """A non-continuous engine (transformers) needs the micro-batcher to form a physical batch at all."""
    api = _sampler_model_api(continuous=False, explicit_generation_keys=())
    assert api.continuous is False
    assert api.batcher is not None
    api.batcher.close()


def test_sampler_adapter_rejects_unmappable_generation_field():
    from evalscope.api.model import GenerateConfig
    from twinkle_agentic.evaluator import UnsupportedCapabilityError
    api = _sampler_model_api(continuous=True, explicit_generation_keys={'frequency_penalty'})
    with pytest.raises(UnsupportedCapabilityError, match='frequency_penalty'):
        api.validate_generation_config(GenerateConfig(frequency_penalty=0.5))


def test_sampler_adapter_rejects_streaming():
    from evalscope.api.model import GenerateConfig
    from twinkle_agentic.evaluator import UnsupportedCapabilityError
    api = _sampler_model_api(continuous=True, explicit_generation_keys={'stream'})
    with pytest.raises(UnsupportedCapabilityError, match='[Ss]treaming'):
        api.validate_generation_config(GenerateConfig(stream=True))
