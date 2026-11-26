# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os

from swift.model import save_checkpoint


def _write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def _read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


class _FakeModel:
    """Simulates a model saved by transformers 5.x: stamps `transformers_version: 5.2.0`
    into config/generation config and writes fresh weights + safetensors index."""

    def __init__(self, model_dir):
        self.model_dir = model_dir

    def save_pretrained(self, output_dir, safe_serialization=True, max_shard_size='5GB', **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        _write_json(
            os.path.join(output_dir, 'config.json'), {
                'architectures': ['Qwen3_5ForConditionalGeneration'],
                'model_type': 'qwen3_5_moe',
                'torch_dtype': 'bfloat16',
                'tie_word_embeddings': False,
                'quantization_config': {
                    'quant_method': 'gptq',
                    'bits': 4
                },
                'transformers_version': '5.2.0',
            })
        _write_json(
            os.path.join(output_dir, 'generation_config.json'), {
                'eos_token_id': 151645,
                'transformers_version': '5.2.0',
            })
        with open(os.path.join(output_dir, 'model.safetensors'), 'wb') as f:
            f.write(b'fresh-weights')
        _write_json(
            os.path.join(output_dir, 'model.safetensors.index.json'), {
                'metadata': {
                    'total_size': 123
                },
                'weight_map': {
                    'model.layers.0.weight': 'model.safetensors'
                },
            })


class _FakeProcessor:
    """Simulates a processor saved by transformers 5.x: stamps the `TokenizersBackend`
    tokenizer class and `transformers_version: 5.2.0`, and adds a fresh token."""

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        _write_json(
            os.path.join(output_dir, 'tokenizer_config.json'), {
                'tokenizer_class': 'TokenizersBackend',
                'transformers_version': '5.2.0',
                'chat_template': 'fresh-template',
                'added_tokens_decoder': {
                    '151643': {
                        'content': '<|orig|>',
                        'special': True
                    },
                    '151646': {
                        'content': '<|fresh|>',
                        'special': True
                    },
                },
            })
        with open(os.path.join(output_dir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
            f.write('fresh-tokenizer')


def _write_original_model(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    _write_json(
        os.path.join(model_dir, 'config.json'), {
            'architectures': ['Qwen3_5ForConditionalGeneration'],
            'model_type': 'qwen3_5_moe',
            'torch_dtype': 'bfloat16',
            'transformers_version': '4.57.1',
        })
    _write_json(
        os.path.join(model_dir, 'generation_config.json'), {
            'eos_token_id': 151645,
            'transformers_version': '4.57.1',
        })
    _write_json(
        os.path.join(model_dir, 'tokenizer_config.json'), {
            'tokenizer_class': 'Qwen2TokenizerFast',
            'transformers_version': '4.57.1',
            'added_tokens_decoder': {
                '151643': {
                    'content': '<|orig|>',
                    'special': True
                },
            },
        })


def test_save_checkpoint_restores_original_transformers_stamps(tmp_path):
    """Regression test for modelscope/ms-swift#8098.

    A model fine-tuned under transformers 5.x must still deploy on vLLM, which
    requires transformers<5. The exported config/tokenizer must therefore carry
    the original model's transformers stamps, while freshly generated weights,
    the safetensors index and genuinely new content are kept.
    """
    src_dir = tmp_path / 'original-model'
    _write_original_model(str(src_dir))
    # A training checkpoint dir whose own files already carry the 5.x stamps:
    # the base model directory must win over it as the restore source.
    adapter_dir = tmp_path / 'checkpoint-100'
    os.makedirs(adapter_dir, exist_ok=True)
    _write_json(os.path.join(str(adapter_dir), 'config.json'), {'transformers_version': '5.2.0'})

    output_dir = tmp_path / 'exported'
    save_checkpoint(_FakeModel(str(src_dir)), _FakeProcessor(), str(output_dir), model_dirs=[str(adapter_dir)])

    config = _read_json(output_dir / 'config.json')
    assert config['transformers_version'] == '4.57.1'
    # Freshly saved values other than the stamp are preserved.
    assert config['quantization_config'] == {'quant_method': 'gptq', 'bits': 4}
    assert config['tie_word_embeddings'] is False
    assert config['architectures'] == ['Qwen3_5ForConditionalGeneration']

    tokenizer_config = _read_json(output_dir / 'tokenizer_config.json')
    assert tokenizer_config['transformers_version'] == '4.57.1'
    assert tokenizer_config['tokenizer_class'] == 'Qwen2TokenizerFast'
    # Added tokens and chat template from the fine-tune are preserved.
    assert tokenizer_config['chat_template'] == 'fresh-template'
    assert '151646' in tokenizer_config['added_tokens_decoder']

    generation_config = _read_json(output_dir / 'generation_config.json')
    assert generation_config['transformers_version'] == '4.57.1'
    assert generation_config['eos_token_id'] == 151645

    # Freshly generated weights and the safetensors index are untouched.
    assert (output_dir / 'model.safetensors').read_bytes() == b'fresh-weights'
    index = _read_json(output_dir / 'model.safetensors.index.json')
    assert index['metadata'] == {'total_size': 123}
    assert (output_dir / 'tokenizer.json').read_text(encoding='utf-8') == 'fresh-tokenizer'


def test_save_checkpoint_keeps_stamps_without_original_files(tmp_path):
    output_dir = tmp_path / 'exported'
    save_checkpoint(_FakeModel(str(tmp_path / 'empty-model')), _FakeProcessor(), str(output_dir), model_dirs=[])

    assert _read_json(output_dir / 'config.json')['transformers_version'] == '5.2.0'
    tokenizer_config = _read_json(output_dir / 'tokenizer_config.json')
    assert tokenizer_config['transformers_version'] == '5.2.0'
    assert tokenizer_config['tokenizer_class'] == 'TokenizersBackend'


def test_save_checkpoint_does_not_downgrade_transformers_5_original(tmp_path):
    src_dir = tmp_path / 'original-model'
    os.makedirs(src_dir, exist_ok=True)
    _write_json(
        os.path.join(str(src_dir), 'config.json'), {
            'model_type': 'qwen3_5_moe',
            'transformers_version': '5.0.0',
        })
    _write_json(
        os.path.join(str(src_dir), 'tokenizer_config.json'), {
            'tokenizer_class': 'TokenizersBackend',
            'transformers_version': '5.0.0',
        })

    output_dir = tmp_path / 'exported'
    save_checkpoint(_FakeModel(str(src_dir)), _FakeProcessor(), str(output_dir))

    assert _read_json(output_dir / 'config.json')['transformers_version'] == '5.2.0'
    assert _read_json(output_dir / 'tokenizer_config.json')['tokenizer_class'] == 'TokenizersBackend'
