# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import tempfile
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from transformers import PretrainedConfig

from swift.model import save_checkpoint


class TextConfig(PretrainedConfig):
    model_type = 'qwen3_5_text'


class DummyProcessor:

    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def save_pretrained(self, output_dir):
        with open(os.path.join(output_dir, 'preprocessor_config.json'), 'w') as f:
            json.dump({'processor': 'multimodal'}, f)


class DummyTokenizer:

    def save_pretrained(self, output_dir):
        with open(os.path.join(output_dir, 'tokenizer_config.json'), 'w') as f:
            json.dump({}, f)


class DummyQwen35Model:

    def __init__(self):
        self.config = PretrainedConfig()
        self.config.architectures = ['Qwen3_5ForConditionalGeneration']
        self.config.text_config = TextConfig()
        self.config.text_config.architectures = ['Qwen3_5ForConditionalGeneration']
        self.model_meta = SimpleNamespace(
            model_arch=SimpleNamespace(language_model=['model.language_model', 'lm_head']), additional_saved_files=[])
        self.model_dir = None

    def state_dict(self):
        return {
            'model.language_model.embed_tokens.weight': torch.ones(2, 2),
            'model.language_model.layers.0.self_attn.q_proj.weight': torch.ones(2, 2) * 2,
            'lm_head.weight': torch.ones(2, 2) * 3,
            'model.visual.patch_embed.weight': torch.ones(2, 2) * 4,
        }


class DummyDefaultModel(DummyQwen35Model):

    def save_pretrained(self, output_dir, safe_serialization=True, max_shard_size='5GB'):
        with open(os.path.join(output_dir, 'model_saved.json'), 'w') as f:
            json.dump({
                'safe_serialization': safe_serialization,
                'max_shard_size': max_shard_size,
            }, f)


def test_save_checkpoint_default_multimodal_export_unchanged():
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = os.path.join(tmp_dir, 'output')
        model_dir = os.path.join(tmp_dir, 'model')
        os.makedirs(output_dir)
        os.makedirs(model_dir)
        with open(os.path.join(model_dir, 'preprocessor_config.json'), 'w') as f:
            json.dump({'source': 'vision'}, f)

        save_checkpoint(
            DummyDefaultModel(),
            DummyProcessor(),
            output_dir,
            safe_serialization=False,
            max_shard_size='1GB',
            model_dirs=[model_dir])

        with open(os.path.join(output_dir, 'model_saved.json')) as f:
            model_saved = json.load(f)
        assert model_saved == {'safe_serialization': False, 'max_shard_size': '1GB'}

        with open(os.path.join(output_dir, 'preprocessor_config.json')) as f:
            processor_config = json.load(f)
        assert processor_config == {'source': 'vision'}
        assert not os.path.exists(os.path.join(output_dir, 'tokenizer_config.json'))


def test_save_checkpoint_export_language_model_only():
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = os.path.join(tmp_dir, 'output')
        model_dir = os.path.join(tmp_dir, 'model')
        os.makedirs(model_dir)
        with open(os.path.join(model_dir, 'preprocessor_config.json'), 'w') as f:
            json.dump({'source': 'vision'}, f)
        with open(os.path.join(model_dir, 'args.json'), 'w') as f:
            json.dump({'source': 'args'}, f)

        save_checkpoint(
            DummyQwen35Model(),
            DummyProcessor(),
            output_dir,
            safe_serialization=True,
            max_shard_size='10GB',
            model_dirs=[model_dir],
            language_model_only=True)

        state_dict = load_file(os.path.join(output_dir, 'model.safetensors'))
        assert set(state_dict) == {
            'model.embed_tokens.weight',
            'model.layers.0.self_attn.q_proj.weight',
            'lm_head.weight',
        }

        with open(os.path.join(output_dir, 'config.json')) as f:
            config = json.load(f)
        assert config['model_type'] == 'qwen3_5_text'
        assert config['architectures'] == ['Qwen3_5ForCausalLM']
        assert os.path.exists(os.path.join(output_dir, 'tokenizer_config.json'))
        assert os.path.exists(os.path.join(output_dir, 'args.json'))
        assert not os.path.exists(os.path.join(output_dir, 'preprocessor_config.json'))
