# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
import json
import pytest
import torch
from safetensors import safe_open
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast
from types import SimpleNamespace

from swift.model.decision_head import KIND_NAMES, StartLuxLetterHead, masked_softmax
from swift.tuners import LoraConfig, Swift

_qwen3_5 = pytest.importorskip(
    'transformers.models.qwen3_5.modeling_qwen3_5', reason='Qwen3.5 requires transformers>=5.2')


def _config():
    return _qwen3_5.Qwen3_5Config(
        text_config={
            'vocab_size': 128,
            'hidden_size': 32,
            'intermediate_size': 64,
            'num_hidden_layers': 1,
            'num_attention_heads': 4,
            'num_key_value_heads': 2,
            'head_dim': 8,
            'layer_types': ['full_attention'],
            'max_position_embeddings': 128,
            'rope_parameters': {
                'rope_type': 'default',
                'rope_theta': 10000.0,
                'partial_rotary_factor': 1.0,
                'mrope_section': [1, 1, 2]
            },
        },
        vision_config={
            'depth': 1,
            'hidden_size': 32,
            'intermediate_size': 64,
            'num_heads': 4,
            'patch_size': 2,
            'spatial_merge_size': 2,
            'out_hidden_size': 32,
            'num_position_embeddings': 16
        },
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123)


@pytest.mark.parametrize('max_shard_size', ['5GB', '1KB'])
def test_startlux_merge_exports_backbone_and_preserves_trained_readout(tmp_path, monkeypatch, max_shard_size):
    merge_module = importlib.import_module('swift.pipelines.export.merge_lora')
    torch.manual_seed(42)
    base_dir = tmp_path / 'base'
    base_dir.mkdir()
    factory = {
        'letter_token_ids': list(range(26)),
        'temperature_by_type': {
            'noul': 1.2,
            'choice': 1.5,
            'score': 1.8
        },
        'prompt_attention': 'causal',
        'auto_think': {
            'gate': 0.7,
            'mix': 0.5
        },
        'wide_choice': {
            'group': 25,
            'keep': 3,
            'residual': 0.001
        },
    }
    (base_dir / 'decision_config.json').write_text(json.dumps(factory))
    base = _qwen3_5.Qwen3_5ForConditionalGeneration(_config())
    base.model_dir = str(base_dir)
    base.model_meta = SimpleNamespace(additional_saved_files=[])
    head = StartLuxLetterHead.from_pretrained(str(base_dir))
    head.head_temperature.fill_(1.3)
    base.scoring_head = head
    model = Swift.prepare_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj']))
    head.bias.requires_grad_(True)
    tokens = torch.tensor([[31, 32, 33, 34]] * 3)
    meta = {'kinds': [0, 1, 2], 'n_options': [2, 3, 5], 'record_index': [0, 1, 2], 'seq_lens': [4] * 3}
    model.eval()
    output = head(None, model(input_ids=tokens, use_cache=False).logits, meta)
    loss = torch.nn.functional.cross_entropy(output.logits, torch.tensor([1, 1, 2]))
    loss.backward()
    torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1).step()
    assert torch.count_nonzero(head.bias) > 0
    with torch.no_grad():
        trained = head(None, model(input_ids=tokens, use_cache=False).logits, meta)
        expected = masked_softmax(trained.logits, trained.option_mask)

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]')))
    monkeypatch.setattr(merge_module, 'prepare_model_template', lambda args:
                        (model, SimpleNamespace(processor=tokenizer)))
    out = tmp_path / 'merged'
    args = SimpleNamespace(
        output_dir=str(out),
        adapters=[str(tmp_path / 'adapter')],
        quant_method=None,
        device_map='cpu',
        task_type='decision',
        safe_serialization=True,
        max_shard_size=max_shard_size)
    merge_module.merge_lora(args)

    for weights in out.glob('*.safetensors'):
        with safe_open(weights, framework='pt') as f:
            assert not any(name.startswith('scoring_head.') for name in f.keys())
    if max_shard_size == '1KB':
        assert (out / 'model.safetensors.index.json').exists()
    exported = json.loads((out / 'decision_config.json').read_text())
    assert exported['auto_think'] == factory['auto_think']
    assert exported['wide_choice'] == factory['wide_choice']
    assert exported['letter_bias'] == head.bias.detach().tolist()
    assert exported['temperature_by_type'] == {
        kind: float(head.head_temperature * head.temperatures[i])
        for i, kind in enumerate(KIND_NAMES)
    }
    reloaded = _qwen3_5.Qwen3_5ForConditionalGeneration.from_pretrained(str(out)).eval()
    restored_head = StartLuxLetterHead.from_pretrained(str(out))
    with torch.no_grad():
        result = restored_head(None, reloaded(input_ids=tokens, use_cache=False).logits, meta)
        actual = masked_softmax(result.logits, result.option_mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert restored_head.load_pretrained(str(out))
    with torch.no_grad():
        result = restored_head(None, reloaded(input_ids=tokens, use_cache=False).logits, meta)
    torch.testing.assert_close(masked_softmax(result.logits, result.option_mask), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('bias', [None, [0] * 25, [True] * 26, ['0'] * 26, [float('nan')] * 26])
def test_startlux_rejects_invalid_native_bias(tmp_path, bias):
    (tmp_path / 'decision_config.json').write_text(
        json.dumps({
            'letter_token_ids': list(range(26)),
            'letter_bias': bias
        }))
    with pytest.raises(ValueError, match='letter_bias'):
        StartLuxLetterHead.from_pretrained(str(tmp_path))
