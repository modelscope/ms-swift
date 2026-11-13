# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
import json
import tempfile
import torch
import unittest
from itertools import product
from pathlib import Path
from safetensors import safe_open
from tokenizers import Tokenizer, models
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from types import SimpleNamespace
from unittest.mock import patch

from swift.model.decision_head import (ClefJointSchemaHead, JevVerbalizerHead, KevPointerHead, LayaHead, OmniJevHead,
                                       StartLuxLetterHead, Thx02ReadoutHead, masked_softmax,
                                       maybe_load_trained_scoring_head)
from swift.tuners import LoraConfig, Swift

HEADS = [
    JevVerbalizerHead, StartLuxLetterHead, ClefJointSchemaHead, Thx02ReadoutHead, OmniJevHead, KevPointerHead, LayaHead
]


def make_head(head_cls, base_dir):
    temperatures = {'noul': 0.8, 'choice': 1.3, 'score': 1.7}
    if head_cls is StartLuxLetterHead:
        (base_dir / 'decision_config.json').write_text(
            json.dumps({
                'letter_token_ids': list(range(26)),
                'temperature_by_type': temperatures
            }))
        head = head_cls.from_pretrained(str(base_dir))
        head.head_temperature.fill_(1.2)
        return head
    if head_cls is JevVerbalizerHead:
        return head_cls(
            bias=[0.] * 24,
            verbalizer_ids=list(range(24)),
            ranges=head_cls.DEFAULT_RANGES,
            temperatures=temperatures,
            head_temperature=1.2)
    if head_cls is ClefJointSchemaHead:
        return head_cls(hidden_size=8, width=12, routing_layers=1, layers=2, heads=3, feedforward=19, dropout=0.2)
    if head_cls is Thx02ReadoutHead:
        return head_cls(hidden_size=8, num_codes=5, temperature=0.7, head_temperature=1.4, temperatures=temperatures)
    if head_cls is OmniJevHead:
        head = head_cls(
            hidden_size=8,
            head_hidden=13,
            ord_hidden=11,
            lm_feats=True,
            ordinal=True,
            temperatures=temperatures,
            noul_bias=0.3,
            head_temperature=1.2)
        head.set_marker_lengths(1, 1)
        return head
    if head_cls is KevPointerHead:
        head = head_cls(hidden_size=8, pointer_dim=5)
        head.temperature.fill_(1.7)
        return head
    return head_cls(
        hidden_size=8,
        head_layers=1,
        dropout=0.2,
        temperatures=[0.8, 1.3, 1.7],
        temperature_by_options={'choice:3-5': 1.9})


def score(model, head, tokens, meta):
    output = model(input_ids=tokens, output_hidden_states=True)
    hidden = output.last_hidden_state if head.encoder_backbone else output.hidden_states[-1]
    result = head(hidden, getattr(output, 'logits', None), meta)
    return result, masked_softmax(result.logits, result.option_mask)


class TestDecisionExport(unittest.TestCase):

    def test_merged_backbone_and_trained_head_roundtrip(self):
        merge_module = importlib.import_module('swift.pipelines.export.merge_lora')
        for head_cls, shard_size in product(HEADS, ['5GB', '1KB']):
            with self.subTest(
                    head=head_cls.head_type, shard_size=shard_size), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(42)
                root = Path(directory)
                base_dir = root / 'base'
                base_dir.mkdir()
                head = make_head(head_cls, base_dir)
                if head.encoder_backbone:
                    base = BertModel(
                        BertConfig(
                            vocab_size=64,
                            hidden_size=8,
                            intermediate_size=16,
                            num_hidden_layers=1,
                            num_attention_heads=2,
                            max_position_embeddings=32))
                    targets = ['query', 'value']
                else:
                    base = Qwen2ForCausalLM(
                        Qwen2Config(
                            vocab_size=64,
                            hidden_size=8,
                            intermediate_size=16,
                            num_hidden_layers=1,
                            num_attention_heads=2,
                            num_key_value_heads=1,
                            max_position_embeddings=32))
                    targets = ['q_proj', 'v_proj']
                model_cls = type(base)
                base.model_dir = str(base_dir)
                base.model_meta = SimpleNamespace(additional_saved_files=[])
                base.scoring_head = head
                if head.needs_output_embeddings:
                    head.set_output_embeddings(base.get_output_embeddings())
                model = Swift.prepare_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=targets))
                head.requires_grad_(True)
                model.eval()
                tokens = torch.arange(36).reshape(3, 12) + 1
                meta = {
                    'n_options': [2, 3, 3],
                    'kinds': [0, 1, 2],
                    'record_index': [0, 1, 2],
                    'counts': [1, 1, 1],
                    'seq_lens': [12] * 3,
                    'input_ids': tokens,
                    'question_spans': [(0, 2)] * 3,
                    'option_spans': [[(2, 4), (4, 6)], [(2, 4), (4, 6), (6, 8)], [(2, 4), (4, 6), (6, 8)]],
                    'option_markers': [[(2, 4)], [(2, 4), (5, 7)], [(2, 4), (5, 7), (8, 10)]],
                    'decide_idx': [1, 1, 1],
                    'opt_idx': [[4, 7], [4, 7, 10], [4, 7, 10]],
                    'readout_positions': [1, 1, 1],
                    'laya_markers': [[4, 7], [4, 7, 10], [4, 7, 10]],
                }
                initial = {k: v.clone() for k, v in head.head_state_dict().items()}
                result, _ = score(model, head, tokens, meta)
                torch.nn.functional.cross_entropy(result.logits, torch.tensor([1, 1, 2])).backward()
                torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.02).step()
                trained_state = {k: v.clone() for k, v in head.head_state_dict().items()}
                self.assertTrue(any(not torch.equal(initial[k], v) for k, v in trained_state.items()))
                with torch.no_grad():
                    _, expected = score(model, head, tokens, meta)

                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]')))
                out = root / 'merged'
                args = SimpleNamespace(
                    output_dir=str(out),
                    adapters=[str(root / 'adapter')],
                    quant_method=None,
                    device_map='cpu',
                    task_type='decision',
                    safe_serialization=True,
                    max_shard_size=shard_size)
                # Use real LoRA merge/save/reload; only replace model/tokenizer discovery.
                with patch.object(
                        merge_module, 'prepare_model_template',
                        return_value=(model, SimpleNamespace(processor=tokenizer))):
                    merge_module.merge_lora(args)

                backbone_files = list(out.glob('model*.safetensors'))
                self.assertTrue(backbone_files)
                for weights in backbone_files:
                    with safe_open(weights, framework='pt') as f:
                        self.assertFalse(any(k.startswith('scoring_head.') for k in f.keys()))
                if shard_size == '1KB':
                    index = json.loads((out / 'model.safetensors.index.json').read_text())
                    self.assertFalse(any(k.startswith('scoring_head.') for k in index['weight_map']))
                self.assertTrue((out / head.HEAD_WEIGHTS_NAME).is_file())
                restored, info = model_cls.from_pretrained(str(out), output_loading_info=True)
                self.assertFalse(info['unexpected_keys'])
                self.assertFalse(info['missing_keys'])
                restored_head = head_cls.from_pretrained(str(out), model=restored, config=restored.config)
                restored.scoring_head = restored_head
                if restored_head.needs_output_embeddings:
                    restored_head.set_output_embeddings(restored.get_output_embeddings())
                maybe_load_trained_scoring_head(restored, str(out))
                restored.eval()
                for name, value in trained_state.items():
                    torch.testing.assert_close(restored_head.head_state_dict()[name], value, rtol=0, atol=0)
                self.assertEqual(restored_head.extra_state_for_save(), head.extra_state_for_save())
                with torch.no_grad():
                    _, actual = score(restored, restored_head, tokens, meta)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
