# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import tempfile
import torch
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from swift.model.decision_branch import OPT_CLOSE, OPT_OPEN
from swift.model.decision_head import load_thx02_decision_config
from swift.model.models.decision import OmniJevModelLoader


def make_model_processor():
    from tokenizers import Tokenizer, models
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({
            '[UNK]': 0,
            'a': 1,
            'b': 2,
            'c': 3
        }, unk_token='[UNK]')),
        unk_token='[UNK]')
    model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_embd=8, n_layer=1, n_head=2))
    return model, SimpleNamespace(tokenizer=tokenizer)


class TestDecisionCheckpointAssets(unittest.TestCase):

    @contextmanager
    def _checkpoint(self):
        # Each parameter combination starts with a clean factory environment and directory.
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            for name in ['OMNIJEV_NEW_TOK_EMB', 'THX02_DECISION_CONFIG']:
                os.environ.pop(name, None)
            yield Path(directory)

    def test_omnijev_checkpoint_keeps_trained_option_embeddings(self):
        with self._checkpoint() as tmp_path:
            os.environ.pop('OMNIJEV_NEW_TOK_EMB', None)
            model, processor = make_model_processor()
            factory = tmp_path / 'factory'
            factory.mkdir()
            rows = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16
            torch.save(rows, factory / 'new_tok_emb.pt')
            OmniJevModelLoader._prepare_option_tokens(model, processor, str(factory))
            token_ids = model.config.omnijev_option_token_ids
            assert token_ids == processor.tokenizer.convert_tokens_to_ids([OPT_OPEN, OPT_CLOSE])
            torch.testing.assert_close(model.get_input_embeddings().weight[token_ids], rows)
            with torch.no_grad():
                model.get_input_embeddings().weight[token_ids[0]].add_(0.25)
            expected = model.get_input_embeddings().weight.detach().clone()
            model.save_pretrained(tmp_path / 'checkpoint')
            processor.tokenizer.save_pretrained(tmp_path / 'checkpoint')

            from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
            restored = GPT2LMHeadModel.from_pretrained(tmp_path / 'checkpoint')
            processor.tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp_path / 'checkpoint')
            # Even a still-exported factory path must not overwrite trained checkpoint rows.
            os.environ['OMNIJEV_NEW_TOK_EMB'] = str(factory / 'new_tok_emb.pt')
            OmniJevModelLoader._prepare_option_tokens(restored, processor, str(tmp_path / 'checkpoint'))
            torch.testing.assert_close(restored.get_input_embeddings().weight, expected, rtol=0, atol=0)

    def test_omnijev_checkpoint_rejects_inconsistent_tokenizer(self):
        for saved_ids in [[4], [4, 4], [4, 99], [4, 5], ['4', '5']]:
            with self.subTest(saved_ids=saved_ids), self._checkpoint() as tmp_path:
                model, processor = make_model_processor()
                model.config.omnijev_option_token_ids = saved_ids
                with self.assertRaisesRegex(ValueError, 'option-token IDs'):
                    OmniJevModelLoader._prepare_option_tokens(model, processor, str(tmp_path))

    def test_omnijev_factory_still_requires_embeddings(self):
        with self._checkpoint() as tmp_path:
            os.environ.pop('OMNIJEV_NEW_TOK_EMB', None)
            model, processor = make_model_processor()
            with self.assertRaisesRegex(FileNotFoundError, 'new_tok_emb.pt'):
                OmniJevModelLoader._prepare_option_tokens(model, processor, str(tmp_path))

    def test_thx02_checkpoint_processor_config_and_override(self):
        with self._checkpoint() as tmp_path:
            os.environ.pop('THX02_DECISION_CONFIG', None)
            saved = {
                'attention_mode': 'noncausal_full_attention',
                'codes': ['A', 'B', 'C'],
                'temperature': 0.7,
                'max_pixels': 131072
            }
            (tmp_path / 'config.json').write_text(json.dumps({'thx02_decision_config': saved}))
            assert load_thx02_decision_config(str(tmp_path)) == saved
            override = dict(saved, codes=['X', 'Y'], max_pixels=262144)
            path = tmp_path / 'override.json'
            path.write_text(json.dumps(override))
            os.environ['THX02_DECISION_CONFIG'] = str(path)
            assert load_thx02_decision_config(str(tmp_path)) == override
            path.write_text('{')
            with self.assertRaises(json.JSONDecodeError):
                load_thx02_decision_config(str(tmp_path))

    def test_thx02_checkpoint_still_validates_processor_config(self):
        for field, value in [('attention_mode', 'causal'), ('codes', [])]:
            with self.subTest(field=field, value=value), self._checkpoint() as tmp_path:
                os.environ.pop('THX02_DECISION_CONFIG', None)
                saved = {'attention_mode': 'noncausal_full_attention', 'codes': ['A', 'B'], 'temperature': 1.0}
                saved[field] = value
                (tmp_path / 'config.json').write_text(json.dumps({'thx02_decision_config': saved}))
                with self.assertRaises(ValueError):
                    load_thx02_decision_config(str(tmp_path))


if __name__ == '__main__':
    unittest.main()
