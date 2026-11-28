# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import tempfile
import torch
import unittest
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from safetensors.torch import save_file
from unittest.mock import patch

from swift.model.decision_head import JevVerbalizerHead, StartLuxLetterHead


def make_head(head_cls):
    size = 26 if head_cls is StartLuxLetterHead else 24
    return head_cls(
        bias=torch.linspace(-0.2, 0.3, size).tolist(),
        verbalizer_ids=list(range(10, 10 + size)),
        ranges=head_cls.DEFAULT_RANGES,
        temperatures={
            'noul': 1.25,
            'choice': 0.75,
            'score': 1.5
        },
        head_temperature=1.2)


def write_factory(directory, head_cls, first_token=1):
    directory.mkdir(exist_ok=True)
    if head_cls is StartLuxLetterHead:
        filename = 'decision_config.json'
        data = {'letter_token_ids': list(range(first_token, first_token + 26)), 'temperature_by_type': {}}
    else:
        filename = 'decision_head.json'
        data = {'bias': [0.] * 24, 'verbalizer_ids': list(range(first_token, first_token + 24))}
    path = directory / filename
    path.write_text(json.dumps(data))
    return path


class TestVerbalizerCheckpoint(unittest.TestCase):

    @contextmanager
    def _checkpoint(self):
        # Each parameter combination starts with a clean factory environment and directory.
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            for name in ['JEV_DECISION_HEAD', 'JEV_DECIDE_CALIBRATION', 'STARTLUX_DECISION_CONFIG']:
                os.environ.pop(name, None)
            yield Path(directory)

    def test_saved_verbalizer_checkpoint_roundtrip(self):
        for head_cls, checkpoint_format in product([JevVerbalizerHead, StartLuxLetterHead],
                                                   ['adapter', 'full', 'sharded']):
            with self.subTest(
                    head_cls=head_cls.__name__, checkpoint_format=checkpoint_format), self._checkpoint() as tmp_path:
                head = make_head(head_cls)
                head.save_pretrained(str(tmp_path), save_weights=checkpoint_format == 'adapter')
                if checkpoint_format != 'adapter':
                    filename = ('model.safetensors'
                                if checkpoint_format == 'full' else 'model-00001-of-00001.safetensors')
                    state = {'scoring_head.bias': head.bias.detach()}
                    save_file(state, str(tmp_path / filename))
                    if checkpoint_format == 'sharded':
                        (tmp_path / 'model.safetensors.index.json').write_text(
                            json.dumps({'weight_map': {
                                'scoring_head.bias': filename
                            }}))
                meta_file = tmp_path / 'scoring_head.json'
                meta = json.loads(meta_file.read_text())
                meta['bias'] = [0.] * len(meta['bias'])
                meta_file.write_text(json.dumps(meta))
                restored = head_cls.from_pretrained(str(tmp_path))
                assert isinstance(restored, head_cls)
                assert restored.load_pretrained(str(tmp_path))
                assert restored.extra_state_for_save() == head.extra_state_for_save()
                torch.testing.assert_close(restored.bias, head.bias, rtol=0, atol=0)
                logits = torch.linspace(-1., 1., 3 * 2 * 64).view(3, 2, 64)
                meta = {'kinds': [0, 1, 2], 'n_options': [2, 3, 4], 'record_index': [0, 1, 2], 'counts': [1, 1, 1]}
                expected = head(None, logits, meta)
                actual = restored(None, logits, meta)
                torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
                assert torch.equal(actual.option_mask, expected.option_mask)

    def test_factory_precedes_saved_metadata(self):
        for head_cls, factory_location in product([JevVerbalizerHead, StartLuxLetterHead], ['local', 'override']):
            with self.subTest(
                    head_cls=head_cls.__name__, factory_location=factory_location), self._checkpoint() as tmp_path:
                head = make_head(head_cls)
                head.save_pretrained(str(tmp_path))
                write_factory(tmp_path, head_cls, first_token=1)
                expected_first_token = 1
                if factory_location == 'override':
                    override = write_factory(tmp_path / 'override', head_cls, first_token=2)
                    env_name = 'STARTLUX_DECISION_CONFIG' if head_cls is StartLuxLetterHead else 'JEV_DECISION_HEAD'
                    os.environ[env_name] = str(override)
                    expected_first_token = 2
                restored = head_cls.from_pretrained(str(tmp_path))
                assert restored.verbalizer_ids[0].item() == expected_first_token
                assert torch.equal(restored.bias, torch.zeros_like(restored.bias))

    def test_missing_factory_and_saved_metadata_still_fails(self):
        for head_cls in [JevVerbalizerHead, StartLuxLetterHead]:
            with self.subTest(head_cls=head_cls.__name__), self._checkpoint() as tmp_path:
                with self.assertRaises(FileNotFoundError):
                    head_cls.from_pretrained(str(tmp_path))

    def test_invalid_factory_is_not_hidden_by_saved_metadata(self):
        for head_cls in [JevVerbalizerHead, StartLuxLetterHead]:
            with self.subTest(head_cls=head_cls.__name__), self._checkpoint() as tmp_path:
                make_head(head_cls).save_pretrained(str(tmp_path))
                write_factory(tmp_path, head_cls).write_text('{')
                with self.assertRaises(json.JSONDecodeError):
                    head_cls.from_pretrained(str(tmp_path))

    def test_jev_factory_subdirectory_still_precedes_saved_metadata(self):
        with self._checkpoint() as tmp_path:
            os.environ.pop('JEV_DECISION_HEAD', None)
            make_head(JevVerbalizerHead).save_pretrained(str(tmp_path))
            write_factory(tmp_path / 'adapter_vllm', JevVerbalizerHead, first_token=3)
            restored = JevVerbalizerHead.from_pretrained(str(tmp_path))
            assert restored.verbalizer_ids[0].item() == 3

    def test_startlux_instant_guard_still_applies(self):
        for attention_source in ['factory', 'config', 'checkpoint']:
            with self.subTest(attention_source=attention_source), self._checkpoint() as tmp_path:
                os.environ.pop('STARTLUX_DECISION_CONFIG', None)
                make_head(StartLuxLetterHead).save_pretrained(str(tmp_path))
                if attention_source != 'checkpoint':
                    factory = write_factory(tmp_path, StartLuxLetterHead)
                if attention_source == 'factory':
                    data = json.loads(factory.read_text())
                    data['prompt_attention'] = 'bidirectional'
                    factory.write_text(json.dumps(data))
                else:
                    (tmp_path / 'config.json').write_text(json.dumps({'jev_prompt_attention': 'bidirectional'}))
                with self.assertRaisesRegex(ValueError, 'Instant'):
                    StartLuxLetterHead.from_pretrained(str(tmp_path))

    def test_startlux_override_uses_factory_directory_for_attention(self):
        with self._checkpoint() as tmp_path:
            make_head(StartLuxLetterHead).save_pretrained(str(tmp_path))
            (tmp_path / 'config.json').write_text(json.dumps({'jev_prompt_attention': 'bidirectional'}))
            factory = write_factory(tmp_path / 'override', StartLuxLetterHead, first_token=2)
            os.environ['STARTLUX_DECISION_CONFIG'] = str(factory)
            restored = StartLuxLetterHead.from_pretrained(str(tmp_path))
            assert restored.verbalizer_ids[0].item() == 2

    def test_saved_metadata_without_global_temperature(self):
        for head_cls in [JevVerbalizerHead, StartLuxLetterHead]:
            with self.subTest(head_cls=head_cls.__name__), self._checkpoint() as tmp_path:
                head = make_head(head_cls)
                head.save_pretrained(str(tmp_path))
                meta_file = tmp_path / 'scoring_head.json'
                meta = json.loads(meta_file.read_text())
                del meta['head_temperature']
                meta_file.write_text(json.dumps(meta))
                restored = head_cls.from_pretrained(str(tmp_path))
                assert restored.head_temperature.item() == 1.0
                assert restored.verbalizer_ids.tolist() == head.verbalizer_ids.tolist()
                torch.testing.assert_close(restored.temperatures, head.temperatures, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
