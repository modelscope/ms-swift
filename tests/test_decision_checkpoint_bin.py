# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import tempfile
import torch
import unittest
from pathlib import Path
from safetensors.torch import save_file
from unittest.mock import patch

from swift.model.decision_head import JevVerbalizerHead, KevPointerHead, StartLuxLetterHead


class TestDecisionBinCheckpoint(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for name in ('JEV_DECISION_HEAD', 'JEV_DECIDE_CALIBRATION', 'STARTLUX_DECISION_CONFIG', 'KEV_HEAD'):
            os.environ.pop(name, None)

    @staticmethod
    def make_head(head_cls):
        if head_cls is KevPointerHead:
            return head_cls(hidden_size=8, pointer_dim=5, temperature=1.7)
        size = 26 if head_cls is StartLuxLetterHead else 24
        return head_cls(
            bias=torch.linspace(-0.2, 0.3, size).tolist(),
            verbalizer_ids=list(range(size)),
            ranges=head_cls.DEFAULT_RANGES,
            temperatures={
                'noul': 0.8,
                'choice': 1.3,
                'score': 1.6
            },
            head_temperature=1.2)

    @staticmethod
    def scores(head):
        head.eval()
        meta = {
            'kinds': [0, 1, 2],
            'n_options': [2, 3, 3],
            'record_index': [0, 1, 2],
            'counts': [1, 1, 1],
            'decide_idx': [0, 0, 0],
            'opt_idx': [[1, 2], [1, 2, 3], [1, 2, 3]],
        }
        hidden = torch.linspace(-1., 1., 3 * 4 * 8).reshape(3, 4, 8)
        logits = torch.linspace(-1., 1., 3 * 4 * 32).reshape(3, 4, 32)
        return head(hidden, logits, meta).logits

    @staticmethod
    def write_weights(directory, state, sharded, safetensors=False):
        name = 'model.safetensors' if safetensors else 'pytorch_model.bin'

        def save(data, path):
            if safetensors:
                save_file(data, str(path))
            else:
                torch.save(data, path)

        if not sharded:
            save(state, directory / name)
            return [name]
        weight_map = {'backbone.weight': 'unneeded-backbone-shard'}
        files = []
        keys = sorted(state)
        for shard, subset in enumerate((keys[::2], keys[1::2])):
            if not subset:
                continue
            filename = f'head-{shard}-{name}'
            files.append(filename)
            save({key: state[key] for key in subset}, directory / filename)
            weight_map.update({key: filename for key in subset})
        (directory / (name + '.index.json')).write_text(json.dumps({'weight_map': weight_map}))
        return files

    def test_bin_roundtrip_restores_weights_and_scores(self):
        for head_cls in (JevVerbalizerHead, StartLuxLetterHead, KevPointerHead):
            for sharded in (False, True):
                with self.subTest(head=head_cls.__name__, sharded=sharded):
                    directory = self.directory / f'{head_cls.__name__}-{sharded}'
                    head = self.make_head(head_cls)
                    head.save_pretrained(str(directory), save_weights=False)
                    state = {'scoring_head.' + key: value for key, value in head.head_state_dict().items()}
                    files = self.write_weights(directory, state, sharded)
                    # The serialized parameters must override the constructor's copy.
                    if head_cls is not KevPointerHead:
                        path = directory / 'scoring_head.json'
                        meta = json.loads(path.read_text())
                        meta['bias'] = [0.] * len(meta['bias'])
                        path.write_text(json.dumps(meta))
                    restored = head_cls.from_pretrained(str(directory))
                    with patch('torch.load', wraps=torch.load) as load:
                        self.assertTrue(restored.load_pretrained(str(directory)))
                    self.assertEqual({Path(call.args[0]).name for call in load.call_args_list}, set(files))
                    for call in load.call_args_list:
                        self.assertEqual(call.kwargs, {'map_location': 'cpu', 'weights_only': True})
                    for key, value in head.head_state_dict().items():
                        torch.testing.assert_close(restored.head_state_dict()[key], value, rtol=0, atol=0)
                    torch.testing.assert_close(self.scores(restored), self.scores(head), rtol=0, atol=0)

    def test_safetensors_precede_bin(self):
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                directory = self.directory / str(sharded)
                head = self.make_head(KevPointerHead)
                head.save_pretrained(str(directory), save_weights=False)
                state = {'scoring_head.' + key: value for key, value in head.head_state_dict().items()}
                self.write_weights(directory, state, sharded, safetensors=True)
                # This invalid lower-priority file must never be read.
                (directory / 'pytorch_model.bin').write_bytes(b'invalid lower-priority checkpoint')
                restored = KevPointerHead.from_pretrained(str(directory))
                with patch('torch.load', side_effect=AssertionError('must prefer safetensors')):
                    self.assertTrue(restored.load_pretrained(str(directory)))
                torch.testing.assert_close(self.scores(restored), self.scores(head), rtol=0, atol=0)

    def test_bin_index_precedes_single_file(self):
        head = self.make_head(KevPointerHead)
        head.save_pretrained(str(self.directory), save_weights=False)
        state = {'scoring_head.' + key: value for key, value in head.head_state_dict().items()}
        self.write_weights(self.directory, state, sharded=True)
        (self.directory / 'pytorch_model.bin').write_bytes(b'invalid lower-priority checkpoint')
        restored = KevPointerHead.from_pretrained(str(self.directory))
        self.assertTrue(restored.load_pretrained(str(self.directory)))
        torch.testing.assert_close(self.scores(restored), self.scores(head), rtol=0, atol=0)

    def test_bin_missing_head_weights_fails(self):
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                directory = self.directory / str(sharded)
                head = self.make_head(KevPointerHead)
                head.save_pretrained(str(directory), save_weights=False)
                self.write_weights(directory, {'backbone.weight': torch.ones(1)}, sharded)
                restored = KevPointerHead.from_pretrained(str(directory))
                with self.assertRaisesRegex(FileNotFoundError, 'No trained scoring_head weights'):
                    restored.load_pretrained(str(directory))

    def test_bin_invalid_head_weights_fail(self):
        for defect in ('missing', 'unexpected', 'legacy_alias'):
            with self.subTest(defect=defect):
                directory = self.directory / defect
                head = self.make_head(KevPointerHead)
                head.save_pretrained(str(directory), save_weights=False)
                state = {'scoring_head.' + key: value for key, value in head.head_state_dict().items()}
                if defect == 'missing':
                    state.pop('scoring_head.q.bias')
                else:
                    key = '_output_embeddings.weight' if defect == 'legacy_alias' else 'unexpected'
                    state['scoring_head.' + key] = torch.ones(1)
                self.write_weights(directory, state, sharded=True)
                restored = KevPointerHead.from_pretrained(str(directory))
                with self.assertRaises(ValueError):
                    restored.load_pretrained(str(directory))


if __name__ == '__main__':
    unittest.main()
