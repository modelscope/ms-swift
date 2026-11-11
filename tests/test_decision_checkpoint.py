# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import pickle
import tempfile
import torch
import unittest
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from safetensors import SafetensorError
from safetensors.torch import save_file
from unittest.mock import patch

from swift.model.decision_head import (ClefJointSchemaHead, KevPointerHead, LayaHead, OmniJevHead, ScoringHead,
                                       Thx02ReadoutHead)

HEADS = [ClefJointSchemaHead, OmniJevHead, KevPointerHead, Thx02ReadoutHead]
FACTORY_ENV = [
    'CLEF_JOINT_HEAD_CONFIG', 'CLEF_JOINT_HEAD_WEIGHTS', 'OMNIJEV_HEAD_META', 'OMNIJEV_HEAD', 'OMNIJEV_ORD', 'KEV_HEAD',
    'THX02_READOUT', 'THX02_DECISION_CONFIG'
]


def make_head(head_cls, factory=False):
    torch.manual_seed(17 if factory else 23)
    if head_cls is ClefJointSchemaHead:
        return head_cls(hidden_size=8, width=12, routing_layers=1, layers=2, heads=3, feedforward=19, dropout=0.2)
    if head_cls is OmniJevHead:
        head = head_cls(
            hidden_size=8,
            head_hidden=16 if factory else 13,
            ord_hidden=8 if factory else 11,
            lm_feats=False,
            ordinal=True,
            temperatures={
                'noul': 0.75,
                'choice': 1.25,
                'score': 1.5
            },
            noul_bias=0.3,
            head_temperature=1.2)
        head.set_marker_lengths(2, 2)
        return head
    if head_cls is KevPointerHead:
        head = head_cls(hidden_size=8, pointer_dim=5)
        head.temperature.fill_(1.7)
        return head
    return head_cls(
        hidden_size=8,
        num_codes=5,
        temperature=0.7,
        head_temperature=1.4,
        temperatures={
            'noul': 0.8,
            'choice': 1.3,
            'score': 1.8
        })


def inputs(head):
    hidden = torch.linspace(-1., 1., 3 * 12 * 8).reshape(3, 12, 8)
    meta = {
        'n_options': [2, 3, 3],
        'kinds': [0, 1, 2],
        'record_index': [0, 1, 2],
        'counts': [1, 1, 1],
        'question_spans': [(0, 2)] * 3,
        'option_spans': [[(2, 4), (4, 6)], [(2, 4), (4, 6), (6, 8)], [(2, 4), (4, 6), (6, 8)]],
        'option_markers': [[(2, 4)], [(2, 4), (5, 7)], [(2, 4), (5, 7), (8, 10)]],
        'decide_idx': [1, 1, 1],
        'opt_idx': [[4, 7], [4, 7, 10], [4, 7, 10]],
        'readout_positions': [1, 1, 1],
        'input_ids': torch.arange(36).reshape(3, 12) % 16,
    }
    if head.needs_output_embeddings:
        embeddings = torch.nn.Linear(8, 16, bias=False)
        with torch.no_grad():
            embeddings.weight.copy_(torch.linspace(-0.2, 0.3, 16 * 8).reshape(16, 8))
        head.set_output_embeddings(embeddings)
    return hidden, meta


def save_checkpoint(head, directory, checkpoint_format):
    head.save_pretrained(str(directory), save_weights=checkpoint_format == 'adapter')
    if checkpoint_format == 'adapter':
        return
    state = {'scoring_head.' + key: value.detach() for key, value in head.state_dict().items()}
    if checkpoint_format == 'full':
        save_file(state, str(directory / 'model.safetensors'))
        return
    # Head tensors span two shards; an unrelated backbone shard need not be read.
    names = sorted(state)
    weight_map = {'backbone.weight': 'unrelated-backbone.safetensors'}
    for shard, keys in enumerate([names[::2], names[1::2]], 1):
        filename = f'model-{shard:05d}-of-00003.safetensors'
        save_file({key: state[key] for key in keys}, str(directory / filename))
        weight_map.update({key: filename for key in keys})
    (directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))


def write_factory(directory, head_cls):
    directory.mkdir(exist_ok=True)
    head = make_head(head_cls, factory=True)
    if head_cls is ClefJointSchemaHead:
        config = dict(hidden_size=8, width=12, routing_layers=1, layers=2, heads=3, feedforward=19, dropout=0.2)
        (directory / 'joint_head_config.json').write_text(json.dumps(config))
        save_file(head.head_state_dict(), str(directory / 'joint_head.safetensors'))
        artifact, env = 'joint_head.safetensors', 'CLEF_JOINT_HEAD_WEIGHTS'
    elif head_cls is OmniJevHead:
        config = {'norm': 'softmax', 'lm_feats': False, 'ordinal': True}
        (directory / 'head_meta.json').write_text(json.dumps(config))
        torch.save(head.head.state_dict(), directory / 'head.pt')
        torch.save(head.ord.state_dict(), directory / 'ord.pt')
        artifact, env = 'head.pt', 'OMNIJEV_HEAD'
    elif head_cls is KevPointerHead:
        torch.save({'head': head.head_state_dict(), 'temperature': 1.7}, directory / 'head.pt')
        artifact, env = 'head.pt', 'KEV_HEAD'
    else:
        config = {'attention_mode': 'noncausal_full_attention', 'codes': ['A', 'B', 'C', 'D', 'E'], 'temperature': 0.7}
        (directory / 'decision_config.json').write_text(json.dumps(config))
        save_file(head.readout.state_dict(), str(directory / 'readout.safetensors'))
        artifact, env = 'readout.safetensors', 'THX02_READOUT'
    return head, artifact, env


class TestDecisionCheckpoint(unittest.TestCase):

    @contextmanager
    def _checkpoint(self):
        # Each parameter combination starts with a clean factory environment and directory.
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            for name in FACTORY_ENV:
                os.environ.pop(name, None)
            yield Path(directory)

    def test_decision_checkpoint_restores_trained_scores(self):
        for head_cls, checkpoint_format in product(HEADS, ['adapter', 'full', 'sharded']):
            with self.subTest(
                    head_cls=head_cls.__name__, checkpoint_format=checkpoint_format), self._checkpoint() as tmp_path:
                head = make_head(head_cls)
                hidden, meta = inputs(head)
                head.eval()
                initial = head(hidden, None, meta).logits.detach().clone()
                # Exercise genuine trained parameters, not just deterministic initialization.
                optimizer = torch.optim.SGD(
                    [p for name, p in head.named_parameters() if not name.startswith('_output_embeddings')], lr=0.02)
                output = head(hidden, None, meta)
                output.logits[output.option_mask].square().mean().backward()
                optimizer.step()
                expected = head(hidden, None, meta)
                assert not torch.equal(initial, expected.logits)
                save_checkpoint(head, tmp_path, checkpoint_format)
                assert not any(key.startswith('_output_embeddings') for key in head.head_state_dict())
                restored = head_cls.from_pretrained(str(tmp_path))
                assert restored.load_pretrained(str(tmp_path))
                restored.eval()
                inputs(restored)
                assert restored.extra_state_for_save() == head.extra_state_for_save()
                for key, value in head.head_state_dict().items():
                    torch.testing.assert_close(restored.head_state_dict()[key], value, rtol=0, atol=0)
                actual = restored(hidden, None, meta)
                for name in ('logits', 'option_mask', 'kinds', 'counts'):
                    torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)

    def test_complete_metadata_without_type_tag_is_readable(self):
        for head_cls in HEADS:
            with self.subTest(head_cls=head_cls.__name__), self._checkpoint() as tmp_path:
                head = make_head(head_cls)
                head.save_pretrained(str(tmp_path))
                # Existing checkpoints did not have a discriminator. Complete old metadata remains valid.
                metadata = head.extra_state_for_save()
                (tmp_path / 'scoring_head.json').write_text(json.dumps(metadata))
                restored = head_cls.from_pretrained(str(tmp_path))
                assert restored.load_pretrained(str(tmp_path))
                assert restored.extra_state_for_save() == metadata

    def test_saved_metadata_rejects_wrong_head_type(self):
        for head_cls in HEADS:
            with self.subTest(head_cls=head_cls.__name__), self._checkpoint() as tmp_path:
                make_head(head_cls).save_pretrained(str(tmp_path))
                path = tmp_path / 'scoring_head.json'
                metadata = json.loads(path.read_text())
                metadata['head_type'] = 'jev'
                path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, 'type|Type'):
                    head_cls.from_pretrained(str(tmp_path))

    def test_checkpoint_overlay_rejects_wrong_head_type(self):
        for head_cls in HEADS:
            with self.subTest(head_cls=head_cls.__name__), self._checkpoint() as tmp_path:
                head = make_head(head_cls)
                head.save_pretrained(str(tmp_path))
                path = tmp_path / 'scoring_head.json'
                metadata = json.loads(path.read_text())
                metadata['head_type'] = 'jev'
                path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, 'type|Type'):
                    head.load_pretrained(str(tmp_path))

    def test_factory_weights_precede_saved_metadata(self):
        for head_cls, override in product(HEADS, [False, True]):
            with self.subTest(head_cls=head_cls.__name__, override=override), self._checkpoint() as tmp_path:
                make_head(head_cls).save_pretrained(str(tmp_path))
                factory, artifact, env = write_factory(tmp_path, head_cls)
                if override:
                    alternate, _, _ = write_factory(tmp_path / 'override', head_cls)
                    with torch.no_grad():
                        for parameter in alternate.parameters():
                            parameter.add_(0.125)
                    if head_cls is OmniJevHead:
                        torch.save(alternate.head.state_dict(), tmp_path / 'override' / artifact)
                        factory.head.load_state_dict(alternate.head.state_dict())
                    elif head_cls is KevPointerHead:
                        torch.save({
                            'head': alternate.head_state_dict(),
                            'temperature': 1.7
                        }, tmp_path / 'override' / artifact)
                        factory = alternate
                    elif head_cls is Thx02ReadoutHead:
                        save_file(alternate.readout.state_dict(), str(tmp_path / 'override' / artifact))
                        factory = alternate
                    else:
                        save_file(alternate.head_state_dict(), str(tmp_path / 'override' / artifact))
                        factory = alternate
                    os.environ[env] = str(tmp_path / 'override' / artifact)
                restored = head_cls.from_pretrained(str(tmp_path))
                for key, value in factory.head_state_dict().items():
                    torch.testing.assert_close(restored.head_state_dict()[key], value, rtol=0, atol=0)

    def test_broken_factory_is_not_hidden_by_metadata(self):
        for head_cls in HEADS:
            with self.subTest(head_cls=head_cls.__name__), self._checkpoint() as tmp_path:
                make_head(head_cls).save_pretrained(str(tmp_path))
                _, artifact, _ = write_factory(tmp_path, head_cls)
                (tmp_path / artifact).write_bytes(b'not a checkpoint')
                with self.assertRaises((SafetensorError, pickle.UnpicklingError)):
                    head_cls.from_pretrained(str(tmp_path))

    def test_legacy_calibration_metadata_with_factory(self):
        for head_cls in [OmniJevHead, KevPointerHead]:
            with self.subTest(head_cls=head_cls.__name__), self._checkpoint() as tmp_path:
                for name in FACTORY_ENV:
                    os.environ.pop(name, None)
                head, _, _ = write_factory(tmp_path, head_cls)
                head.save_pretrained(str(tmp_path))
                if head_cls is KevPointerHead:
                    legacy = {'temperature': 2.25}
                else:
                    legacy = {
                        'head_temperature': 1.75,
                        'temperatures': {
                            'noul': 0.7,
                            'choice': 1.2,
                            'score': 1.6
                        },
                        'noul_bias': 0.4
                    }
                (tmp_path / 'scoring_head.json').write_text(json.dumps(legacy))
                head._apply_calibration_meta(legacy)
                restored = head_cls.from_pretrained(str(tmp_path))
                assert restored.load_pretrained(str(tmp_path))
                hidden, meta = inputs(head)
                inputs(restored)
                head.eval()
                restored.eval()
                torch.testing.assert_close(
                    restored(hidden, None, meta).logits, head(hidden, None, meta).logits, rtol=0, atol=0)

    def test_laya_checkpoint_constructor_contract(self):
        for head_layers, legacy in product([0, 2], [False, True]):
            with self.subTest(head_layers=head_layers, legacy=legacy), self._checkpoint() as tmp_path:
                os.environ.pop('LAYA_CONFIG', None)
                head = LayaHead(
                    hidden_size=8,
                    head_layers=head_layers,
                    dropout=0.1 if legacy else 0.2,
                    temperatures=[0.7, 1.3, 1.7],
                    temperature_by_options={'choice:3-5': 0.6})
                head.save_pretrained(str(tmp_path))
                if legacy:
                    (tmp_path / 'scoring_head.json').write_text(
                        json.dumps({
                            'hidden_size': 8,
                            'head_layers': head_layers,
                            'temperature': head.temperature.tolist(),
                            'temperature_by_options': head.temperature_by_options,
                        }))
                restored = LayaHead.from_pretrained(str(tmp_path))
                assert restored.load_pretrained(str(tmp_path))
                assert restored.extra_state_for_save() == head.extra_state_for_save()
                head.eval()
                restored.eval()
                hidden = torch.linspace(-1., 1., 2 * 6 * 8).reshape(2, 6, 8)
                meta = {
                    'n_options': [2, 3],
                    'kinds': [0, 1],
                    'record_index': [0, 1],
                    'counts': [1, 1],
                    'laya_markers': [[1, 3], [1, 3, 5]]
                }
                torch.testing.assert_close(
                    restored(hidden, None, meta).logits, head(hidden, None, meta).logits, rtol=0, atol=0)

    def test_new_head_inherits_checkpoint_reconstruction(self):
        with self._checkpoint() as tmp_path:

            class NewHead(ScoringHead):
                head_type = 'checkpoint-contract-test'

                def __init__(self, gain):
                    super().__init__()
                    self.gain = gain
                    self.bias = torch.nn.Parameter(torch.tensor([0.25]))

                def extra_state_for_save(self):
                    return {'gain': self.gain}

            head = NewHead(gain=1.75)
            head.save_pretrained(str(tmp_path))
            restored = NewHead.from_pretrained(str(tmp_path))
            assert restored.load_pretrained(str(tmp_path))
            assert restored.gain == 1.75
            torch.testing.assert_close(restored.bias, head.bias, rtol=0, atol=0)

    def test_backbone_embedding_keeps_canonical_checkpoint_key(self):
        for head_cls, prefix in product([ClefJointSchemaHead, OmniJevHead], ['', 'wrapped.']):
            with self.subTest(head_cls=head_cls.__name__, prefix=prefix), self._checkpoint() as tmp_path:
                head = make_head(head_cls)
                hidden, meta = inputs(head)
                if head_cls is OmniJevHead:
                    head.lm_feats = True
                    head.set_marker_lengths(1, 1)
                model = torch.nn.Module()
                model.lm_head = head._output_embeddings
                model.scoring_head = head
                assert head._modules['_output_embeddings'] is model.lm_head
                assert head._output_embeddings is model.lm_head
                state = model.state_dict(prefix=prefix)
                assert prefix + 'lm_head.weight' in state
                assert not any('_output_embeddings' in key for key in state)
                assert not any('_output_embeddings' in key for key in head.head_state_dict())
                # Safetensors must see one owner for the shared tensor and save it successfully.
                save_file(state, str(tmp_path / 'model.safetensors'))
                head.eval()
                output = head(hidden, None, meta)
                output.logits[output.option_mask].square().mean().backward()
                assert model.lm_head.weight.grad is not None
                assert torch.isfinite(model.lm_head.weight.grad).all()
                assert model.lm_head.weight.grad.abs().sum() > 0

    def test_reconstructed_head_requires_checkpoint_weights(self):
        for head_cls in HEADS:
            with self.subTest(head_cls=head_cls.__name__), self._checkpoint() as tmp_path:
                make_head(head_cls).save_pretrained(str(tmp_path), save_weights=False)
                restored = head_cls.from_pretrained(str(tmp_path))
                with self.assertRaisesRegex(FileNotFoundError, 'weights'):
                    restored.load_pretrained(str(tmp_path))

    def test_initialized_head_accepts_calibration_only_overlay(self):
        for use_factory in [False, True]:
            with self.subTest(use_factory=use_factory), self._checkpoint() as tmp_path:
                os.environ.pop('KEV_HEAD', None)
                if use_factory:
                    write_factory(tmp_path / 'factory', KevPointerHead)
                    head = KevPointerHead.from_pretrained(str(tmp_path / 'factory'))
                else:
                    head = make_head(KevPointerHead)
                original_weights = {key: value.clone() for key, value in head.head_state_dict().items()}
                calibrated = make_head(KevPointerHead)
                calibrated.temperature.fill_(2.5)
                calibrated.save_pretrained(str(tmp_path / 'calibration'), save_weights=False)
                assert head.load_pretrained(str(tmp_path / 'calibration'))
                assert float(head.temperature) == 2.5
                for key, value in original_weights.items():
                    torch.testing.assert_close(head.head_state_dict()[key], value, rtol=0, atol=0)

    def test_legacy_embedding_alias_requires_migration(self):
        for head_cls, checkpoint_format, has_canonical_key in product([ClefJointSchemaHead, OmniJevHead],
                                                                      ['full', 'sharded'], [False, True]):
            with self.subTest(
                    head_cls=head_cls.__name__, checkpoint_format=checkpoint_format,
                    has_canonical_key=has_canonical_key), self._checkpoint() as tmp_path:
                head = make_head(head_cls)
                inputs(head)
                head.save_pretrained(str(tmp_path), save_weights=False)
                state = {'scoring_head.' + key: value for key, value in head.head_state_dict().items()}
                state['scoring_head._output_embeddings.weight'] = head._output_embeddings.weight.detach().clone()
                if has_canonical_key:
                    state['lm_head.weight'] = head._output_embeddings.weight.detach().clone()
                filename = 'model.safetensors' if checkpoint_format == 'full' else 'model-00001-of-00001.safetensors'
                save_file(state, str(tmp_path / filename))
                if checkpoint_format == 'sharded':
                    (tmp_path / 'model.safetensors.index.json').write_text(
                        json.dumps({'weight_map': {
                            key: filename
                            for key in state
                        }}))
                # Dropping only the alias is unsafe: older safetensors saves could have deduplicated
                # away the canonical backbone key, leaving that embedding randomly initialized.
                with self.assertRaisesRegex(ValueError, '(?i)legacy.*migrate'):
                    head.load_pretrained(str(tmp_path))

    def test_adapter_embedding_alias_requires_migration(self):
        for head_cls in [ClefJointSchemaHead, OmniJevHead]:
            with self.subTest(head_cls=head_cls.__name__), self._checkpoint() as tmp_path:
                head = make_head(head_cls)
                inputs(head)
                head.save_pretrained(str(tmp_path))
                state = head.head_state_dict()
                state['_output_embeddings.weight'] = head._output_embeddings.weight.detach().clone()
                save_file(state, str(tmp_path / 'scoring_head.safetensors'))
                with self.assertRaisesRegex(ValueError, '(?i)legacy.*migrate'):
                    head.load_pretrained(str(tmp_path))


if __name__ == '__main__':
    unittest.main()
