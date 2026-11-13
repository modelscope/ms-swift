# Copyright (c) ModelScope Contributors. All rights reserved.
"""Portable lifecycle contracts; constructor fixtures do not run Flash kernels."""
import copy
import importlib.util
import os
import sys
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch  # noqa: E402
from torch import nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from test_ling3_compat import mock_modules
finally:
    sys.path.pop(0)

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ling3_rope_cpu', ROOT / 'swift/model/models/ling3_rope.py')
rope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rope)


class RopeTests(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = SimpleNamespace(
            hidden_size=1536,
            num_hidden_layers=24,
            qk_rope_head_dim=64,
            rope_theta=6000000,
            rope_scaling=None,
            rope_parameters={'rope_theta': 6000000},
            auto_map={'AutoModelForCausalLM': 'modeling_bailing_moe_v3.BailingMoeV3ForCausalLM'},
            tie_word_embeddings=False,
            max_position_embeddings=256,
            _attn_implementation='flash_attention_2')
        self.remote = ModuleType('_ling3_remote_cpu')

        class Base(nn.Module):
            _supports_flash_attn_2 = True
            _supports_flash_attn = False

        class Rotary(nn.Module):

            def __init__(self, config, device=None):
                super().__init__()
                self.register_buffer('inv_freq', torch.ones(32), persistent=False)

        class MLA(nn.Module):

            def __init__(self, config):
                super().__init__()
                self.config = config
                if isinstance(config.rope_scaling, dict):
                    assert config.rope_scaling['factor'] == 1.0

        class Model(Base):
            _tied_weights_keys = ['lm_head.weight']

            def __init__(self, config):
                super().__init__()
                self.rotary = Rotary(config)
                self.attention = MLA(config)

        Model.__module__ = self.remote.__name__
        self.remote.BailingMoeV3RotaryEmbedding = Rotary
        self.remote.BailingMoeV3MultiLatentAttention = MLA
        self.remote.BailingMoeV3PreTrainedModel = Base
        self.model_cls = Model
        self.base = Base
        self.rotary = Rotary
        self.originals = (Rotary.__init__, MLA.__init__, Base.__init__, Model._tied_weights_keys)
        tf = ModuleType('transformers')
        tf.__version__ = '5.14.1'
        dynamic = ModuleType('transformers.dynamic_module_utils')
        dynamic.get_class_from_dynamic_module = lambda *args, **kwargs: Model
        self.tf = tf
        modules = {'transformers': tf, dynamic.__name__: dynamic, self.remote.__name__: self.remote}
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock_modules(modules))

    def context(self):
        return rope.ling3_rope_construction(self.directory.name, self.config, {})

    def tearDown(self):
        self.assertIs(self.rotary.__init__, self.originals[0])
        self.assertIs(self.remote.BailingMoeV3MultiLatentAttention.__init__, self.originals[1])
        self.assertIs(self.base.__init__, self.originals[2])
        self.assertNotIn('__init__', self.base.__dict__)
        self.assertFalse(self.base._supports_flash_attn)
        self.assertIs(self.model_cls._tied_weights_keys, self.originals[3])

    def test_tf4_context_is_noop(self):
        self.tf.__version__ = '4.57.1'
        before = copy.deepcopy(vars(self.config))
        with self.context() as finalize:
            self.assertIsNone(finalize)
            self.assertIs(self.rotary.__init__, self.originals[0])
        self.assertEqual(vars(self.config), before)

    def test_instance_capability_and_nonpersistent_buffers(self):
        with self.context() as finalize:
            model = self.model_cls(self.config)
            finalize(model)
            self.assertTrue(model._supports_flash_attn)
            self.assertFalse(self.base._supports_flash_attn)
            self.assertEqual(model.rotary.state_dict(), {})
            self.assertFalse(list(model.parameters()))
        self.assertTrue(model._supports_flash_attn)

    def test_dtype_moves_and_meta_materialization(self):
        with self.context() as finalize:
            model = self.model_cls(self.config)
            finalize(model)
        reference = 1.0 / (6000000**(torch.arange(0, 64, 2).float() / 64))
        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            model.to(dtype=dtype)
            expected = reference.double() if dtype == torch.float64 else reference
            torch.testing.assert_close(model.rotary.inv_freq, expected, rtol=0, atol=0)
        model.to('meta')
        model.to_empty(device='cpu')
        torch.testing.assert_close(model.rotary.inv_freq, reference.double(), rtol=0, atol=0)
        self.assertEqual(model.rotary.state_dict(), {})

    def test_missing_factor_uses_leaf_copy(self):
        self.config.rope_scaling = {'rope_type': 'default'}
        with self.context() as finalize:
            model = self.model_cls(self.config)
            finalize(model)
        self.assertIs(model.attention.config, self.config)
        self.assertNotIn('factor', self.config.rope_scaling)

    def test_exception_restores_constructors_and_releases_lock(self):

        def failing_init(self, *args, **kwargs):
            raise RuntimeError('injected base construction failure')

        with patch.object(self.base, '__init__', failing_init):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with self.context():
                    self.model_cls(self.config)
            self.assertIs(self.base.__init__, failing_init)
        with self.context() as finalize:
            finalize(self.model_cls(self.config))

    def test_nested_and_foreign_thread_rejected(self):
        errors = []
        with self.context() as finalize:
            with self.assertRaisesRegex(RuntimeError, 'nested or concurrent'):
                with self.context():
                    pass

            def foreign():
                try:
                    self.model_cls(self.config)
                except RuntimeError as error:
                    errors.append(str(error))

            thread = threading.Thread(target=foreign)
            thread.start()
            thread.join()
            self.assertEqual(len(errors), 1)
            self.assertIn('controlled thread', errors[0])
            finalize(self.model_cls(self.config))

    def test_unsupported_version_scaling_and_flash_rejected(self):
        for field, value in (('_attn_implementation', 'flash_attention_3'), ('rope_scaling', {
                'rope_type': 'linear',
                'factor': 2.0
        })):
            before = getattr(self.config, field)
            setattr(self.config, field, value)
            with self.assertRaises(NotImplementedError):
                with self.context():
                    pass
            setattr(self.config, field, before)
        self.tf.__version__ = '5.20.0'
        with self.assertRaises(NotImplementedError):
            with self.context():
                pass

    def test_unverified_version_precedes_invalid_rope_parameters(self):
        self.tf.__version__ = '5.20.0'
        self.config.rope_scaling = 'invalid parameters must not mask the version error'
        with self.assertRaisesRegex(NotImplementedError, 'has only been verified with'):
            with self.context():
                self.fail('unverified version entered construction')
        self.assertEqual(self.originals, (self.rotary.__init__, self.remote.BailingMoeV3MultiLatentAttention.__init__,
                                          self.base.__init__, self.model_cls._tied_weights_keys))

    def test_legacy_flash_capability_required(self):
        with patch.object(self.base, '_supports_flash_attn_2', False):
            with self.assertRaisesRegex(ValueError, 'must declare'):
                with self.context():
                    pass

    def test_eager_does_not_install_capability_bridge(self):
        self.config._attn_implementation = 'eager'
        with self.context() as finalize:
            self.assertNotIn('__init__', self.base.__dict__)
            model = self.model_cls(self.config)
            finalize(model)
            self.assertFalse(model._supports_flash_attn)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
