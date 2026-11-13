# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU loader and adapter lifecycle contracts; spies do not run FLA kernels."""
import ast
import importlib.util
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_test(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'tests/models/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import shared CPU fixtures normally; do not recursively execute another suite.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import test_ling3_compat as base
finally:
    sys.path.pop(0)

mock_modules = base.mock_modules
compat = base.compat
apply_rotary_pos_emb_interleave = base.apply_rotary_pos_emb_interleave
chunk_kda = base.chunk_kda


class Shell(nn.Module):

    def forward(self, input_ids=None, position_ids=None, **kwargs):
        return input_ids


def shell():
    model = Shell()
    model.config = SimpleNamespace(
        architectures=['BailingMoeV3ForCausalLM'],
        num_nextn_predict_layers=0,
        hidden_size=1536,
        num_hidden_layers=24,
        q_lora_rank=256,
        rope_scaling=None,
        use_cache=False,
        _attn_implementation='flash_attention_2')
    kda, _, _ = base.kda()
    model.model = SimpleNamespace(layers=[
        SimpleNamespace(attention=kda, attention_layer_type='linear_attention'),
        SimpleNamespace(attention=base.MLA(), attention_layer_type='attention')
    ])

    return model


def dependencies(version='4.57.1', missing=False, bad_conv=False):
    names = ('swift', 'swift.model', 'swift.model.models', 'swift.model.patcher', 'transformers',
             'transformers.modeling_utils', 'transformers.utils', 'transformers.utils.import_utils', 'fla',
             'fla.modules', 'fla.modules.conv')
    modules = {name: ModuleType(name) for name in names}
    tree = ast.parse((ROOT / 'swift/model/patcher.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'patch_module_forward')
    ns = {'MethodType': MethodType}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<Swift patcher>', 'exec'), ns)
    modules['swift.model.patcher'].patch_module_forward = ns['patch_module_forward']
    modules['swift.model.models.ling3_compat'] = compat
    rope = ModuleType('swift.model.models.ling3_rope')
    rope.SUPPORTED_TRANSFORMERS = ('4.57.1', '5.14.1')
    rope.ling3_rope_construction = lambda *args: nullcontext(None)
    modules[rope.__name__] = rope
    modules['transformers'].__version__ = version
    modules['transformers.modeling_utils'].ALL_ATTENTION_FUNCTIONS = {'sdpa': lambda *args, **kwargs: None}
    modules['transformers.utils.import_utils'].is_torch_fx_available = lambda: True
    if not missing:
        conv = ModuleType('fla.modules.conv.causal_conv1d')
        conv.causal_conv1d = (lambda x: x) if bad_conv else (lambda **kwargs: (kwargs['x'], None))
        modules[conv.__name__] = conv
    return modules


def loader(model):

    class ModelLoader:

        def get_model(self, *args):
            return model

    tree = ast.parse((ROOT / 'swift/model/models/llm.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BailingHybridLoader')
    messages = []
    ns = dict(
        ModelLoader=ModelLoader,
        PretrainedConfig=object,
        Processor=object,
        Any=object,
        logger=SimpleNamespace(warning=messages.append),
        __package__='swift.model.models')
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual BailingHybridLoader>', 'exec'), ns)
    return ns['BailingHybridLoader'](), messages


def unmodified(model):
    attentions = [layer.attention for layer in model.model.layers]
    convs = [getattr(attentions[0], name) for name in ('q_conv1d', 'k_conv1d', 'v_conv1d')]
    return [(dict(module.__dict__), getattr(module, 'forward', None))
            for module in [model, model.model, *attentions, *convs]]


class LoaderAndAdapterTests(unittest.TestCase):

    def test_eager_future_mask_guard(self):
        x = torch.zeros(1, 3, 4)
        noncausal = torch.zeros(1, 1, 3, 3)
        with self.assertRaisesRegex(ValueError, 'future token'):
            compat._validate_eager_causal_mask(x, noncausal)
        causal = noncausal.masked_fill(torch.ones(3, 3, dtype=torch.bool).triu(1), float('-inf'))
        compat._validate_eager_causal_mask(x, causal)
        nan_mask = causal.clone()
        nan_mask[..., 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            compat._validate_eager_causal_mask(x, nan_mask)
        for mask in (torch.ones(1, 3), torch.ones(1, 1, 3, 3, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                compat._validate_eager_causal_mask(x, mask)
        q = torch.ones(1, 1, 3, 2)
        for mask, leaks in ((noncausal, True), (causal, False)):
            v = torch.randn(1, 1, 3, 2, requires_grad=True)
            y = torch.nn.functional.scaled_dot_product_attention(q, q, v, attn_mask=mask)
            grad = torch.autograd.grad(y[:, :, 0].sum(), v)[0]
            self.assertEqual(bool(grad[:, :, 1:].abs().sum() > 0), leaks)

    def test_registry_failure_preserves_instance(self):
        modules = dependencies()
        del modules['transformers.modeling_utils'].ALL_ATTENTION_FUNCTIONS
        with mock_modules(modules):
            model = shell()
            before = unmodified(model)
            with self.assertRaisesRegex(RuntimeError, 'registry'):
                compat.install_ling3_compat(model)
            self.assertEqual(before, unmodified(model))

    def test_patch_failure_rolls_back_all_instances(self):
        modules = dependencies()
        original = modules['swift.model.patcher'].patch_module_forward
        count = []

        def failing(module, forward):
            original(module, forward)
            count.append(module)
            if len(count) == 2:
                raise RuntimeError('injected patch failure')

        modules['swift.model.patcher'].patch_module_forward = failing
        with mock_modules(modules):
            model = shell()
            before = unmodified(model)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                compat.install_ling3_compat(model)
            self.assertEqual(before, unmodified(model))

    def test_supported_loader_install_idempotence_and_parameter_identity(self):
        model = shell()
        parameters = [(name, id(p), p.shape) for layer in model.model.layers
                      for name, p in layer.attention.named_parameters()]
        with mock_modules(dependencies()):
            instance, messages = loader(model)
            self.assertIs(instance.get_model('.', model.config, None, {}), model)
            forward = model.model.layers[0].attention.forward
            compat.install_ling3_compat(model)
            self.assertIs(model.model.layers[0].attention.forward, forward)
        self.assertFalse(messages)
        self.assertTrue(model.model._supports_packing_metadata)
        self.assertEqual(parameters, [(name, id(p), p.shape) for layer in model.model.layers
                                      for name, p in layer.attention.named_parameters()])
        self.assertFalse(hasattr(model.model.layers[0].attention, '_swift_ling3_origin_forward'))

    def test_certified_versions_install_and_dispatch(self):
        # Contract migration: 5.14.1 is certified only with the required APIs.
        for version in ('4.57.1', '5.14.1'):
            with self.subTest(version=version), mock_modules(dependencies(version=version)):
                model = shell()
                attention = model.model.layers[0].attention
                original = attention.forward
                parameters = [id(p) for p in attention.parameters()]
                self.assertFalse(getattr(model, '_swift_ling3_compat', False))
                self.assertIsNone(compat.install_ling3_compat(model))
                self.assertIsNot(attention.forward, original)
                self.assertTrue(model.model._supports_packing_metadata)
                x = torch.randn(1, 3, 4, requires_grad=True)
                output = attention(x, position_ids=torch.arange(3)[None])[0]
                output.sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(attention.q_conv1d.calls)
                self.assertEqual(parameters, [id(p) for p in attention.parameters()])
                installed = attention.forward
                compat.install_ling3_compat(model)
                self.assertIs(attention.forward, installed)

    def test_required_adapter_failure_stops_tiny_loading(self):
        for options in ({'missing': True}, {'bad_conv': True}, {'version': '5.20.0'}):
            with self.subTest(options=options), mock_modules(dependencies(**options)):
                model = shell()
                before = unmodified(model)
                instance, messages = loader(model)
                with self.assertRaises(RuntimeError):
                    instance.get_model('.', model.config, None, {})
                self.assertEqual(before, unmodified(model))
                self.assertFalse(messages)

    def test_preflight_failure_has_no_partial_mutation(self):
        for options in ({'missing': True}, {'bad_conv': True}, {'version': '5.20.0'}):
            with self.subTest(options=options), mock_modules(dependencies(**options)):
                model = shell()
                before = unmodified(model)
                with self.assertRaises(RuntimeError):
                    compat.install_ling3_compat(model)
                self.assertEqual(before, unmodified(model))
        with mock_modules(dependencies()), patch.dict(globals(), {'chunk_kda': lambda q: q}):
            model = shell()
            before = unmodified(model)
            with self.assertRaises(RuntimeError):
                compat.install_ling3_compat(model)
            self.assertEqual(before, unmodified(model))
        with mock_modules(dependencies()):
            model = shell()
            model.model.layers[0].attention.v_conv1d._patched = True
            before = unmodified(model)
            with self.assertRaisesRegex(ValueError, 'conflicting patch'):
                compat.install_ling3_compat(model)
            self.assertEqual(before, unmodified(model))

    def test_other_models_and_configurations_are_not_adapted(self):
        for architecture, width in (('BailingMoeV2_5ForCausalLM', 1536), ('BailingMoeV2ForCausalLM', 1536),
                                    ('BailingMoeV3ForCausalLM', 2560)):
            with self.subTest(architecture=architecture), mock_modules(dependencies()):
                model = shell()
                model.config.architectures = [architecture]
                model.config.hidden_size = width
                instance, messages = loader(model)
                with patch(
                        'swift.model.models.ling3_rope.ling3_rope_construction',
                        side_effect=AssertionError('non-target entered tiny construction')):
                    instance.get_model('.', model.config, None, {})
                self.assertFalse(getattr(model, '_swift_ling3_compat', False))
                self.assertFalse(hasattr(model, '_swift_ling3_packing_handle'))
                self.assertFalse(messages)

    def test_mtp_preserves_loading_without_claiming_compatibility(self):
        for version in ('4.57.1', '5.14.1'):
            with self.subTest(version=version), mock_modules(dependencies(version=version)):
                model = shell()
                model.config.num_nextn_predict_layers = 1
                instance, _ = loader(model)
                # Cache extension is unchanged; this fixture tests Loader routing.
                extensions = []
                instance._patch_dynamic_cache_auto_extend = lambda: extensions.append(True)
                self.assertIs(instance.get_model('.', model.config, None, {}), model)
                self.assertEqual(extensions, [True])
                self.assertFalse(getattr(model, '_swift_ling3_compat', False))
                self.assertFalse(getattr(model.model, '_supports_packing_metadata', False))
                inputs = torch.ones(1, 4, dtype=torch.long)
                self.assertIs(model(input_ids=inputs, position_ids=torch.arange(4)[None]), inputs)
                with self.assertRaisesRegex(NotImplementedError, 'MTP'):
                    compat.install_ling3_compat(model)
                model.model._supports_packing_metadata = True
                positions = torch.tensor([[0, 1, 0, 1]])
                with mock_modules({'swift.utils': SimpleNamespace(get_packed_seq_params=base.helper)}):
                    with self.assertRaisesRegex(ValueError, 'MTP'):
                        model(input_ids=inputs, position_ids=positions, **base.helper(positions))

    def test_padding_cache_semantics_and_unpadded_eval_delegation(self):
        for config_cache, explicit in ((True, False), (False, None), (True, None), (False, True)):
            model, kernel, _ = base.kda()
            model.eval()
            model.config = SimpleNamespace(use_cache=config_cache)
            mask, positions = base.padded('right')
            kwargs = dict(
                attention_mask=mask, position_ids=positions, use_cache=explicit, original=None, chunk_kda=kernel)
            if config_cache if explicit is None else explicit:
                with self.assertRaisesRegex(ValueError, 'use_cache=False'):
                    compat._kda_forward(model, torch.randn(2, 5, 4), **kwargs)
            else:
                compat._kda_forward(model, torch.randn(2, 5, 4), **kwargs)
        with self.assertRaisesRegex(ValueError, 'historical cache'):
            compat._kda_forward(model, torch.randn(2, 5, 4), past_key_value=(), **dict(kwargs, use_cache=False))
        delegated = []
        compat._kda_forward(
            model,
            torch.randn(1, 5, 4),
            use_cache=False,
            original=lambda *args, **kw: delegated.append(kw),
            chunk_kda=kernel)
        self.assertEqual(len(delegated), 1)

    @unittest.skipUnless(os.environ.get('LING3_MODEL_SOURCE'), 'requires official source, no weights')
    def test_packing_module_isolation_in_shared_process(self):
        sentinel = ModuleType('swift.utils')
        with mock_modules({'swift.utils': sentinel}):
            module = load_test('test_ling3_packing')
            self.assertIs(sys.modules['swift.utils'], sentinel)
            module.setUpModule()
            try:
                self.assertIs(sys.modules['swift.utils'], module.utils)
                raise RuntimeError('expected fixture interruption')
            except RuntimeError:
                pass
            finally:
                module.tearDownModule()
            self.assertIs(sys.modules['swift.utils'], sentinel)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
