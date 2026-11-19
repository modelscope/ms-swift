# Copyright (c) ModelScope Contributors. All rights reserved.
"""Portable CPU adapter contracts. Kernel spies do not validate FLA kernels."""
import ast
import copy
import importlib.util
import os
import sys
import typing
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {'TORCH_DEVICE_BACKEND_AUTOLOAD': '0'}):
    import torch
import torch.nn.functional as F  # noqa: E402
from ling3_remote_forward_fixture import remote_forward  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('swift.model.models._ling3_adapter_cpu',
                                              ROOT / 'swift/model/models/ling3_compat.py')
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)
tree = ast.parse((ROOT / 'swift/utils/transformers_utils.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_packed_seq_params')
namespace = {**vars(typing), 'torch': torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<Swift metadata helper>', 'exec'), namespace)
helper = namespace['get_packed_seq_params']


def apply_rotary_pos_emb_interleave(q, k, c, s):
    return q, k


def chunk_kda(**kwargs):
    return kwargs['v'], None


def setUpModule():
    unittest.addModuleCleanup(torch.set_rng_state, torch.get_rng_state())


class ConvSpy(nn.Module):

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, cu_seqlens=None, **kwargs):
        self.calls.append(cu_seqlens)
        return x, None


def kda():
    m = nn.Module()
    m.training = True
    m.mode = 'chunk'
    m.no_kda_lora = True
    m.head_dim = m.head_k_dim = 2
    m.safe_gate = True
    m.lower_bound = -5.
    for name in ('q_proj', 'k_proj', 'v_proj', 'f_proj', 'g_proj'):
        setattr(m, name, nn.Linear(4, 4, bias=False))
    for name in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
        setattr(m, name, ConvSpy())
    m.b_proj = nn.Linear(4, 2, bias=False)
    m.A_log = nn.Parameter(torch.zeros(2))
    m.dt_bias = nn.Parameter(torch.zeros(4))
    m.o_norm = lambda x, gate: x * torch.sigmoid(gate)
    m.o_proj = nn.Linear(4, 4, bias=False)
    calls = []

    def chunk(**kw):
        calls.append(kw)
        # Differentiable segmented stub for boundary and gradient checks.
        z = (kw['v'] + kw['q'] * kw['k'] + kw['g']) * kw['beta'].unsqueeze(-1)
        cu = kw.get('cu_seqlens')
        bounds = [0, z.shape[1]] if cu is None else cu.tolist()
        return torch.cat([z[:, a:b].cumsum(1) for a, b in zip(bounds[:-1], bounds[1:])], 1), None

    return m, chunk, calls


def padded(side, lengths=(3, 5)):
    mask = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
    if side == 'left':
        mask = mask.flip(-1)
    pos = (mask.long().cumsum(-1) - 1).clamp_min(0)
    return mask.long(), pos


class MLA(nn.Module):

    def __init__(self, backend='sdpa'):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=backend, rope_interleave=True)
        self.q_lora_rank = None
        self.qk_head_dim = 4
        self.qk_nope_head_dim = 2
        self.v_head_dim = 2
        self.qk_rope_head_dim = 2
        self.kv_lora_rank = 4
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(8, 6, bias=False)
        self.kv_a_layernorm = nn.Identity()
        self.kv_b_proj = nn.Linear(4, 8, bias=False)
        self.num_heads = 2
        self.is_causal = True
        self.scaling = .5
        self.attention_dropout = 0.
        self.g_proj = None
        self.dense = nn.Linear(4, 8, bias=False)
        self._old_forward = MethodType(remote_forward, self)

        self._adapted_forward = MethodType(
            compat._make_mla_forward(self._old_forward, {'sdpa': self.cpu_attention}), self)

    @staticmethod
    def cpu_attention(module, q, k, v, attention_mask, **kw):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=attention_mask is None, scale=kw['scaling']).transpose(1,
                                                                                                                2), None

    def forward(self, x, mask=None, **kwargs):
        return self._adapted_forward(x, (None, None), mask, **kwargs)


@contextmanager
def mock_modules(modules):
    """Restore only mocked names; keep unrelated imports and Torch registrations."""
    missing = object()
    originals = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class AdapterTests(unittest.TestCase):

    def test_module_overrides_preserve_unrelated_imports(self):
        name, imported_name = '_ling3_mocked_module', '_ling3_imported_module'
        original, replacement, imported = ModuleType(name), ModuleType(name), ModuleType(imported_name)
        self.addCleanup(sys.modules.pop, imported_name, None)
        with mock_modules({name: original}):
            with mock_modules({name: replacement}):
                sys.modules[imported_name] = imported
            self.assertIs(sys.modules[name], original)
            self.assertIs(sys.modules[imported_name], imported)

    def test_mla_cache_prefill_and_single_query_match_full_sequence(self):

        class Cache:

            def __init__(self):
                self.key = self.value = None

            def update(self, key, value, layer_idx, cache_kwargs):
                self.key = key if self.key is None else torch.cat((self.key, key), dim=2)
                self.value = value if self.value is None else torch.cat((self.value, value), dim=2)
                return self.key, self.value

        torch.manual_seed(11)
        module = MLA().eval()
        module.layer_idx = 0
        x = torch.randn(1, 4, 8)
        full = module(x, position_ids=torch.arange(4)[None])[0]
        cache = Cache()
        prefix = module(x[:, :3], past_key_values=cache, position_ids=torch.arange(3)[None])[0]
        last = module(
            x[:, 3:],
            torch.zeros(1, 1, 1, 4),
            past_key_values=cache,
            cache_position=torch.tensor([3]),
            position_ids=torch.tensor([[3]]))[0]
        torch.testing.assert_close(prefix, full[:, :3])
        torch.testing.assert_close(last, full[:, 3:])
        self.assertEqual(cache.key.shape[2], 4)

    def test_padding_valid_outputs_and_pad_input_gradients(self):
        for training, side in ((True, 'left'), (True, 'right'), (False, 'left'), (False, 'right')):
            torch.manual_seed(4)
            m, kernel, calls = kda()
            m.train(training)
            mask, pos = padded(side)
            x = torch.randn(2, 5, 4, requires_grad=training)
            y = compat._kda_forward(m, x, attention_mask=mask, position_ids=pos, original=None, chunk_kda=kernel)[0]
            refs = []
            for row in range(2):
                one = x[row, mask[row].bool()][None]
                cu = torch.tensor([0, one.shape[1]])
                refs.append(compat._kda_chunk_forward(m, one, cu, chunk_kda=kernel)[0][0])
            torch.testing.assert_close(y[mask.bool()], torch.cat(refs))
            self.assertTrue(torch.equal(calls[0]['cu_seqlens'], torch.tensor([0, 3, 8])))
            self.assertTrue(calls[0]['use_gate_in_kernel'])
            self.assertTrue(calls[0]['use_qk_l2norm_in_kernel'])
            self.assertIsNone(calls[0]['initial_state'])
            if training:
                y[mask.bool()].sum().backward()
                self.assertEqual(int(torch.count_nonzero(x.grad[~mask.bool()])), 0)
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_padding_ignores_pad_content(self):
        m, kernel, _ = kda()
        for side in ('left', 'right'):
            mask, pos = padded(side)
            x = torch.randn(2, 5, 4)
            alt = x.clone()
            alt[~mask.bool()] = 100

            def f(t):
                return compat._kda_forward(
                    m, t, attention_mask=mask, position_ids=pos, original=None, chunk_kda=kernel)[0]

            self.assertTrue(torch.equal(f(x)[mask.bool()], f(alt)[mask.bool()]))

    def test_padding_singletons_backward_and_boundary_identity(self):
        m, kernel, calls = kda()
        mask, pos = padded('left', (1, 2))
        x = torch.randn(2, 2, 4, requires_grad=True)
        compat._kda_forward(
            m, x, attention_mask=mask, position_ids=pos, original=None, chunk_kda=kernel)[0].sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for conv in (m.q_conv1d, m.k_conv1d, m.v_conv1d):
            self.assertIs(conv.calls[0], calls[0]['cu_seqlens'])

    def test_padding_rejects_invalid_contracts(self):
        for mutation in ('internal', 'empty', 'binary', 'positions', 'cache', 'mode', 'packing'):
            with self.subTest(mutation=mutation):
                m, kernel, _ = kda()
                mask, pos = padded('right')
                kw = {}
                if mutation == 'internal':
                    mask[0] = torch.tensor([1, 0, 1, 0, 0])
                if mutation == 'empty':
                    mask[0] = 0
                if mutation == 'binary':
                    mask[0, 0] = 2
                if mutation == 'positions':
                    pos[0, 1] = 7
                if mutation == 'cache':
                    kw['past_key_value'] = object()
                if mutation == 'mode':
                    m.mode = 'fused_recurrent'
                if mutation == 'packing':
                    kw['cu_seqlens'] = torch.tensor([0, 8])
                with self.assertRaises(ValueError):
                    compat._kda_forward(
                        m,
                        torch.randn(2, 5, 4),
                        attention_mask=mask,
                        position_ids=pos,
                        original=None,
                        chunk_kda=kernel,
                        **kw)

    def test_right_broadcast_positions_and_left_rejection(self):
        m, kernel, _ = kda()
        for side in ('right', 'left'):
            mask, _ = padded(side)

            def f():
                return compat._kda_forward(
                    m,
                    torch.randn(2, 5, 4),
                    attention_mask=mask,
                    position_ids=torch.arange(5)[None],
                    original=None,
                    chunk_kda=kernel)

            if side == 'right':
                f()
            else:
                with self.assertRaises(ValueError):
                    f()

    def test_no_pad_dispatch_preserved(self):
        for training, length, cache, delegate in ((True, 1, None, False), (True, 64, None, False),
                                                  (True, 65, None, True), (False, 64, None, True), (True, 64, object(),
                                                                                                    True)):
            m, kernel, calls = kda()
            m.train(training)
            old = []

            def original(*a, **kw):
                old.append(kw)
                return a[0], None, cache

            compat._kda_forward(m, torch.randn(1, length, 4), past_key_value=cache, original=original, chunk_kda=kernel)
            self.assertEqual(bool(old), delegate)
            self.assertEqual(bool(calls), not delegate)

    def test_checkpoint_recomputes_local_padding_metadata(self):
        for reentrant in (False, True):
            m, kernel, calls = kda()
            mask, pos = padded('left')
            x = torch.randn(2, 5, 4, requires_grad=True)

            def f(t):
                return compat._kda_forward(
                    m, t, attention_mask=mask, position_ids=pos, original=None, chunk_kda=kernel)[0]

            checkpoint(f, x, use_reentrant=reentrant).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreaterEqual(len(calls), 2)
            for kw in calls:
                self.assertTrue(torch.equal(kw['cu_seqlens'], torch.tensor([0, 3, 8])))

    def test_packed_helper_and_padding_not_misclassified(self):
        utils = ModuleType('swift.utils')
        utils.get_packed_seq_params = helper
        with mock_modules({'swift.utils': utils}):
            for lengths in ((1, 1), (3, 2), (63, 64, 65)):
                pos = torch.cat([torch.arange(n) for n in lengths])[None]
                params = compat._packed_params(torch.zeros(1, sum(lengths), 4), pos, None, {})
                self.assertEqual(params['cu_seq_lens_q'].tolist(), [0] + list(torch.tensor(lengths).cumsum(0).tolist()))
            mask, pos = padded('left')
            self.assertEqual(compat._packed_params(torch.zeros(2, 5, 4), pos, mask, {}), {})
            with self.assertRaises(ValueError):
                compat._packed_params(torch.zeros(1, 4, 4), None, None, {'cu_seqlens': torch.tensor([0, 4])})

    def test_mla_suffix_causality_and_pad_isolation(self):
        torch.manual_seed(7)
        m = MLA().train()
        x = torch.randn(1, 6, 8, requires_grad=True)
        alt = x.detach().clone()
        alt[:, 4:] += 30
        a = m(x)[0]
        b = m(alt)[0]
        torch.testing.assert_close(a[:, :4], b[:, :4])
        grad = torch.autograd.grad(a[:, :4].sum(), x)[0]
        self.assertEqual(int(torch.count_nonzero(grad[:, 4:])), 0)
        for side in ('left', 'right'):
            mask, pos = padded(side)
            x = torch.randn(2, 5, 8, requires_grad=True)
            alt = x.detach().clone()
            alt[~mask.bool()] = 30
            allowed = torch.ones(5, 5, dtype=torch.bool).tril()[None, None] & mask[:, None, None, :].bool()
            a = m(x, allowed, position_ids=pos)[0]
            b = m(alt, allowed, position_ids=pos)[0]
            self.assertTrue(torch.equal(a[mask.bool()], b[mask.bool()]))

    def test_singleton_conv_uses_autograd_path(self):
        module = SimpleNamespace(
            training=True, weight=nn.Parameter(torch.randn(4, 1, 3)), bias=None, activation='silu', backend='triton')
        delegated = []
        kernels = []

        def original(*a, **kw):
            delegated.append(True)
            return a[0], None

        def conv(**kw):
            kernels.append(kw)
            return kw['x'] * kw['weight'].sum(), None

        fn = compat._singleton_forward(original, conv)
        x = torch.randn(1, 2, 4, requires_grad=True)
        fn(module, x, cu_seqlens=torch.tensor([0, 1, 2]))[0].sum().backward()
        self.assertFalse(delegated)
        self.assertEqual(len(kernels), 1)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_installer_instance_lifecycle_and_conflict(self):
        patcher_tree = ast.parse((ROOT / 'swift/model/patcher.py').read_text())
        patch_node = next(
            n for n in patcher_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'patch_module_forward')
        ns = {'MethodType': MethodType}
        exec(compile(ast.Module(body=[patch_node], type_ignores=[]), '<Swift instance patch>', 'exec'), ns)
        modules = {
            n: ModuleType(n)
            for n in ('swift', 'swift.model', 'swift.model.models', 'swift.model.patcher', 'transformers',
                      'transformers.modeling_utils', 'fla', 'fla.modules', 'fla.modules.conv',
                      'fla.modules.conv.causal_conv1d', 'swift.model.models.ling3_rope')
        }
        modules['swift.model.patcher'].patch_module_forward = ns['patch_module_forward']
        rope_spec = importlib.util.spec_from_file_location('ling3_rope_fixture',
                                                           ROOT / 'swift/model/models/ling3_rope.py')
        rope_fixture = importlib.util.module_from_spec(rope_spec)
        rope_spec.loader.exec_module(rope_fixture)
        modules['swift.model.models.ling3_rope'].transformers_compatibility = rope_fixture.transformers_compatibility
        modules['transformers'].__version__ = '4.57.1'
        modules['transformers.modeling_utils'].ALL_ATTENTION_FUNCTIONS = {'sdpa': lambda *args, **kwargs: None}
        modules['fla.modules.conv.causal_conv1d'].causal_conv1d = lambda **kw: (kw['x'], None)

        class Shell(nn.Module):

            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    num_nextn_predict_layers=0, hidden_size=1536, num_hidden_layers=24, q_lora_rank=256)
                attn, _, _ = kda()
                self.model = SimpleNamespace(layers=[
                    SimpleNamespace(attention=attn, attention_layer_type='linear_attention'),
                    SimpleNamespace(attention=MLA(), attention_layer_type='attention')
                ])

        with mock_modules(modules):
            first = Shell()
            second = Shell()
            second.config.hidden_size = 2560
            with self.assertRaisesRegex(NotImplementedError, 'tiny'):
                compat.install_ling3_compat(second)
            self.assertFalse(getattr(second, '_swift_ling3_compat', False))
            second.config.hidden_size = 1536
            compat.install_ling3_compat(first)
            forward = first.model.layers[0].attention.forward
            compat.install_ling3_compat(first)
            self.assertIs(first.model.layers[0].attention.forward, forward)
            self.assertFalse(getattr(second, '_swift_ling3_compat', False))
            self.assertTrue(first.model._supports_packing_metadata)
            second.model.layers[1].attention._patched = True
            with self.assertRaises(ValueError):
                compat.install_ling3_compat(second)


class AttentionCompatContractTests(unittest.TestCase):

    def test_packing_record_validates_incompatible_leaf_calls(self):
        positions = torch.tensor([[0, 1, 0, 1]])
        hidden = torch.zeros(1, 4, 8)
        params = helper(positions)
        with mock_modules({'swift.utils': SimpleNamespace(get_packed_seq_params=helper)}):
            validated = compat._validate_packed_params(hidden, positions, None, params)
            record = compat._PackingRecord(positions, hidden, validated)
            with patch.object(compat, '_validate_packed_params', wraps=compat._validate_packed_params) as validate:
                actual = compat._packed_params(hidden, positions, None, {'_swift_ling3_packing_record': record})
                self.assertIs(actual['cu_seq_lens_q'], validated['cu_seq_lens_q'])
                validate.assert_not_called()
                for incompatible in (hidden.to('meta'), hidden[:, :-1]):
                    with self.assertRaisesRegex(ValueError, 'aligned with position_ids'):
                        compat._packed_params(incompatible, positions, None, {'_swift_ling3_packing_record': record})
                metadata = dict(validated, _swift_ling3_packing_record=record)
                compat._packed_params(hidden, positions, None, metadata)
                self.assertEqual(validate.call_count, 3)
                metadata = dict(validated, _swift_ling3_packing_record=record)
                metadata['cu_seq_lens_q'] = torch.tensor([0, 1, 4], dtype=validated['cu_seq_lens_q'].dtype)
                with self.assertRaisesRegex(ValueError, 'Conflicting Ling3 cu_seq_lens_q'):
                    compat._packed_params(hidden, positions, None, metadata)

    def test_verified_kda_avoids_repeated_keep_mask_reduction(self):
        positions = torch.tensor([[0, 1, 0, 1]])
        hidden = torch.randn(1, 4, 4)
        mask = torch.ones(1, 4)
        model, kernel, _ = kda()
        record = compat._PackingRecord(positions, hidden, helper(positions), mask)
        original_bool = torch.Tensor.__bool__
        calls = []

        def counted(value):
            calls.append(value)
            return original_bool(value)

        with patch.object(torch.Tensor, '__bool__', counted):
            actual = compat._kda_forward(
                model,
                hidden,
                mask,
                position_ids=positions,
                original=lambda *a, **kw: None,
                chunk_kda=kernel,
                _swift_ling3_packing_record=record)[0]
        self.assertEqual(calls, [])
        with mock_modules({'swift.utils': SimpleNamespace(get_packed_seq_params=helper)}):
            expected = compat._kda_forward(
                model, hidden, mask, position_ids=positions, original=lambda *a, **kw: None, chunk_kda=kernel)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_packed_kda_rejects_changed_or_mutated_keep_mask(self):
        positions = torch.tensor([[0, 1, 0, 1]])
        hidden = torch.randn(1, 4, 4)
        model, kernel, _ = kda()
        for mutate in (False, True):
            mask = torch.ones(1, 4)
            record = compat._PackingRecord(positions, hidden, helper(positions), mask)
            leaf_mask = mask if mutate else mask.clone()
            leaf_mask[0, 0] = 0
            with self.assertRaisesRegex(ValueError, 'all-valid leaf attention mask'):
                compat._kda_forward(
                    model,
                    hidden,
                    leaf_mask,
                    position_ids=positions,
                    original=None,
                    chunk_kda=kernel,
                    _swift_ling3_packing_record=record)

    def test_forward_scoped_metadata_reaches_mla_varlen_dispatch(self):
        boundaries = []

        def varlen(module, q, k, v, mask, **kwargs):
            cu = kwargs.get('cu_seq_lens_q')
            cu = cu.tolist() if cu is not None else [0, q.shape[2]]
            boundaries.append(cu)
            output = torch.cat([
                torch.nn.functional.scaled_dot_product_attention(
                    q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], is_causal=True, scale=kwargs['scaling'])
                for a, b in zip(cu, cu[1:])
            ], 2)
            return output.transpose(1, 2), None

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Module()])
                attention = MLA('flash_attention_2')
                attention.forward = MethodType(
                    compat._make_mla_forward(attention._old_forward, {'flash_attention_2': varlen}), attention)
                self.model.layers[0].attention = attention
                self.model._supports_packing_metadata = True
                self.config = SimpleNamespace(
                    num_nextn_predict_layers=0, _attn_implementation='flash_attention_2', use_cache=False)
                compat.patch_ling3_packing(self)

            def forward(self, inputs_embeds=None, position_ids=None, **kwargs):
                # Deliberately drop kwargs, as the frozen remote call sites do.
                return self.model.layers[0].attention(
                    hidden_states=inputs_embeds,
                    position_embeddings=(None, None),
                    attention_mask=None,
                    position_ids=position_ids)[0]

        torch.manual_seed(57)
        model = Model()
        reference = Model()
        reference.load_state_dict(model.state_dict())
        positions = torch.tensor([[0, 1, 0, 1]])
        x = torch.randn(1, 4, 8, requires_grad=True)
        with mock_modules({'swift.utils': SimpleNamespace(get_packed_seq_params=helper)}):
            with patch.object(compat, '_validate_packed_params', wraps=compat._validate_packed_params) as full:
                output = model(inputs_embeds=x, position_ids=positions, **helper(positions))
                self.assertEqual(full.call_count, 1)
            self.assertTrue(boundaries, 'MLA did not dispatch to the varlen backend')
            self.assertEqual(boundaries[-1], [0, 2, 4])
            other = torch.cat([
                reference(inputs_embeds=x[:, a:b], position_ids=torch.arange(b - a)[None]) for a, b in ((0, 2), (2, 4))
            ], 1)
            torch.testing.assert_close(output, other, rtol=0, atol=0)
            grad = torch.autograd.grad(output[:, 2:].sum(), x)[0]
            self.assertEqual(float(grad[:, :2].abs().max()), 0.)
        self.assertFalse(model.model.layers[0].attention._forward_pre_hooks)

    def test_real_cache_and_legacy_alias(self):
        from transformers.cache_utils import DynamicCache

        torch.manual_seed(51)
        module = MLA().eval()
        module.layer_idx = 0
        x = torch.randn(1, 4, 8)
        full = module(x, position_ids=torch.arange(4)[None])[0]
        for key in ('past_key_values', 'past_key_value'):
            cache = DynamicCache()
            prefix = module(x[:, :3], position_ids=torch.arange(3)[None], **{key: cache})[0]
            last = module(
                x[:, 3:],
                torch.zeros(1, 1, 1, 4),
                position_ids=torch.tensor([[3]]),
                cache_position=torch.tensor([3]),
                **{key: cache})[0]
            torch.testing.assert_close(prefix, full[:, :3])
            torch.testing.assert_close(last, full[:, 3:])
            self.assertEqual(cache.get_seq_length(), 4)

    def test_unknown_dispatch_and_decorator_rejected_before_install(self):
        from functools import wraps
        from ling3_remote_forward_fixture import remote_forward

        def unsupported(self, hidden_states, **kwargs):
            return hidden_states

        with self.assertRaisesRegex(ValueError, 'dispatch'):
            compat._make_mla_forward(unsupported, {})

        @wraps(remote_forward)
        def unknown(*args, **kwargs):
            return remote_forward(*args, **kwargs)

        with self.assertRaisesRegex(ValueError, 'decorator'):
            compat._make_mla_forward(unknown, {})
        from inspect import unwrap
        from transformers.utils.deprecation import deprecate_kwarg

        changed = deprecate_kwarg('different_name', new_name='past_key_values', version='4.58')(unwrap(remote_forward))
        with self.assertRaisesRegex(ValueError, 'decorator arguments'):
            compat._make_mla_forward(changed, {})

    def test_wrapper_preserves_remote_globals_and_instance_state(self):
        from inspect import unwrap
        from ling3_remote_forward_fixture import remote_forward

        raw = unwrap(remote_forward)
        original = raw.__globals__['eager_attention_forward']
        first = MLA().eval()
        second = MLA().eval()
        second.load_state_dict(first.state_dict())
        before = {name: id(value) for name, value in first.named_parameters()}
        x = torch.randn(1, 4, 8, requires_grad=True)
        a = first(x, position_ids=torch.arange(4)[None])[0]
        b = second(x, position_ids=torch.arange(4)[None])[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.sum().backward()
        self.assertIs(raw.__globals__['eager_attention_forward'], original)
        self.assertEqual(before, {name: id(value) for name, value in first.named_parameters()})


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
