# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU-safe contracts extracted from the actual patched source, without backend imports."""
import ast
import copy
import math
import os
import sys
import typing
import unittest
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {'TORCH_DEVICE_BACKEND_AUTOLOAD': '0'}):
    import torch

import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn.utils.rnn import pad_sequence  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

SWIFT = Path(__file__).resolve().parents[2]
MODEL_DIR = os.environ.get('LING3_MODEL_SOURCE')
MODEL = Path(MODEL_DIR) / 'modeling_bailing_moe_v3.py' if MODEL_DIR else None
if MODEL is not None and not MODEL.is_file():
    raise FileNotFoundError('LING3_MODEL_SOURCE has no modeling_bailing_moe_v3.py')


def method(path, cls, name, namespace, remove_import=None):
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls) if cls else tree
    node = copy.deepcopy(next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    if remove_import:
        node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom) or n.module != remove_import]
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


NS = dict(
    typing.__dict__,
    torch=torch,
    nn=nn,
    F=F,
    math=math,
    partial=partial,
    rearrange=rearrange,
    MoeV3ModelOutputWithPast=object,
    Cache=object)
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from test_ling3_compat import compat, helper, mock_modules
finally:
    sys.path.pop(0)

utils = ModuleType('swift.utils')
utils.get_packed_seq_params = helper


def setUpModule():
    unittest.addModuleCleanup(torch.set_rng_state, torch.get_rng_state())
    stack = ExitStack()
    unittest.addModuleCleanup(stack.close)
    stack.enter_context(mock_modules({'swift.utils': utils}))


def tearDownModule():
    unittest.doModuleCleanups()


base = SWIFT / 'swift/template/base.py'
methods = {
    name: method(base, 'Template', name, {
        **NS, 'pad_sequence': pad_sequence
    })
    for name in ('packing_row', '_data_collator', '_pad_sequence')
}
NativeTemplate = type('NativeTemplateMethods', (), methods)
template_tree = ast.parse((SWIFT / 'swift/template/templates/llm.py').read_text())
ling = next(n for n in template_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Ling3Template')
collator = copy.deepcopy(next(n for n in ling.body if isinstance(n, ast.FunctionDef) and n.name == '_data_collator'))
collator.body = [n for n in collator.body if not isinstance(n, ast.ImportFrom)]
class_node = ast.ClassDef(
    name='Ling3Template',
    bases=[ast.Name(id='Template', ctx=ast.Load())],
    keywords=[],
    body=[collator],
    decorator_list=[])
module = ast.fix_missing_locations(ast.Module(body=[class_node], type_ignores=[]))
tns = dict(Template=NativeTemplate, get_packed_seq_params=helper)
exec(compile(module, 'actual_ling3_template_collator', 'exec'), tns)
Ling3Template = tns['Ling3Template']
install_hook = compat.patch_ling3_packing


def template(padding_free=True):
    t = Ling3Template()
    t.tokenizer = SimpleNamespace(pad_token_id=0)
    t.padding_side = 'right'
    t.is_training = True
    t.use_megatron = False
    t.padding_free = padding_free
    t.sequence_parallel_size = 1
    t._handle_megatron_cp = lambda batch: None
    t._data_collator_mm_data = lambda batch: {}
    return t


def batch(lengths=(4, 3)):
    rows = [
        dict(
            input_ids=list(range(1, n + 1)),
            labels=[-100] + list(range(2, n + 1)),
            loss_scale=[0.] + [1.] * (n - 1),
            length=n) for n in lengths
    ]
    return template()._data_collator(rows)


class SparseSentinel(nn.Module):
    pass


class AttentionSpy(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.1))
        self.calls = []
        original = self.forward

        def forward(module, hidden_states, **kwargs):
            params = compat._packed_params(hidden_states, kwargs.get('position_ids'), kwargs.get('attention_mask'),
                                           kwargs)
            kwargs.update(params)
            if params:
                kwargs['cu_seqlens'] = params['cu_seq_lens_q']
            return original(hidden_states, **kwargs)

        self.forward = MethodType(forward, self)

    def forward(self, hidden_states, **kwargs):
        self.calls.append(kwargs)
        return hidden_states * self.weight, None, None


dns = {**NS, 'BailingMoeV3SparseMoeBlock': SparseSentinel}
decoder_forward = method(MODEL, 'BailingMoeV3DecoderLayer', 'forward', dns) if MODEL is not None else None
model_forward = method(MODEL, 'BailingMoeV3Model', 'forward', dns) if MODEL is not None else None


class Decoder(nn.Module):
    forward = decoder_forward

    def __init__(self, kind):
        super().__init__()
        self.config = SimpleNamespace(num_experts=None)
        self.attention_layer_type = kind
        self.attention = AttentionSpy()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = nn.Identity()


class Core(nn.Module):
    forward = model_forward
    _supports_packing_metadata = True

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            output_router_logits=False,
            use_cache=False,
            use_return_dict=False,
            layer_group_size=4)
        self._use_flash_attention_2 = True
        self._use_sdpa = False
        self.gradient_checkpointing = False
        self.num_nextn_predict_layers = 0
        self.layers = nn.ModuleList([Decoder('linear_attention'), Decoder('attention')])
        self.word_embeddings = nn.Embedding(32, 4)
        self.norm = nn.Identity()
        self.rotary_emb = lambda hidden, pos: (torch.ones(1, pos.shape[-1], 2), torch.zeros(1, pos.shape[-1], 2))


class CausalShell(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = Core()
        self.config = SimpleNamespace(
            _attn_implementation='flash_attention_2', use_cache=False, num_nextn_predict_layers=0)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=None,
                **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)


def cpu_attention(module, query, key, value, mask, scaling, **kwargs):
    cu = kwargs.get('cu_seq_lens_q')
    bounds = [0, query.shape[2]] if cu is None else cu.tolist()
    parts = [
        F.scaled_dot_product_attention(
            query[:, :, a:b], key[:, :, a:b], value[:, :, a:b], is_causal=module.is_causal, scale=scaling)
        for a, b in zip(bounds[:-1], bounds[1:])
    ]
    return torch.cat(parts, 2).transpose(1, 2), None


mlans = dict(
    NS,
    FlashAttentionKwargs=dict,
    eager_attention_forward=cpu_attention,
    ALL_ATTENTION_FUNCTIONS={
        'sdpa': cpu_attention,
        'flash_attention_2': cpu_attention
    },
    apply_rotary_pos_emb_interleave=lambda q, k, cos, sin: (q, k))
mla_forward = method(MODEL, 'BailingMoeV3MultiLatentAttention', 'forward', mlans) if MODEL is not None else None
if mla_forward is not None:
    from transformers.utils.deprecation import deprecate_kwarg

    mla_forward = deprecate_kwarg('past_key_value', new_name='past_key_values', version='4.58')(mla_forward)


class MLAFixture(nn.Module):
    forward = mla_forward

    def __init__(self, backend):
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
        self.forward = MethodType(compat._make_mla_forward(self.forward, mlans['ALL_ATTENTION_FUNCTIONS']), self)


class ConvSpy(nn.Module):

    def __init__(self):
        super().__init__()
        self.boundaries = []

    def forward(self, x, cu_seqlens=None, **kwargs):
        self.boundaries.append(cu_seqlens)
        return x, None


def kda_fixture():
    calls = []

    def core(name):

        def call(**kwargs):
            calls.append((name, kwargs))
            return kwargs['v'], None

        return call

    namespace = dict(NS, chunk_kda=core('chunk'), fused_recurrent_kda=core('recurrent'))
    forward = method(MODEL, 'BailingMoeV3KimiDeltaAttention', 'forward', namespace)
    cls = type('ActualKDAForwardFixture', (nn.Module, ), {'forward': forward})
    module = cls()
    module.mode = 'chunk'
    module.no_kda_lora = True
    module.head_dim = 2
    module.head_k_dim = 2
    module.safe_gate = True
    module.lower_bound = -5.
    module.layer_idx = 0
    for name in ('q_proj', 'k_proj', 'v_proj', 'f_proj', 'g_proj'):
        setattr(module, name, nn.Linear(4, 4))
    module.b_proj = nn.Linear(4, 2)
    module.A_log = nn.Parameter(torch.zeros(2))
    module.dt_bias = nn.Parameter(torch.zeros(4))
    module.q_conv1d = ConvSpy()
    module.k_conv1d = ConvSpy()
    module.v_conv1d = ConvSpy()
    module.o_norm = lambda o, g: o
    module.o_proj = nn.Identity()
    module.forward = MethodType(compat._make_kda_forward(module.forward, namespace['chunk_kda']), module)
    return module, calls


@unittest.skipUnless(MODEL is not None, 'requires official remote source, not weights or devices')
class PackingTests(unittest.TestCase):

    def test_actual_kda_short_training_and_shared_boundaries(self):
        for length in (1, 2, 63, 64, 65):
            m, calls = kda_fixture()
            m.train()
            x = torch.randn(1, length, 4, requires_grad=True)
            cu = torch.tensor([0, length], dtype=torch.int32)
            m(x, position_ids=torch.arange(length)[None], cu_seqlens=cu)[0].sum().backward()
            self.assertEqual(calls[0][0], 'chunk')
            self.assertIs(calls[0][1]['cu_seqlens'], cu)
            for conv in (m.q_conv1d, m.k_conv1d, m.v_conv1d):
                self.assertIs(conv.boundaries[0], cu)
            self.assertTrue(calls[0][1]['safe_gate'])
            self.assertEqual(calls[0][1]['lower_bound'], -5.)
        m, calls = kda_fixture()
        m.eval()
        m(torch.ones(1, 2, 4))
        self.assertEqual(calls[0][0], 'recurrent')

    def test_actual_mla_dispatch_cpu_reference_causality(self):
        torch.manual_seed(5)
        m = MLAFixture('sdpa').train()
        x = torch.randn(1, 5, 8, requires_grad=True)
        z = x.detach().clone()
        z[:, 3:] += 10
        pos = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))
        a = m(x, pos, None)[0]
        b = m(z, pos, None)[0]
        torch.testing.assert_close(a[:, :3], b[:, :3], rtol=1e-5, atol=1e-6)
        grad = torch.autograd.grad(a[:, :3].sum(), x)[0]
        self.assertEqual(float(grad[:, 3:].abs().max()), 0)

    def test_actual_mla_metadata_cpu_oracle_isolation(self):
        torch.manual_seed(6)
        m = MLAFixture('flash_attention_2').train()
        x = torch.randn(1, 5, 8, requires_grad=True)
        meta = helper(torch.tensor([[0, 1, 0, 1, 2]]))
        pos = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))
        a = m(x, pos, None, position_ids=torch.tensor([[0, 1, 0, 1, 2]]), **meta)[0]
        changed = x.detach().clone()
        changed[:, :2] += 10
        b = m(changed, pos, None, position_ids=torch.tensor([[0, 1, 0, 1, 2]]), **meta)[0]
        torch.testing.assert_close(a[:, 2:], b[:, 2:], rtol=1e-5, atol=1e-6)
        grad = torch.autograd.grad(a[:, 2:].sum(), x)[0]
        self.assertEqual(float(grad[:, :2].abs().max()), 0)

    def test_actual_modified_call_chain_and_checkpoint(self):
        for reentrant in (None, False, True):
            model = CausalShell()
            install_hook(model)
            if reentrant is not None:
                model.model.gradient_checkpointing = True
                model.model._gradient_checkpointing_func = partial(checkpoint, use_reentrant=reentrant)
            b = batch()
            b.pop('input_ids')
            b.pop('labels')
            b.pop('loss_scale')
            x = torch.randn(1, 7, 4, requires_grad=True)
            y = model(inputs_embeds=x, **b)[0]
            y.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            for layer in model.model.layers:
                self.assertTrue(layer.attention.calls)
                for call in layer.attention.calls:
                    self.assertTrue(torch.equal(call['cu_seqlens'], b['cu_seq_lens_q']))
                    self.assertTrue(torch.equal(call['cu_seq_lens_q'], b['cu_seq_lens_q']))
                    self.assertEqual(call['max_length_q'], b['max_length_q'])

    def test_explicit_canonical_not_overwritten(self):
        model = CausalShell()
        install_hook(model)
        b = batch()
        canonical = b['cu_seq_lens_q'].clone()
        b['cu_seqlens'] = canonical
        model(**b)
        for layer in model.model.layers:
            self.assertTrue(torch.equal(layer.attention.calls[0]['cu_seqlens'], canonical))

    def test_explicit_none_input_ids_with_inputs_embeds(self):
        model = CausalShell()
        install_hook(model)
        b = batch()
        b['input_ids'] = None
        model(inputs_embeds=torch.ones(1, 7, 4), **b)

    def test_positional_inputs_keep_existing_signature(self):
        model = CausalShell()
        install_hook(model)
        b = batch()
        ids = b.pop('input_ids')
        pos = b.pop('position_ids')
        model(ids, None, pos, **b)

    def test_unpacking_unchanged_and_hook_idempotent(self):
        model = CausalShell()
        install_hook(model)
        install_hook(model)
        self.assertEqual(len(model._forward_pre_hooks), 1)
        b = template(False)._data_collator(
            [dict(input_ids=[1, 2], labels=[-100, 2]),
             dict(input_ids=[1, 2, 3], labels=[-100, 2, 3])])
        self.assertNotIn('cu_seq_lens_q', b)
        model(**b)
        for layer in model.model.layers:
            self.assertNotIn('cu_seqlens', layer.attention.calls[0])

    def test_keep_mask_semantics(self):
        model = CausalShell()
        install_hook(model)
        b = batch()
        model(**dict(b, attention_mask=torch.ones_like(b['position_ids'])))
        bad = torch.ones_like(b['position_ids'])
        bad[0, -1] = 0
        with self.assertRaisesRegex(ValueError, 'padding'):
            model(**dict(b, attention_mask=bad))
        with self.assertRaisesRegex(ValueError, '2-D'):
            model(**dict(b, attention_mask=torch.zeros(1, 1, 7, 7)))

    def test_invalid_metadata_fail_fast(self):
        cases = []
        b = batch()
        b['cu_seq_lens_q'] = torch.tensor([1, 4, 7], dtype=torch.int32)
        cases.append(b)
        b = batch()
        b['cu_seq_lens_k'] = torch.tensor([0, 3, 7], dtype=torch.int32)
        cases.append(b)
        b = batch()
        b['max_length_q'] = 6
        cases.append(b)
        b = batch()
        b.pop('max_length_k')
        cases.append(b)
        b = batch()
        b['position_ids'][0, 2] = 9
        cases.append(b)
        b = batch()
        b['cu_seqlens'] = torch.tensor([0, 7], dtype=torch.int32)
        cases.append(b)
        for b in cases:
            model = CausalShell()
            install_hook(model)
            with self.assertRaises(ValueError):
                model(**b)

    def test_cache_backend_and_old_model_rejected_for_packing(self):
        for mutate in (lambda m: setattr(m.config, '_attn_implementation', 'sdpa'),
                       lambda m: setattr(m.config, 'use_cache', True),
                       lambda m: setattr(m.model, '_supports_packing_metadata', False)):
            model = CausalShell()
            install_hook(model)
            mutate(model)
            with self.assertRaises(ValueError):
                model(**batch())
        model = CausalShell()
        install_hook(model)
        with self.assertRaises(ValueError):
            model(**dict(batch(), past_key_values=object()))
        model = CausalShell()
        install_hook(model)
        model.config.use_cache = True
        with self.assertRaises(ValueError):
            model(**dict(batch(), use_cache=None))

    def test_missing_packed_metadata_rejected(self):
        model = CausalShell()
        install_hook(model)
        b = batch()
        for k in ('cu_seq_lens_q', 'cu_seq_lens_k', 'max_length_q', 'max_length_k'):
            b.pop(k)
        with self.assertRaisesRegex(ValueError, 'explicit sequence metadata'):
            model(**b)


class NativePackingCollatorTests(unittest.TestCase):

    def test_native_metadata_loss_boundaries(self):
        b = batch()
        self.assertEqual(b['position_ids'].tolist(), [[0, 1, 2, 3, 0, 1, 2]])
        self.assertEqual(b['cu_seq_lens_q'].tolist(), [0, 4, 7])
        self.assertIs(b['cu_seq_lens_q'], b['cu_seq_lens_k'])
        self.assertEqual(b['cu_seq_lens_q'].dtype, torch.int32)
        self.assertEqual(int(b['max_length_q']), 4)
        self.assertEqual(torch.roll(b['labels'], -1, -1)[0, 3].item(), -100)
        self.assertEqual(torch.roll(b['labels'], -1, -1)[0, -1].item(), -100)
        self.assertEqual(b['loss_scale'][0, 4].item(), 0)

    def test_length_matrix_metadata(self):
        for lengths in ((1, ), (1, 1, 1), (1, 3), (3, 1), (63, 64, 65), (127, 2)):
            b = batch(lengths)
            expected = [0]
            for n in lengths:
                expected.append(expected[-1] + n)
            self.assertEqual(b['cu_seq_lens_q'].tolist(), expected)
            self.assertEqual(int(b['max_length_q']), max(lengths))


class PackingShell(nn.Module):
    """Only the outer hook runs; no model or attention computation."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            _attn_implementation='flash_attention_2', use_cache=False, num_nextn_predict_layers=0)
        self.model = SimpleNamespace(_supports_packing_metadata=True)
        self.forward_calls = 0

    def forward(self, input_ids=None, **kwargs):
        self.forward_calls += 1
        return input_ids


class PackingMaskBoundaryTests(unittest.TestCase):

    def setUp(self):
        utils = ModuleType('swift.utils')
        utils.get_packed_seq_params = helper
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock_modules({'swift.utils': utils}))

    def model(self):
        model = PackingShell()
        model.model._supports_packing_metadata = True
        compat.patch_ling3_packing(model)
        return model

    def test_training_rejects_external_4d_before_forward(self):
        positions = torch.tensor([[0, 1, 0, 1]])
        documents = torch.tensor([0, 0, 1, 1])
        block = torch.ones(4, 4, dtype=torch.bool).tril() & (documents[:, None] == documents[None, :])
        for mask in (block[None, None], torch.zeros(1, 1, 4, 4).masked_fill(~block[None, None], float('-inf')),
                     torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4, dtype=torch.bool)):
            for metadata in ({}, helper(positions)):
                model = self.model().train()
                with self.subTest(
                        dtype=mask.dtype, metadata=bool(metadata)), self.assertRaisesRegex(ValueError, 'external 4-D'):
                    model(
                        input_ids=torch.ones(1, 4, dtype=torch.long),
                        position_ids=positions,
                        attention_mask=mask,
                        **metadata)
                self.assertEqual(model.forward_calls, 0)

    def test_training_none_and_2d_inputs_are_preserved(self):
        inputs = torch.ones(1, 4, dtype=torch.long)
        for mask, positions in ((None, [[0, 1, 2, 3]]), (torch.ones(1, 4), [[0, 1, 2, 3]]),
                                (torch.tensor([[0, 0, 1, 1]]), [[0, 0, 0, 1]]), (torch.tensor([[1, 1, 0,
                                                                                                0]]), [[0, 1, 0, 0]])):
            model = self.model().train()
            self.assertIs(model(input_ids=inputs, position_ids=torch.tensor(positions), attention_mask=mask), inputs)
            self.assertEqual(model.forward_calls, 1)

    def test_standard_collator_uses_supported_training_masks(self):
        rows = [
            dict(input_ids=[1, 2], labels=[-100, 2], length=2),
            dict(input_ids=[3, 4, 5], labels=[-100, 4, 5], length=3)
        ]
        for side in ('left', 'right'):
            collator = template(padding_free=False)
            collator.padding_side = side
            inputs = collator._data_collator(copy.deepcopy(rows))
            self.assertEqual(inputs['attention_mask'].ndim, 2)
            self.model().train()(**inputs)
        packed = batch()
        self.assertTrue(packed.get('attention_mask') is None or packed['attention_mask'].ndim == 2)
        self.model().train()(**packed)

    def test_entry_restriction_does_not_reject_internal_mla_causal_mask(self):
        x = torch.zeros(1, 4, 4)
        mask = torch.zeros(1, 1, 4, 4).masked_fill(torch.ones(4, 4, dtype=torch.bool).triu(1), float('-inf'))
        compat._validate_eager_causal_mask(x, mask)
        self.assertEqual(compat._packed_params(x, torch.arange(4)[None], mask, {}), {})

    def test_reset_4d_without_metadata_rejected_at_both_entries(self):
        positions = torch.tensor([[0, 1, 0, 1]])
        masks = (torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4, dtype=torch.bool),
                 torch.zeros(1, 1, 4, 4).masked_fill(torch.ones(4, 4, dtype=torch.bool).triu(1), float('-inf')))
        for mask in masks:
            with self.subTest(dtype=mask.dtype, entry='model'), self.assertRaises(ValueError):
                self.model()(input_ids=torch.ones(1, 4, dtype=torch.long), position_ids=positions, attention_mask=mask)
            with self.subTest(dtype=mask.dtype, entry='module'), self.assertRaises(ValueError):
                compat._packed_params(torch.zeros(1, 4, 4), positions, mask, {})

    def test_explicit_packing_rejects_block_diagonal_mask(self):
        positions = torch.tensor([[0, 1, 0, 1]])
        documents = torch.tensor([0, 0, 1, 1])
        allowed = torch.ones(4, 4, dtype=torch.bool).tril() & (documents[:, None] == documents[None, :])
        metadata = helper(positions)
        for mask in (allowed[None, None], torch.zeros(1, 1, 4, 4).masked_fill(~allowed[None, None], float('-inf'))):
            with self.subTest(dtype=mask.dtype), self.assertRaises(ValueError):
                self.model()(
                    input_ids=torch.ones(1, 4, dtype=torch.long),
                    position_ids=positions,
                    attention_mask=mask,
                    **metadata)
            with self.assertRaises(ValueError):
                compat._packed_params(torch.zeros(1, 4, 4), positions, mask, metadata)

    def test_explicit_packing_metadata_mask_contract(self):
        positions = torch.tensor([[0, 1, 0, 1]])
        metadata = helper(positions)
        for mask in (None, torch.ones(1, 4)):
            self.model()(
                input_ids=torch.ones(1, 4, dtype=torch.long), position_ids=positions, attention_mask=mask, **metadata)
            params = compat._packed_params(torch.zeros(1, 4, 4), positions, mask, metadata)
            self.assertEqual(params['cu_seq_lens_q'].tolist(), [0, 2, 4])
        for mask in (torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), torch.tensor([[0, 1, 1, 1]])):
            with self.subTest(entry='model'), self.assertRaises(ValueError):
                self.model()(
                    input_ids=torch.ones(1, 4, dtype=torch.long),
                    position_ids=positions,
                    attention_mask=mask,
                    **metadata)
            with self.subTest(entry='module'), self.assertRaises(ValueError):
                compat._packed_params(torch.zeros(1, 4, 4), positions, mask, metadata)
        partial = dict(metadata)
        partial.pop('max_length_k')
        with self.assertRaisesRegex(ValueError, 'together'):
            self.model()(input_ids=torch.ones(1, 4, dtype=torch.long), position_ids=positions, **partial)

    def test_none_and_all_valid_2d_still_require_outer_metadata(self):
        positions = torch.tensor([[0, 1, 0, 1]])
        for mask in (None, torch.ones(1, 4)):
            with self.assertRaisesRegex(ValueError, 'explicit sequence metadata'):
                self.model()(input_ids=torch.ones(1, 4, dtype=torch.long), position_ids=positions, attention_mask=mask)
            # Local reconstruction remains necessary when the remote model drops masks.
            params = compat._packed_params(torch.zeros(1, 4, 4), positions, mask, {})
            self.assertEqual(params['cu_seq_lens_q'].tolist(), [0, 2, 4])

    def test_ordinary_2d_padding_is_not_packing(self):
        for mask, positions in (([[0, 0, 1, 1]], [[0, 0, 0, 1]]), ([[0, 0, 1, 1]], [[1, 1, 0,
                                                                                     1]]), ([[1, 1, 0,
                                                                                              0]], [[0, 1, 0, 0]])):
            mask, positions = torch.tensor(mask), torch.tensor(positions)
            inputs = torch.ones(1, 4, dtype=torch.long)
            self.assertIs(self.model()(input_ids=inputs, position_ids=positions, attention_mask=mask), inputs)
            self.assertEqual(compat._packed_params(torch.zeros(1, 4, 4), positions, mask, {}), {})

    def test_eval_4d_masks_and_cache_positions_are_preserved(self):
        masks = (torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4, dtype=torch.bool).tril())
        for positions in (torch.arange(4)[None], torch.tensor([[0, 0, 1, 2]]), torch.arange(8, 12)[None]):
            for mask in masks:
                inputs = torch.ones(1, 4, dtype=torch.long)
                model = self.model().eval()
                self.assertIs(model(input_ids=inputs, position_ids=positions, attention_mask=mask), inputs)
                self.assertEqual(compat._packed_params(torch.zeros(1, 4, 4), positions, mask, {}), {})
        inputs = torch.ones(1, 1, dtype=torch.long)
        model = self.model().eval()
        self.assertIs(
            model(
                input_ids=inputs,
                position_ids=torch.tensor([[8]]),
                attention_mask=torch.zeros(1, 1, 1, 9),
                past_key_values=object(),
                use_cache=True), inputs)

    def test_eval_4d_padding_is_not_rejected_as_packing(self):
        causal = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
        for keep, positions in (([0, 0, 1, 1], [1, 1, 0, 1]), ([1, 1, 0, 0], [0, 1, 0, 0])):
            allowed = causal & torch.tensor(keep, dtype=torch.bool)[None, None, None, :]
            for mask in (allowed, torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, float('-inf'))):
                inputs = torch.ones(1, 4, dtype=torch.long)
                positions_tensor = torch.tensor([positions])
                with self.subTest(keep=keep, dtype=mask.dtype):
                    self.assertIs(
                        self.model().eval()(input_ids=inputs, position_ids=positions_tensor, attention_mask=mask),
                        inputs)
                    self.assertEqual(compat._packed_params(torch.zeros(1, 4, 4), positions_tensor, mask, {}), {})


class ForwardScopedPackingTests(unittest.TestCase):

    @staticmethod
    def model(optimized=True):

        class Leaf(nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))
                self.register_buffer('scale', torch.ones(()))

            def forward(self, hidden, mask, positions, **kwargs):
                params = compat._packed_params(hidden, positions, mask, kwargs)
                cu = params.get('cu_seq_lens_q', torch.tensor([0, hidden.shape[1]]))
                # Segmented differentiable proxy for boundary plumbing, not KDA.
                return torch.cat([hidden[:, a:b].cumsum(1) for a, b in zip(cu.tolist(),
                                                                           cu.tolist()[1:])], 1) * self.weight

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Module() for _ in range(24)])
                for layer in self.model.layers:
                    layer.attention = Leaf()
                self.model._supports_packing_metadata = True
                self.config = SimpleNamespace(
                    num_nextn_predict_layers=0, _attn_implementation='flash_attention_2', use_cache=False)
                self.checkpoint_mode = None
                self.fail = False
                self.nested = False

            def forward(self, inputs_embeds=None, position_ids=None, attention_mask=None, **kwargs):
                hidden = inputs_embeds
                for layer in self.model.layers:
                    if self.checkpoint_mode is None:
                        hidden = layer.attention(hidden, attention_mask, position_ids)
                    else:
                        hidden = checkpoint(
                            layer.attention, hidden, attention_mask, position_ids, use_reentrant=self.checkpoint_mode)
                    if self.fail:
                        raise ValueError('injected forward failure')
                    if self.nested:
                        self(inputs_embeds=inputs_embeds, position_ids=position_ids, **helper(position_ids))
                return hidden

        model = Model()
        if optimized:
            compat.patch_ling3_packing(model)
        return model

    def test_full_validation_count_and_direct_leaf_fallback(self):
        from unittest.mock import patch

        positions = torch.tensor([[0, 1, 0, 1]])
        hidden = torch.randn(1, 4, 2)
        current = self.model(False)
        candidate = self.model()
        validate = compat._validate_packed_params
        with patch.object(compat, '_validate_packed_params', wraps=validate) as calls:
            compat._packed_params(hidden, positions, None, helper(positions), require_metadata=True)
            expected = current(inputs_embeds=hidden, position_ids=positions)
            self.assertEqual(calls.call_count, 25)
        with patch.object(compat, '_validate_packed_params', wraps=validate) as calls:
            actual = candidate(inputs_embeds=hidden, position_ids=positions, **helper(positions))
            self.assertEqual(calls.call_count, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with patch.object(compat, '_validate_packed_params', wraps=validate) as calls:
            candidate.model.layers[0].attention(hidden, None, positions)
            self.assertEqual(calls.call_count, 1)

    def test_batches_checkpoint_recomputation_and_gradients(self):
        for mode in (None, True, False):
            candidate = self.model()
            reference = self.model(False)
            candidate.checkpoint_mode = reference.checkpoint_mode = mode
            first = torch.tensor([[0, 1, 0, 1]])
            second = torch.tensor([[0, 0, 1, 2, 3]])
            x = torch.randn(1, 4, 2, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            out = candidate(inputs_embeds=x, position_ids=first, **helper(first))
            candidate(inputs_embeds=torch.randn(1, 5, 2), position_ids=second, **helper(second))
            other = reference(inputs_embeds=y, position_ids=first)
            out.sum().backward()
            other.sum().backward()
            torch.testing.assert_close(out, other, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
            for a, b in zip(candidate.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            self.assertTrue(all(not layer.attention._forward_pre_hooks for layer in candidate.model.layers))

    def test_exception_nested_and_existing_hook_cleanup(self):
        model = self.model()
        positions = torch.tensor([[0, 1, 0, 1]])
        x = torch.randn(1, 4, 2)
        leaf = model.model.layers[0].attention
        handle = leaf.register_forward_pre_hook(lambda *args: None)
        for field in ('fail', 'nested'):
            setattr(model, field, True)
            with self.assertRaises((ValueError, RuntimeError)):
                model(inputs_embeds=x, position_ids=positions, **helper(positions))
            setattr(model, field, False)
            self.assertEqual(list(leaf._forward_pre_hooks), [handle.id])
        handle.remove()
        model(inputs_embeds=x, position_ids=positions, **helper(positions))

    def test_install_state_and_reload_isolation(self):
        model = self.model()
        ids = [id(p) for p in model.parameters()]
        buffers = [id(b) for b in model.buffers()]
        keys = list(model.state_dict())
        forward = model.forward
        compat.patch_ling3_packing(model)
        self.assertIs(model.forward, forward)
        other = self.model()
        other.load_state_dict(model.state_dict())
        for positions in (torch.arange(4)[None], torch.tensor([[0, 1, 0, 1]])):
            x = torch.randn(1, 4, 2)
            a = model(inputs_embeds=x, position_ids=positions, **helper(positions))
            b = other(inputs_embeds=x, position_ids=positions, **helper(positions))
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(ids, [id(p) for p in model.parameters()])
        self.assertEqual(buffers, [id(b) for b in model.buffers()])
        self.assertEqual(keys, list(model.state_dict()))

    def test_record_mutation_falls_back_and_inputs_stay_validated(self):
        from unittest.mock import patch

        positions = torch.tensor([[0, 1, 0, 1]])
        hidden = torch.randn(1, 4, 2)
        params = helper(positions)
        record = compat._PackingRecord(positions, hidden, params)
        positions[0, 1] = 2
        with self.assertRaises(ValueError):
            compat._packed_params(hidden, positions, None, {'_swift_ling3_packing_record': record})
        model = self.model()
        positions = torch.tensor([[0, 1, 0, 1]])
        for key in ('cu_seq_lens_q', 'max_length_q'):
            metadata = helper(positions)
            metadata[key] = metadata[key] + 1
            with self.assertRaises(ValueError):
                model(inputs_embeds=hidden, position_ids=positions, **metadata)
        with self.assertRaises(ValueError):
            model(inputs_embeds=hidden, position_ids=positions, attention_mask=torch.ones(1, 1, 4, 4))
        with self.assertRaises(ValueError):
            model(inputs_embeds=hidden, position_ids=positions, _swift_ling3_packing_record=record)

    def test_failed_hook_installation_preserves_forward(self):
        from unittest.mock import patch

        model = self.model(False)
        before = model.forward.__func__
        with patch.object(model, 'register_forward_pre_hook', side_effect=RuntimeError('injected install failure')):
            with self.assertRaisesRegex(RuntimeError, 'install failure'):
                compat.patch_ling3_packing(model)
        self.assertIs(model.forward.__func__, before)
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(hasattr(model, '_swift_ling3_packing_handle'))

    def test_concurrent_forward_rejected_without_removing_owner_hooks(self):
        import threading

        model = self.model()
        positions = torch.tensor([[0, 1, 0, 1]])
        entered = threading.Event()
        release = threading.Event()
        failures = []
        leaf = model.model.layers[0].attention

        def pause(*args):
            if threading.current_thread() is not thread:
                return
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test synchronization timeout')

        handle = leaf.register_forward_pre_hook(pause)

        def first():
            try:
                model(inputs_embeds=torch.ones(1, 4, 2), position_ids=positions, **helper(positions))
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=first)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaisesRegex(RuntimeError, 'concurrent'):
                model(inputs_embeds=torch.ones(1, 4, 2), position_ids=positions, **helper(positions))
            self.assertGreater(len(leaf._forward_pre_hooks), 1)
            for other in (model, self.model()):
                out = other(inputs_embeds=torch.ones(1, 4, 2), position_ids=torch.arange(4)[None])
                self.assertEqual(out.shape, (1, 4, 2))
            out = self.model()(inputs_embeds=torch.ones(1, 4, 2), position_ids=positions, **helper(positions))
            self.assertEqual(out.shape, (1, 4, 2))
        finally:
            release.set()
            thread.join(5)
            handle.remove()
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(all(not layer.attention._forward_pre_hooks for layer in model.model.layers))

    def test_ordinary_and_single_document_never_install_leaf_hooks(self):
        from unittest.mock import patch

        model = self.model()
        leaf = model.model.layers[0].attention
        hidden = torch.ones(1, 4, 2)
        plain = torch.arange(4)[None]
        packed = torch.tensor([[0, 1, 0, 1]])
        calls = (
            (packed, None, helper(packed)),
            (plain, None, {}),
            (plain, None, helper(plain)),
            (torch.tensor([[1, 1, 0, 1]]), torch.tensor([[0, 0, 1, 1]]), {}),
            (torch.tensor([[0, 1, 0, 0]]), torch.tensor([[1, 1, 0, 0]]), {}),
            (packed, None, helper(packed)),
        )
        with patch.object(leaf, 'register_forward_pre_hook', wraps=leaf.register_forward_pre_hook) as register:
            for (positions, mask, metadata), count in zip(calls, (1, 1, 1, 1, 1, 2)):
                model(inputs_embeds=hidden, position_ids=positions, attention_mask=mask, **metadata)
                self.assertEqual(register.call_count, count)
                self.assertFalse(leaf._forward_pre_hooks)

    def test_signature_and_inference_fallback(self):
        from inspect import signature

        reference = self.model(False)
        before = signature(reference.forward)
        compat.patch_ling3_packing(reference)
        self.assertEqual(before, signature(reference.forward))
        with torch.inference_mode():
            positions = torch.tensor([[0, 1, 0, 1]])
            output = reference(inputs_embeds=torch.ones(1, 4, 2), position_ids=positions, **helper(positions))
            self.assertTrue(torch.isfinite(output).all())

    def test_padding_packing_alternation(self):
        model = self.model()
        reference = self.model(False)
        for mask, positions in ((torch.tensor([[0, 0, 1, 1]]), torch.tensor([[1, 1, 0, 1]])),
                                (torch.tensor([[1, 1, 0, 0]]), torch.tensor([[0, 1, 0,
                                                                              0]])), (None, torch.tensor([[0, 1, 0,
                                                                                                           1]]))):
            x = torch.randn(1, 4, 2)
            metadata = helper(positions) if mask is None else {}
            a = model(inputs_embeds=x, position_ids=positions, attention_mask=mask, **metadata)
            b = reference(inputs_embeds=x, position_ids=positions, attention_mask=mask)
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
