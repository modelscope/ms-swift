import torch
import unittest
from transformers import DynamicCache, PretrainedConfig
from transformers.cache_utils import Cache
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swift.model.models import qwen


class _CurrentLayer:

    def __init__(self):
        self.update_conv_state = Mock(name='layer_update_conv_state')
        self.update_recurrent_state = Mock(name='layer_update_recurrent_state')


class _CurrentCache:

    def __init__(self, previous=False, num_layers=1):
        self.layers = [_CurrentLayer() for _ in range(num_layers)]
        self.has_previous_state = Mock(name='has_previous_state', return_value=previous)
        self.update_conv_state = Mock(name='update_conv_state')
        self.update_recurrent_state = Mock(name='update_recurrent_state')


class _LegacyCache:

    def __init__(self, previous=False, num_layers=1):
        self.conv_states = [None] * num_layers
        self.recurrent_states = [None] * num_layers
        self.has_previous_state = previous


class _GenericCache:

    def __init__(self, num_layers=1):
        self.layers = [object() for _ in range(num_layers)]
        self.has_previous_state = Mock(
            name='generic_has_previous_state', side_effect=AssertionError('generic cache must not be queried'))
        self.update_conv_state = Mock(name='generic_update_conv_state')
        self.update_recurrent_state = Mock(name='generic_update_recurrent_state')


class _FakeGatedDeltaNet:
    _apply_mask_to_padding_states = staticmethod(lambda hidden_states, attention_mask: hidden_states)

    def __init__(self, layer_idx=0):
        self.layer_idx = layer_idx
        self.conv_kernel_size = 4
        self.num_k_heads = 1
        self.num_v_heads = 1
        self.head_k_dim = 2
        self.head_v_dim = 2
        self.key_dim = 2
        self.value_dim = 2
        self.activation = 'silu'
        self.conv1d = SimpleNamespace(weight=torch.ones(6, 1, 4), bias=None)
        self.A_log = torch.zeros(1)
        self.dt_bias = torch.zeros(1)
        self.in_proj_qkv = Mock(name='in_proj_qkv', side_effect=lambda hidden: torch.cat([hidden] * 3, dim=-1))
        self.in_proj_z = Mock(name='in_proj_z', side_effect=lambda hidden: hidden)
        self.in_proj_b = Mock(
            name='in_proj_b', side_effect=lambda hidden: torch.zeros(*hidden.shape[:-1], self.num_v_heads))
        self.in_proj_a = Mock(
            name='in_proj_a', side_effect=lambda hidden: torch.zeros(*hidden.shape[:-1], self.num_v_heads))
        self.out_proj = Mock(name='out_proj', side_effect=lambda hidden: hidden)
        self.norm = Mock(name='norm', side_effect=lambda hidden, gate: hidden)
        self._swift_fla_causal_conv1d_fn = Mock(name='causal_conv', side_effect=lambda *, x, **kwargs: (x, None))
        self.chunk_gated_delta_rule = Mock(name='chunk_gated_delta_rule', side_effect=self._chunk)
        self.output_final_states = []

    def _chunk(self, query, key, value, **kwargs):
        self.output_final_states.append(kwargs['output_final_state'])
        final_state = torch.full((value.shape[0], value.shape[2], value.shape[3], value.shape[3]), 7.0)
        return value, final_state if kwargs['output_final_state'] else None


class TestQwen35CacheLayout(unittest.TestCase):

    def test_cache_none(self):
        self.assertEqual(qwen._get_qwen3_5_linear_cache_layout(None, 0), (None, False))

    def test_real_current_cache_updates_default_state_slot(self):
        if not hasattr(Cache, 'update_conv_state'):
            self.skipTest('Transformers does not expose the current linear-attention Cache API')

        config = PretrainedConfig(num_hidden_layers=2, layer_types=['linear_attention', 'full_attention'])
        cache = DynamicCache(config=config)
        layout, previous = qwen._get_qwen3_5_linear_cache_layout(cache, 0)
        self.assertEqual(layout, qwen._QWEN3_5_CURRENT_CACHE)
        self.assertFalse(previous)

        conv_state = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        recurrent_state = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
        qwen._update_qwen3_5_linear_cache_state(cache, layout, 0, conv_state, recurrent=False)
        qwen._update_qwen3_5_linear_cache_state(cache, layout, 0, recurrent_state, recurrent=True)

        stored_conv = cache.layers[0].conv_states
        stored_recurrent = cache.layers[0].recurrent_states
        if isinstance(stored_conv, dict):
            stored_conv = stored_conv[0]
            stored_recurrent = stored_recurrent[0]
        torch.testing.assert_close(stored_conv, conv_state)
        torch.testing.assert_close(stored_recurrent, recurrent_state)
        self.assertTrue(cache.has_previous_state(0))

        self.assertEqual(qwen._get_qwen3_5_linear_cache_layout(cache, 1), (None, False))

    def test_current_layout_uses_public_container_and_selected_layer(self):
        cache = _CurrentCache(previous=False, num_layers=3)
        layout, previous = qwen._get_qwen3_5_linear_cache_layout(cache, 1)
        self.assertEqual(layout, qwen._QWEN3_5_CURRENT_CACHE)
        self.assertFalse(previous)
        cache.has_previous_state.assert_called_once_with(1)

        conv_state = torch.tensor([1.0])
        recurrent_state = torch.tensor([2.0])
        qwen._update_qwen3_5_linear_cache_state(cache, layout, 1, conv_state, recurrent=False)
        qwen._update_qwen3_5_linear_cache_state(cache, layout, 1, recurrent_state, recurrent=True)

        cache.update_conv_state.assert_called_once_with(conv_state, 1)
        cache.update_recurrent_state.assert_called_once_with(recurrent_state, 1)
        for layer in cache.layers:
            layer.update_conv_state.assert_not_called()
            layer.update_recurrent_state.assert_not_called()

    def test_current_layout_takes_precedence_over_legacy(self):
        cache = _CurrentCache(previous=False)
        cache.conv_states = [None]
        cache.recurrent_states = [None]

        layout, previous = qwen._get_qwen3_5_linear_cache_layout(cache, 0)
        self.assertEqual(layout, qwen._QWEN3_5_CURRENT_CACHE)
        self.assertFalse(previous)
        state = torch.tensor([1.0])
        qwen._update_qwen3_5_linear_cache_state(cache, layout, 0, state, recurrent=False)

        cache.update_conv_state.assert_called_once_with(state, 0)
        self.assertIsNone(cache.conv_states[0])

    def test_legacy_layout_updates_only_selected_layer(self):
        cache = _LegacyCache(previous=False, num_layers=3)
        layout, previous = qwen._get_qwen3_5_linear_cache_layout(cache, 1)
        self.assertEqual(layout, qwen._QWEN3_5_LEGACY_CACHE)
        self.assertFalse(previous)

        conv_state = torch.tensor([1.0])
        recurrent_state = torch.tensor([2.0])
        qwen._update_qwen3_5_linear_cache_state(cache, layout, 1, conv_state, recurrent=False)
        qwen._update_qwen3_5_linear_cache_state(cache, layout, 1, recurrent_state, recurrent=True)

        self.assertEqual(cache.conv_states, [None, conv_state, None])
        self.assertEqual(cache.recurrent_states, [None, recurrent_state, None])

    def test_generic_invalid_and_out_of_range_caches_are_not_queried(self):
        generic = _GenericCache()
        self.assertEqual(qwen._get_qwen3_5_linear_cache_layout(generic, 0), (None, False))
        generic.has_previous_state.assert_not_called()

        current = _CurrentCache(previous=False)
        self.assertEqual(qwen._get_qwen3_5_linear_cache_layout(current, -1), (None, False))
        self.assertEqual(qwen._get_qwen3_5_linear_cache_layout(current, 1), (None, False))
        current.has_previous_state.assert_not_called()

        current.update_conv_state = None
        self.assertEqual(qwen._get_qwen3_5_linear_cache_layout(current, 0), (None, False))
        current.has_previous_state.assert_not_called()

    def test_recognized_callable_errors_propagate(self):

        class CacheError(RuntimeError):
            pass

        for error in (TypeError('bad signature'), CacheError('broken cache')):
            with self.subTest(error=type(error).__name__):
                cache = _CurrentCache(previous=False)
                cache.has_previous_state.side_effect = error
                with self.assertRaises(type(error)) as raised:
                    qwen._get_qwen3_5_linear_cache_layout(cache, 0)
                self.assertIs(raised.exception, error)
                cache.has_previous_state.assert_called_once_with(0)

    def test_legacy_requires_non_callable_previous_state(self):
        cache = _LegacyCache(previous=False)
        cache.has_previous_state = Mock(return_value=False)
        self.assertEqual(qwen._get_qwen3_5_linear_cache_layout(cache, 0), (None, False))
        cache.has_previous_state.assert_not_called()

    def test_legacy_rejects_non_list_state_containers_without_query(self):
        for states in ((None, ), {0: None}):
            with self.subTest(container=type(states).__name__):
                cache = SimpleNamespace(
                    conv_states=states,
                    recurrent_states=states,
                    has_previous_state=Mock(side_effect=AssertionError('invalid legacy cache must not be queried')),
                )
                self.assertEqual(qwen._get_qwen3_5_linear_cache_layout(cache, 0), (None, False))
                cache.has_previous_state.assert_not_called()


class TestQwen35CacheForward(unittest.TestCase):

    @staticmethod
    def _hidden(seq_len=2):
        return torch.arange(seq_len * 2, dtype=torch.float32).reshape(1, seq_len, 2) / 10

    def _forward(self, mod, cache, seq_len=2, cache_position=None):
        return qwen._run_qwen3_5_gated_delta_net_sequence_parallel_forward(
            mod,
            self._hidden(seq_len),
            cache_params=cache,
            cache_position=cache_position,
            cu_seq_lens_q=torch.tensor([0, seq_len], dtype=torch.int32),
        )

    def test_no_cache_and_generic_cache_match_without_state_work(self):
        generic = _GenericCache()
        no_cache_mod = _FakeGatedDeltaNet()
        generic_mod = _FakeGatedDeltaNet()

        with patch.object(qwen, '_ensure_linear_attention_kernels') as ensure, \
                patch.object(qwen.sequence_parallel, 'enabled', return_value=False), \
                patch.object(qwen.F, 'pad', wraps=qwen.F.pad) as pad:
            no_cache_output = self._forward(no_cache_mod, None)
            generic_output = self._forward(generic_mod, generic)

        torch.testing.assert_close(no_cache_output, generic_output)
        self.assertEqual(no_cache_mod.output_final_states, [False])
        self.assertEqual(generic_mod.output_final_states, [False])
        self.assertEqual(pad.call_count, 0)
        self.assertEqual(ensure.call_count, 2)
        generic.has_previous_state.assert_not_called()
        generic.update_conv_state.assert_not_called()
        generic.update_recurrent_state.assert_not_called()

    def test_current_empty_cache_uses_public_updates_for_single_token_prefill(self):
        cache = _CurrentCache(previous=False)
        mod = _FakeGatedDeltaNet()

        with patch.object(qwen, '_ensure_linear_attention_kernels'), \
                patch.object(qwen.sequence_parallel, 'enabled', return_value=False), \
                patch.object(qwen.F, 'pad', wraps=qwen.F.pad) as pad:
            output = self._forward(mod, cache, seq_len=1, cache_position=torch.tensor([0]))

        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(mod.output_final_states, [True])
        self.assertEqual(pad.call_count, 1)
        cache.has_previous_state.assert_called_once_with(0)
        cache.update_conv_state.assert_called_once()
        cache.update_recurrent_state.assert_called_once()
        self.assertEqual(cache.update_conv_state.call_args.args[1], 0)
        self.assertEqual(cache.update_recurrent_state.call_args.args[1], 0)
        cache.layers[0].update_conv_state.assert_not_called()
        cache.layers[0].update_recurrent_state.assert_not_called()

    def test_legacy_empty_cache_updates_selected_layer(self):
        cache = _LegacyCache(previous=False, num_layers=2)
        mod = _FakeGatedDeltaNet(layer_idx=1)

        with patch.object(qwen, '_ensure_linear_attention_kernels'), \
                patch.object(qwen.sequence_parallel, 'enabled', return_value=False):
            output = self._forward(mod, cache)

        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(mod.output_final_states, [True])
        self.assertIsNone(cache.conv_states[0])
        self.assertIsNone(cache.recurrent_states[0])
        self.assertIsNotNone(cache.conv_states[1])
        self.assertIsNotNone(cache.recurrent_states[1])

    def test_initialized_cache_fails_before_kernel_setup_projection_or_mutation(self):
        for cache_cls in (_CurrentCache, _LegacyCache):
            for seq_len in (1, 2):
                with self.subTest(cache=cache_cls.__name__, seq_len=seq_len):
                    cache = cache_cls(previous=True)
                    mod = _FakeGatedDeltaNet()
                    with patch.object(
                            qwen, '_ensure_linear_attention_kernels',
                            side_effect=AssertionError('kernel setup called')) as ensure:
                        with self.assertRaisesRegex(NotImplementedError, 'initialized cache state'):
                            self._forward(mod, cache, seq_len=seq_len, cache_position=torch.arange(seq_len))

                    ensure.assert_not_called()
                    mod.in_proj_qkv.assert_not_called()
                    if isinstance(cache, _CurrentCache):
                        cache.update_conv_state.assert_not_called()
                        cache.update_recurrent_state.assert_not_called()
                        cache.has_previous_state.assert_called_once_with(0)
                    else:
                        self.assertEqual(cache.conv_states, [None])
                        self.assertEqual(cache.recurrent_states, [None])


class TestQwen35PatchDispatch(unittest.TestCase):

    def test_dense_and_moe_keep_distinct_origins_and_share_custom_path(self):
        dense_calls = []
        moe_calls = []

        def dense_origin(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None, **kwargs):
            dense_calls.append((self, hidden_states, cache_params, cache_position, attention_mask, kwargs))
            return 'dense-origin'

        def moe_origin(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None, **kwargs):
            moe_calls.append((self, hidden_states, cache_params, cache_position, attention_mask, kwargs))
            return 'moe-origin'

        class Dense:
            forward = dense_origin

        class Moe:
            forward = moe_origin

        modules = {
            'transformers.models.qwen3_5.modeling_qwen3_5':
            SimpleNamespace(Qwen3_5GatedDeltaNet=Dense, apply_mask_to_padding_states=lambda hidden, mask: hidden),
            'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe':
            SimpleNamespace(Qwen3_5MoeGatedDeltaNet=Moe, apply_mask_to_padding_states=lambda hidden, mask: hidden),
        }

        def custom_forward(mod, hidden_states, **kwargs):
            return f'{type(mod).__name__}-custom'

        with patch.object(qwen, 'import_module', side_effect=lambda name: modules[name]), \
                patch.object(qwen, '_run_qwen3_5_gated_delta_net_sequence_parallel_forward',
                             side_effect=custom_forward) as custom, \
                patch.object(qwen.sequence_parallel, 'enabled', return_value=False) as sp_enabled, \
                patch.object(qwen.sequence_parallel, 'rp_world_size', 1):
            qwen._patch_qwen3_5_linear_attention_sequence_parallel()
            dense = Dense()
            moe = Moe()
            dense_cache = object()
            moe_cache = object()

            self.assertEqual(
                dense.forward('dense-hidden', cache_params=dense_cache, cache_position='dense-position'),
                'dense-origin')
            self.assertEqual(
                moe.forward('moe-hidden', cache_params=moe_cache, cache_position='moe-position'), 'moe-origin')
            self.assertEqual(dense.forward('dense-packed', cu_seq_lens_q=torch.tensor([0, 1])), 'Dense-custom')
            self.assertEqual(moe.forward('moe-packed', cu_seq_lens_q=torch.tensor([0, 1])), 'Moe-custom')
            sp_enabled.return_value = True
            self.assertEqual(dense.forward('dense-sp'), 'Dense-custom')
            self.assertEqual(moe.forward('moe-sp'), 'Moe-custom')

        self.assertEqual(dense_calls, [(dense, 'dense-hidden', dense_cache, 'dense-position', None, {})])
        self.assertEqual(moe_calls, [(moe, 'moe-hidden', moe_cache, 'moe-position', None, {})])
        self.assertEqual(custom.call_count, 4)
        self.assertEqual([call.args[0] for call in custom.call_args_list], [dense, moe, dense, moe])


if __name__ == '__main__':
    unittest.main()
