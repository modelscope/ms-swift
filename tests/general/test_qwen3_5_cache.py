import torch
import unittest
from transformers import DynamicCache, PretrainedConfig
from transformers.cache_utils import Cache
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swift.model.models import qwen


class _CurrentLayer:

    def __init__(self):
        self.update_conv_state = Mock()
        self.update_recurrent_state = Mock()


class _CurrentCache:

    def __init__(self, previous=False):
        self.layers = [_CurrentLayer()]
        self.has_previous_state = Mock(return_value=previous)
        self.update_conv_state = Mock()
        self.update_recurrent_state = Mock()


class _LegacyCache:

    def __init__(self, previous=False, num_layers=1):
        self.conv_states = [None] * num_layers
        self.recurrent_states = [None] * num_layers
        self.has_previous_state = previous


class TestQwen35Cache(unittest.TestCase):

    def test_current_cache_uses_public_container_api(self):
        if not hasattr(Cache, 'update_conv_state'):
            self.skipTest('Transformers does not expose the current linear-attention cache API')

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

        cache = _CurrentCache()
        cache.conv_states = [None]
        cache.recurrent_states = [None]
        layout, _ = qwen._get_qwen3_5_linear_cache_layout(cache, 0)
        cache.has_previous_state.assert_called_once_with(0)
        state = torch.tensor([1.0])
        qwen._update_qwen3_5_linear_cache_state(cache, layout, 0, state, recurrent=False)
        cache.update_conv_state.assert_called_once_with(state, 0)
        cache.layers[0].update_conv_state.assert_not_called()
        self.assertIsNone(cache.conv_states[0])

    def test_legacy_cache_updates_only_selected_layer(self):
        cache = _LegacyCache(num_layers=3)
        layout, previous = qwen._get_qwen3_5_linear_cache_layout(cache, 1)
        self.assertEqual(layout, qwen._QWEN3_5_LEGACY_CACHE)
        self.assertFalse(previous)

        conv_state = torch.tensor([1.0])
        recurrent_state = torch.tensor([2.0])
        qwen._update_qwen3_5_linear_cache_state(cache, layout, 1, conv_state, recurrent=False)
        qwen._update_qwen3_5_linear_cache_state(cache, layout, 1, recurrent_state, recurrent=True)
        self.assertEqual(cache.conv_states, [None, conv_state, None])
        self.assertEqual(cache.recurrent_states, [None, recurrent_state, None])

    def test_generic_cache_is_not_queried_or_mutated(self):
        cache = SimpleNamespace(
            layers=[object()],
            has_previous_state=Mock(side_effect=AssertionError('generic cache must not be queried')),
            update_conv_state=Mock(),
            update_recurrent_state=Mock(),
        )
        self.assertEqual(qwen._get_qwen3_5_linear_cache_layout(cache, 0), (None, False))
        cache.has_previous_state.assert_not_called()
        cache.update_conv_state.assert_not_called()
        cache.update_recurrent_state.assert_not_called()

    def test_initialized_cache_fails_before_kernel_setup_or_mutation(self):
        for cache in (_CurrentCache(previous=True), _LegacyCache(previous=True)):
            with self.subTest(cache=type(cache).__name__):
                mod = SimpleNamespace(layer_idx=0)
                with patch.object(
                        qwen, '_ensure_linear_attention_kernels',
                        side_effect=AssertionError('kernel setup called')) as ensure:
                    with self.assertRaisesRegex(NotImplementedError, 'initialized cache state'):
                        qwen._run_qwen3_5_gated_delta_net_sequence_parallel_forward(
                            mod, torch.empty(1, 1, 1), cache_params=cache)

                ensure.assert_not_called()
                if isinstance(cache, _CurrentCache):
                    cache.has_previous_state.assert_called_once_with(0)
                    cache.update_conv_state.assert_not_called()
                    cache.update_recurrent_state.assert_not_called()
                else:
                    self.assertEqual(cache.conv_states, [None])
                    self.assertEqual(cache.recurrent_states, [None])


if __name__ == '__main__':
    unittest.main()
