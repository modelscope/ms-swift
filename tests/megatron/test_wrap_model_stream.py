# Copyright (c) ModelScope Contributors. All rights reserved.
import types
import unittest
from unittest import mock

try:
    import swift.megatron.utils.megatron_lm_utils as megatron_lm_utils
    MEGATRON_UNAVAILABLE = None
except Exception as e:
    # CI does not install megatron, and a mismatched mcore dependency can fail with anything from
    # ImportError to AttributeError, so any failure here means the tests below cannot run.
    megatron_lm_utils = None
    MEGATRON_UNAVAILABLE = f'Megatron dependencies not available: {e}'


@unittest.skipIf(MEGATRON_UNAVAILABLE is not None, MEGATRON_UNAVAILABLE or '')
class TestWrapModelStream(unittest.TestCase):
    """DDP / Megatron-FSDP wrapping should only use a side stream when CUDA graphs need one."""

    def _wrap(self, cuda_graph_impl, use_megatron_fsdp=False):
        M = megatron_lm_utils
        model = mock.MagicMock()
        model.parameters.return_value = []
        model.config = types.SimpleNamespace(cuda_graph_impl=cuda_graph_impl)
        args = types.SimpleNamespace(
            fp16=False,
            bf16=False,
            use_megatron_fsdp=use_megatron_fsdp,
            overlap_grad_reduce=False,
            overlap_param_gather_with_optimizer_step=False,
            data_parallel_random_init=False)
        current_stream = mock.MagicMock(name='current_stream')
        side_stream = mock.MagicMock(name='side_stream')
        dp_cls = mock.MagicMock(name='DP')
        dp_name = 'megatron_FSDP' if use_megatron_fsdp else 'DDP'
        mpu = mock.MagicMock()
        mpu.get_data_parallel_world_size.return_value = 1
        with mock.patch.object(M.torch.cuda, 'current_device', return_value=0), \
             mock.patch.object(M.torch.cuda, 'current_stream', return_value=current_stream), \
             mock.patch.object(M.torch.cuda, 'Stream', return_value=side_stream) as stream_cls, \
             mock.patch.object(M.torch.cuda, 'stream') as stream_ctx, \
             mock.patch.object(M, 'mpu', mpu), \
             mock.patch.object(M, dp_name, dp_cls):
            wrapped = M.wrap_model(args, [model])
        self.assertEqual(wrapped, [dp_cls.return_value])
        return current_stream, side_stream, stream_cls, stream_ctx

    def test_eager_uses_current_stream(self):
        for use_megatron_fsdp in (False, True):
            with self.subTest(use_megatron_fsdp=use_megatron_fsdp):
                current_stream, _, stream_cls, stream_ctx = self._wrap('none', use_megatron_fsdp)
                stream_cls.assert_not_called()
                stream_ctx.assert_called_once_with(current_stream)
                current_stream.wait_stream.assert_not_called()

    def test_cuda_graph_keeps_side_stream(self):
        for cuda_graph_impl in ('local', 'transformer_engine', 'full_iteration'):
            with self.subTest(cuda_graph_impl=cuda_graph_impl):
                current_stream, side_stream, stream_cls, stream_ctx = self._wrap(cuda_graph_impl)
                stream_cls.assert_called_once_with()
                stream_ctx.assert_called_once_with(side_stream)
                side_stream.wait_stream.assert_called_once_with(current_stream)
                current_stream.wait_stream.assert_called_once_with(side_stream)


if __name__ == '__main__':
    unittest.main()
