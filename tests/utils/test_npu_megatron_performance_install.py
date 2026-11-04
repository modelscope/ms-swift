# Copyright (c) ModelScope Contributors. All rights reserved.
"""Check installation through the actual Swift Megatron entry point."""

import pytest
from transformers.utils import is_torch_npu_available


@pytest.mark.skipif(not is_torch_npu_available(), reason='Requires Ascend NPU')
def test_installation_and_idempotence():
    import importlib
    import megatron_adaptor  # noqa: F401
    from megatron.core.fusions import fused_bias_swiglu
    from megatron.core.optimizer import clip_grads
    from megatron.core.transformer.moe import experts, moe_utils, token_dispatcher
    from transformer_engine.pytorch import optimizers

    original_swiglu_functions = (
        fused_bias_swiglu.weighted_bias_swiglu_impl,
        experts.weighted_bias_swiglu_impl,
    )

    import swift.megatron  # noqa: F401
    from swift.model.npu_patch import megatron_moe as moe
    from swift.model.npu_patch import megatron_optimizer as optimizer

    optimizer.patch_megatron_optimizer()
    original_permute = moe._ORIGINAL_PERMUTE
    moe.patch_megatron_moe()
    assert moe._ORIGINAL_PERMUTE is original_permute
    assert moe_utils.permute is token_dispatcher.permute is moe.permute
    assert original_swiglu_functions == (
        fused_bias_swiglu.weighted_bias_swiglu_impl,
        experts.weighted_bias_swiglu_impl,
    )
    for name in ('multi_tensor_l2norm', 'multi_tensor_unscale_l2norm', 'multi_tensor_scale',
                 'multi_tensor_scale_tensor'):
        module_name = 'multi_tensor_l2norm' if 'l2norm' in name else 'multi_tensor_scale'
        module = importlib.import_module('transformer_engine.pytorch.optimizers.' + module_name)
        assert getattr(module, name) is getattr(optimizers, name) is getattr(optimizer, name)
    assert clip_grads.l2_norm_impl is optimizer.multi_tensor_l2norm
    assert clip_grads.multi_tensor_scale_impl is optimizer.multi_tensor_scale
    assert clip_grads.multi_tensor_scale_tensor_impl is optimizer.multi_tensor_scale_tensor
