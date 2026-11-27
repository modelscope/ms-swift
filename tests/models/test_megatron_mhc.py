# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import pytest
import sys
import torch
from types import SimpleNamespace


def test_glm_cann_mhc_matches_megatron(monkeypatch):
    pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('Requires an Ascend NPU')
    pytest.importorskip('megatron_adaptor')
    import swift.megatron  # Apply the Ascend adaptations before constructing MCore modules.
    from swift.model.npu_patch import megatron_mhc
    hyper_connection = pytest.importorskip('megatron.core.transformer.hyper_connection')
    if not hasattr(hyper_connection, 'native_h_post_bda_fp32'):
        pytest.skip('Requires Megatron-Core with the GLM-5.3 mcore-bridge patch')
    hc = hyper_connection.HyperConnectionModule
    if megatron_mhc.get_cann_mhc_ops() is None:
        pytest.skip('Requires CANN mHC pre/post kernels')

    config = SimpleNamespace(
        hf_model_type='glm5_next',
        num_residual_streams=4,
        hidden_size=4096,
        mhc_sinkhorn_iterations=20,
        mhc_init_gating_factor=0.5,
        mhc_norm_eps_inside_sqrt=True,
        mhc_keep_mappings_in_fp32=True,
        layernorm_epsilon=1e-5,
        use_fused_mhc=False,
        perform_initialization=True,
        sequence_parallel=True,
    )
    # Other tests may also initialize Swift in this process. Restore the class after this test.
    monkeypatch.setattr(hc, '__init__',
                        hc.__init__.__wrapped__ if getattr(hc.__init__, '_glm53_cann_mhc', False) else hc.__init__)
    native_init = hc.__init__
    native = hc(config, 1).to('npu')
    generator = torch.Generator(device='npu').manual_seed(139)
    with torch.no_grad():
        native.mapping_proj.weight.uniform_(-1 / 128, 1 / 128, generator=generator)
        native.bias.uniform_(-0.6, 0.6, generator=generator)
    with monkeypatch.context() as patch:
        patch.setattr(megatron_mhc, 'get_cann_mhc_ops', lambda: None)
        patch.setattr(hc, '__init__', native_init)
        megatron_mhc.patch_megatron_mhc()
        fallback = hc(config, 1)
        assert fallback._forward_normal.__func__ is native._forward_normal.__func__
        assert fallback._h_post_bda_op is native._h_post_bda_op
    with monkeypatch.context() as patch:
        patch.setattr(sys, 'argv', ['megatron', 'pt', '--enable_npu_model_patch', 'false'])
        megatron_mhc.patch_megatron_mhc()
        assert hc.__init__ is native_init
    megatron_mhc.patch_megatron_mhc()
    init = hc.__init__
    megatron_mhc.patch_megatron_mhc()
    assert hc.__init__ is init
    fused = hc(config, 1).to('npu')
    pre_patched = getattr(fused._forward_normal, 'func', None) is megatron_mhc._cann_pre
    post_patched = getattr(fused._h_post_bda_op, 'func', None) is megatron_mhc._cann_post
    assert pre_patched and post_patched
    fused.load_state_dict(native.state_dict())

    # Batch > 1 catches mistakes in the sequence-major to batch-major conversion.
    def random_tensor(width):
        return torch.randn(128, 2, width, device='npu', generator=generator).bfloat16()

    x = random_tensor(4 * config.hidden_size).requires_grad_()
    branch = random_tensor(config.hidden_size).requires_grad_()
    upstream = random_tensor(4 * config.hidden_size)

    # Older MCore returns the residual by default. Newer versions make it opt-in.
    forward_kwargs = {'return_residual': True} if 'return_residual' in inspect.signature(hc.forward).parameters else {}

    def forward_and_grads(module):
        collapsed, h_res, h_post, residual = module(x, **forward_kwargs)
        merged = module.fused_h_res_h_post_bda(h_res, residual, h_post, (collapsed + branch, None), 0., True, True)
        grads = torch.autograd.grad(merged, (x, branch, *module.parameters()), upstream)
        return (collapsed, h_res, h_post, merged, *grads)

    for actual, expected in zip(forward_and_grads(fused), forward_and_grads(native)):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
        assert relative_l2.item() < 0.03
