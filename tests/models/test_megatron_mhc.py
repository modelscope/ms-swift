# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import pytest
import sys
import torch
from types import SimpleNamespace
from unittest.mock import Mock


def test_disabled_patch_skips_mcore(monkeypatch):
    from swift.model.npu_patch import megatron_mhc
    load_module = Mock()
    monkeypatch.setattr(sys, 'argv', ['megatron', 'pt', '--enable_npu_model_patch', 'false'])
    monkeypatch.setattr(megatron_mhc, 'import_optional_module', load_module)
    megatron_mhc.patch_megatron_mhc()
    load_module.assert_not_called()


@pytest.mark.parametrize('model_type', ['glm5_next', 'other_model'])
def test_native_methods_preserved(monkeypatch, model_type):
    from swift.model.npu_patch import megatron_mhc

    class HyperConnection:
        _forward_normal = Mock()
        _h_post_bda_op = Mock()

        def __init__(self, config, layer_number):
            self.n = 4
            self.sinkhorn_iterations = 20
            self.sinkhorn_eps = self.compute_h_eps = 1e-6

    module = SimpleNamespace(HyperConnectionModule=HyperConnection)
    monkeypatch.setattr(megatron_mhc, 'import_optional_module', lambda name: module)
    # GLM falls back when the operators are missing. Other models keep their methods even if both exist.
    get_ops = Mock(return_value=None if model_type == 'glm5_next' else (Mock(), Mock()))
    monkeypatch.setattr(megatron_mhc, 'get_cann_mhc_ops', get_ops)
    native_init = HyperConnection.__init__
    native_pre, native_post = HyperConnection._forward_normal, HyperConnection._h_post_bda_op
    megatron_mhc.patch_megatron_mhc()
    patched_init = HyperConnection.__init__
    assert patched_init is not native_init
    megatron_mhc.patch_megatron_mhc()
    assert HyperConnection.__init__ is patched_init

    config = SimpleNamespace(hf_model_type=model_type, mhc_norm_eps_inside_sqrt=True, mhc_keep_mappings_in_fp32=True)
    layer = HyperConnection(config, 1)
    assert layer._forward_normal is native_pre
    assert layer._h_post_bda_op is native_post
    if model_type == 'glm5_next':
        get_ops.assert_called_once_with()
    else:
        get_ops.assert_not_called()


def test_glm_cann_mhc_matches_megatron(monkeypatch):
    pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('Requires an Ascend NPU')
    pytest.importorskip('megatron_adaptor')
    from swift.model.npu_patch import megatron_mhc
    hyper_connection = pytest.importorskip('megatron.core.transformer.hyper_connection')
    if not hasattr(hyper_connection, 'native_h_post_bda_fp32'):
        pytest.skip('Requires Megatron-Core with the GLM-5.3 mcore-bridge patch')
    hc = hyper_connection.HyperConnectionModule
    ops = megatron_mhc.get_cann_mhc_ops()
    if ops is None:
        pytest.skip('Requires CANN mHC pre/post kernels')
    import swift.megatron  # Apply the Ascend adaptations before constructing MCore modules.

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
    native = hc(config, 1).to('npu')
    generator = torch.Generator(device='npu').manual_seed(139)
    with torch.no_grad():
        native.mapping_proj.weight.uniform_(-1 / 128, 1 / 128, generator=generator)
        native.bias.uniform_(-0.6, 0.6, generator=generator)
    cann_pre, cann_post = (Mock(wraps=op) for op in ops)
    monkeypatch.setattr(megatron_mhc, 'get_cann_mhc_ops', lambda: (cann_pre, cann_post))
    megatron_mhc.patch_megatron_mhc()
    fused = hc(config, 1).to('npu')
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

    expected_values = forward_and_grads(native)
    actual_values = forward_and_grads(fused)
    cann_pre.assert_called_once()
    cann_post.assert_called_once()
    for actual, expected in zip(actual_values, expected_values):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
        assert relative_l2.item() < 0.03
