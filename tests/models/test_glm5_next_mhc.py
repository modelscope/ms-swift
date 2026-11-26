# Copyright (c) ModelScope Contributors. All rights reserved.
"""Compare the GLM mHC block and its gradients to native Transformers on a single NPU."""
import argparse
import inspect
from types import SimpleNamespace


def check(args):
    import torch
    from transformers.models.glm5_next import modeling_glm5_next as glm

    from swift.model.npu_patch import glm5_next_mhc as mhc
    original = inspect.unwrap(glm.Glm5NextTextHyperConnection.forward)
    pre, post, _, _ = mhc.make_forwards(args.mode, original, mhc.selected_backend())
    if args.mode in ('post', 'both'):
        decoder = inspect.unwrap(glm.Glm5NextTextDecoderLayer.forward)
        mhc.with_fused_post(decoder, post)
    torch.npu.set_device(args.device)
    torch.manual_seed(1234)
    torch.npu.manual_seed_all(1234)
    device = torch.device(args.device)
    config = SimpleNamespace(hidden_size=4096, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=args.norm_eps)
    hc = glm.Glm5NextTextHyperConnection(config).to(device=device)
    hc.requires_grad_(args.full_grad)
    params = (hc.fn, hc.scale, hc.base) if args.full_grad else ()
    with torch.no_grad():
        bound = (config.hc_mult * config.hidden_size)**-0.5
        hc.fn.uniform_(-bound, bound)
        hc.scale.copy_(torch.tensor([0.6, 0.9, 0.7], device=device))
        hc.base.uniform_(-0.6, 0.6)

    def random(shape, scale=1.0):
        return (torch.randn(shape, device=device) * scale).to(torch.bfloat16)

    d = config.hidden_size
    x = random((1, args.seq_len, 4, d))
    a, b = random((8, d), 0.016), random((d, 8), 0.035)
    upstream = random(x.shape)

    def run(pre_op, post_op):
        x0, a0, b0 = [t.detach().requires_grad_() for t in (x, a, b)]
        post_w, comb, collapsed = pre_op(hc, x0)
        # Small trainable low-rank branch. A nonzero B exercises both gradients.
        linear = torch.nn.functional.linear
        branch = 0.5 * collapsed + linear(linear(collapsed, a0), b0)
        y = post_op(x0, comb, branch, post_w)
        grads = torch.autograd.grad(y, (x0, a0, b0, *params), grad_outputs=upstream)
        return tuple(t.detach() for t in (post_w, comb, collapsed, y, *grads))

    ref = run(original, mhc.native_post)
    actual = run(pre, post)
    ok = True
    names = ('post', 'comb', 'h_in', 'output', 'dx', 'dA', 'dB')
    if args.full_grad:
        names += ('d_fn', 'd_scale', 'd_base')
    for name, expected, observed in zip(names, ref, actual):
        if expected.shape != observed.shape or expected.dtype != observed.dtype:
            raise AssertionError(
                f'{name}: expected {expected.shape}/{expected.dtype}, got {observed.shape}/{observed.dtype}')
        delta = observed.float() - expected.float()
        rel = (delta.norm() / expected.float().norm().clamp_min(1e-12)).item()
        print(f'{name:8s}  relative_L2={rel:.6g}  max_abs={delta.abs().max().item():.6g}')
        ok = ok and rel <= args.max_rel_l2
    if not ok:
        raise AssertionError('FAIL: inspect the failing outputs/gradients before enabling in training')
    print('PASS: one BF16 mHC block and low-rank branch on the selected NPU')


def test_cann_full_grad(monkeypatch):
    import pytest
    import torch
    from transformers.utils import is_torch_npu_available

    if not is_torch_npu_available() or not torch.npu.is_available():
        pytest.skip('Requires an Ascend NPU')
    pytest.importorskip('transformers.models.glm5_next.modeling_glm5_next')
    from swift.model.npu_patch import glm5_next_mhc as mhc
    package = mhc.cann_package()
    if package is None or any(mhc.lookup_op(package, name) is None for name in ('mhc_pre_sinkhorn', 'mhc_post')):
        pytest.skip('Requires CANN mHC operators')
    monkeypatch.setenv('GLM_MHC_BACKEND', 'cann')
    check(argparse.Namespace(mode='both', device='npu:0', seq_len=1024, norm_eps=1e-5, full_grad=True, max_rel_l2=0.03))


def main():
    parser = argparse.ArgumentParser(description='Compare fused/native GLM mHC forward and gradients on one NPU')
    parser.add_argument('--probe', action='store_true', help='Show the selected implementation for each operation')
    parser.add_argument('--full-grad', action='store_true', help='Also compare gradients of fn, scale and base')
    parser.add_argument('--mode', choices=('pre', 'post', 'both'), default='both')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument(
        '--norm-eps',
        type=float,
        default=1e-5,
        help="RMS epsilon for this synthetic check. Training reads the HF module's actual epsilon.")
    parser.add_argument(
        '--max-rel-l2',
        type=float,
        default=0.03,
        help='Diagnostic per-tensor relative-L2 threshold, not an end-to-end acceptance criterion')
    args = parser.parse_args()
    if args.probe:
        from swift.model.npu_patch import glm5_next_mhc as mhc
        backend = mhc.selected_backend()
        package = mhc.cann_package()
        for name in ('mhc_pre_sinkhorn', 'mhc_post'):
            _, implementation = mhc.resolve_op(package, name, backend)
            print(f'{name}: {implementation}')
    else:
        check(args)


if __name__ == '__main__':
    main()
