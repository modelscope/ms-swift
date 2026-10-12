# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
import unittest
from unittest.mock import patch

from swift.optimizers.muonclip import MuonClip, _MaxLogitsTracker


class TestMuonClip(unittest.TestCase):

    def setUp(self):
        # These optimizer tests do not need global attention instrumentation.
        tracker = patch.object(_MaxLogitsTracker, 'enable_all')
        tracker.start()
        self.addCleanup(tracker.stop)

    def test_muon_updates_match_reference(self):
        for shape in ((3, 5), (5, 3)):
            for nesterov in (False, True):
                with self.subTest(shape=shape, nesterov=nesterov):
                    generator = torch.Generator().manual_seed(42)
                    param = torch.nn.Parameter(torch.randn(shape, generator=generator))
                    expected = param.detach().clone()
                    buffer = torch.zeros_like(param)
                    momentum, lr = 0.9, 0.1
                    optimizer = MuonClip([param],
                                         lr=lr,
                                         momentum=momentum,
                                         nesterov=nesterov,
                                         weight_decay=0,
                                         qk_clip_enabled=False)

                    for step in range(1, 4):
                        grad = torch.randn(shape, generator=generator)
                        param.grad = grad.clone()
                        buffer = momentum * buffer + grad
                        # Moonlight combines Nesterov momentum before orthogonalization.
                        direction = grad.add(buffer, alpha=momentum) if nesterov else buffer
                        orth = MuonClip.newton_schulz(direction)
                        expected.add_(orth * (0.2 * math.sqrt(max(shape))), alpha=-lr)

                        optimizer.step(max_logits=0.0)

                        torch.testing.assert_close(param, expected)
                        torch.testing.assert_close(param.grad, grad, rtol=0, atol=0)
                        torch.testing.assert_close(optimizer.state[param]['momentum_buffer'], buffer)
                        self.assertEqual(optimizer.state[param]['step'], step)

    def test_muon_update_norm_is_invariant_to_gradient_scale(self):
        generator = torch.Generator().manual_seed(42)
        grad = torch.randn((8, 4), generator=generator)
        for nesterov in (False, True):
            with self.subTest(nesterov=nesterov):
                norms = []
                for scale in (1.0, 100.0):
                    param = torch.nn.Parameter(torch.zeros_like(grad))
                    optimizer = MuonClip([param], lr=1.0, nesterov=nesterov, weight_decay=0, qk_clip_enabled=False)
                    param.grad = grad * scale
                    optimizer.step(max_logits=0.0)
                    norms.append(param.detach().norm())
                # Allow rounding differences from the BF16 Newton-Schulz iterations.
                torch.testing.assert_close(norms[1] / norms[0], torch.tensor(1.0), rtol=0.03, atol=0)

    def test_non_muon_updates_match_sgd(self):
        for shape, apply_muon in (((4, ), True), ((3, 5), False)):
            for nesterov in (False, True):
                with self.subTest(shape=shape, apply_muon=apply_muon, nesterov=nesterov):
                    generator = torch.Generator().manual_seed(42)
                    param = torch.nn.Parameter(torch.randn(shape, generator=generator))
                    expected = torch.nn.Parameter(param.detach().clone())
                    optimizer = MuonClip([{
                        'params': [param],
                        'apply_muon': apply_muon
                    }],
                                         lr=0.1,
                                         momentum=0.9,
                                         nesterov=nesterov,
                                         weight_decay=0,
                                         qk_clip_enabled=False)
                    reference = torch.optim.SGD([expected], lr=0.1, momentum=0.9, nesterov=nesterov)
                    for _ in range(3):
                        grad = torch.randn(shape, generator=generator)
                        param.grad = grad.clone()
                        expected.grad = grad.clone()
                        optimizer.step(max_logits=0.0)
                        reference.step()
                        torch.testing.assert_close(param, expected)
                        torch.testing.assert_close(param.grad, grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
