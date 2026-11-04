# Copyright (c) ModelScope Contributors. All rights reserved.
import io
import torch
import unittest
from copy import deepcopy
from transformers.utils import is_bitsandbytes_available

from swift.optimizers.galore import GaLoreAdafactor, GaLoreAdamW
from swift.optimizers.galore.galore_projector import GaLoreProjector


def _optimizer(cls, parameter, projection):
    kwargs = {'scale_parameter': False, 'relative_step': False} if cls is GaLoreAdafactor else {}
    return cls([{
        'params': [parameter],
        'rank': 2,
        'update_proj_gap': 4,
        'scale': 0.7,
        'proj_type': projection
    }],
               lr=0.01,
               **kwargs)


def _roundtrip(state):
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    return torch.load(stream, map_location='cpu', weights_only=True)


class TestGaLoreCheckpoint(unittest.TestCase):

    def _check_resume(self, cls, projection, device='cpu', shape=(8, 4)):
        generator = torch.Generator().manual_seed(42)
        parameter = torch.nn.Parameter(torch.randn(*shape, generator=generator, device='cpu').to(device))
        optimizer = _optimizer(cls, parameter, projection)
        gradients = [torch.randn(*shape, generator=generator).to(device) for _ in range(7)]
        for gradient in gradients[:3]:
            parameter.grad = gradient.clone()
            optimizer.step()
        saved = _roundtrip(optimizer.state_dict())
        # Serialization must not replace the live projector or alter the caller's checkpoint.
        self.assertIsInstance(optimizer.state[parameter]['projector'], GaLoreProjector)
        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = _optimizer(cls, restored_parameter, projection)
        restored.load_state_dict(saved)
        self.assertIsInstance(saved['state'][0]['projector'], dict)
        # Resume before the next SVD refresh, then also cross the refresh boundary.
        for gradient in gradients[3:]:
            parameter.grad = gradient.clone()
            restored_parameter.grad = gradient.clone()
            optimizer.step()
            restored.step()
            torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)

    def test_cpu_resume(self):
        for cls in [GaLoreAdamW, GaLoreAdafactor]:
            for projection in ['std', 'reverse_std', 'left', 'right', 'full']:
                with self.subTest(optimizer=cls.__name__, projection=projection):
                    self._check_resume(cls, projection)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_cpu_mapped_checkpoint_resumes_on_cuda(self):
        for cls in [GaLoreAdamW, GaLoreAdafactor]:
            for projection in ['std', 'full']:
                with self.subTest(optimizer=cls.__name__, projection=projection):
                    self._check_resume(cls, projection, 'cuda')

    @unittest.skipUnless(torch.cuda.is_available() and is_bitsandbytes_available(), 'CUDA and bitsandbytes required')
    def test_8bit_resume(self):
        from swift.optimizers.galore import GaLoreAdamW8bit
        self._check_resume(GaLoreAdamW8bit, 'std', 'cuda', shape=(2048, 8))

    def test_unprojected_state_and_legacy_projector(self):
        for cls in [GaLoreAdamW, GaLoreAdafactor]:
            with self.subTest(optimizer=cls.__name__):
                parameter = torch.nn.Parameter(torch.ones(8, 4))
                optimizer = _optimizer(cls, parameter, 'std')
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                legacy = deepcopy(optimizer.state_dict())
                legacy['state'][0]['projector'] = deepcopy(optimizer.state[parameter]['projector'])
                restored = _optimizer(cls, torch.nn.Parameter(parameter.detach().clone()), 'std')
                restored.load_state_dict(legacy)
                self.assertIsInstance(next(iter(restored.state.values()))['projector'], GaLoreProjector)
                _roundtrip(restored.state_dict())
                plain_parameter = torch.nn.Parameter(torch.ones(3))
                kwargs = {'scale_parameter': False, 'relative_step': False} if cls is GaLoreAdafactor else {}
                plain = cls([plain_parameter], lr=0.01, **kwargs)
                plain_parameter.grad = torch.ones_like(plain_parameter)
                plain.step()
                plain.load_state_dict(_roundtrip(plain.state_dict()))
                self.assertNotIn('projector', plain.state[plain_parameter])


if __name__ == '__main__':
    unittest.main()
