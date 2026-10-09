# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest
from torch import nn

from swift.optimizers.galore import GaLoreAdamW
from swift.optimizers.galore.galore_projector import GaLoreProjector


class TestGaloreProjector(unittest.TestCase):

    def test_projection_matches_explicit_matrix_projection(self):
        cases = [(shape, rank, mode) for shape in ((4, 8), (8, 4), (4, 4)) for rank in (2, 4, 128)
                 for mode in ('std', 'reverse_std')]
        cases.append(((64, 256), 128, 'reverse_std'))
        for shape, rank, mode in cases:
            with self.subTest(shape=shape, rank=rank, mode=mode):
                torch.manual_seed(42)
                gradient = torch.randn(*shape)
                projector = GaLoreProjector(rank, scale=0.5, proj_type=mode)
                actual = projector.project_back(projector.project(gradient, 0))
                u, _, vh = torch.linalg.svd(gradient, full_matrices=False)
                project_right = shape[0] >= shape[1] if mode == 'std' else shape[0] < shape[1]
                if project_right:
                    basis = vh[:rank]
                    expected = (gradient @ basis.T) @ basis
                else:
                    basis = u[:, :rank]
                    expected = basis @ (basis.T @ gradient)
                self.assertEqual(actual.shape, gradient.shape)
                torch.testing.assert_close(actual, expected * 0.5)

    def test_optimizer_matches_explicit_projection_direction(self):
        cases = (((4, 8), 4, 'right'), ((64, 256), 128, 'right'), ((8, 4), 4, 'left'), ((4, 4), 4, 'left'))
        for shape, rank, direction in cases:
            with self.subTest(shape=shape, rank=rank):
                torch.manual_seed(42)
                parameter = nn.Parameter(torch.randn(*shape))
                reference = nn.Parameter(parameter.detach().clone())
                optimizers = [
                    GaLoreAdamW([{
                        'params': [p],
                        'rank': rank,
                        'update_proj_gap': 2,
                        'scale': 0.5,
                        'proj_type': mode,
                    }],
                                lr=0.01,
                                weight_decay=0.1) for p, mode in ((parameter, 'reverse_std'), (reference, direction))
                ]
                for _ in range(3):
                    for p, optimizer in zip((parameter, reference), optimizers):
                        p.square().mean().backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    torch.testing.assert_close(parameter, reference)


if __name__ == '__main__':
    unittest.main()
