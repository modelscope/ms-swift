"""Text fast paths must preserve values and the zero-gradient vision graph."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from transformers import Qwen3_5Config

from swift.dev.template.families.qwen import Qwen3_5Template


def test_dummy_preserves_embeddings_and_zero_vision_gradient():
    class Visual(torch.nn.Module):
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, pixels, grid_thw):
            assert pixels.shape == (16, 1536)
            assert grid_thw.tolist() == [[1, 4, 4]]
            return SimpleNamespace(pooler_output=pixels * self.weight)

    visual = Visual()
    embeddings = torch.randn(1, 8, 16, requires_grad=True)
    processor = Mock()
    result = Qwen3_5Template._get_inputs_embeds_hf(embeddings, {}, visual, processor, Qwen3_5Config())
    torch.testing.assert_close(result, embeddings, rtol=0, atol=0)
    result.sum().backward()
    assert visual.weight.grad is not None and visual.weight.grad.item() == 0
    torch.testing.assert_close(embeddings.grad, torch.ones_like(embeddings))
    processor.image_processor.assert_not_called()
