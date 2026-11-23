"""Text fast paths must preserve positions, values and the zero-gradient vision graph."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

from swift.dev.template.families.qwen import Qwen2VLTemplate, Qwen3_5Template


@pytest.mark.parametrize("mask", [None, [1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0]])
def test_text_positions_match_transformers(mask):
    template = object.__new__(Qwen3_5Template)
    model = SimpleNamespace(config=Qwen3_5Config())
    template._get_get_rope_index = lambda: partial(Qwen3_5Model.get_rope_index, model)
    inputs = {"input_ids": torch.tensor([[10, 11, 12, 13]]), "mm_token_type_ids": torch.zeros(1, 4, dtype=torch.long)}
    if mask is not None:
        inputs["attention_mask"] = torch.tensor([mask])
    expected = Qwen2VLTemplate._get_position_ids(template, inputs)["position_ids"]
    actual = template._get_position_ids(inputs)["position_ids"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("key", ["image_grid_thw", "video_grid_thw", "mm_token_type_ids"])
def test_multimodal_positions_keep_original_path(key, monkeypatch):
    template = object.__new__(Qwen3_5Template)
    original = Mock(return_value={"position_ids": "multimodal"})
    monkeypatch.setattr(Qwen2VLTemplate, "_get_position_ids", original)
    result = template._get_position_ids({key: torch.ones(1, 3, dtype=torch.long)})
    assert result == {"position_ids": "multimodal"}
    original.assert_called_once()


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
