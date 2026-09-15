import math
from types import SimpleNamespace

import torch
from PIL import Image

from swift import InferRequest
from swift.model import get_processor
from swift.template import get_template
from swift.template.templates.deepseek import DeepseekV41Template


MODEL_ID = 'deepseek-ai/DeepSeek-V4.1-Flash'


def _get_template():
    processor = get_processor(MODEL_ID, model_type='deepseek_v41')
    return get_template(processor)


def test_deepseek_v41_image_preprocessing_layout():
    config = SimpleNamespace(
        patch_size=14,
        downsample_ratio=3,
        max_image_tokens=1024,
        min_pixels=0,
        max_wh_ratio=None,
    )
    image = Image.new('RGB', (112, 84), (255, 127, 0))
    patches, grid_thw, token_types = DeepseekV41Template._process_image(image, config)

    assert grid_thw == (1, 6, 8)
    assert patches.shape == (48, 3, 14, 14)
    assert patches.dtype == torch.bfloat16
    assert token_types.tolist() == [
        DeepseekV41Template.IMAGE_START,
        DeepseekV41Template.IMAGE,
        DeepseekV41Template.IMAGE,
        DeepseekV41Template.IMAGE,
        DeepseekV41Template.IMAGE_NEW_LINE,
        DeepseekV41Template.IMAGE,
        DeepseekV41Template.IMAGE,
        DeepseekV41Template.IMAGE,
        DeepseekV41Template.IMAGE_NEW_LINE,
        DeepseekV41Template.IMAGE_END,
    ]


def test_deepseek_v41_image_encoding_and_collation():
    template = _get_template()
    image = Image.new('RGB', (112, 84), (255, 0, 0))
    image_row = template.encode(
        InferRequest(messages=[{'role': 'user', 'content': '<image>Describe this image.'}], images=[image]))
    text_row = template.encode(InferRequest(messages=[{'role': 'user', 'content': 'Hello.'}]))

    grid = image_row['image_grid_thw'][0]
    n_vit_h, n_vit_w = grid[-2:].tolist()
    ratio = template.config.vision_config.downsample_ratio
    n_llm_h, n_llm_w = math.ceil(n_vit_h / ratio), math.ceil(n_vit_w / ratio)
    expected_image_tokens = n_llm_h * (n_llm_w + 1) + 2
    image_mask = image_row['image_token_types'] >= 0

    assert image_row['pixel_values'].shape[0] == n_vit_h * n_vit_w
    assert image_mask.sum().item() == expected_image_tokens
    assert torch.all(torch.tensor(image_row['input_ids'])[image_mask] == template.config.image_token_id)
    assert torch.all(text_row['image_token_types'] == DeepseekV41Template.TEXT)

    batch = template.data_collator([image_row, text_row])
    assert batch['image_token_types'].shape == batch['input_ids'].shape
    assert torch.all(batch['image_token_types'][1] == DeepseekV41Template.TEXT)
    assert batch['image_grid_thw'].shape == (1, 3)
