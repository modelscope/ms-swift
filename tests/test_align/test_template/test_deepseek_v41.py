import json
import math
import sys
import torch
from PIL import Image
from types import SimpleNamespace

from swift import InferRequest
from swift.model import get_processor
from swift.template import get_template
from swift.template.templates.deepseek import DeepseekV41Template

MODEL_ID = 'deepseek-ai/DeepSeek-V4.1-Flash'


def test_deepseek_v41_config_is_process_local(tmp_path):
    import transformers.models.deepseek_v4.configuration_deepseek_v4 as v4_config_module
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    from swift.model.models.deepseek import DeepseekV41Loader

    text_config = DeepseekV4Config(num_hidden_layers=4).to_dict()
    text_config.update(
        model_type='deepseek_v41_text',
        num_hidden_layers=4,
        mlp_layer_types=['moe'] * 4,
        compress_ratios=[0, 1, 2, 4],
    )
    text_config.pop('layer_types', None)
    config_dict = {
        'model_type': 'deepseek_v41',
        'architectures': ['DeepseekV41ForCausalLM'],
        'text_config': text_config,
        'vision_config': {
            'model_type': 'deepseek_v41_vision'
        },
        'image_token_id': 1,
    }
    with open(tmp_path / 'config.json', 'w') as config_file:
        json.dump(config_dict, config_file)

    loader = object.__new__(DeepseekV41Loader)
    loader.auto_config_cls = None
    ratio_mapping = dict(v4_config_module._COMPRESS_RATIO_TO_LAYER_TYPE)
    vllm_modules = {name for name in sys.modules if name.startswith('vllm')}

    config = loader.get_config(str(tmp_path))

    assert config.text_config.compress_ratios == [0, 1, 2, 4]
    assert config.text_config.layer_types == [
        'sliding_attention',
        'compressed_sparse_attention',
        'heavily_compressed_attention',
        'compressed_sparse_attention',
    ]
    assert v4_config_module._COMPRESS_RATIO_TO_LAYER_TYPE == ratio_mapping
    assert {name for name in sys.modules if name.startswith('vllm')} == vllm_modules


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
        InferRequest(messages=[{
            'role': 'user',
            'content': '<image>Describe this image.'
        }], images=[image]))
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
