"""
Circle-RoPE: Circular Rotary Position Embedding for Vision-Language Models

This package provides Circle-RoPE implementation for Qwen2.5-VL models.

## Usage (Non-invasive Integration with ms-swift)

### Method 1: Using --custom_register_path (Recommended)
```bash
swift sft \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --model_type qwen2_5_vl_circle_rope \
    --custom_register_path /path/to/circle_rope/register.py \
    --local_repo_path /path/to/circle_rope \
    ...
```

### Method 2: Direct registration in your script
```python
import sys
sys.path.insert(0, '/path/to/circle_rope')
from register import register_model
# The model is automatically registered when importing
```

## Components:
    - circle_rope.py: Core Circle-RoPE implementation
    - modular_qwen2_5_vl_circle_rope.py: Qwen2.5-VL model with Circle-RoPE
    - register.py: Non-invasive model registration for ms-swift
    - config.json: Example configuration file with Circle-RoPE settings
"""

# from .register import get_model_tokenizer_qwen2_5_vl_circle_rope
# from .modular_qwen2_5_vl_circle_rope import Qwen2_5_VLForConditionalGeneration_CircleRoPE
#
# __all__ = [
#     'get_model_tokenizer_qwen2_5_vl_circle_rope',
#     'Qwen2_5_VLForConditionalGeneration_CircleRoPE'
# ]
#
# __version__ = '1.0.0'
