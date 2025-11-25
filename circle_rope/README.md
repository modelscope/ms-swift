# Circle-RoPE for ms-swift

Circle-RoPE (Circular Rotary Position Embedding) implementation for Qwen2.5-VL models, integrated with ms-swift framework using **non-invasive registration**.

## Overview

This implementation provides a custom position encoding mechanism for vision-language models that maps spatial-temporal coordinates to a circular projection, enabling better handling of video and multi-image inputs.

## Features

- ✅ **Non-invasive Integration**: Uses ms-swift's `--custom_register_path` mechanism
- ✅ **Flexible Configuration**: Customizable Circle-RoPE parameters via `model_config_override`
- ✅ **AGE Strategy Support**: Multiple attention gradient editing strategies (strategy_2, strategy_3, strategy_4)
- ✅ **Compatible with Qwen2.5-VL**: Works with all Qwen2.5-VL models (3B, 7B, 32B, 72B)

## File Structure

```
circle_rope/
├── circle_rope.py                          # Core Circle-RoPE implementation
├── modular_qwen2_5_vl_circle_rope.py      # Modified Qwen2.5-VL model classes
├── register.py                             # Model registration for ms-swift
├── config.json                             # Example configuration
├── __init__.py                             # Package initialization
├── README.md                               # This file
└── exp/
    └── train.sh                            # Training script example
```

## Installation

No installation required! Simply use the files in this directory.

### Prerequisites

- ms-swift (latest version)
- transformers >= 4.49
- qwen_vl_utils >= 0.0.6
- decord

## Usage

### Method 1: Using Training Script (Recommended)

```bash
cd circle_rope/exp
bash train.sh
```

The script automatically handles:
- Model registration via `--custom_register_path`
- Local repository path via `--local_repo_path`
- Circle-RoPE configuration

### Method 2: Manual Command

```bash
swift sft \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --model_type qwen2_5_vl_circle_rope \
    --custom_register_path /path/to/circle_rope/register.py \
    --local_repo_path /path/to/circle_rope \
    --dataset 'AI-ModelScope/coco#20000' \
    --train_type lora \
    --torch_dtype bfloat16 \
    ...
```

### Method 3: Custom Configuration Override

You can override Circle-RoPE parameters at runtime:

```bash
swift sft \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --model_type qwen2_5_vl_circle_rope \
    --custom_register_path /path/to/circle_rope/register.py \
    --local_repo_path /path/to/circle_rope \
    --model_config_override '{"circle_rope": {"alpha": 0.7, "radius": 15, "AGE_mode": "strategy_2"}}' \
    ...
```

## Configuration Parameters

### Circle-RoPE Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.5 | Nonlinear coefficient (0-1), controls distribution density |
| `radius` | float/str | 10 | Circle radius. Can be float or "auto", "auto-{scale}" |
| `method` | str | "circle" | Projection method: "circle" or "no_circle" |
| `AGE_mode` | str | "strategy_4" | Attention gradient editing strategy |
| `move_to_origin` | bool | True | Whether to center coordinates at origin |
| `move_to_positive` | bool/str/float | False | Move coordinates to positive axis |
| `dff_rate` | float/bool | False | Differential rate for blending |

### AGE Strategies

- **strategy_2**: First 18 layers use Circle-RoPE, last 18 use original RoPE
- **strategy_3**: First 18 layers use original RoPE, last 18 use Circle-RoPE
- **strategy_4**: Alternating layers (even: Circle-RoPE, odd: original RoPE)

## How It Works

### 1. Non-invasive Registration

The `register.py` file uses ms-swift's plugin system:

```python
register_model(
    ModelMeta(
        'qwen2_5_vl_circle_rope',
        [ModelGroup([Model(model_path='qwen2_5_vl_circle_rope')])],
        TemplateType.qwen2_5_vl,
        get_model_tokenizer_qwen2_5_vl_circle_rope,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen2_5_VLForConditionalGeneration_CircleRoPE'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video', 'circle-rope'],
        is_multimodal=True,
    ))
```

### 2. Configuration Override

The loader function applies Circle-RoPE config via `model_config_override`:

```python
model_config_override = {
    'auto_map': {
        'AutoConfig': 'modular_qwen2_5_vl_circle_rope.Qwen2_5_VLConfig_CircleRoPE',
        'AutoModelForCausalLM': 'modular_qwen2_5_vl_circle_rope.Qwen2_5_VLForConditionalGeneration_CircleRoPE',
    },
    'circle_rope': {
        'alpha': 0.5,
        'radius': 10,
        'method': 'circle',
        'AGE_mode': 'strategy_4',
        'move_to_origin': True,
    }
}
```

### 3. Trust Remote Code

The custom model classes are loaded via `trust_remote_code=True`:
- `modular_qwen2_5_vl_circle_rope.py` contains the modified model
- `circle_rope.py` provides the position encoding logic

## Integration with ms-swift

This implementation uses **zero code modification** to ms-swift framework:

1. **Custom Register Path**: `--custom_register_path` loads the registration
2. **Local Repo Path**: `--local_repo_path` makes modules importable
3. **Model Config Override**: Dynamic configuration via `model_config_override`
4. **Trust Remote Code**: Transformers loads custom model files

## Troubleshooting

### Issue: Module not found

**Solution**: Ensure `--local_repo_path` points to the `circle_rope` directory

```bash
--local_repo_path /absolute/path/to/circle_rope
```

### Issue: Model not registered

**Solution**: Verify `--custom_register_path` is correct

```bash
--custom_register_path /absolute/path/to/circle_rope/register.py
```

### Issue: Circle-RoPE not applied

**Solution**: Check that:
1. `--model_type qwen2_5_vl_circle_rope` is set
2. Log shows: "Using Circle-RoPE configuration: ..."

## Example Output

When correctly configured, you should see:

```
Circle-RoPE Directory: /path/to/circle_rope
Register Path: /path/to/circle_rope/register.py
...
Using Circle-RoPE configuration: {'auto_map': {...}, 'circle_rope': {...}}
Circle-RoPE model registered successfully. Use --model_type qwen2_5_vl_circle_rope
```

## Citation

If you use Circle-RoPE in your research, please cite:

```bibtex
@article{circle_rope_2024,
  title={Circle-RoPE: Circular Rotary Position Embedding for Vision-Language Models},
  author={Your Name},
  year={2024}
}
```

## License

Copyright (c) Alibaba, Inc. and its affiliates.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
