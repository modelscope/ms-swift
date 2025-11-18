# Tensor Input Support

MS-Swift supports loading and processing tensor files (`.pt` format) as input modality, similar to how images, videos, and audios are handled. This feature is particularly useful for specialized applications such as:

- Medical imaging and report generation (e.g., loading pre-processed scan tensors)
- Scientific data visualization
- Custom feature representations
- Pre-computed embeddings or feature maps

## Overview

The tensor support follows the same pattern as other multimodal inputs in MS-Swift:

- Use `<tensor>` tags in messages (similar to `<image>`, `<video>`, `<audio>`)
- Provide tensor file paths via the `tensors` parameter
- Tensors are automatically converted to PIL Images for model processing
- Supports both single tensors and batched tensors

## Supported Tensor Formats

The tensor loader supports various tensor shapes:

| Tensor Shape | Description | Output |
|--------------|-------------|--------|
| `(C, H, W)` | Single image tensor (e.g., `(3, 224, 224)` for RGB) | 1 PIL Image |
| `(B, C, H, W)` | Batch of image tensors (e.g., `(4, 3, 224, 224)`) | B PIL Images |
| `(1, H, W)` | Single channel (grayscale) tensor | 1 PIL Image (grayscale) |
| `(H, W)` | 2D tensor (grayscale) | 1 PIL Image (grayscale) |

**Supported channel formats:**
- RGB: 3 channels `(3, H, W)`
- Grayscale: 1 channel `(1, H, W)` or 2D `(H, W)`

**Value ranges:**
- Tensors with values in `[0, 1]` are automatically scaled to `[0, 255]`
- Tensors with values in `[0, 255]` are used directly
- All tensors are converted to uint8 format

## Usage Examples

### Basic Usage

#### Method 1: Using `tensors` parameter

```python
from swift.llm import InferRequest

request = InferRequest(
    messages=[
        {
            "role": "user",
            "content": "Analyze this tensor data: <tensor>"
        }
    ],
    tensors=["path/to/tensor.pt"]
)
```

#### Method 2: Using content with tensor type

```python
from swift.llm import InferRequest

request = InferRequest(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "tensor", "tensor": "path/to/tensor.pt"},
                {"type": "text", "text": "Analyze this tensor data."}
            ]
        }
    ]
)
```

### Multiple Tensors

```python
request = InferRequest(
    messages=[
        {
            "role": "user",
            "content": "Compare these tensors: <tensor><tensor>"
        }
    ],
    tensors=["tensor1.pt", "tensor2.pt"]
)
```

### Mixed Media Types

Tensors can be combined with images, videos, and audios:

```python
request = InferRequest(
    messages=[
        {
            "role": "user",
            "content": "Here's an image: <image> and tensor data: <tensor>"
        }
    ],
    images=["photo.jpg"],
    tensors=["data.pt"]
)
```

## Dataset Format

### Training Data Format

Tensors follow the same dataset format as other multimodal data:

#### Supervised Fine-tuning

```jsonl
{"messages": [{"role": "user", "content": "<tensor>Analyze this medical scan"}, {"role": "assistant", "content": "This scan shows..."}], "tensors": ["/path/to/scan.pt"]}
{"messages": [{"role": "user", "content": "Compare <tensor><tensor>"}, {"role": "assistant", "content": "The first tensor shows..."}], "tensors": ["/path/to/tensor1.pt", "/path/to/tensor2.pt"]}
```

#### Pre-training

```jsonl
{"messages": [{"role": "assistant", "content": "<tensor> represents a normal cardiac scan"}], "tensors": ["/path/to/cardiac_scan.pt"]}
```

#### RLHF (DPO/ORPO/CPO/SimPO)

```jsonl
{"messages": [{"role": "user", "content": "<tensor>What does this show?"}, {"role": "assistant", "content": "This is a detailed analysis..."}], "tensors": ["/path/to/scan.pt"], "rejected_response": "I don't know"}
```

#### Mixed Modality

```jsonl
{"messages": [{"role": "user", "content": "<image><tensor>Compare the image and tensor"}, {"role": "assistant", "content": "The image shows... while the tensor indicates..."}], "images": ["/path/to/image.jpg"], "tensors": ["/path/to/tensor.pt"]}
```

### Command Line Usage

Use tensors directly with command line parameters:

```bash
# Training with tensor dataset
swift sft \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --dataset /path/to/tensor_dataset.jsonl \
    --max_length 2048

# The dataset should contain 'tensors' field
# Example: {"messages": [...], "tensors": ["tensor.pt"]}
```

## Creating Tensor Files

### Example: Saving Tensors

```python
import torch

# Create a sample RGB tensor (simulating image data)
tensor = torch.randn(3, 224, 224)

# Normalize to [0, 1] range
tensor = torch.clamp((tensor + 1) / 2, 0, 1)

# Save the tensor
torch.save(tensor, "sample_tensor.pt")
```

### Example: Batch Tensor

```python
import torch

# Create a batch of tensors
batch_tensor = torch.randn(4, 3, 224, 224)
batch_tensor = torch.clamp((batch_tensor + 1) / 2, 0, 1)

torch.save(batch_tensor, "batch_tensor.pt")
```

### Example: Medical Imaging Data

```python
import torch
import numpy as np
from PIL import Image

# Load a medical scan (e.g., from DICOM, NIfTI, etc.)
# Assuming you have a numpy array from your medical imaging library
scan_array = np.load("medical_scan.npy")  # Shape: (H, W) or (H, W, C)

# Convert to tensor
if len(scan_array.shape) == 2:  # Grayscale
    tensor = torch.from_numpy(scan_array).unsqueeze(0)  # Add channel dim
elif len(scan_array.shape) == 3:  # Multi-channel
    tensor = torch.from_numpy(scan_array).permute(2, 0, 1)  # HWC -> CHW

# Normalize to [0, 1] if needed
tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

# Save for use with MS-Swift
torch.save(tensor, "medical_scan.pt")
```

## Custom Template with Tensor-to-Image Conversion

For advanced use cases, you can create custom templates that handle tensor conversion:

```python
from typing import List, Literal
from swift.llm.template.base import Template, Context
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.vision_utils import load_tensor


class TensorToImageTemplate(Template):
    """Custom template that converts tensors to images."""

    def replace_tag(self, media_type: Literal['image', 'video', 'audio', 'tensor'],
                    index: int, inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'tensor':
            # Convert tensor to images using built-in method
            return self.replace_tensor2image(
                load_tensor,
                inputs,
                lambda i: self.image_placeholder
            )
        else:
            return super().replace_tag(media_type, index, inputs)
```

## Loading Tensors Programmatically

```python
from swift.llm.template.vision_utils import load_tensor

# Load a tensor from file
tensor = load_tensor("path/to/tensor.pt")

# Load from URL
tensor = load_tensor("https://example.com/tensor.pt")

# The loaded tensor is always on CPU
print(f"Tensor shape: {tensor.shape}")
print(f"Tensor device: {tensor.device}")  # cpu
```

## Technical Details

### Tensor Loading Pipeline

1. **Load**: Tensor file is loaded using `torch.load()` with `map_location='cpu'`
2. **Validate**: Checks that the loaded object is a `torch.Tensor`
3. **Convert**: Tensor is converted to PIL Image(s) using `_tensor_to_images()`
4. **Process**: Images are processed like regular image inputs

### Conversion Process

The tensor-to-image conversion follows these steps:

1. **Shape Detection**: Determines if tensor is 2D, 3D, or 4D
2. **Batch Handling**: For 4D tensors, each batch item is converted separately
3. **Channel Formatting**: Converts from `(C, H, W)` to `(H, W, C)` for RGB
4. **Normalization**: Values in `[0, 1]` are scaled to `[0, 255]`
5. **Type Casting**: Converts to `uint8` format
6. **PIL Conversion**: Creates PIL Images (RGB or grayscale mode)

### Integration with Template System

Tensors are integrated into the template system:

- **Special Token**: `<tensor>` tag in messages
- **Placeholder**: `Template.tensor_placeholder = ['<tensor>']`
- **Index Tracking**: `inputs.tensor_idx` tracks current tensor position
- **Multimodal Detection**: Tensors contribute to `inputs.is_multimodal`

## Use Cases

### Medical Report Generation

```python
# Training dataset for medical report generation
dataset = [
    {
        "messages": [
            {"role": "user", "content": "<tensor>Generate a report for this scan"},
            {"role": "assistant", "content": "Findings: Normal cardiac function..."}
        ],
        "tensors": ["scans/patient001_cardiac.pt"]
    },
    # ... more examples
]
```

### Scientific Data Analysis

```python
# Analyzing scientific measurements
request = InferRequest(
    messages=[
        {
            "role": "user",
            "content": "<tensor>Analyze this spectral data and identify anomalies"
        }
    ],
    tensors=["spectra/sample_001.pt"]
)
```

### Feature Representation Learning

```python
# Using pre-computed features
request = InferRequest(
    messages=[
        {
            "role": "user",
            "content": "<tensor>Classify this feature representation"
        }
    ],
    tensors=["features/embedding_vector.pt"]
)
```

## Limitations

1. **Format**: Only `.pt` files (PyTorch tensor format) are supported
2. **Channels**: Limited to 1 (grayscale) or 3 (RGB) channels
3. **Dimensions**: Tensors must be 2D, 3D, or 4D
4. **Memory**: Large batch tensors may consume significant memory
5. **Conversion**: Tensors are always converted to images for model processing

## Troubleshooting

### Common Issues

**Issue**: `ValueError: Expected a torch.Tensor, but got <type>`
- **Solution**: Ensure the `.pt` file contains a PyTorch tensor, not other objects

**Issue**: `ValueError: Unsupported tensor shape: torch.Size([...])`
- **Solution**: Reshape your tensor to supported formats (2D, 3D, or 4D)

**Issue**: `ValueError: Unsupported number of channels: X`
- **Solution**: Convert to 1 (grayscale) or 3 (RGB) channels

**Issue**: Images appear too dark/bright
- **Solution**: Normalize your tensor values to [0, 1] range before saving

### Debugging

Enable debug mode to see tensor processing details:

```bash
export SWIFT_DEBUG=1
python your_script.py
```

## References

- [Custom Dataset Documentation](Custom-dataset.md)
- [Multimodal Training Best Practices](../BestPractices/Rapidly-Training-VL-model.md)
- [Template System Documentation](../Instruction/Template.md)

## Example Scripts

Complete example scripts are available in the `examples/` directory:

- `examples/tensor_example.py`: Basic tensor loading and usage
- `examples/tensor_template_example.py`: Custom template implementation
