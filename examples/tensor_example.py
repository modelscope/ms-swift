#!/usr/bin/env python3

"""
Example script demonstrating .pt file support in MS-Swift.

This script shows how to use the new <tensor> tag to handle .pt files
in the same way that <image> and <video> tags are handled.
"""

import torch
import tempfile
import os
from swift.llm.template.template_inputs import InferRequest, StdTemplateInputs
from swift.llm.template.vision_utils import load_tensor


def create_sample_tensor():
    """Create a sample tensor for testing."""
    # Create a sample RGB image tensor (3, 224, 224)
    tensor = torch.randn(3, 224, 224)
    # Normalize to [0, 1] range to simulate image data
    tensor = torch.clamp((tensor + 1) / 2, 0, 1)
    return tensor


def create_batch_tensor():
    """Create a batch of sample tensors for testing."""
    # Create a batch of RGB image tensors (4, 3, 224, 224)
    tensor = torch.randn(4, 3, 224, 224)
    # Normalize to [0, 1] range to simulate image data
    tensor = torch.clamp((tensor + 1) / 2, 0, 1)
    return tensor


def demo_tensor_loading():
    """Demonstrate tensor loading functionality."""
    print("=== Tensor Loading Demo ===")
    
    # Create sample tensors
    single_tensor = create_sample_tensor()
    batch_tensor = create_batch_tensor()
    
    # Save tensors to temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        single_tensor_path = os.path.join(temp_dir, "single_tensor.pt")
        batch_tensor_path = os.path.join(temp_dir, "batch_tensor.pt")
        
        torch.save(single_tensor, single_tensor_path)
        torch.save(batch_tensor, batch_tensor_path)
        
        print(f"Saved single tensor to: {single_tensor_path}")
        print(f"Single tensor shape: {single_tensor.shape}")
        
        print(f"Saved batch tensor to: {batch_tensor_path}")
        print(f"Batch tensor shape: {batch_tensor.shape}")
        
        # Load tensors using our new function
        loaded_single = load_tensor(single_tensor_path)
        loaded_batch = load_tensor(batch_tensor_path)
        
        print(f"Loaded single tensor shape: {loaded_single.shape}")
        print(f"Loaded batch tensor shape: {loaded_batch.shape}")
        
        # Verify tensors are identical
        assert torch.allclose(single_tensor, loaded_single), "Single tensor loading failed!"
        assert torch.allclose(batch_tensor, loaded_batch), "Batch tensor loading failed!"
        
        print("✓ Tensor loading verification passed!")


def demo_tensor_in_messages():
    """Demonstrate using tensors in message format."""
    print("\n=== Tensor in Messages Demo ===")
    
    # Create sample tensor
    tensor = create_sample_tensor()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tensor_path = os.path.join(temp_dir, "sample_tensor.pt")
        torch.save(tensor, tensor_path)
        
        # Method 1: Using tensors parameter
        request1 = InferRequest(
            messages=[
                {
                    "role": "user",
                    "content": "Please analyze this tensor data: <tensor>"
                }
            ],
            tensors=[tensor_path]
        )
        
        print("Method 1 - Using tensors parameter:")
        print(f"  Messages: {request1.messages}")
        print(f"  Tensors: {request1.tensors}")
        
        # Method 2: Using content with tensor type
        request2 = InferRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tensor",
                            "tensor": tensor_path
                        },
                        {
                            "type": "text",
                            "text": "Please analyze this tensor data."
                        }
                    ]
                }
            ]
        )
        
        print("\nMethod 2 - Using content with tensor type:")
        print(f"  Messages: {request2.messages}")
        print(f"  Tensors: {request2.tensors}")


def demo_mixed_media():
    """Demonstrate using tensors with other media types."""
    print("\n=== Mixed Media Demo ===")
    
    # Create sample tensor
    tensor = create_sample_tensor()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tensor_path = os.path.join(temp_dir, "sample_tensor.pt")
        torch.save(tensor, tensor_path)
        
        # Mix tensors with images and text
        request = InferRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here's an image: "
                        },
                        {
                            "type": "image",
                            "image": "https://example.com/image.jpg"
                        },
                        {
                            "type": "text",
                            "text": " and here's tensor data: "
                        },
                        {
                            "type": "tensor",
                            "tensor": tensor_path
                        },
                        {
                            "type": "text",
                            "text": " Please compare them."
                        }
                    ]
                }
            ]
        )
        
        print("Mixed media request:")
        print(f"  Messages: {request.messages}")
        print(f"  Images: {request.images}")
        print(f"  Tensors: {request.tensors}")
        
        # Convert to StdTemplateInputs to show multimodal detection
        std_inputs = StdTemplateInputs(
            messages=request.messages,
            images=request.images,
            tensors=request.tensors
        )
        
        print(f"  Is multimodal: {std_inputs.is_multimodal}")


if __name__ == "__main__":
    try:
        demo_tensor_loading()
        demo_tensor_in_messages()
        demo_mixed_media()
        print("\n🎉 All tensor demos completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        raise