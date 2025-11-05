#!/usr/bin/env python3

"""
Example template demonstrating tensor to image conversion in MS-Swift.

This shows how to create a custom template that handles <tensor> tags
and converts tensors to images, similar to how video templates work.
"""

from typing import List, Literal
from swift.llm.template.base import Template, Context
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.vision_utils import load_tensor


class TensorToImageTemplate(Template):
    """
    Example template that converts tensor files to images.
    
    This template demonstrates how to handle <tensor> tags by:
    1. Loading tensors from .pt files
    2. Converting them to PIL Images
    3. Processing them as regular images
    
    This is similar to how video templates convert video frames to images.
    """
    
    def replace_tag(self, media_type: Literal['image', 'video', 'audio', 'tensor'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """
        Override replace_tag to handle tensor conversion.
        
        When a <tensor> tag is encountered, this method:
        1. Loads the tensor from the .pt file
        2. Converts it to PIL Images using replace_tensor2image
        3. Returns image placeholders for the converted images
        """
        if media_type == 'tensor':
            # Use the replace_tensor2image method to convert tensor to images
            return self.replace_tensor2image(
                load_tensor, 
                inputs, 
                lambda i: self.image_placeholder  # Each tensor becomes an image
            )
        else:
            # Use the default handling for other media types
            return super().replace_tag(media_type, index, inputs)


def demo_tensor_template():
    """Demonstrate the tensor template functionality."""
    import torch
    import tempfile
    import os
    from swift.llm.template.template_inputs import StdTemplateInputs
    
    print("=== Tensor Template Demo ===")
    
    # Create a sample tensor that looks like image data
    # Shape: (3, 64, 64) - a small RGB image
    sample_tensor = torch.rand(3, 64, 64)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tensor_path = os.path.join(temp_dir, "sample_image_tensor.pt")
        torch.save(sample_tensor, tensor_path)
        
        print(f"Created sample tensor: {sample_tensor.shape}")
        print(f"Saved to: {tensor_path}")
        
        # Create template inputs with tensor
        inputs = StdTemplateInputs(
            messages=[{"role": "user", "content": "Analyze this tensor: <tensor>"}],
            tensors=[tensor_path]
        )
        
        print(f"Is multimodal: {inputs.is_multimodal}")
        print(f"Input messages: {inputs.messages}")
        print(f"Input tensors: {inputs.tensors}")
        
        # Create our custom template
        template = TensorToImageTemplate(
            processor=None,  # Not needed for this demo
            template_meta=None  # Not needed for this demo
        )
        
        # Test tensor to image conversion
        try:
            # Simulate the tensor conversion process
            inputs.tensor_idx = 0
            inputs.image_idx = 0
            inputs.images = []  # Start with no images
            
            # This would be called during template processing
            result = template.replace_tag('tensor', 0, inputs)
            
            print(f"✓ Tensor replacement result: {result}")
            print(f"✓ Number of images created: {len(inputs.images)}")
            print(f"✓ Image types: {[type(img).__name__ for img in inputs.images]}")
            
        except Exception as e:
            print(f"❌ Tensor conversion failed: {e}")
            raise


def demo_batch_tensor_template():
    """Demonstrate batch tensor handling."""
    import torch
    import tempfile
    import os
    from swift.llm.template.template_inputs import StdTemplateInputs
    
    print("\n=== Batch Tensor Template Demo ===")
    
    # Create a batch of tensor images
    # Shape: (4, 3, 64, 64) - 4 RGB images
    batch_tensor = torch.rand(4, 3, 64, 64)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tensor_path = os.path.join(temp_dir, "batch_tensor.pt")
        torch.save(batch_tensor, tensor_path)
        
        print(f"Created batch tensor: {batch_tensor.shape}")
        print(f"Saved to: {tensor_path}")
        
        # Create template inputs
        inputs = StdTemplateInputs(
            messages=[{"role": "user", "content": "Analyze these tensor images: <tensor>"}],
            tensors=[tensor_path]
        )
        
        # Create template
        template = TensorToImageTemplate(
            processor=None,
            template_meta=None
        )
        
        # Test batch tensor conversion
        try:
            inputs.tensor_idx = 0
            inputs.image_idx = 0
            inputs.images = []
            
            result = template.replace_tag('tensor', 0, inputs)
            
            print(f"✓ Batch tensor replacement result: {result}")
            print(f"✓ Number of images created from batch: {len(inputs.images)}")
            print(f"✓ Image types: {[type(img).__name__ for img in inputs.images]}")
            
            # Should create 4 images from the batch tensor
            assert len(inputs.images) == 4, f"Expected 4 images, got {len(inputs.images)}"
            print("✓ Batch processing verification passed!")
            
        except Exception as e:
            print(f"❌ Batch tensor conversion failed: {e}")
            raise


if __name__ == "__main__":
    try:
        demo_tensor_template()
        demo_batch_tensor_template()
        print("\n🎉 All tensor template demos completed successfully!")
        print("\n📝 Usage Summary:")
        print("1. Add <tensor> tags to your messages")
        print("2. Provide .pt files in the 'tensors' parameter")
        print("3. Tensors are automatically converted to images")
        print("4. Process them like regular images in your model")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        raise