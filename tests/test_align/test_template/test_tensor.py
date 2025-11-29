import os
import tempfile
from typing import List

import torch
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['SWIFT_DEBUG'] = '1'


def test_load_tensor():
    """Test basic tensor loading functionality."""
    from swift.llm.template.vision_utils import load_tensor

    # Create a sample tensor
    sample_tensor = torch.randn(3, 224, 224)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        tensor_path = tmp_file.name
        torch.save(sample_tensor, tensor_path)

        try:
            # Test loading
            loaded_tensor = load_tensor(tensor_path)
            assert isinstance(loaded_tensor, torch.Tensor), "Loaded object is not a tensor"
            assert torch.allclose(sample_tensor, loaded_tensor), "Tensor values don't match"
            assert loaded_tensor.shape == (3, 224, 224), f"Unexpected shape: {loaded_tensor.shape}"
            print("✓ test_load_tensor passed")
        finally:
            os.unlink(tensor_path)


def test_load_batch_tensor():
    """Test loading batch tensors."""
    from swift.llm.template.vision_utils import load_tensor

    # Create a batch tensor
    batch_tensor = torch.randn(4, 3, 64, 64)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        tensor_path = tmp_file.name
        torch.save(batch_tensor, tensor_path)

        try:
            # Test loading
            loaded_tensor = load_tensor(tensor_path)
            assert isinstance(loaded_tensor, torch.Tensor), "Loaded object is not a tensor"
            assert torch.allclose(batch_tensor, loaded_tensor), "Tensor values don't match"
            assert loaded_tensor.shape == (4, 3, 64, 64), f"Unexpected shape: {loaded_tensor.shape}"
            print("✓ test_load_batch_tensor passed")
        finally:
            os.unlink(tensor_path)


def test_tensor_to_images_single():
    """Test converting a single tensor to image."""
    from swift.llm.template.base import Template

    # Create a mock template to access the conversion methods
    template = Template(processor=None, template_meta=None)

    # Create a single RGB tensor (3, 64, 64)
    single_tensor = torch.rand(3, 64, 64)

    # Convert to images
    images = template._tensor_to_images(single_tensor)

    assert isinstance(images, list), "Output should be a list"
    assert len(images) == 1, f"Expected 1 image, got {len(images)}"
    assert isinstance(images[0], Image.Image), "Output should contain PIL Images"
    assert images[0].size == (64, 64), f"Unexpected image size: {images[0].size}"
    assert images[0].mode == 'RGB', f"Unexpected image mode: {images[0].mode}"
    print("✓ test_tensor_to_images_single passed")


def test_tensor_to_images_batch():
    """Test converting a batch of tensors to images."""
    from swift.llm.template.base import Template

    template = Template(processor=None, template_meta=None)

    # Create a batch of RGB tensors (4, 3, 64, 64)
    batch_tensor = torch.rand(4, 3, 64, 64)

    # Convert to images
    images = template._tensor_to_images(batch_tensor)

    assert isinstance(images, list), "Output should be a list"
    assert len(images) == 4, f"Expected 4 images, got {len(images)}"
    assert all(isinstance(img, Image.Image) for img in images), "All outputs should be PIL Images"
    assert all(img.size == (64, 64) for img in images), "All images should have size (64, 64)"
    assert all(img.mode == 'RGB' for img in images), "All images should be RGB"
    print("✓ test_tensor_to_images_batch passed")


def test_tensor_to_images_grayscale():
    """Test converting grayscale tensors to images."""
    from swift.llm.template.base import Template

    template = Template(processor=None, template_meta=None)

    # Create a grayscale tensor (1, 64, 64)
    gray_tensor = torch.rand(1, 64, 64)

    # Convert to images
    images = template._tensor_to_images(gray_tensor)

    assert isinstance(images, list), "Output should be a list"
    assert len(images) == 1, f"Expected 1 image, got {len(images)}"
    assert isinstance(images[0], Image.Image), "Output should contain PIL Images"
    assert images[0].size == (64, 64), f"Unexpected image size: {images[0].size}"
    assert images[0].mode == 'L', f"Expected grayscale (L) mode, got: {images[0].mode}"
    print("✓ test_tensor_to_images_grayscale passed")


def test_tensor_to_images_2d():
    """Test converting 2D tensors to images."""
    from swift.llm.template.base import Template

    template = Template(processor=None, template_meta=None)

    # Create a 2D tensor (64, 64)
    tensor_2d = torch.rand(64, 64)

    # Convert to images
    images = template._tensor_to_images(tensor_2d)

    assert isinstance(images, list), "Output should be a list"
    assert len(images) == 1, f"Expected 1 image, got {len(images)}"
    assert isinstance(images[0], Image.Image), "Output should contain PIL Images"
    assert images[0].size == (64, 64), f"Unexpected image size: {images[0].size}"
    assert images[0].mode == 'L', f"Expected grayscale (L) mode, got: {images[0].mode}"
    print("✓ test_tensor_to_images_2d passed")


def test_tensor_normalization():
    """Test that tensor values are properly normalized."""
    from swift.llm.template.base import Template
    import numpy as np

    template = Template(processor=None, template_meta=None)

    # Test normalized values [0, 1]
    normalized_tensor = torch.rand(3, 32, 32)  # Already in [0, 1]
    images = template._tensor_to_images(normalized_tensor)
    img_array = np.array(images[0])
    assert img_array.min() >= 0 and img_array.max() <= 255, "Image values should be in [0, 255]"

    # Test values that need scaling
    scaled_tensor = torch.rand(3, 32, 32) * 0.5  # In [0, 0.5]
    images = template._tensor_to_images(scaled_tensor)
    img_array = np.array(images[0])
    assert img_array.min() >= 0 and img_array.max() <= 255, "Image values should be in [0, 255]"

    print("✓ test_tensor_normalization passed")


def test_template_inputs_with_tensors():
    """Test StdTemplateInputs with tensor support."""
    from swift.llm.template.template_inputs import StdTemplateInputs

    # Create inputs with tensors
    inputs = StdTemplateInputs(
        messages=[{"role": "user", "content": "Analyze this: <tensor>"}],
        tensors=["path/to/tensor.pt"]
    )

    # Verify tensor_idx is initialized
    assert hasattr(inputs, 'tensor_idx'), "tensor_idx should be initialized"
    assert inputs.tensor_idx == 0, f"tensor_idx should be 0, got {inputs.tensor_idx}"

    # Verify multimodal detection
    assert inputs.is_multimodal, "Should be detected as multimodal with tensors"

    # Test with single string (should be converted to list)
    inputs2 = StdTemplateInputs(
        messages=[{"role": "user", "content": "<tensor>"}],
        tensors="single_tensor.pt"
    )
    assert isinstance(inputs2.tensors, list), "Single tensor should be converted to list"
    assert len(inputs2.tensors) == 1, f"Expected 1 tensor, got {len(inputs2.tensors)}"

    print("✓ test_template_inputs_with_tensors passed")


def test_infer_request_with_tensors():
    """Test InferRequest with tensor support."""
    from swift.llm.template.template_inputs import InferRequest

    # Test Method 1: Using tensors parameter
    request1 = InferRequest(
        messages=[{"role": "user", "content": "Analyze this: <tensor>"}],
        tensors=["tensor1.pt", "tensor2.pt"]
    )
    assert len(request1.tensors) == 2, f"Expected 2 tensors, got {len(request1.tensors)}"

    # Test Method 2: Using content with tensor type
    request2 = InferRequest(
        messages=[{
            "role": "user",
            "content": [
                {"type": "tensor", "tensor": "tensor.pt"},
                {"type": "text", "text": "Analyze this."}
            ]
        }]
    )
    assert isinstance(request2.messages, list), "Messages should be a list"

    # Test single string conversion
    request3 = InferRequest(
        messages=[{"role": "user", "content": "<tensor>"}],
        tensors="single.pt"
    )
    assert isinstance(request3.tensors, list), "Single tensor should be converted to list"

    print("✓ test_infer_request_with_tensors passed")


def test_mixed_media():
    """Test using tensors with other media types."""
    from swift.llm.template.template_inputs import StdTemplateInputs

    inputs = StdTemplateInputs(
        messages=[{"role": "user", "content": "<image><tensor><video>"}],
        images=["image.jpg"],
        tensors=["tensor.pt"],
        videos=["video.mp4"]
    )

    assert inputs.is_multimodal, "Should be multimodal"
    assert len(inputs.images) == 1, "Should have 1 image"
    assert len(inputs.tensors) == 1, "Should have 1 tensor"
    assert len(inputs.videos) == 1, "Should have 1 video"
    assert inputs.tensor_idx == 0, "tensor_idx should be initialized"

    print("✓ test_mixed_media passed")


def test_dataset_preprocessor_tensor_support():
    """Test that dataset preprocessor supports tensors."""
    from swift.llm.dataset.preprocessor.core import RowPreprocessor

    preprocessor = RowPreprocessor()

    # Check that tensors is in standard_keys
    assert 'tensors' in RowPreprocessor.standard_keys, "tensors should be in standard_keys"

    # Check that column mapping works
    assert 'tensor' in preprocessor.columns, "tensor column should be mapped"
    assert preprocessor.columns['tensor'] == 'tensors', "tensor should map to tensors"
    assert 'tensors' in preprocessor.columns, "tensors column should be mapped"
    assert preprocessor.columns['tensors'] == 'tensors', "tensors should map to tensors"

    print("✓ test_dataset_preprocessor_tensor_support passed")


def test_special_tokens():
    """Test that tensor special tokens are properly registered."""
    from swift.llm.template.base import Template

    # Check that <tensor> is in special_tokens
    assert '<tensor>' in Template.special_tokens, "<tensor> should be in special_tokens"

    # Check that tensors is in special_keys
    assert 'tensors' in Template.special_keys, "tensors should be in special_keys"

    # Check that tensor_placeholder exists
    assert hasattr(Template, 'tensor_placeholder'), "Template should have tensor_placeholder"
    assert Template.tensor_placeholder == ['<tensor>'], "tensor_placeholder should be ['<tensor>']"

    print("✓ test_special_tokens passed")


if __name__ == '__main__':
    print("=== Running Tensor Support Tests ===\n")

    try:
        # Basic loading tests
        test_load_tensor()
        test_load_batch_tensor()

        # Tensor to image conversion tests
        test_tensor_to_images_single()
        test_tensor_to_images_batch()
        test_tensor_to_images_grayscale()
        test_tensor_to_images_2d()
        test_tensor_normalization()

        # Template inputs tests
        test_template_inputs_with_tensors()
        test_infer_request_with_tensors()
        test_mixed_media()

        # Integration tests
        test_dataset_preprocessor_tensor_support()
        test_special_tokens()

        print("\n🎉 All tensor support tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
