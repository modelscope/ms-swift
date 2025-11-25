"""
Debug script to trace the data pipeline and identify where pixel_values becomes list[list]
"""

import sys
import torch
from transformers import AutoConfig, AutoProcessor

def check_model_loading():
    """Check if model loads correctly"""
    print("=" * 80)
    print("Step 1: Check Model Loading")
    print("=" * 80)

    try:
        # Import V2 classes
        from circle_rope.modular_qwen2_5_vl_circle_rope_v2 import (
            Qwen2_5_VLConfig_CircleRoPE_V2,
            Qwen2_5_VLForConditionalGeneration_CircleRoPE_V2
        )

        print("✅ Import successful")

        # Check config type
        print(f"Config model_type: {Qwen2_5_VLConfig_CircleRoPE_V2.model_type}")
        assert Qwen2_5_VLConfig_CircleRoPE_V2.model_type == "qwen2_5_vl", \
            f"model_type should be 'qwen2_5_vl' but got '{Qwen2_5_VLConfig_CircleRoPE_V2.model_type}'"
        print("✅ model_type is correct: qwen2_5_vl")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_visual_component(model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
    """Check if visual component is correctly initialized"""
    print("\n" + "=" * 80)
    print("Step 2: Check Visual Component")
    print("=" * 80)

    try:
        from circle_rope.modular_qwen2_5_vl_circle_rope_v2 import (
            Qwen2_5_VLConfig_CircleRoPE_V2,
            Qwen2_5_VLModel_CircleRoPE_V2
        )

        print(f"Loading config from {model_path}...")
        config = Qwen2_5_VLConfig_CircleRoPE_V2.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        config.circle_rope = {
            "circle_r": 10000,
            "base": 10000,
            "mrope_section": [16, 24, 24]
        }

        print("Creating model (this may take a while)...")
        print("Note: This is just for testing structure, not loading weights")

        # Check if visual exists in parent
        from transformers import Qwen2_5_VLModel
        temp_model = Qwen2_5_VLModel(config)

        print(f"✅ Parent model has visual: {hasattr(temp_model, 'visual')}")
        print(f"✅ Parent model has language_model: {hasattr(temp_model, 'language_model')}")

        # Now check our model
        our_model = Qwen2_5_VLModel_CircleRoPE_V2(config)
        print(f"✅ Our model has visual: {hasattr(our_model, 'visual')}")
        print(f"✅ Our model has language_model: {hasattr(our_model, 'language_model')}")
        print(f"✅ Our model has rope_deltas: {hasattr(our_model, 'rope_deltas')}")

        # Check methods
        methods = ['get_image_features', 'get_video_features', 'get_rope_index']
        for method in methods:
            has_it = hasattr(our_model, method)
            print(f"✅ Our model has {method}: {has_it}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_template_compatibility():
    """Check if template recognizes the model correctly"""
    print("\n" + "=" * 80)
    print("Step 3: Check Template Compatibility")
    print("=" * 80)

    try:
        from swift.llm import get_template
        from circle_rope.modular_qwen2_5_vl_circle_rope_v2 import (
            Qwen2_5_VLConfig_CircleRoPE_V2
        )

        # Create a dummy config
        config = Qwen2_5_VLConfig_CircleRoPE_V2()
        print(f"Config class: {config.__class__.__name__}")
        print(f"Config model_type: {config.model_type}")

        # Check if model_type matches expected
        if config.model_type == "qwen2_5_vl":
            print("✅ model_type matches original, template should work")
        else:
            print(f"⚠️  model_type is '{config.model_type}', may cause template issues")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_processor():
    """Check processor behavior"""
    print("\n" + "=" * 80)
    print("Step 4: Check Processor")
    print("=" * 80)

    try:
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"Loading processor from {model_path}...")

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Processor loaded: {type(processor).__name__}")

        # Test with a dummy input
        from PIL import Image
        import numpy as np

        # Create a dummy image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_img},
                {"type": "text", "text": "Test"}
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[dummy_img], return_tensors="pt")

        print(f"✅ Processor output keys: {list(inputs.keys())}")
        if 'pixel_values' in inputs:
            pv = inputs['pixel_values']
            print(f"✅ pixel_values type: {type(pv)}")
            print(f"✅ pixel_values shape: {pv.shape if isinstance(pv, torch.Tensor) else 'N/A'}")

            # Check if it's a proper tensor
            if isinstance(pv, torch.Tensor):
                print("✅ pixel_values is a proper Tensor")
            elif isinstance(pv, list):
                print(f"⚠️  pixel_values is a list with {len(pv)} elements")
                if len(pv) > 0:
                    print(f"    First element type: {type(pv[0])}")
            else:
                print(f"❌ pixel_values is unexpected type: {type(pv)}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 80)
    print("Circle-RoPE V2 Data Pipeline Debug")
    print("=" * 80)
    print()

    tests = [
        ("Model Loading", check_model_loading),
        ("Visual Component", lambda: check_visual_component()),
        ("Template Compatibility", check_template_compatibility),
        ("Processor", check_processor),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("Debug Summary")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Total: {passed}/{total} checks passed")

    if passed == total:
        print("\n✅ All checks passed! The issue is likely in ms-swift integration.")
        print("\nNext steps:")
        print("1. Check your training config file")
        print("2. Verify custom_register_path is correct")
        print("3. Check if model_config_override is properly set")
    else:
        print(f"\n⚠️  {total - passed} check(s) failed.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
