#!/usr/bin/env python3
"""
Download Omega17Exp Model from HuggingFace (Private Model Support)

This script downloads the Omega17Exp model from HuggingFace to a local directory.
It ensures all custom files (tokenizer, config, model) are properly downloaded.

Usage:
    # Set token via environment variable (recommended)
    export HF_TOKEN=hf_xxxxxxxxxxxxx
    python download_model.py --model_id YOUR_HF_MODEL_ID --output_dir ./model
    
    # Or pass token directly
    python download_model.py --model_id YOUR_HF_MODEL_ID --output_dir ./model --token hf_xxxxx
    
Example:
    export HF_TOKEN=your_token_here
    python download_model.py --model_id arpitsh018/omega17exp-prod-v1.1 --output_dir ./model
"""

# Default model ID for Omega17Exp
DEFAULT_MODEL_ID = "arpitsh018/omega17exp-prod-v1.1"

import argparse
import os
import sys
from pathlib import Path


def download_model(model_id: str, output_dir: str, token: str = None):
    """
    Download model from HuggingFace Hub to local directory.
    
    Args:
        model_id: HuggingFace model ID (e.g., 'your-org/omega17-exp')
        output_dir: Local directory to save the model
        token: HuggingFace token for private models (optional)
    """
    try:
        from huggingface_hub import snapshot_download, hf_hub_download, login
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub -q")
        from huggingface_hub import snapshot_download, hf_hub_download, login
    
    # Get token from argument, environment, or prompt
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if token:
        print("🔑 Using HuggingFace token for authentication")
        try:
            login(token=token)
        except Exception:
            pass  # Token will be passed directly to snapshot_download
    else:
        print("⚠️  No HuggingFace token provided. Private models will fail.")
        print("   Set HF_TOKEN environment variable or use --token argument")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"DOWNLOADING OMEGA17EXP MODEL")
    print(f"{'='*60}")
    print(f"Model ID: {model_id}")
    print(f"Output: {output_path.absolute()}")
    print(f"{'='*60}\n")
    
    # Download the entire model repository
    print("📥 Downloading model from HuggingFace Hub...")
    print("   This may take a while depending on model size...\n")
    
    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=str(output_path),
            local_dir_use_symlinks=False,  # Copy files instead of symlinks
            token=token,
            resume_download=True,  # Resume if interrupted
        )
        
        print(f"\n✅ Model downloaded successfully!")
        print(f"   Location: {local_dir}")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check if the model ID is correct")
        print("2. For private models, use --token YOUR_HF_TOKEN")
        print("3. Make sure you have internet connection")
        return None
    
    # Verify required files
    print("\n📋 Verifying downloaded files...")
    required_files = [
        "config.json",
        "tokenizer_config.json",
    ]
    
    custom_files = [
        "tokenization_omega17.py",
        "configuration_omega17_exp.py",
        "modeling_omega17_exp.py",
    ]
    
    all_files = list(output_path.glob("*"))
    print(f"   Found {len(all_files)} files/folders")
    
    # Check required files
    missing = []
    for f in required_files:
        if (output_path / f).exists():
            print(f"   ✅ {f}")
        else:
            print(f"   ❌ {f} (MISSING)")
            missing.append(f)
    
    # Check custom files
    for f in custom_files:
        if (output_path / f).exists():
            print(f"   ✅ {f}")
        else:
            print(f"   ⚠️  {f} (not found - may be in different location)")
    
    # Check for model weights
    safetensors = list(output_path.glob("*.safetensors"))
    bin_files = list(output_path.glob("*.bin"))
    
    if safetensors:
        print(f"   ✅ Found {len(safetensors)} .safetensors file(s)")
    elif bin_files:
        print(f"   ✅ Found {len(bin_files)} .bin file(s)")
    else:
        print("   ⚠️  No model weights found in root (may be sharded)")
    
    if missing:
        print(f"\n⚠️  Warning: {len(missing)} required files missing")
    else:
        print(f"\n✅ All required files present!")
    
    print(f"\n{'='*60}")
    print(f"Model ready at: {output_path.absolute()}")
    print(f"{'='*60}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download Omega17Exp model from HuggingFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_id", type=str, default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./model",
        help="Local directory to save the model"
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace token for private models"
    )
    
    args = parser.parse_args()
    
    # Download
    result = download_model(
        model_id=args.model_id,
        output_dir=args.output_dir,
        token=args.token
    )
    
    if result:
        print("\n📝 Next steps:")
        print(f"   1. Run training:")
        print(f"      python train_lora_sft.py --model_path {args.output_dir} --dataset your_data.jsonl")
        print(f"\n   2. Or use MS-SWIFT CLI:")
        print(f"      swift sft --model {args.output_dir} --model_type omega17_exp --dataset alpaca-en")


if __name__ == "__main__":
    main()
