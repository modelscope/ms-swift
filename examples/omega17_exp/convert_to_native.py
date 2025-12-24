#!/usr/bin/env python3
"""
Convert Omega17Exp model to use native Qwen3MoeForCausalLM architecture.

This script updates the model's config.json to use native Qwen3MoeForCausalLM
instead of Omega17ExpForCausalLM, enabling ~2x faster training by using
transformers' native optimized code paths.

Usage:
    python convert_to_native.py --model_dir ./model
    
After conversion:
    - config.json will use "architectures": ["Qwen3MoeForCausalLM"]
    - Custom modeling files are kept as backup but not used
    - Training will use native optimized implementation
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def convert_to_native(model_dir: str, backup: bool = True):
    """
    Convert Omega17Exp model to use native Qwen3MoeForCausalLM architecture.
    
    Args:
        model_dir: Path to the model directory
        backup: Whether to backup original config.json
    """
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    
    if not config_path.exists():
        print(f"❌ config.json not found at {config_path}")
        return False
    
    # Read current config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check current architecture
    current_arch = config.get('architectures', [])
    print(f"Current architecture: {current_arch}")
    
    if 'Qwen3MoeForCausalLM' in current_arch:
        print("✅ Model already uses native Qwen3MoeForCausalLM architecture")
        return True
    
    if 'Omega17ExpForCausalLM' not in current_arch:
        print(f"⚠️  Unknown architecture: {current_arch}")
        print("   Expected 'Omega17ExpForCausalLM'")
        return False
    
    # Backup original config
    if backup:
        backup_path = model_path / "config.json.omega17exp.backup"
        shutil.copy(config_path, backup_path)
        print(f"📁 Backed up original config to {backup_path}")
    
    # Update architecture
    config['architectures'] = ['Qwen3MoeForCausalLM']
    
    # Update model_type
    config['model_type'] = 'qwen3_moe'
    
    # Remove auto_map (no longer needed with native architecture)
    if 'auto_map' in config:
        del config['auto_map']
        print("   Removed auto_map (using native implementation)")
    
    # Write updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Updated config.json:")
    print(f"   - architectures: ['Qwen3MoeForCausalLM']")
    print(f"   - model_type: 'qwen3_moe'")
    print(f"\n🚀 Model now uses NATIVE optimized implementation!")
    print(f"   Training will be ~2x faster")
    
    return True


def revert_to_custom(model_dir: str):
    """
    Revert model to use custom Omega17ExpForCausalLM architecture.
    
    Args:
        model_dir: Path to the model directory
    """
    model_path = Path(model_dir)
    backup_path = model_path / "config.json.omega17exp.backup"
    config_path = model_path / "config.json"
    
    if not backup_path.exists():
        print(f"❌ Backup not found at {backup_path}")
        return False
    
    shutil.copy(backup_path, config_path)
    print(f"✅ Reverted to original Omega17ExpForCausalLM architecture")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert Omega17Exp model to native Qwen3MoeForCausalLM architecture'
    )
    parser.add_argument(
        '--model_dir', '-m',
        type=str,
        default='./model',
        help='Path to model directory (default: ./model)'
    )
    parser.add_argument(
        '--revert',
        action='store_true',
        help='Revert to original Omega17ExpForCausalLM architecture'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup of original config'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OMEGA17EXP NATIVE ARCHITECTURE CONVERTER")
    print("=" * 60)
    
    if args.revert:
        print(f"\n📂 Model directory: {args.model_dir}")
        print("\n🔄 Reverting to custom architecture...")
        success = revert_to_custom(args.model_dir)
    else:
        print(f"\n📂 Model directory: {args.model_dir}")
        print("\n🔧 Converting to native Qwen3MoeForCausalLM architecture...")
        success = convert_to_native(args.model_dir, backup=not args.no_backup)
    
    print("\n" + "=" * 60)
    if success:
        if args.revert:
            print("✅ REVERT COMPLETE!")
        else:
            print("✅ CONVERSION COMPLETE!")
            print("\nNow run training with:")
            print("   swift sft --model ./model --model_type omega17_exp ...")
    else:
        print("❌ OPERATION FAILED")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
