#!/usr/bin/env python3
"""
Omega17Exp LoRA SFT Fine-tuning Script

This script provides a production-ready way to fine-tune the Omega17ExpForCausalLM
model using LoRA (Low-Rank Adaptation) for Supervised Fine-Tuning (SFT).

Requirements:
    pip install transformers-usf-om-vl-exp-v0
    pip install ms-swift[llm]

Usage:
    python train_lora_sft.py \
        --model_path /path/to/omega17-exp \
        --dataset alpaca-en \
        --output_dir ./output/omega17_lora

For RunPod:
    CUDA_VISIBLE_DEVICES=0 python train_lora_sft.py --model_path /workspace/model --dataset your_dataset.jsonl
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import torch


def register_omega17_model():
    """Register the Omega17Exp model with MS-SWIFT."""
    from swift.llm.model.register import MODEL_MAPPING
    
    # Skip if already registered
    if 'omega17_exp' in MODEL_MAPPING:
        print("✅ Omega17Exp model already registered")
        return
    
    # Import the omega17 module to trigger registration
    # This imports all the proper model registration from omega17.py
    from swift.llm.model.model import omega17
    
    if 'omega17_exp' in MODEL_MAPPING:
        print("✅ Omega17Exp model registered successfully!")
    else:
        raise RuntimeError(
            "Failed to register Omega17Exp model. "
            "Make sure swift.llm.model.model.omega17 is properly installed."
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Omega17Exp model using LoRA SFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the Omega17Exp model or HuggingFace model ID"
    )
    parser.add_argument(
        "--model_type", type=str, default="omega17_exp",
        help="Model type for MS-SWIFT"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name (MS-SWIFT built-in) or path to JSONL file"
    )
    parser.add_argument(
        "--val_dataset", type=str, default=None,
        help="Validation dataset path (optional)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir", type=str, default="./output/omega17_lora_sft",
        help="Output directory for checkpoints"
    )
    
    # LoRA configuration
    parser.add_argument(
        "--lora_rank", type=int, default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=128,
        help="LoRA alpha (typically 2x rank)"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Target modules for LoRA"
    )
    
    # Training parameters
    parser.add_argument(
        "--max_length", type=int, default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=16,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.03,
        help="Warmup ratio"
    )
    
    # Quantization
    parser.add_argument(
        "--use_qlora", action="store_true",
        help="Use QLoRA (4-bit quantization)"
    )
    parser.add_argument(
        "--quant_bits", type=int, default=4, choices=[4, 8],
        help="Quantization bits for QLoRA"
    )
    
    # Optimization
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", default=True,
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--torch_dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Training precision"
    )
    
    # Logging and saving
    parser.add_argument(
        "--logging_steps", type=int, default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--save_total_limit", type=int, default=3,
        help="Maximum number of checkpoints to keep"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500,
        help="Evaluate every N steps"
    )
    
    # DeepSpeed
    parser.add_argument(
        "--deepspeed", type=str, default=None,
        help="Path to DeepSpeed config file"
    )
    
    # Misc
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--report_to", type=str, nargs="+", default=["tensorboard"],
        help="Reporting integrations"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 60)
    print("OMEGA17EXP LORA SFT FINE-TUNING")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA Rank: {args.lora_rank}, Alpha: {args.lora_alpha}")
    print(f"Batch Size: {args.batch_size} x {args.gradient_accumulation_steps} (grad accum)")
    print(f"Max Length: {args.max_length}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"QLoRA: {args.use_qlora}")
    print("=" * 60 + "\n")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No GPU detected! Training will be very slow.")
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Register custom model
    print("\n📦 Registering Omega17Exp model...")
    register_omega17_model()
    
    # Import MS-SWIFT
    from swift.llm import SftArguments, sft_main
    
    # Build training arguments
    print("\n⚙️  Building training arguments...")
    
    dataset_list = [args.dataset]
    if args.val_dataset:
        dataset_list.append(args.val_dataset)
    
    sft_args = SftArguments(
        # Model
        model=args.model_path,
        model_type=args.model_type,
        
        # Dataset
        dataset=dataset_list,
        
        # Training type
        train_type='lora',
        
        # LoRA
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        
        # Quantization
        quant_bits=args.quant_bits if args.use_qlora else None,
        
        # Training
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        
        # Optimizer
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        
        # Precision
        torch_dtype=args.torch_dtype,
        
        # Memory optimization
        gradient_checkpointing=args.gradient_checkpointing,
        
        # Logging
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        
        # DeepSpeed
        deepspeed=args.deepspeed,
        
        # Misc
        seed=args.seed,
        report_to=args.report_to,
    )
    
    # Start training
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    result = sft_main(sft_args)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED!")
    print(f"   Output directory: {args.output_dir}")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
