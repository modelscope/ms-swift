#!/usr/bin/env bash
# Transformers-only HF training for Nemotron 3.5 Transcribe. NeMo is not required.
# Requires transformers>5.21 (5.22 release pending).
# pip install mamba-ssm causal-conv1d>=1.2.0
#
# FLEURS Cantonese is downloaded from ModelScope. Evaluate the base model first:
# python examples/models/nemotron_transcribe/evaluate_fleurs.py \
#   --model ${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B} \
#   --output baseline.jsonl
# Then run the same command on the exported adapter/merged checkpoint.
#
# lora_llm trains LoRA on the language model and the full audio tower/aligner.

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
SWIFT_AUDIO_LOAD_BACKEND=soundfile_pyav \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model ${MODEL:-nv-community/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B} \
    --dataset 'google/fleurs:yue_hant_hk_train' \
    --val_dataset 'google/fleurs:yue_hant_hk_validation' \
    --load_from_cache_file true \
    --tuner_type lora_llm \
    --lora_rank 8 \
    --lora_alpha 16 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --max_length 8192 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1