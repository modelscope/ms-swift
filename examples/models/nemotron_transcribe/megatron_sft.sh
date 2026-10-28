#!/usr/bin/env bash
# Requires Megatron-Core with HybridModel, Transformer Engine and mamba-ssm.
# PyPI nemo-toolkit 3.0.0 does not include ParallelExpertEncoderPT. Tested setup:
# pip install 'nemo-toolkit[asr]==3.0.0' 'nemo-automodel==0.7.0' 'peft==0.20.0'
# pip install --no-deps --force-reinstall \
#   'git+https://github.com/NVIDIA-NeMo/Speech.git@68a12fc5e469614abd23b25ff9cfcb822aa0be93' \
#   'git+https://github.com/NVIDIA-NeMo/Automodel.git@309a10098e3ea3d9b79d4b96fa397db40908b751' \
#   'lhotse==2.0.0a6'
# FLEURS Cantonese is downloaded from ModelScope. The base model does not list
# Cantonese as a supported language, making held-out transcription a useful
# before/after fine-tuning check. Keep the test split out of training.
#
# Before training, measure baseline CER and save predictions:
# python examples/models/nemotron_transcribe/evaluate_fleurs.py \
#   --model nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --output baseline.jsonl
# After training, run the same command with --model pointing to the merged
# checkpoint and --output finetuned.jsonl. Lower CER is better; inspect the saved
# predictions as well. CER ignores whitespace and retains punctuation.
#
# Cantonese is outside the published language set, so both the acoustic
# encoder and the language-side LoRA are trained.
# MTP, CP, FP8/FP4, packing and padding-free are unsupported in this version.
# Do not use all-linear: it includes 128 routed experts and unsupported Mamba LoRA targets.

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
SWIFT_AUDIO_LOAD_BACKEND=soundfile_pyav \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron sft \
    --model ${MODEL:-nv-community/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B} \
    --dataset 'google/fleurs:yue_hant_hk_train' \
    --val_dataset 'google/fleurs:yue_hant_hk_validation' \
    --load_from_cache_file true \
    --tuner_type lora_llm \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules linear_qkv linear_proj proj \
    --freeze_vit false \
    --freeze_aligner false \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --context_parallel_size 1 \
    --mtp_num_layers auto \
    --mtp_loss_scaling_factor 0.1 \
    --megatron_extra_kwargs '{"moe_router_bias_update_rate":0.0}' \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --num_train_epochs 3 \
    --max_length 8192 \
    --lr 5e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 5e-6 \
    --recompute_granularity selective \
    --padding_free false \
    --packing false \
    --attention_backend flash \
    --logging_steps 1 \
    --eval_steps 200 \
    --save_steps 1000 \
    --save_safetensors true \
    --save_missing_weights true \
    --merge_lora true \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR:-megatron_output/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-yue}

# Keep --merge_lora true and --save_missing_weights true so the evaluation model
# is complete and the original MTP, CTC and _extra_state tensors are preserved.
