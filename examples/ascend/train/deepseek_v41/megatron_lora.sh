#!/usr/bin/env bash
# DeepSeek-V4.1 text-only LoRA on Ascend NPU, with frozen Engram tables on CPU.
# Load CANN and the matching NPU/Megatron environment before running this script.
# Tested on a four-layer checkpoint with 8 Ascend 950DT devices; full-model training
# requires separate memory planning and validation. See README.md in this directory.
set -eo pipefail

MODEL=${MODEL:?Set MODEL to a local DeepSeek-V4.1 checkpoint directory}
DATASET=${DATASET:-'AI-ModelScope/alpaca-gpt4-data-zh#500'}
ENGRAM_TOKENIZER_MAP=${ENGRAM_TOKENIZER_MAP:-"${MODEL}/engram_tokenizer_map.json"}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/deepseek-v41-npu-lora}
MAX_LENGTH=${MAX_LENGTH:-2048}
TRAIN_ITERS=${TRAIN_ITERS:-10}
SAVE_STEPS=${SAVE_STEPS:-5}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export TOKENIZERS_PARALLELISM=false
export NGRAM_EMBEDDING_CPU_OFFLOAD=1

# The map must match the checkpoint's tokenizer, hash layer IDs and vocabulary.
# Prepare it with the matching MCore tools/engram/generate_tokenizer_map.py.
megatron sft \
    --model "${MODEL}" \
    --model_type deepseek_v41 \
    --dataset "${DATASET}" \
    --split_dataset_ratio 0 \
    --language_model_only true \
    --engram_tokenizer_map "${ENGRAM_TOKENIZER_MAP}" \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --expert_tensor_parallel_size 1 \
    --sequence_parallel false \
    --use_cpu_initialization true \
    --bf16 true \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --target_regex '^(?!.*dspark).*\.(linear_q_down_proj|linear_q_up_proj|linear_proj)$' \
    --micro_batch_size 1 \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --padding_free true \
    --packing true \
    --packing_length "${MAX_LENGTH}" \
    --packing_strategy binpack \
    --packing_num_proc 1 \
    --attention_backend auto \
    --train_iters "${TRAIN_ITERS}" \
    --lr 1e-4 \
    --lr_decay_style constant \
    --weight_decay 0 \
    --seed 42 \
    --recompute_granularity none \
    --gradient_accumulation_fusion false \
    --moe_grouped_gemm true \
    --moe_permute_fusion false \
    --masked_softmax_fusion false \
    --bias_activation_fusion false \
    --bias_dropout_fusion false \
    --apply_rope_fusion false \
    --save_safetensors false \
    --merge_lora false \
    --save_steps "${SAVE_STEPS}" \
    --eval_iters 0 \
    --dataset_num_proc 1 \
    --dataloader_num_workers 0 \
    --logging_steps 1 \
    --report_to tensorboard \
    --check_model false \
    --output_dir "${OUTPUT_DIR}"
