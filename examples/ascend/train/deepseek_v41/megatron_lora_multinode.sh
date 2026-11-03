#!/usr/bin/env bash
# DeepSeek-V4.1 full-depth text-only LoRA on 2 hosts with 8 Ascend NPUs each.
# Based on the validated TP1/PP4/EP4 run; load the matching CANN/NPU environment.
# Set NODE_RANK=0/1 and the same MASTER_ADDR on both hosts. See README.md.
set -eo pipefail

MODEL=${MODEL:?Set MODEL to a local DeepSeek-V4.1 checkpoint directory}
DATASET=${DATASET:-'AI-ModelScope/alpaca-gpt4-data-zh#500'}
ENGRAM_TOKENIZER_MAP=${ENGRAM_TOKENIZER_MAP:-"${MODEL}/engram_tokenizer_map.json"}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/deepseek-v41-npu-lora-multinode}
MAX_LENGTH=${MAX_LENGTH:-256}
TRAIN_ITERS=${TRAIN_ITERS:-10}
SAVE_STEPS=${SAVE_STEPS:-5}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NNODES=2
export NODE_RANK=${NODE_RANK:?Set NODE_RANK to 0 or 1}
export MASTER_ADDR=${MASTER_ADDR:?Set MASTER_ADDR to the reachable address of node 0}
export MASTER_PORT=${MASTER_PORT:-29500}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-600}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
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
    --pipeline_model_parallel_size 4 \
    --context_parallel_size 1 \
    --expert_model_parallel_size 4 \
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
    --padding_free false \
    --packing false \
    --attention_backend unfused \
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
    --add_version false \
    --output_dir "${OUTPUT_DIR}"
