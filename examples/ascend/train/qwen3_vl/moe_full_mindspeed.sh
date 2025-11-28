# 16 * 64GiB Ascend A3
# Modified from https://github.com/modelscope/ms-swift/blob/main/examples/models/qwen3_vl/mcore_full.sh
PYTORCH_NPU_ALLOC_CONF='expandable_segments:True' \
MULTI_STREAM_MEMORY_REUSE=2 \
OMP_NUM_THREADS=14 \
NPROC_PER_NODE=16 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
megatron sft \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
              'swift/VideoChatGPT:Generic#2000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-VL-30B-A3B-Instruct \
    --eval_interval 500 \
    --save_interval 500 \
    --max_length 4096 \
    --packing true \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --attention_backend flash
    # --moe_permute_fusion true
    # --optimizer_cpu_offload true
    # --use_precision_aware_optimizer true
    # --optimizer_offload_fraction 0.2
