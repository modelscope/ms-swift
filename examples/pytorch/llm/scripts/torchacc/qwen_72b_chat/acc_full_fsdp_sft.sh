# Experimental environment: 4 * 8*A100
# 80GB GPU memory
# Note: TorchAcc is currently only available internally.

export USE_TORCHACC=1
export XLA_FLAGS='--xla_multiheap_size_constraint_per_heap=4831838208 --xla_disable_hlo_passes=all-gather-combiner,all-reduce-combiner,reduce-scatter-combiner'
export XLA_IR_SHAPE_CACHE_SIZE=100000000
export XLA_ALLOCATOR_FRACTION=0.97

# Note: You need to set the correct MASTER_ADDR, MASTER_PORT and NODE_RANK for each node.

MASTER_ADDR=127.0.0.1 \
MASTER_PORT=12456 \
NODE_RANK=0 \
NNODES=4 \
NPROC_PER_NODE=8 \
swift sft \
    --model_type qwen-72b-chat \
    --model_layer_cls_name QWenBlock \
    --dataset codefuse-python-en \
    --sft_type full \
    --output_dir output \
    --num_train_epochs 1 \
    --max_length 1024 \
    --batch_size 1 \
    --use_flash_attn true \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing no \
    --tuner_backend 'peft' \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 100 \
    --metric_warmup_step 0.1 \
    --report_to 'none'
    --fsdp_num 32
