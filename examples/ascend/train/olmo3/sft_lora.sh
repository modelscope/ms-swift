export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NPROC_PER_NODE=16

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export HCCL_CONNECT_TIMEOUT=600

MODEL_PATH="your model path"
DATASET_PATH="your dataset path"
OUTPUT_DIR="your output dir"

swift sft \
    --model "${MODEL_PATH}" \
    --model_type olmo3 \
    --template olmo3_think \
    --dataset "${DATASET_PATH}" \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --num_train_epochs 1 \
    --max_length 4096 \
    --padding_free false \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --logging_steps 5 \
    --save_steps 200 \
    --save_total_limit 2 \
    --output_dir "${OUTPUT_DIR}" \
    --report_to none
