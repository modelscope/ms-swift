# 22GB
NPROC_PER_NODE=$ARNOLD_WORKER_GPU \
NNODES=$ARNOLD_WORKER_NUM \
NODE_RANK=$ARNOLD_ID \
MASTER_ADDR=$ARNOLD_WORKER_0_HOST \
MASTER_PORT=$ARNOLD_WORKER_0_PORT \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /mnt/bn/medical-mllm-lf/share/hf_models/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 \
    --train_type full \
    --dataset '/mnt/bn/douyin-health-mllm-lf/shengzhiwang/work/FG-VLM-Dataset/ms-swift/data/no_cot/invasive_train_no_cot.json' \
              '/mnt/bn/douyin-health-mllm-lf/shengzhiwang/work/FG-VLM-Dataset/ms-swift/data/no_cot/yanghu_train_no_cot.json' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --eval_steps -1 \
    --save_steps 50 \
    --save_total_limit 100 \
    --logging_steps 5 \
    --max_length 16384 \
    --output_dir /mnt/bn/douyin-health-mllm-lf/shengzhiwang/work/FG-VLM-Dataset/API_process/Model_test/train_data/data/0304_version/train_no_cot/ckpts \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --attn_impl flash_attn \
    --deepspeed zero2 \
    --report_to wandb 
