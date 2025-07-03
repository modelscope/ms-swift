# 8 * 80GiB
# Corresponding Megatron-SWIFT script reference:
# https://github.com/modelscope/ms-swift/tree/main/examples/train/megatron/base_to_chat.sh
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model Qwen/Qwen2.5-14B \
    --train_type full \
    --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --max_steps 2000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --packing true \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit -1 \
    --save_only_model true \
    --output_dir output/Qwen2.5-14B \
    --deepspeed zero2 \
    --attn_impl flash_attn
