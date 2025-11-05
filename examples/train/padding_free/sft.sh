# Supported multimodal models reference:
# https://github.com/modelscope/ms-swift/blob/main/examples/train/packing/qwen2_5_vl.sh
# without padding_free: 4 * 60GiB, 26h
# padding_free: 4 * 44GiB, 13h
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2.5-7B \
    --train_type full \
    --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output/Qwen2.5-7B \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attn \
    --padding_free true
