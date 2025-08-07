swift export \
    --model Qwen/Qwen2.5-7B \
    --dataset 'AI-ModelScope/ruozhiba:all' \
    --max_length 8192 \
    --dataset_num_proc 64 \
    --to_cached_dataset true \
    --split_dataset_ratio 0.01 \
    --use_chat_template false \
    --loss_scale all \
    --output_dir ./pretrain_cached_dataset

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift pt \
    --model Qwen/Qwen2.5-7B \
    --train_type full \
    --cached_dataset './pretrain_cached_dataset' \
    --num_train_epochs 3 \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --packing true \
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
    --attn_impl flash_attn
