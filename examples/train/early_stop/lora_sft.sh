# 22GB
# Change: https://github.com/modelscope/ms-swift/blob/main/swift/plugin/callback.py
# If you have custom implementations
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --split_dataset_ratio 0.1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --early_stop_interval 3 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --metric_for_best_model loss \

# a sample result
# Train:  83%|██████████████████████████████████████████████████████████████████████████████████████████▊                  | 10/12 [00:42<00:06,  3.14s/it]
#{'eval_loss': 4.26491737, 'eval_token_acc': 0.57142857, 'eval_runtime': 20.3945, 'eval_samples_per_second': 0.049, 'eval_steps_per_second': 0.049, 'epoch': 2.5, 'global_step/max_steps': '10/12', 'percentage': '83.33%', 'elapsed_time': '1m 2s', 'remaining_time': '12s'}
#Val: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 28.85it/s]
#[INFO:swift] Saving model checkpoint to output/xxx/checkpoint-10
#[INFO:swift] Training stop because of eval metric is stable at step 10
