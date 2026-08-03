export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --config_file "./examples/ascend/train/qwen3_5/fsdp2.json" \
    swift/cli/sft.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_length 2048 \
    --num_train_epochs 2 \
    --eval_strategy no \
    --save_steps 500 \
    --logging_steps 1 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output \
    --attn_impl 'flash_attention_2' \
    --packing true
