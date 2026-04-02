# hardware: Atlas 900 A2
# For NPU, in Transformers versions 5.0 and above, it is recommended to disable
# cpu_ram_efficient_loading in fsdp.json to avoid timeout issues at the first
# synchronization point caused by inter-card desynchronization when loading the model.
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --config_file "./examples/ascend/train/qwen3_lora_fsdp/fsdp.json" \
    swift/cli/sft.py \
    --model 'Qwen/Qwen3-32B' \
    --tuner_type lora \
    --dataset 'swift/self-cognition#1000' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_length 1200 \
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
