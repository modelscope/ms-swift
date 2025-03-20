# 25GiB
# vit/merger lr 2e-6; llm lora lr 1e-4
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset 'AI-ModelScope/coco#20000' \
    --train_type custom \
    --optimizer custom \
    --external_plugins 'examples/train/multimodal/custom_tuner/custom_plugin.py' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4
