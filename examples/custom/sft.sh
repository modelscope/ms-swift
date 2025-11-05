# sh examples/custom/sft.sh
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --custom_register_path examples/custom/dataset.py \
                           examples/custom/model.py \
    --model AI-ModelScope/Nemotron-Mini-4B-Instruct \
    --train_type lora \
    --dataset swift/stsb \
    --split_dataset_ratio 0.01 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_length 2048 \
    --output_dir output \
    --dataset_num_proc 4
