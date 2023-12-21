CUDA_VISIBLE_DEVICES=6,7 \
swift sft \
	--model_type qwen-72b-chat \
	--dataset codefuse-python-en \
	--sft_type lora \
    --output_dir output \
    --num_train_epochs 1 \
    --max_length 1024 \
    --batch_size 1 \
    --use_flash_attn true \
    --preprocess_num_proc 4 \
    --gradient_accumulation_steps 1 \
    --eval_steps 2000000 \
    --save_steps 2000000