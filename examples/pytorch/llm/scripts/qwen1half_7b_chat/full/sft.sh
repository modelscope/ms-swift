# Experimental environment: A100
# 80GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen1half-7b-chat \
    --sft_type full \
    --train_dataset_sample -1 \
    --eval_steps 1000 \
    --output_dir output \
    --num_train_epochs 1 \
    --max_length 4096 \
    --learning_rate 1e-5 \
    --use_flash_attn true \
    --save_only_model true \
    --dataset codefuse-evol-instruction-zh \
    --preprocess_num_proc 4 \
