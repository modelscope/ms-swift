# Experimental environment: 4 * A100
# 4 * 50GB GPU memory
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type qwen-audio-chat \
    --sft_type full \
    --train_dataset_sample -1 \
    --eval_steps 100 \
    --output_dir output \
    --num_train_epochs 1 \
    --max_length 2048 \
    --learning_rate 1e-5 \
    --use_flash_attn true \
    --save_only_model true \
    --dataset aishell1-mini-zh \
    --lazy_tokenize true \
