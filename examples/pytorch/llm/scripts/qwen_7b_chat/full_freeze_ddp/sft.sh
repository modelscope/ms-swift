# Experimental environment: 2 * A100
# 2 * 80GB GPU memory
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model_type qwen-7b-chat \
    --sft_type full \
    --train_dataset_sample -1 \
    --eval_steps 100 \
    --output_dir output \
    --num_train_epochs 1 \
    --max_length 4096 \
    --learning_rate 1e-5 \
    --use_flash_attn true \
    --save_only_model true \
    --dataset codefuse-evol-instruction-zh \
    --freeze_parameters 0.25 \
    --additional_trainable_parameters transformer.wte \
    --preprocess_num_proc 4 \
