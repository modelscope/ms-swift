CUDA_VISIBLE_DEVICES=0 \
python src/llm_infer.py \
    --model_type internlm-7b-chat \
    --sft_type lora \
    --template_type internlm \
    --dtype bf16 \
    --ckpt_dir "runs/internlm-7b-chat/vx_xxx/checkpoint-xxx" \
    --eval_human false \
    --dataset jd-zh \
    --train_dataset_sample -1 \
    --max_length 2048 \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 50 \
    --top_p 0.9 \
    --do_sample true \
