CUDA_VISIBLE_DEVICES=0 \
python src/llm_infer.py \
    --model_type seqgpt-560m \
    --sft_type full \
    --template_type default-generation \
    --dtype bf16 \
    --ckpt_dir "runs/seqgpt-560m/vx_xxx/checkpoint-xxx" \
    --eval_human false \
    --dataset cmnli-zh \
    --dataset_sample 20000 \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 50 \
    --top_p 0.9 \
    --do_sample true \
