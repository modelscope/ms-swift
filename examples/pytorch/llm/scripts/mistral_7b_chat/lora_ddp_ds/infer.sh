# Experimental environment: A10
# If you want to merge LoRA weight and save it, you need to set `--merge_lora_and_save true`.
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --model_id_or_path AI-ModelScope/Mistral-7B-Instruct-v0.1 \
    --model_revision master \
    --sft_type lora \
    --template_type llama \
    --dtype bf16 \
    --ckpt_dir "output/mistral-7b-chat/vx_xxx/checkpoint-xxx" \
    --eval_human false \
    --dataset leetcode-python-en \
    --max_length 4096 \
    --max_new_tokens 2048 \
    --temperature 0.9 \
    --top_k 20 \
    --top_p 0.9 \
    --repetition_penalty 1.05 \
    --do_sample true \
    --merge_lora_and_save false \
