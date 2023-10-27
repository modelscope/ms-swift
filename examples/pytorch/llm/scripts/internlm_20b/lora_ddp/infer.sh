# Experimental environment: A100
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --model_id_or_path Shanghai_AI_Laboratory/internlm-20b \
    --model_revision master \
    --sft_type lora \
    --template_type default-generation \
    --dtype bf16 \
    --ckpt_dir "output/internlm-20b/vx_xxx/checkpoint-xxx" \
    --eval_human false \
    --dataset jd-sentiment-zh \
    --max_length 2048 \
    --max_new_tokens 2048 \
    --temperature 0.9 \
    --top_k 20 \
    --top_p 0.9 \
    --do_sample true \
    --merge_lora_and_save false \
