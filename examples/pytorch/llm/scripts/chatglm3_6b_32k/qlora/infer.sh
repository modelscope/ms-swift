# Experimental environment: A10, 3090
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --ckpt_dir "output/chatglm3-6b-32k/v4-20240125-235528/checkpoint-11278" \
    --load_dataset_config true \
    --max_length 4096 \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --top_p 0.7 \
    --eval_human true \
    --repetition_penalty 1. \
    --do_sample true \
    --infer_backend pt \
    --merge_lora_and_save false