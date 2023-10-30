# Experimental environment: A10
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python rome_infer.py \
    --model_id_or_path modelscope/Llama-2-13b-chat-ms \
    --model_revision master \
    --template_type llama \
    --dtype bf16 \
    --eval_human true \
    --max_length 4096 \
    --max_new_tokens 2048 \
    --temperature 0.9 \
    --top_k 20 \
    --top_p 0.9 \
    --do_sample true \
    --rome_request_file rome_example/request.json