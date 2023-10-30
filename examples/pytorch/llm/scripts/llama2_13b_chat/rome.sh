# Experimental environment: A10
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python rome_infer.py \
    --model_id_or_path modelscope/Llama-2-13b-chat-ms \
    --model_revision master \
    --template_type default-generation \
    --dtype bf16 \
    --eval_human true \
    --max_new_tokens 128 \
    --temperature 0.1 \
    --top_k 50 \
    --top_p 0.9 \
    --do_sample true \
    --rome_request_file rome_example/request.json
