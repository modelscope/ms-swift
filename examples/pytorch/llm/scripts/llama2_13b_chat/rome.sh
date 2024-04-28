# Experimental environment: A10
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python rome_infer.py \
    --model_id_or_path modelscope/Llama-2-13b-chat-ms \
    --model_revision master \
    --template_type AUTO \
    --dtype AUTO \
    --max_new_tokens 128 \
    --temperature 0.1 \
    --top_p 0.7 \
    --do_sample true \
    --rome_request_file rome_example/request.json
