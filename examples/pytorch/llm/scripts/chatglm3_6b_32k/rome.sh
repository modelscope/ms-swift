# Experimental environment: A10
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python rome_infer.py \
    --model_id_or_path ZhipuAI/chatglm3-6b-32k \
    --model_revision master \
    --template_type chatglm3 \
    --dtype AUTO \
    --eval_human true \
    --max_new_tokens 128 \
    --temperature 0.1 \
    --top_k 50 \
    --top_p 0.9 \
    --do_sample true \
    --rome_request_file rome_example/request.json
