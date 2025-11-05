# test env: lmdeploy 0.9.2.post1
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --infer_backend lmdeploy \
    --val_dataset AI-ModelScope/alpaca-gpt4-data-zh#1000 \
    --max_new_tokens 512
