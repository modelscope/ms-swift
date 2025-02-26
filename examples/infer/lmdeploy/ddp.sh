NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend lmdeploy \
    --val_dataset AI-ModelScope/alpaca-gpt4-data-zh#1000 \
    --max_new_tokens 2048
