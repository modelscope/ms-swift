# 18GB
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --infer_backend pt \
    --val_dataset AI-ModelScope/alpaca-gpt4-data-zh#1000 \
    --max_batch_size 16 \
    --max_new_tokens 512
