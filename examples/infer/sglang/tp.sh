CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --model Qwen/Qwen3-8B \
    --infer_backend sglang \
    --val_dataset AI-ModelScope/alpaca-gpt4-data-zh#2000 \
    --max_new_tokens 2048 \
    --sglang_context_length 8192 \
    --sglang_tp_size 2 \
    --write_batch_size 1000
