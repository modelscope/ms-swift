NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --infer_backend pt \
    --val_dataset AI-ModelScope/LaTeX_OCR#1000 \
    --max_batch_size 16 \
    --max_new_tokens 512
