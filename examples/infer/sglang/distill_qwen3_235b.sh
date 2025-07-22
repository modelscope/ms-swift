CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift infer \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --infer_backend sglang \
    --val_dataset liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT \
    --sglang_context_length 12000 \
    --sglang_tp_size 8 \
    --write_batch_size 10000 \
    --result_path distill_qwen3_235b.jsonl
