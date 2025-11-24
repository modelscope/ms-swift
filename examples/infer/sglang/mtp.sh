CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model ZhipuAI/GLM-4.5-Air \
    --sglang_tp_size 4 \
    --infer_backend sglang \
    --val_dataset AI-ModelScope/alpaca-gpt4-data-zh#100 \
    --sglang_context_length 8192 \
    --max_new_tokens 2048 \
    --sglang_mem_fraction_static 0.7 \
    --sglang_speculative_algorithm EAGLE \
    --sglang_speculative_eagle_topk 1 \
    --sglang_speculative_num_steps 3 \
    --sglang_speculative_num_draft_tokens 4
