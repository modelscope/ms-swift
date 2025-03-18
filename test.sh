CUDA_VISIBLE_DEVICES=0 \
swift eval \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --eval_backend Native \
    --infer_backend pt \
    --eval_limit 10 \
    --eval_dataset gsm8k \
    --dataset_args '{"gsm8k": {"few_shot_num": 0}}'