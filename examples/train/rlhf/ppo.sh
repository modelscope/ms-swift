CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=. \
python swift/cli/rlft.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --model_type qwen2_5 \
    --reward_type agent \
    --dataset swift/ToolBench#5000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --max_length 2048 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.05 \
    --temperature 0.5 \
    --gradient_accumulation_steps 4
