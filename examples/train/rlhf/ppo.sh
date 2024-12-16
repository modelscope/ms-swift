CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=. \
python swift/cli/rlft.py \
    --model /mnt/workspace/yzhao/tastelikefeet/swift/output/pythia-1b-deduped/v24-20241216-185813/checkpoint-400 \
    --model_type qwen2_5 \
    --reward_type agent \
    --template default \
    --dataset swift/ToolBench:ppo#5000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 1 \
    --max_length 2048 \
    --num_train_epochs 3 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.05 \
    --temperature 0.5
