# Reinforce++-Baseline in https://arxiv.org/abs/2501.03262

CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
    --rlhf_type grpo \
    --model_id_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset AI-MO/NuminaMath-TIR#10000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 4 \
    --beta 0.04 \
    --num_generations 4 \
    --max_prompt_length 1024 \
    --max_completion_length 512 \
    --temperature 1.0 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --advantage_estimator reinforce_plus_plus \
    --kl_in_reward true \
    --scale_rewards batch
