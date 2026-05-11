# REAL, https://arxiv.org/abs/2602.05630

CUDA_VISIBLE_DEVICES=2 \
swift rollout \
    --model Qwen/Qwen2.5-0.5B-Instruct

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
MASTER_PORT=29600 \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --save_safetensors true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --dataset 'AI-MO/NuminaMath-TIR#4000' \
    --num_train_epochs 1 \
    --micro_batch_size 8 \
    --global_batch_size 128 \
    --num_generations 8 \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --max_completion_length 2048 \
    --tuner_type full \
    --gradient_accumulation_fusion false \
    --lr 2e-6 \
    --bf16 true \
    --beta 0.001 \
    --dynamic_sample false \
    --loss_type real \
    --logging_steps 1 \
    --recompute_granularity selective \
    --finetune \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --padding_free true \
    --attention_backend flash \
    --no_save_optim \
    --no_save_rng \
    --temperature 0.6 \
    --system """You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}.""" \
    --log_completions true \
    --eval_steps 100 \
    --save_steps 100
