# DP size = world_size // (context_parallel_size * tensor_model_parallel_size * pipeline_model_parallel_size)
#         = 8 // (1 * 1 * 1) = 8

# NOTE: global_batch_size and micro_batch_size are completion-level
# global_batch_size = micro_batch_size * DP size * gradient_accumulation_steps (128)
# generation_batch_size = global_batch_size * steps_per_generation (128 * 4 = 512)
# num_of_prompt_to_rollout = generation_batch_size / num_generations (512 / 8 = 64)
# num_of_prompt_to_train = generation_batch_size / num_generations (128 / 8 = 16)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MAX_PIXELS=602112 \
MASTER_PORT=29600 \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --dataset AI-ModelScope/clevr_cogen_a_train#10000 \
    --max_epochs 1 \
    --global_batch_size 128 \
    --micro_batch_size 4 \
    --steps_per_generation 4 \
    --num_generations 8 \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 10240 \
    --max_length 8192 \
    --max_completion_length 2048 \
    --train_type full \
    --lr 1e-6 \
    --bf16 true \
    --beta 0.001 \
    --importance_sampling_level token \
    --epsilon 0.2 \
    --epsilon_high 0.2 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --sleep_level 2 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
    --log_interval 1 \
    --recompute_granularity selective \
    --finetune \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim \
    --no_save_rng \
    --attention_backend flash \
    --temperature 1.0 \
    --system examples/train/grpo/prompt.txt \
    --padding_free true \
    --log_completions true \
    --wandb_project megatron_swift \
    --wandb_exp_name megatron_grpo \
    --train_iters 100 \
    --eval_interval 1000 \
    --save_interval 1000
