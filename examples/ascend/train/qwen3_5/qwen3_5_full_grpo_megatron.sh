export PYTHONPATH=$PYTHONPATH:<your_local_megatron_lm_path>
export MEGATRON_LM_PATH=<your_local_megatron_lm_path>
export VLLM_ASCEND_ENABLE_NZ=0
export USE_MCORE_GDN=1
#export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NPROC_PER_NODE=16 \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-35B-A3B \
    --save_safetensors true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 8 \
    --pipeline_model_parallel_size 2 \
    --dataset AI-MO/NuminaMath-TIR#1000 \
    --num_train_epochs 1 \
    --global_batch_size 16 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 4 \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_tensor_parallel_size 4 \
    --vllm_max_model_len 16384 \
    --max_length 8192 \
    --max_completion_length 8192\
    --tuner_type full \
    --lr 5e-5 \
    --bf16 true \
    --beta 0.00 \
    --importance_sampling_level sequence \
    --gradient_accumulation_fusion false \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --sleep_level 2 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
    --optimizer_cpu_offload true \
    --logging_steps 1 \
    --recompute_granularity selective \
    --finetune \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim \
    --no_save_rng \
    --attention_backend flash \
    --temperature 1.0 \
    --padding_free true \
    --sequence_parallel true \
    --log_completions true \
    --report_to tensorboard
