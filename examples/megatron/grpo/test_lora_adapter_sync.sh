# Test: Megatron GRPO LoRA with adapter-only sync (sleep_level=1)
# This test verifies the _move_adapter_to_vllm path (peft_format export)
# After first full sync, subsequent syncs only transfer LoRA deltas.

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MAX_PIXELS=602112
export WANDB_MODE=disabled

PYTHON=/mnt/nas2/anaconda3/envs/vllm_main/bin/python

$PYTHON -m torch.distributed.run \
    --nproc_per_node 4 \
    --master_port 29701 \
    /mnt/nas2/hujinghan.hjh/workspace/swift/swift/cli/_megatron/rlhf.py \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --save_safetensors true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --dataset AI-ModelScope/clevr_cogen_a_train#1000 \
    --num_train_epochs 1 \
    --global_batch_size 16 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 4 \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 4096 \
    --max_length 2048 \
    --max_completion_length 1024 \
    --tuner_type lora \
    --lora_rank 8 \
    --vllm_enable_lora true \
    --lr 5e-5 \
    --bf16 true \
    --beta 0.001 \
    --importance_sampling_level token \
    --epsilon 0.2 \
    --epsilon_high 0.2 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --sleep_level 1 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
    --logging_steps 1 \
    --recompute_granularity selective \
    --finetune \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --no_save_optim \
    --no_save_rng \
    --attention_backend flash \
    --temperature 1.0 \
    --system examples/train/grpo/prompt.txt \
    --padding_free true \
    --train_iters 5 \
    --eval_steps 1000 \
    --save_steps 1000
