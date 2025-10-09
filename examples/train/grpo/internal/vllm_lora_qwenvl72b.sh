# pip install math_verify # reward function
# GPU memory: 8 * 80GiB

MAX_PIXELS=602112 \
WANDB_API_KEY=xxx \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen2.5-VL-72B-Instruct \
  --train_type lora \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.5 \
  --vllm_max_model_len 8192 \
  --vllm_tensor_parallel_size 4 \
  --dataset lmms-lab/multimodal-open-r1-8k-verified#1000 \
  --load_from_cache_file true \
  --external_plugins examples/train/grpo/plugin/plugin.py \
  --reward_funcs external_r1v_acc format \
  --reward_weights 1 0.1 \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --num_train_epochs 1 \
  --max_length 8192 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --eval_steps 500 \
  --save_steps 500 \
  --learning_rate 1e-6 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --max_completion_length 2048 \
  --num_generations 8 \
  --deepspeed zero3 \
  --temperature 1.1 \
  --top_p 1.0 \
  --top_k 80 \
  --log_completions true \
  --async_generate false \
  --offload_optimizer true \
  --offload_model true \
  --move_model_batches 40 \
  --sleep_level 1 \
  --report_to wandb \
  --system examples/train/grpo/prompt.txt
