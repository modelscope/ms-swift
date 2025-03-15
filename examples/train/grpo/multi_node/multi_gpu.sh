WANDB_API_KEY=4a43517a837d20517a01d3ba1f0cc90cdf8a7b82 \
MAX_PIXELS=1003520 \
NPROC_PER_NODE=7 \
swift rlhf \
	--rlhf_type grpo \
	--model Qwen/Qwen2-VL-7B-Instruct \
	--train_type lora \
	--dataset AI-ModelScope/chartqa_digit_r1v_format \
	--torch_dtype bfloat16 \
	--system examples/train/grpo/prompt.txt \
	--num_train_epochs 1 \
	--max_length 2048 \
	--per_device_train_batch_size 7 \
	--per_device_eval_batch_size 7 \
	--learning_rate 1e-6 \
	--save_total_limit 2 \
	--logging_steps 5 \
	--output_dir output \
	--warmup_ratio 0.05 \
	--dataloader_num_workers 4 \
	--max_completion_length 1024 \
	--reward_funcs accuracy format \
	--num_generations 49 \
	--use_vllm true \
	--vllm_gpu_memory_utilization 0.5 \
	--deepspeed zero3 \
	--temperature 1.0 \
	--top_p 0.85 \
	--report_to wandb