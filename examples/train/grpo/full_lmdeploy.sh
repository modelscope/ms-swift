# A800 * 8
# pip install lmdeploy==0.6.4
# exp link: https://wandb.ai/tastelikefeet/grpo_perf_test?nw=nwuseryuzezyz
# In exp no `--system 'examples/train/grpo/prompt.txt'`, so the format reward is not correct and there are speed diffs with this script
# important args: --num_infer_workers 2 --num_iterations 2 --use_lmdeploy true --async_generate true
# if forward/backward error: pip install deepspeed==0.14.5
# and change deepspeed zero3.json stage3_prefetch_bucket_size=0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B \
    --reward_funcs accuracy format \
    --use_lmdeploy true \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --max_completion_length 1536 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --learning_rate 1e-6 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 60 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --async_generate true \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 2 \
    --num_infer_workers 2 \
