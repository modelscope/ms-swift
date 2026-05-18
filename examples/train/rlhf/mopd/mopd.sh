NPROC_PER_NODE=7 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3-8B-Base \
    --teacher_model_group Qwen/Qwen3-32B,AI-ModelScope/Skywork-Reward-Llama-3.1-8B-v0.2  \
    --use_mopd true \
    --tuner_type full \
    --dataset open-thoughts/OpenThoughts3-1.2M#10000 \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 16000 \
    --max_completion_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 64 \
    --dataset_num_proc 4 \
    --deepspeed zero3 \
    --teacher_deepspeed zero3 \

