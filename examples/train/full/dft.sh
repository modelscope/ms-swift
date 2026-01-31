# 4*80G
# exp: https://github.com/modelscope/ms-swift/pull/5355
# export HARD_GATING_PROBABILITY_THRESHOLD=0.3 将低于概率阈值的token进行掩码,不参与loss计算 https://arxiv.org/abs/2601.09195
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model Qwen/Qwen2.5-Math-1.5B \
    --tuner_type full \
    --dataset AI-MO/NuminaMath-CoT#100000 \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --enable_dft_loss true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 32 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.1 \
    --deepspeed zero2 \
    --dataloader_num_workers 4
