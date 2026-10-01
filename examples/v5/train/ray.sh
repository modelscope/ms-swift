# v5 SFT launched through Ray on the transformers backend (mode=ray, 2 data-parallel workers).
#
# Unlike the torchrun path, there is NO NPROC_PER_NODE and no torch.distributed.run: `--mode ray` makes
# initialize_twinkle build a 'model' DeviceGroup that 2 Ray workers join, while the driver process only
# orchestrates (it holds no GPU itself, so `--nproc_per_node` must be given explicitly). Each batch the
# driver forms is scattered across the workers via forward_backward(dispatch='slice_dp'), so
# per_device_train_batch_size must be >= the dp size (2). The transformers backend has no TP/PP weight
# sharding, so the Ray mesh is pure data parallelism.
CUDA_VISIBLE_DEVICES=0,1 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --mode ray \
    --nproc_per_node 2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --logging_steps 5 \
    --save_steps 100 \
    --output_dir output \
    --report_to tensorboard
