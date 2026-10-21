# v5 SFT launched through Ray on the transformers backend (mode=ray, 2 data-parallel workers).
#
# Unlike the torchrun path, there is NO NPROC_PER_NODE and no torch.distributed.run: `--mode ray` makes
# initialize_twinkle build a 'model' DeviceGroup that 2 Ray workers join, while the driver process only
# orchestrates (it holds no GPU itself, so `--nproc_per_node` must be given explicitly). Batch semantics
# match torchrun: per_device_train_batch_size is PER DP RANK. The driver's DataLoader builds the GLOBAL
# batch (per_device x dp = 2 x 2 = 4 here) and forward_backward(dispatch='slice_dp') scatters it to
# per_device rows on each worker, so the global batch is per_device x dp x gradient_accumulation_steps --
# the same as under torchrun. The transformers backend has no TP/PP weight sharding, so the Ray mesh is
# pure data parallelism.
#
# --attn_impl flash_attention_2 is pinned: on the validation box (torch 2.11 + cuDNN 9.19) the default
# SDPA path picks a cuDNN attention kernel whose bf16 backward returns NaN on longer rows -- clean
# through ~step 25, then grad_norm goes non-finite and loss becomes nan. FA2 avoids it (and is standard
# for multi-GPU long-sequence training); `--attn_impl eager` is the fallback where flash-attn is absent.
CUDA_VISIBLE_DEVICES=0,1 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
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
