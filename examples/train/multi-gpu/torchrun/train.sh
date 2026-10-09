# v5 SFT under torchrun: data parallel (dp2), with the sequence-parallel (Ulysses) variant noted.
#
# Setting NPROC_PER_NODE makes swift/cli/main.py relaunch the dev CLI under `torch.distributed.run`
# (mode='local'): each of the 2 ranks runs the same recipe over its shard, and the transformers backend
# builds a pure data-parallel mesh from the world size. `--parallel_spec dp2` states that layout
# explicitly (it is the default here). per_device_train_batch_size must be >= the dp size, since the
# driver batch is scattered across the dp ranks.
#
# SEQUENCE PARALLEL (Ulysses) instead splits each sequence along the token dim inside attention: add
# `--sequence_parallel_size 2` (TemplateConfig). A hybrid dp x sp layout uses parallel_spec, e.g.
# `--parallel_spec dp2sp2` under NPROC_PER_NODE=4. SP is a pure re-partition, so the loss equals the
# single-process loss on the same global batch.
#
# MEMORY SHARDING is a flag on this same command, not a separate launch form. Plain dp2 (above) keeps a
# full replica per rank; to shard optimizer state / gradients / parameters across the ranks add
# `--deepspeed zero2` (optim+grads) or `--deepspeed zero3` (params too), or `--fsdp` / `--fsdp fsdp2`
# with `--fsdp_config` for torch FSDP. DDP is the default (no flag); `device_map` model-parallel is not
# a training path here. The launcher and every other flag are unchanged.
#
# --attn_impl flash_attention_2 is pinned on purpose. On the box these examples were validated on
# (torch 2.11 + cuDNN 9.19) the default SDPA path selects a cuDNN attention kernel whose bf16 BACKWARD
# returns NaN on the longer alpaca rows: training is clean through ~step 25, then grad_norm goes
# non-finite (it drops out of the log, the tracker discards non-finite scalars) and loss becomes nan.
# FA2 avoids that kernel and is the usual choice for multi-GPU long-sequence training anyway. An
# environment without flash-attn can use `--attn_impl eager`, which also sidesteps cuDNN.
NPROC_PER_NODE=2 \
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
    --parallel_spec dp2 \
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
