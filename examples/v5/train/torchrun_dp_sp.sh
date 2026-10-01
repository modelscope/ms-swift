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
NPROC_PER_NODE=2 \
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
