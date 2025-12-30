#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model 'Qwen/Qwen3-8B' \
    --dataset 'swift/self-cognition#1000' \ \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --fsdp './examples/train/activation_cpu_offload/fsdp2.json'

# activation_cpu_offload=true
# {'loss': 1.13790035, 'grad_norm': 1.41501045, 'learning_rate': 5e-05, 'token_acc': 0.83174487, 'epoch': 0.04, 'global_step/max_steps': '1/27', 'percentage': '3.70%', 'elapsed_time': '3m 36s', 'remaining_time': '1h 33m 43s', 'memory(GiB)': 32.54, 'train_speed(iter/s)': 0.004623}
# {'loss': 0.94536996, 'grad_norm': 0.85681218, 'learning_rate': 9.649e-05, 'token_acc': 0.84959215, 'epoch': 0.19, 'global_step/max_steps': '5/27', 'percentage': '18.52%', 'elapsed_time': '17m 16s', 'remaining_time': '1h 16m 1s', 'memory(GiB)': 39.92, 'train_speed(iter/s)': 0.004823}
# {'loss': 0.68646059, 'grad_norm': 0.25970718, 'learning_rate': 7.679e-05, 'token_acc': 0.85168261, 'epoch': 0.37, 'global_step/max_steps': '10/27', 'percentage': '37.04%', 'elapsed_time': '34m 34s', 'remaining_time': '58m 46s', 'memory(GiB)': 39.92, 'train_speed(iter/s)': 0.00482}

# activation_cpu_offload=false
# OOM
# {'loss': 1.13790035, 'grad_norm': 1.41472316, 'learning_rate': 5e-05, 'token_acc': 0.83174487, 'epoch': 0.04, 'global_step/max_steps': '1/27', 'percentage': '3.70%', 'elapsed_time': '46s', 'remaining_time': '20m 1s', 'memory(GiB)': 61.79, 'train_speed(iter/s)': 0.021641}
# Train:  11%|████████████                                                                                                 | 3/27 [01:52<14:28, 36.19s/it
# ...
# [rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 320.00 MiB. GPU 1 has a total capacity of 63.59 GiB of which 0 bytes is free. Of the allocated memory 55.85 GiB is allocated by PyTorch, and 3.64 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
