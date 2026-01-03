#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model 'Qwen/Qwen3-8B' \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_checkpointing false \
    --weight_decay 0.1 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 5 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system You\ are\ a\ helpful\ assistant. \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --fsdp './examples/train/activation_cpu_offload/fsdp2.json'


#  --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500'
# activation_cpu_offload=true

# {'loss': 2.1327579, 'grad_norm': 1.72890568, 'learning_rate': 8.346e-05, 'token_acc': 0.58396158, 'epoch': 0.32, 'global_step/max_steps': '5/16', 'percentage': '31.25%', 'elapsed_time': '5m 28s', 'remaining_time': '12m 2s', 'memory(GiB)': 24.8, 'train_speed(iter/s)': 0.015218}
# Train:  31%|██████████████████████████████████████▍                                                                                    | 5/16 [05:28<11:41, 63.77s/it][INFO:swift] Saving model checkpoint to /model/ljl/output/v45-20251231-160511/checkpoint-5
# {'loss': 1.51323957, 'grad_norm': 0.39210615, 'learning_rate': 3.455e-05, 'token_acc': 0.62368014, 'epoch': 0.64, 'global_step/max_steps': '10/16', 'percentage': '62.50%', 'elapsed_time': '10m 22s', 'remaining_time': '6m 13s', 'memory(GiB)': 24.87, 'train_speed(iter/s)': 0.016054}
# Train:  62%|████████████████████████████████████████████████████████████████████████████▎                                             | 10/16 [10:22<05:37, 56.26s/it][INFO:swift] Saving model checkpoint to /model/ljl/output/v45-20251231-160511/checkpoint-10
# {'loss': 1.36127844, 'grad_norm': 0.30676287, 'learning_rate': 1.09e-06, 'token_acc': 0.64411869, 'epoch': 0.96, 'global_step/max_steps': '15/16', 'percentage': '93.75%', 'elapsed_time': '15m 6s', 'remaining_time': '1m 0s', 'memory(GiB)': 24.87, 'train_speed(iter/s)': 0.016547}
# ...
# {'train_runtime': 962.7184, 'train_samples_per_second': 0.519, 'train_steps_per_second': 0.017, 'train_loss': 1.61728384, 'token_acc': 0.62789828, 'epoch': 1.0, 'global_step/max_steps': '16/16', 'percentage': '100.00%', 'elapsed_time': '16m 2s', 'remaining_time': '0s', 'memory(GiB)': 24.87, 'train_speed(iter/s)': 0.016624}
# Train: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [16:02<00:00, 60.16s/it]


# activation_cpu_offload=false

# {'loss': 2.15452981, 'grad_norm': 1.7536869, 'learning_rate': 0.0001, 'token_acc': 0.61792799, 'epoch': 0.06, 'global_step/max_steps': '1/16', 'percentage': '6.25%', 'elapsed_time': '46s', 'remaining_time': '11m 39s', 'memory(GiB)': 26.14, 'train_speed(iter/s)': 0.021458}
# {'loss': 2.13306689, 'grad_norm': 1.7279824, 'learning_rate': 8.346e-05, 'token_acc': 0.58295639, 'epoch': 0.32, 'global_step/max_steps': '5/16', 'percentage': '31.25%', 'elapsed_time': '2m 55s', 'remaining_time': '6m 26s', 'memory(GiB)': 26.59, 'train_speed(iter/s)': 0.028456}
# Train:  31%|██████████████████████████████████████▍                                                                                    | 5/16 [02:55<05:59, 32.65s/it][INFO:swift] Saving model checkpoint to /model/ljl/output/v44-20251231-155036/checkpoint-5
# {'loss': 1.51308346, 'grad_norm': 0.39151499, 'learning_rate': 3.455e-05, 'token_acc': 0.62377399, 'epoch': 0.64, 'global_step/max_steps': '10/16', 'percentage': '62.50%', 'elapsed_time': '5m 18s', 'remaining_time': '3m 10s', 'memory(GiB)': 27.73, 'train_speed(iter/s)': 0.031432}
# Train:  62%|████████████████████████████████████████████████████████████████████████████▎                                             | 10/16 [05:18<02:51, 28.58s/it][INFO:swift] Saving model checkpoint to /model/ljl/output/v44-20251231-155036/checkpoint-10
# {'loss': 1.36132231, 'grad_norm': 0.30557585, 'learning_rate': 1.09e-06, 'token_acc': 0.64442776, 'epoch': 0.96, 'global_step/max_steps': '15/16', 'percentage': '93.75%', 'elapsed_time': '7m 57s', 'remaining_time': '31s', 'memory(GiB)': 27.96, 'train_speed(iter/s)': 0.031437}
# ...
# {'train_runtime': 507.5282, 'train_samples_per_second': 0.985, 'train_steps_per_second': 0.032, 'train_loss': 1.61732693, 'token_acc': 0.63051608, 'epoch': 1.0, 'global_step/max_steps': '16/16', 'percentage': '100.00%', 'elapsed_time': '8m 27s', 'remaining_time': '0s', 'memory(GiB)': 27.96, 'train_speed(iter/s)': 0.031543}
# Train: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [08:27<00:00, 31.70s/it]
