#!/bin/bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model 'Qwen/Qwen3-8B' \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh' \
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
    --max_length 4096 \
    --output_dir output \
    --system You\ are\ a\ helpful\ assistant. \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --fsdp './examples/ascend/activation_cpu_offload/fsdp2.json'

#  --dataset AI-ModelScope/LongAlpaca-12k
# activation_cpu_offload=false

# {'loss': 2.93329144, 'grad_norm': 2.44835496, 'learning_rate': 0.0001, 'token_acc': 0.56405613, 'epoch': 0.06, 'global_step/max_steps': '1/16', 'percentage': '6.25%', 'elapsed_time': '8s', 'remaining_time': '2m 6s', 'memory(GiB)': 24.8, 'train_speed(iter/s)': 0.118837}
# {'loss': 2.93490505, 'grad_norm': 2.63550186, 'learning_rate': 8.346e-05, 'token_acc': 0.58979954, 'epoch': 0.32, 'global_step/max_steps': '5/16', 'percentage': '31.25%', 'elapsed_time': '28s', 'remaining_time': '1m 2s', 'memory(GiB)': 57.91, 'train_speed(iter/s)': 0.175644}
# Train:  31%|███████████████████████████████████                                                                             | 5/16 [00:28<00:57,  5.22s/it][INFO:swift] Saving model checkpoint to /model/ljl/project/ms-swift/output/v60-20260202-130514/checkpoint-5
# {'loss': 1.61339226, 'grad_norm': 1.05343676, 'learning_rate': 3.455e-05, 'token_acc': 0.63342983, 'epoch': 0.64, 'global_step/max_steps': '10/16', 'percentage': '62.50%', 'elapsed_time': '51s', 'remaining_time': '31s', 'memory(GiB)': 58.02, 'train_speed(iter/s)': 0.192856}
# Train:  62%|█████████████████████████████████████████████████████████████████████▍                                         | 10/16 [00:51<00:27,  4.66s/it][INFO:swift] Saving model checkpoint to /model/ljl/project/ms-swift/output/v60-20260202-130514/checkpoint-10
# {'loss': 1.32472887, 'grad_norm': 0.60581738, 'learning_rate': 1.09e-06, 'token_acc': 0.64779323, 'epoch': 0.96, 'global_step/max_steps': '15/16', 'percentage': '93.75%', 'elapsed_time': '1m 13s', 'remaining_time': '4s', 'memory(GiB)': 58.02, 'train_speed(iter/s)': 0.204973}
# Train:  94%|████████████████████████████████████████████████████████████████████████████████████████████████████████       | 15/16 [01:13<00:04,  4.12s/it][INFO:swift] Saving model checkpoint to /model/ljl/project/ms-swift/output/v60-20260202-130514/checkpoint-15
# Train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [01:17<00:00,  4.25s/it][INFO:swift] Saving model checkpoint to /model/ljl/project/ms-swift/output/v60-20260202-130514/checkpoint-16
# {'train_runtime': 79.7064, 'train_samples_per_second': 6.311, 'train_steps_per_second': 0.201, 'train_loss': 1.91648413, 'token_acc': 0.68027888, 'epoch': 1.0, 'global_step/max_steps': '16/16', 'percentage': '100.00%', 'elapsed_time': '1m 19s', 'remaining_time': '0s', 'memory(GiB)': 58.02, 'train_speed(iter/s)': 0.200728}
# Train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [01:19<00:00,  4.98s/it]


#  --dataset AI-ModelScope/LongAlpaca-12k
# "activation_cpu_offload": true

# {'loss': 2.93329144, 'grad_norm': 2.44853568, 'learning_rate': 0.0001, 'token_acc': 0.56405613, 'epoch': 0.06, 'global_step/max_steps': '1/16', 'percentage': '6.25%', 'elapsed_time': '26s', 'remaining_time': '6m 43s', 'memory(GiB)': 24.62, 'train_speed(iter/s)': 0.037168}
# {'loss': 2.93512678, 'grad_norm': 2.6212213, 'learning_rate': 8.346e-05, 'token_acc': 0.5895268, 'epoch': 0.32, 'global_step/max_steps': '5/16', 'percentage': '31.25%', 'elapsed_time': '1m 21s', 'remaining_time': '2m 58s', 'memory(GiB)': 26.93, 'train_speed(iter/s)': 0.061631}
# Train:  31%|███████████████████████████████████                                                                             | 5/16 [01:21<02:30, 13.67s/it][INFO:swift] Saving model checkpoint to /model/ljl/project/ms-swift/output/v59-20260202-125158/checkpoint-5
# {'loss': 1.61200867, 'grad_norm': 1.05091298, 'learning_rate': 3.455e-05, 'token_acc': 0.63310818, 'epoch': 0.64, 'global_step/max_steps': '10/16', 'percentage': '62.50%', 'elapsed_time': '2m 20s', 'remaining_time': '1m 24s', 'memory(GiB)': 26.93, 'train_speed(iter/s)': 0.0712}
# Train:  62%|█████████████████████████████████████████████████████████████████████▍                                         | 10/16 [02:20<01:11, 11.97s/it][INFO:swift] Saving model checkpoint to /model/ljl/project/ms-swift/output/v59-20260202-125158/checkpoint-10
# {'loss': 1.32489185, 'grad_norm': 0.60476321, 'learning_rate': 1.09e-06, 'token_acc': 0.64746468, 'epoch': 0.96, 'global_step/max_steps': '15/16', 'percentage': '93.75%', 'elapsed_time': '3m 11s', 'remaining_time': '12s', 'memory(GiB)': 26.94, 'train_speed(iter/s)': 0.078265}
# Train:  94%|████████████████████████████████████████████████████████████████████████████████████████████████████████       | 15/16 [03:11<00:10, 10.03s/it][INFO:swift] Saving model checkpoint to /model/ljl/project/ms-swift/output/v59-20260202-125158/checkpoint-15
# Train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [03:20<00:00,  9.65s/it][INFO:swift] Saving model checkpoint to /model/ljl/project/ms-swift/output/v59-20260202-125158/checkpoint-16
# {'train_runtime': 202.2537, 'train_samples_per_second': 2.487, 'train_steps_per_second': 0.079, 'train_loss': 1.91632293, 'token_acc': 0.67729084, 'epoch': 1.0, 'global_step/max_steps': '16/16', 'percentage': '100.00%', 'elapsed_time': '3m 22s', 'remaining_time': '0s', 'memory(GiB)': 26.94, 'train_speed(iter/s)': 0.078996}
# Train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [03:22<00:00, 12.66s/it]
