# Env: 8 * H800
# Max Length: 65536
# GPU Memory: 8 * 80GiB, Training Speed 18.97s/it
NPROC_PER_NODE=8 \
CELOSS_PARALLEL_SIZE=2048 \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --dataset 'AI-ModelScope/LongAlpaca-12k' \
    --load_from_cache_file true \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 2 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --save_steps 50 \
    --max_length 65535 \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --logging_steps 1 \
    --use_logits_to_keep false \
    --sequence_parallel_size 4 \
    --padding_free true
# Train:   1%|          | 1/189 [02:41<8:25:06, 161.21s/it]

# {'loss': '1.461', 'grad_norm': '1.705', 'learning_rate': '1e-05', 'token_acc': '0.6399', 'epoch': '0.016', 'global_step/max_steps': '1/189', 'elapsed_time': '2m 41s', 'remaining_time': '8h 25m 20s', 'memory(GiB)': '67.95', 'train_speed(s/it)': '161.3'}
# {'loss': '1.484', 'grad_norm': '1.666', 'learning_rate': '2e-05', 'token_acc': '0.6324', 'epoch': '0.032', 'global_step/max_steps': '2/189', 'elapsed_time': '3m 55s', 'remaining_time': '6h 6m 35s', 'memory(GiB)': '68.05', 'train_speed(s/it)': '117.6'}
# {'loss': '1.477', 'grad_norm': '1.767', 'learning_rate': '3e-05', 'token_acc': '0.6311', 'epoch': '0.048', 'global_step/max_steps': '3/189', 'elapsed_time': '4m 48s', 'remaining_time': '4h 57m 21s', 'memory(GiB)': '68.05', 'train_speed(s/it)': '95.92'}
# {'loss': '1.498', 'grad_norm': '1.683', 'learning_rate': '4e-05', 'token_acc': '0.6355', 'epoch': '0.064', 'global_step/max_steps': '4/189', 'elapsed_time': '5m 36s', 'remaining_time': '4h 18m 56s', 'memory(GiB)': '68.05', 'train_speed(s/it)': '83.98'}
# {'loss': '1.422', 'grad_norm': '1.528', 'learning_rate': '5e-05', 'token_acc': '0.6329', 'epoch': '0.08', 'global_step/max_steps': '5/189', 'elapsed_time': '6m 17s', 'remaining_time': '3h 51m 17s', 'memory(GiB)': '68.05', 'train_speed(s/it)': '75.42'}
# {'loss': '1.322', 'grad_norm': '1.203', 'learning_rate': '6e-05', 'token_acc': '0.6457', 'epoch': '0.096', 'global_step/max_steps': '6/189', 'elapsed_time': '6m 53s', 'remaining_time': '3h 30m 1s', 'memory(GiB)': '68.05', 'train_speed(s/it)': '68.86'}
# ...
# Train:  24%|██▍       | 46/189 [17:53<45:13, 18.97s/it]
# {'loss': '1.049', 'grad_norm': '0.6095', 'learning_rate': '9.035e-05', 'token_acc': '0.7018', 'epoch': '0.736', 'global_step/max_steps': '46/189', 'elapsed_time': '17m 54s', 'remaining_time': '55m 38s', 'memory(GiB)': '68.06', 'train_speed(s/it)': '23.34'}
