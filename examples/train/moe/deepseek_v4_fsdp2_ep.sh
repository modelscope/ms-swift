# DeepSeek-V4-Flash full fine-tuning with FSDP2 + Expert Parallel
# 8 * 80GiB
# Expert Parallel shards MoE experts across GPUs via FSDP2 + All-to-All dispatch,
# and offloads expert optimizer states to CPU to save GPU memory.
# --ep_size 8: shard experts across 8 GPUs (one expert partition per GPU)
# --offload_expert_optimizer: keep expert AdamW states on CPU
# --expert_optimizer_backend torch: use PyTorch CPUAdam (alternative: deepspeed for DeepSpeedCPUAdam)

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --nproc_per_node=8 \
    --master_port 29500 \
    -m swift.cli.sft \
    --model /personal/model/DeepSeek-V4-Flash-BF16 \
    --tuner_type full \
    --dataset /personal/data/OpenHermes-2.5 \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --eval_strategy no \
    --torch_dtype bfloat16 \
    --save_steps 5000 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --router_aux_loss_coef 1e-3 \
    --dataloader_num_workers 0 \
    --dataloader_persistent_workers false \
    --fsdp '{"fsdp":"full_shard auto_wrap offload","fsdp_config":{"fsdp_version":2,"reshard_after_forward":true,"auto_wrap_policy":"TRANSFORMER_BASED_WRAP","cpu_ram_efficient_loading":false,"state_dict_type":"FULL_STATE_DICT","activation_checkpointing":true}}' \
    --ep_size 8 \
    --gradient_checkpointing false \
    --offload_expert_optimizer true \
    --expert_optimizer_backend torch \
    --expert_optimizer_dtype bf16 \
    --dataset_num_proc 16 \
    --max_length 128000 \
    --truncation_strategy right \
    --output_dir /root/output