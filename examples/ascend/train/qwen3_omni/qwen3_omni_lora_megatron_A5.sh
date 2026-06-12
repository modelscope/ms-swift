# 8 * 96GiB Ascend A5

export USE_MCORE_GDN=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export HCCL_CONNECT_TIMEOUT=600

MODEL_PATH=Qwen/Qwen3-Omni-30B-A3B-Instruct
DATASET_PATH='AI-ModelScope/MAmmoTH-VL-Instruct-12M#1000'

NPROC_PER_NODE=8 \
megatron sft \
    --model ${MODEL_PATH} \
    --save_safetensors false \
    --dataset ${DATASET_PATH} \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --sequence_parallel true \
    --micro_batch_size 4 \
    --global_batch_size 32 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --num_train_epochs 1 \
    --output_dir megatron_output/Qwen3-omni \
    --save_steps 2000 \
    --max_length 4096 \
    --system 'You are a helpful assistant.' \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --gradient_accumulation_fusion false \
    --masked_softmax_fusion false \
    --attention_backend flash \
    --padding_free false \
    --model_author swift \
    --model_name swift-robot



