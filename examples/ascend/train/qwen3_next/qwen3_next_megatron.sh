export TASK_QUEUE_ENABLE=2
NPROC_PER_NODE=8 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --export_model_parallel_size 4 \
    --model_type qwen3_next \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 4 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --num_train_epochs 1 \
    --output_dir megatron_output/Qwen3-Next-Instruct \
    --save_interval 100 \
    --max_length 1024 \
    --system 'You are a helpful assistant.' \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --gradient_accumulation_fusion false \
    --masked_softmax_fusion false \
    --model_author swift \
    --model_name swift-robot
