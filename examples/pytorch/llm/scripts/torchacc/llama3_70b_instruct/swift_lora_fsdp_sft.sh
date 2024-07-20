# Experimental environment: 4 * A100

export USE_TORCH_XLA=0

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MASTER_PORT=29500 \
swift sft \
    --model_id_or_path LLM-Research/Meta-Llama-3-70B-Instruct \
    --model_revision master \
    --sft_type lora \
    --dataset codefuse-python-en \
    --template_type llama3 \
    --dtype AUTO \
    --output_dir output \
    --ddp_backend nccl \
    --train_dataset_sample -1 \
    --tuner_backend 'peft' \
    --num_train_epochs 1 \
    --max_length 2048 \
    --batch_size 4 \
    --use_flash_attn true \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --dataset_test_ratio 0 \
    --save_strategy no \
    --eval_steps 2000000 \
    --save_steps 2000000 \
    --logging_steps 100 \
    --acc_steps 100 \
    --preprocess_num_proc 1 \
    --metric_warmup_step 0.1 \
    --deepspeed default-zero3 \
