nnodes=2
nproc_per_node=4

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --master_port 29500 \
    --nproc_per_node=$nproc_per_node \
    --nnodes=$nnodes \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    swift/cli/sft.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type full \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#20000' \
              'AI-ModelScope/alpaca-gpt4-data-en#20000' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 32 / $nproc_per_node / $nnodes) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2
