nproc_per_node=4

NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift pt \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type full \
    --dataset swift/chinese-c4 \
    --torch_dtype bfloat16 \
    --streaming true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 256 / $nproc_per_node) \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --deepspeed zero3 \
    --max_length 8192 \
    --max_steps 100000
