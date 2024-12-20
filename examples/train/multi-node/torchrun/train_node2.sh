CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --master_port 29500 --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=xxx.xxx.xxx.xxx \
    swift/cli/sft.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset swift/self-cognition#1000 \
    --num_train_epochs 1 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --model_author swift \
    --model_name swift-robot
