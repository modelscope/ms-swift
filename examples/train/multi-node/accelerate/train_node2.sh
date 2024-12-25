CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --config_file ./examples/train/multi-node/accelerate/multi_node.yaml --machine_rank 1 \
    swift/cli/sft.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset 'swift/self-cognition#1000' \
    --num_train_epochs 1 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --model_author swift \
    --model_name swift-robot
