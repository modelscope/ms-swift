export CUDA_VISIBLE_DEVICES=0
export USE_OPENCOMPASS_EVALUATOR=True

swift sample \
    --model ./output/Qwen2.5-Math-7B-Instruct/v40-20250126-161112/checkpoint-20 \
    --orm_model math \
    --sampler_type mcts \
    --sampler_engine vllm \
    --output_dir ./output/sampler/mcts \
    --system ./examples/sampler/system_prompt.txt \
    --stop_words ки \
    --dataset ./datasets/competition_math/small_test.jsonl \
    --num_return_sequences 2 \
    --process_reward_rate 0 \
    --max_new_tokens 2048

## Train
# nproc_per_node=8
# NPROC_PER_NODE=$nproc_per_node \
# swift sft \
#     --model Qwen/Qwen2.5-Math-7B-Instruct \
#     --train_type full \
#     --torch_dtype bfloat16 \
#     --dataset 'datasets/gen_V5.jsonl' \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 1 \
#     --learning_rate 1e-5 \
#     --gradient_accumulation_steps $(expr 128 / $nproc_per_node) \
#     --eval_steps 1000 \
#     --save_steps 10 \
#     --save_total_limit 100 \
#     --max_length 10000 \
#     --logging_steps 5 \
#     --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
#     --deepspeed zero3
