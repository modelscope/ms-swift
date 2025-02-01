export CUDA_VISIBLE_DEVICES=2
export USE_OPENCOMPASS_EVALUATOR=True

swift sample \
    --model ./output/Qwen2.5-Math-7B-Instruct/v40-20250126-161112/checkpoint-20 \
    --orm_model math \
    --sampler_type mcts \
    --sampler_engine vllm \
    --output_dir ./output/sampler/vllm_mcts \
    --system ./examples/sampler/system_prompt.txt \
    --stop_words ки \
    --dataset ./datasets/competition_math/small_test.jsonl \
    --num_return_sequences 2 \
    --process_reward_rate 0 \
    --max_new_tokens 2048
