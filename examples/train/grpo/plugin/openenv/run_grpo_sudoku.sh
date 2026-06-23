#!/usr/bin/env bash

# ============================================================
#  Swift GRPO training with OpenEnv TextArena Sudoku
#
#  Prerequisites:
#    1. Start Sudoku server (separate terminal):
#       TEXTARENA_ENV_ID=Sudoku-v0 MAX_CONCURRENT_ENVS=8 \
#         python examples/train/grpo/plugin/openenv/start_sudoku_server.py
#
#    2. This script uses colocate mode:
#       - vLLM and training share the same GPUs
#       - No separate rollout server needed
#
#  Environment:  TextArena Sudoku (local server, port 8000)
#  Model:        Qwen3.5-4B (enable_thinking=false)
#  Scheduler:    SudokuScheduler (multi-turn, content diff tracking)
#  Multi-turn:   max_turns=20 (20 moves per game)
#  Rewards:      5-component (empty_cell/valid_move/repetition/progress/correct)
#  Hints:        Board parsing + guaranteed moves + candidates
#
# ============================================================

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-4B \
    --dataset examples/train/grpo/plugin/openenv/sudoku.jsonl#1000 \
    --external_plugins examples/train/grpo/plugin/openenv/sudoku_scheduler.py \
    --enable_thinking false \
    --torch_dtype bfloat16 \
    --max_completion_length 256 \
    --max_length 8192 \
    --learning_rate 5e-6 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --num_generations 4 \
    --generation_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --temperature 1 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_max_model_len 12288 \
    --vllm_gpu_memory_utilization 0.35 \
    --gradient_checkpointing true \
    --use_gym_env true \
    --multi_turn_scheduler sudoku_scheduler \
    --max_turns 20 \
    --save_strategy steps \
    --save_steps 50 \
    --logging_steps 1 \
    --log_completions true \
    --report_to tensorboard swanlab
