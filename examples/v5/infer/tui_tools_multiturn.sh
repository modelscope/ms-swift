# v5 interactive tool-calling agent (multi-turn) on multiple GPUs.
#
# No --dataset => the REPL opens. --tools sandbox exposes the LocalEnv's own default tools
# (run_command / write_file / read_file); the sandbox is a local LocalEnv by default because
# --sandbox_template is unset (naming a template would switch it to an AgentEnv microVM instead).
# Tools require --max_turns: each human turn runs twinkle's MultiTurnRollout intra-turn loop -- the
# model calls tools, reads their results, and continues until it answers or hits max_turns /
# max_trajectory_tokens. The tool loop returns a finished trajectory, so this path prints the answer
# whole (no token streaming) and traces the tools it invoked.
#
# Multi-GPU is vLLM TP=2 (dp=1), which the interactive guard allows; --sandbox_num_envs 1 leases a
# single workspace to the one interactive conversation.
CUDA_VISIBLE_DEVICES=4,5,6,7 \
USE_SWIFT_V5=1 \
swift infer \
    --model Qwen/Qwen3.5-4B \
    --infer_backend vllm \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.9 \
    --tools sandbox \
    --max_turns 8 \
    --max_trajectory_tokens 8192 \
    --sandbox_num_envs 1 \
    --sandbox_command_timeout 60 \
    --max_new_tokens 2048 \
    --temperature 0.7
