# v5 AGENTIC GRPO: multi-turn rollout with TOOL CALLING, LoRA, on 2 GPUs through Ray + vLLM.
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI. `--rlhf_type grpo` dispatches to
# swift.dev.recipe.grpo.run_grpo. This builds on ../grpo/grpo.sh (read it first for the GRPO objective); what
# makes it AGENTIC is `--max_turns` (a multi-turn rollout) + `--tools` (the model can call tools mid-episode).
#
# What an agentic rollout does: instead of one shot generation, each episode runs turn by turn up to
# --max_turns. The policy generates; if it emits a tool call, the tool runs inside a sandbox env and its
# observation is fed back as a `tool` message; the policy then continues from there. The WHOLE multi-turn
# trajectory's response tokens are trained, and GRPO scores the final trajectory with --orm (group-relative
# advantage over --num_generations sibling episodes of the same prompt, exactly as in grpo.sh). This is how a
# policy learns to INTERLEAVE reasoning with tool use (run code, read/write files) rather than answer blindly.
#
# Tools (`--tools`): names registered 'tool' plugins, resolved through the PluginRegistry.
#   - `sandbox` (built-in, used below) exposes the sandbox env's OWN default tools -- run_command / write_file
#     / read_file -- bound to the exact workspace this episode leased. No plugin code to write.
#   - custom tools: write a ToolPlugin subclass in a .py, register it with @PluginRegistry.register('tool',
#     'myname'), load it with `--external_plugins mytools.py`, then select it with `--tools myname`. Each tool
#     implements __call__(name, arguments)->str (the observation fed back) and tool_info()->the OpenAI-shaped
#     schema. AVOID eval/exec in a custom tool -- the sandbox already caps memory/time; keep tools narrow.
#   - `--tools` REQUIRES `--max_turns`: tools are injected per turn, so a single-turn rollout has no turn to
#     call them in and validate refuses the combination.
#
# Sandbox env: with no `--sandbox_template` the pool is local `LocalEnv` slots (light, in-process, a fresh
# workspace per slot) -- what runs below. Naming `--sandbox_template <id>` switches to an `AgentEnv` microVM
# (E2B; boots on first use, needs the control-plane URL). `--sandbox_num_envs` is the pool size: one env is
# leased per CONCURRENT episode, so it also caps how many trajectories roll out at once. `--sandbox_command_
# timeout` / `--sandbox_memory_limit_gb` bound each tool call (a model-emitted command cannot run away).
#
# Length budgets: per-turn generation length is --max_completion_length (the sampling max_tokens); the WHOLE
# trajectory is capped by --max_trajectory_tokens (optional). Multi-turn trajectories are longer than a single
# completion, so --max_length must fit the accumulated turns.
#
# Placement: same as grpo.sh (Ray-only, weight-syncable vLLM sampler, --nproc_per_node required). Each episode
# is admitted per-sample on the streaming driver and runs turn-by-turn on its own thread, so multi-turn overlaps
# naturally; concurrency is bounded by the sandbox env pool's leases.
#
# HOW TO READ THIS RUN: judge convergence by the REWARD curve exactly as in grpo.sh (`step N loss= reward=`,
# tensorboard train/total_reward climbing). `train/completion_length` now reflects the MULTI-TURN trajectory
# length (tool observations included). As the policy learns to use tools well, reward rises and the trajectory
# length typically stabilises. There are no agentic-specific tensorboard tags.
#
# DEVICES: pick FREE cards -- this box keeps GPU 0 busy, so the default below uses 1,3 (colocate). Adjust
# CUDA_VISIBLE_DEVICES and --vllm_gpu_memory_utilization to the cards/memory you actually have.
#
# This is a real training scaffold (a full pass over a 5000-row slice). For a fast smoke run shrink the bounded
# dimensions only: a smaller policy (Qwen/Qwen2.5-0.5B-Instruct), a tiny slice (`#64`), --max_steps 4, a short
# --max_completion_length, and --sandbox_num_envs 1. Keep --max_turns and --tools sandbox -- they are what make
# the rollout agentic; dropping them collapses it back to the single-turn ../grpo/grpo.sh.
CUDA_VISIBLE_DEVICES=1,3 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --system "You are a helpful math assistant. You may run Python in a sandbox to verify computations. Put your final answer within \boxed{}." \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --orm accuracy \
    --max_turns 3 \
    --tools sandbox \
    --sandbox_num_envs 2 \
    --sandbox_command_timeout 60 \
    --sandbox_memory_limit_gb 2.0 \
    --max_trajectory_tokens 3072 \
    --nproc_per_node 2 \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --num_generations 8 \
    --advantage_estimator grpo \
    --epsilon 0.2 \
    --num_iterations 1 \
    --beta 0.04 \
    --temperature 1.0 \
    --max_completion_length 1024 \
    --max_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --log_completions true \
    --gradient_checkpointing true \
    --output_dir output \
    --report_to tensorboard
