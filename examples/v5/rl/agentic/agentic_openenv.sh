# v5 AGENTIC GRPO with a CUSTOM OpenEnv environment: the same run as agentic.sh, but the sandbox is
# the bundled `coding_env` package instead of the built-in LocalEnv. Read agentic.sh first -- the GRPO
# objective, tools, length budgets and placement are identical; only the environment backend changes.
#
# What coding_env is: a minimal OpenEnv-spec environment package living right next to this script
# (coding_env/__init__.py exports the Action class, coding_env/server/terminal_environment.py carries
# the Environment class -- the two shapes OpenEnv discovery looks for). It gives the policy ONE tool,
# run_command, against a persistent per-episode workspace (write files with heredocs, run them with
# `python file.py`). With --tools sandbox the tool plugin binds whatever tools the chosen env exposes,
# so the model here sees only run_command instead of LocalEnv's run_command/write_file/read_file.
#
# Wiring: --sandbox_kind openenv --sandbox_env_name coding_env makes builders/sandbox.py construct
# twinkle_agentic's OpenEnv(env_name='coding_env'), which imports the package and discovers the two
# classes in-process (no OpenEnv server, no extra pip package). The import needs the package on
# PYTHONPATH, so this script prepends its own directory -- run it from anywhere, zero manual setup.
# That covers the default single-node form, where ray.init() starts the local cluster as a child of
# this process and the workers inherit its PYTHONPATH. When ATTACHING to an existing cluster
# (RAY_ADDRESS set / `ray start` ran separately), workers inherit the raylet's environment instead:
# there, make coding_env importable at cluster level (pip install -e the package, or export
# PYTHONPATH before `ray start`) -- the driver-side prepend alone does not reach those workers.
#
# Env tuning goes through --sandbox_env_kwargs (a JSON dict forwarded verbatim to the environment's
# constructor), NOT --sandbox_command_timeout/--sandbox_memory_limit_gb: those two flags only feed the
# LocalEnv/AgentEnv builders and would be dead here. The values below match coding_env's own defaults
# (TerminalEnvironment(command_timeout=60, memory_limit_gb=2.0)) and are stated explicitly so the
# knobs are visible. Scoring is unchanged: the env itself returns reward 0 every step; --orm accuracy
# scores the finished trajectory exactly as in agentic.sh.
#
# DEVICES / smoke: same guidance as agentic.sh -- pick free cards, and for a fast smoke shrink only the
# bounded dimensions (smaller policy, `#64` slice, --max_steps 4, --sandbox_num_envs 1). Keep
# --max_turns, --tools sandbox and the two openenv flags -- they are what make this run agentic-on-
# OpenEnv; the sibling `openenv_client` kind (needs --sandbox_api_url) connects to a standalone
# OpenEnv server instead of embedding the package.
# MEASURED: not run — coding_env module not installed (2026-11-24)
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=1,3 \
swift rl \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-4B \
    --attn_impl flash_attention_2 \
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
    --sandbox_kind openenv \
    --sandbox_env_name coding_env \
    --sandbox_env_kwargs '{"command_timeout": 60, "memory_limit_gb": 2.0}' \
    --sandbox_num_envs 2 \
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
