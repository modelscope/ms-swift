# v5 multi-turn tool-calling rollout over a DATASET with an EXTERNAL tool plugin, no reward scoring,
# GRPO-format dump.
#
# Tools are a plugin kind, selected by registered name via --tools. Instead of the built-in `sandbox`
# (run_command/write_file/read_file), this loads examples/v5/infer/custom_tools.py through --external_plugins
# and selects the `calculator` tool registered there -- the bring-your-own-tool path. Tool calling is
# inherently multi-turn (the model emits a call, reads the observation back, then answers), so --max_turns
# caps the rounds, and a math dataset like NuminaMath pairs naturally with a calculator. NO reward channel
# here (no --reward_funcs / --prm_model / --orm_model): scores are null and reward is left to training.
#
# --output_format grpo writes one row per prompt holding the whole sampled group as an offline GRPO corpus:
# every candidate's full multi-turn trajectory (--num_samples of them, messages intact) plus its rollout
# tokens in an NPZ sidecar beside the jsonl -- input_ids / labels / completion_mask (the trainable mask) /
# rollout_logprobs (the old-policy logps) / loss_mask -- with the relative path embedded in the row. The
# group-relative advantage is NOT computed here: it is cheap and belongs to the training step, which
# recomputes it from the stored rewards + logps. With no reward channel the row's scores are null (reward
# is left to training); add --reward_funcs / --prm_model / --orm_model to store per-candidate rewards too.
# grpo requires --num_samples >= 2 and --save_rollout_tokens true (the logps only exist when that forces
# logprob computation). Multimodal inputs are not stored as tensors -- the row keeps the dataset's image
# columns + messages, so training re-encodes the vision inputs (deterministic for identical images).
#
# No reward model => no dedicated DeviceGroup, but a multi-DP sampler still needs Ray to drive the
# replicas: --mode ray --nproc_per_node 4 gives sampler dp=4 across GPUs [0,1,2,3] (tp=pp=cp=1).
# --sandbox_num_envs sizes the LocalEnv pool to the concurrency (one env leased per episode); the calculator
# ignores the leased env, but the pool is still built whenever any tool is named.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_SWIFT_V5=1 \
swift infer \
    --model Qwen/Qwen3.5-4B \
    --infer_backend vllm \
    --dataset 'AI-MO/NuminaMath-TIR#500' \
    --external_plugins examples/v5/infer/custom_tools.py \
    --tools calculator \
    --max_turns 8 \
    --max_trajectory_tokens 8192 \
    --sandbox_num_envs 4 \
    --num_samples 8 \
    --output_format grpo \
    --save_rollout_tokens true \
    --mode ray \
    --nproc_per_node 4 \
    --vllm_gpu_memory_utilization 0.85 \
    --max_new_tokens 2048 \
    --temperature 1.0 \
    --result_path ./output/tools_rollout_grpo.jsonl
