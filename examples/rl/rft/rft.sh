# v5 Rejection-sampling Fine-Tuning (RFT / RAFT / ReST) with LoRA, on 2 GPUs through Ray + vLLM.
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI. `--rlhf_type rft` dispatches to
# swift.dev.recipe.run_rft.
#
# What RFT does: it reuses GRPO's rollout + reward machinery for a simpler purpose than policy gradient.
# Each round: sample --rft_num_samples completions per prompt from the CURRENT policy (the trained weights
# are synced into the vLLM sampler before each rollout, so the behaviour policy tracks the trained one),
# score them with the --orm reward, KEEP only the good ones, and run plain cross-entropy SFT on the kept
# set. Repeating over --rft_iterations rounds -- each re-sampling from the just-improved policy -- is the
# rejection-sampling bootstrap: no advantage estimator, no reference model, no importance ratio. RFT sets
# the SFT cross_entropy loss, NOT an RL loss.
#
# HOW TO READ THIS RUN: the SFT loss on the kept subset does fall, but the signal that shows the BOOTSTRAP
# working is the REWARD of the freshly generated rollouts climbing round over round -- the policy is sampling
# better answers, so more of them clear the filter. Watch:
#   - console head signal:  `step N  loss=...  reward=<mean total reward of the round's rollouts>`;
#   - tensorboard scalars:  `train/total_reward` (+ per-channel `train/<orm>_reward`), `train/completion_length`.
# The reward is fed from the WHOLE generated rollout BEFORE rejection (the driver-side CompletionRewardMetric
# shared with GRPO), so it tracks sample quality, not just the kept subset the loss trains on.
# NOTE: no --log_completions here -- completion logging is GRPO-only (it lives in GRPOLoop's rollout assembly,
# which RFT's rejection-sampling fit bypasses), so on RFT it is a dead flag the config validator rejects.
#
# Selection (--rft_select):
#   best_of_n (default) -- keep the single highest-reward completion per prompt (always keeps one, so a
#                          round never comes up empty; used here);
#   threshold           -- keep every completion scoring >= --rft_threshold;
#   top_k               -- keep the --rft_top_k best per prompt.
# --rft_max_samples_per_prompt caps how many a single prompt contributes. An empty kept set (possible with
# threshold) skips that round's SFT pass with a warning rather than failing. A round whose kept set is
# smaller than train_batch_size (per_device_train_batch_size * dp_size) also skips with a warning, so keep
# --rft_num_samples comfortably above that width.
#
# Reward: --orm accuracy is MathAccuracy (backed by math_verify); it reads the dataset's `solution` column as
# ground truth and the completion's \boxed{} answer. This example uses the real hub dataset
# AI-MO/NuminaMath-TIR (the same one GRPO trains on); the `#2000` suffix takes a 2000-row slice.
#
# Placement: RFT is Ray-only with a vLLM sampler -- the SAME requirement as GRPO/PPO. The dev CLI forces
# use_vllm=true, vllm_mode (colocate by default) and mode=ray for --rlhf_type rft; --nproc_per_node IS
# required because in Ray mode the driver orchestrates and holds no GPU. vllm_mode=colocate shares the GPUs
# between sampler and training (CUDA-IPC weight sync); use vllm_mode=disaggregated for separate device groups.
#
# Step budget: RFT is bounded by --rft_iterations rounds over the prompt set (each round = one generation
# pass + one SFT pass over the kept completions), NOT by a toy --max_steps. --gradient_accumulation_steps 1
# keeps every mini-batch its own optimizer step; raise it for a larger effective batch (over a full round the
# trailing partial accumulation window is dropped, which is negligible at this scale).
#
# DEVICES: pick FREE cards -- this box keeps GPU 0/2 busy, so the default below uses 1,3. Adjust
# CUDA_VISIBLE_DEVICES and --vllm_gpu_memory_utilization to the cards/memory you actually have.
#
# This is a real training scaffold, not a 2-step demo. For a fast smoke run, shrink everything at once: a
# smaller policy (Qwen/Qwen2.5-1.5B-Instruct), a tiny slice (`#64`), --rft_iterations 1, and cap --max_steps 4.
CUDA_VISIBLE_DEVICES=1,3 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type rft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset 'AI-MO/NuminaMath-TIR#2000' \
    --system "You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}." \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --orm accuracy \
    --nproc_per_node 2 \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --rft_num_samples 8 \
    --rft_select best_of_n \
    --rft_iterations 3 \
    --temperature 1.0 \
    --max_completion_length 2048 \
    --max_length 3072 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --gradient_checkpointing true \
    --output_dir output \
    --report_to tensorboard
