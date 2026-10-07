# v5 Group-Relative Policy Optimization (GRPO) with LoRA, on 2 GPUs through Ray + vLLM.
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI. `--rlhf_type grpo` dispatches to
# swift.dev.recipe.run_grpo.
#
# What GRPO does: it is a policy-gradient method with NO critic/value model. Each step: sample
# --num_generations completions per prompt from the CURRENT policy (the trained weights are synced into the
# vLLM sampler before every rollout, so the behaviour policy tracks the trained one), score each completion
# with the --orm reward, then turn a prompt's group of rewards into a GROUP-RELATIVE advantage
# (r - mean(group)) / std(group) -- the group mean is the baseline a critic would otherwise learn. The
# policy is then updated with a PPO-style CLIPPED surrogate (--epsilon) on that advantage, plus an optional
# KL-to-reference penalty (--beta). Averaging over the group is what makes it critic-free: a completion that
# scores above its siblings is pushed up, one below is pushed down.
#
# HOW TO READ THIS RUN (the point of the example): judge convergence by the REWARD curve, NOT by the loss.
# A policy-gradient loss oscillates around zero by construction and hides the trend. Every step's record is
# assembled from twinkle.metric components, so watch:
#   - console head signal:  `step N  loss=...  reward=<mean total reward>`  (the reward should climb);
#   - tensorboard scalars:  `train/total_reward` (+ per-channel `train/<orm>_reward`), `train/completion_length`,
#                           and the GRPOMetric policy stats `train/clip_ratio`, `train/approx_kl`,
#                           `train/policy_confidence`. Add `--log_entropy true` (logging-only: no entropy bonus,
#                           so training is unchanged) to also surface `train/entropy` -- the entropy-collapse
#                           diagnostic; without it the forward never computes per-token entropy, so that one
#                           scalar is absent by design.
# The reward comes from the driver-side CompletionRewardMetric; the policy stats from the model-registered
# GRPOMetric -- a single source each, no hand-folded duplicates.
#
# num_generations is the GROUP SIZE and must be >= 2 (a lone completion has no siblings to be relative to).
# The advantage is non-degenerate only when a group's rewards VARY: if all --num_generations completions of a
# prompt get the same reward, std(group)=0 and that group contributes zero gradient. The rollout therefore
# samples STOCHASTICALLY (--temperature below), so siblings differ and the group-relative signal is real.
#
# Reward: --orm accuracy is MathAccuracy (backed by math_verify); it reads the dataset's `solution` column as
# ground truth and the completion's \boxed{} answer. This example uses the real hub dataset
# AI-MO/NuminaMath-TIR (the same one examples/train/grpo/internal/real.sh trains on); the `#5000` suffix takes
# a 5000-row slice. Add more channels (e.g. `--orm accuracy format`) and weight them with `--orm_weights`; a
# frozen reward MODEL (instead of a rule) is added with `--orm <model_id>` and lands on its own GPUs.
#
# KL reference (--beta): GRPO's default beta is 0.04, so a reference term is ON by default. Under LoRA the
# reference is the policy's OWN base model with the adapter DISABLED -- it reuses the policy actor, so it
# needs NO second model and NO extra GPUs. Pass `--beta 0` to drop the reference entirely (a full-parameter
# run instead loads a separate frozen reference on its own DeviceGroup).
#
# Placement: GRPO is Ray-only with a weight-syncable vLLM (or SGLang) sampler. The dev CLI forces
# use_vllm=true, vllm_mode (colocate by default) and mode=ray for --rlhf_type grpo, so those need not be
# passed; --nproc_per_node IS required because in Ray mode the driver orchestrates and holds no GPU.
#   - vllm_mode=colocate (below): the sampler SHARES the trainer GPUs (CUDA-IPC weight sync, with a
#     sleep/wake device hand-over each step). --vllm_gpu_memory_utilization must leave room for training.
#   - vllm_mode=disaggregated: the sampler gets its OWN GPUs (NCCL weight sync); required for --async_mode
#     (one_step_off / fully_async), which overlaps rollout with training.
#
# DEVICES: pick FREE cards -- this box keeps GPU 0/2 busy, so the default below uses 1,3. Adjust
# CUDA_VISIBLE_DEVICES and --vllm_gpu_memory_utilization to the cards/memory you actually have.
#
# This is a real training scaffold (a full pass over a 5000-row slice, --num_train_epochs 1), not a 2-step
# demo, so it downloads a 7B policy and runs a while. For a fast smoke run, shrink everything at once:
# a smaller policy (Qwen/Qwen2.5-1.5B-Instruct), a tiny slice (`#64`), and cap `--max_steps 4`.
CUDA_VISIBLE_DEVICES=1,3 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
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
    --num_generations 8 \
    --advantage_estimator grpo \
    --epsilon 0.2 \
    --num_iterations 1 \
    --beta 0.04 \
    --temperature 1.0 \
    --max_completion_length 2048 \
    --max_length 3072 \
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
