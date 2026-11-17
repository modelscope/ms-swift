# v5 GRPO, ASYNC (overlapping rollout) with LoRA -- a feature sub-example of ../grpo.sh, on 3 GPUs.
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI. `--rlhf_type grpo` dispatches to
# swift.dev.recipe.grpo.run_grpo. Read ../grpo.sh FIRST: this file changes only the ROLLOUT CADENCE, not the
# GRPO objective (group-relative advantage, clipped surrogate, --orm accuracy, --beta KL). Everything GRPO
# means is explained there; here we only turn on async.
#
# What async does: by default GRPO is SYNCHRONOUS (--async_mode none) -- it admits one rollout window,
# DRAINS it (nothing left in flight), trains on it, publishes the new weights to the sampler, and only then
# generates the next window. The sampler sits idle while the trainer runs and vice-versa. Async OVERLAPS
# them: the sampler keeps generating the NEXT window on its own GPUs while the trainer updates on the
# current one. The knob is a STALENESS LADDER -- how many policy versions a rollout may run ahead of the
# weights it is trained against:
#   - none          staleness 0: strictly on-policy, no overlap (the synchronous default, ../grpo.sh).
#   - one_step_off  staleness 1: train version v while the sampler already generates v+1 (below).
#   - fully_async   staleness up to --max_staleness N: the sampler runs several versions ahead, absorbing a
#                   deeper skew (set --async_mode fully_async --max_staleness N; N>=1).
# A sample is never TRAINED more than max_staleness versions after it was GENERATED -- the streaming
# driver drops a ready group that has fallen further behind (counted in dropped_stale).
#
# Why staleness > 0 needs two companions (validate refuses the run without them -- they are not optional):
#   1. --vllm_mode disaggregated: overlap needs the sampler on its OWN GPUs. 'colocate' time-shares ONE
#      DeviceGroup between trainer and sampler (a sleep/wake hand-over each step), so it physically cannot
#      generate while training -- validate rejects colocate + an overlapping async_mode.
#   2. --rollout_importance_sampling_mode: GRPO trains on RAW sampled tokens, so a stale (off-policy) batch
#      needs a token-level importance-sampling correction (token_truncate / sequence_mask) or the ratio is
#      silently wrong. (PPO's clipped surrogate and the distillation family's teacher target already bound
#      the update, so they do NOT require this -- it is a GRPO-specific off-policy fix.)
# A weight sync holds the sampler's generation (the engine freezes running requests in place, or waits for
# them where it cannot), swaps the weights, then releases it: one episode may span versions, and is
# stamped with the version it was admitted under.
#
# Device budget (disaggregated): total GPUs = --nproc_per_node (trainer) + sampler_world_size
# (--vllm_tensor_parallel_size, default 1) on DISJOINT ranks. Below: nproc_per_node 2 (trainer) + 1 (sampler)
# = 3 cards, so CUDA_VISIBLE_DEVICES lists three.
#
# HOW TO READ THIS RUN: judge convergence by the REWARD curve exactly as in ../grpo.sh (`step N loss= reward=`,
# tensorboard train/total_reward climbing). Async ADDS a set of staleness scalars (bare keys, no train/
# prefix):
#   - policy_version: how many weight syncs reached the sampler. This is the signature the overlap is REAL
#     -- it must ADVANCE every window (1,2,3,...).
#   - version_span_mean / max_partial_span: the observed policy-version skew of the batch. The INVARIANT is a
#     CEILING -- <= max_staleness on EVERY step (1 here). They are NOT expected to be non-zero: whether a
#     generation is slow enough to be consumed a version later is timing-dependent, so a fast run
#     legitimately shows 0. Read them as "never above the bound", not "must be positive".
#   - partial_ratio: the fraction of the batch generated under an older version; dropped_stale: samples
#     discarded for falling behind; trainer_idle_ratio: the share of time the trainer waited for a batch.
# So the async proof is policy_version advancing WHILE every span stays <= max_staleness -- not a non-zero
# span. (At staleness 0, ../grpo.sh, every span is EXACTLY 0 by construction.)
#
# DEVICES: pick FREE cards -- this box keeps GPU 0 busy, so the default below uses 1,3,5. Adjust
# CUDA_VISIBLE_DEVICES, --nproc_per_node and --vllm_tensor_parallel_size to the cards you actually have.
#
# This is a real training scaffold (a full pass over a 5000-row slice). For a fast smoke run shrink the
# bounded dimensions only: a smaller policy (Qwen/Qwen2.5-0.5B-Instruct), a tiny slice (`#64`),
# --nproc_per_node 1 (then 2 cards total: 1 trainer + 1 sampler), and --max_steps 4. Keep --async_mode,
# --vllm_mode disaggregated and --rollout_importance_sampling_mode -- they are the legality-defining flags;
# dropping them would silently turn this back into the synchronous ../grpo.sh and prove nothing about async.
CUDA_VISIBLE_DEVICES=1,3,5 \
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
    --vllm_mode disaggregated \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.5 \
    --async_mode one_step_off \
    --rollout_importance_sampling_mode token_truncate \
    --parameter_sync_step 1 \
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
