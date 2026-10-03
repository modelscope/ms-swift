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
# Selection (--rft_select):
#   best_of_n (default) -- keep the single highest-reward completion per prompt (always keeps one, so a
#                          round never comes up empty; used here);
#   threshold           -- keep every completion scoring >= --rft_threshold;
#   top_k               -- keep the --rft_top_k best per prompt.
# --rft_max_samples_per_prompt caps how many a single prompt contributes. An empty kept set (possible with
# threshold) skips that round's SFT pass with a warning rather than failing.
#
# Reward: --orm accuracy is MathAccuracy (backed by math_verify); it reads the dataset's `solution` column
# as ground truth and the completion's \boxed{} answer, so examples/v5/rl/data/rft_math.jsonl ships both.
# Add more channels (e.g. `--orm accuracy format`) and weight them with `--orm_weights`.
#
# Placement: RFT is Ray-only with a vLLM sampler -- the SAME requirement as GRPO/PPO. The dev CLI forces
# use_vllm=true, vllm_mode (colocate by default) and mode=ray for --rlhf_type rft, so those need not be
# passed; --nproc_per_node IS required because in Ray mode the driver orchestrates and holds no GPU.
# vllm_mode=colocate shares the 2 GPUs between the sampler and training (CUDA-IPC weight sync); use
# vllm_mode=server for disaggregated device groups.
#
# Step budget (design-B mini-batching -- read this before changing the numbers below):
#   Each round rolls out (prompts * --rft_num_samples) = 4 * 4 = 16 completions; best_of_n keeps exactly
#   one per prompt, so 4 selected completions per round (deterministic for best_of_n). A training
#   mini-batch is per_device_train_batch_size * dp_size = 1 * 2 = 2 rows, so each round yields 4 / 2 = 2
#   mini-batches, and --rft_iterations 3 rounds yield 6 mini-batches in total.
#   --gradient_accumulation_steps 1 makes every mini-batch its own optimizer step, so the run takes exactly
#   6 optimizer steps -- hence --max_steps 6 (= --save_steps 6), which binds precisely on the last
#   mini-batch. ga=1 is chosen deliberately: the optimizer-step boundary follows twinkle's "one micro-step
#   late" rule ((micro_step-1) % ga == 0 and micro_step > 1, i.e. syncs at micro_step = ga+1, 2ga+1, ...),
#   so with ga>1 the trailing partial accumulation window is never flushed. E.g. ga=2 over these same 6
#   mini-batches would step only at micro_step 3 and 5 (2 optimizer steps) and silently drop micro-batches
#   5-6, making --max_steps a dead knob. If you raise ga, size the total mini-batch count as a multiple of
#   ga AND expect the last window to drop, or just keep ga=1 for a short illustrative run.
CUDA_VISIBLE_DEVICES=0,1 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type rft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset examples/v5/rl/data/rft_math.jsonl \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --orm accuracy \
    --nproc_per_node 2 \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --rft_num_samples 4 \
    --rft_select best_of_n \
    --rft_iterations 3 \
    --max_completion_length 256 \
    --max_length 1024 \
    --max_steps 6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --save_steps 6 \
    --save_total_limit 2 \
    --output_dir output \
    --report_to tensorboard
