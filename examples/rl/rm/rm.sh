# v5 Reward Model (RM) training with LoRA, on 2 GPUs through Ray.
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI. `--rlhf_type rm` dispatches to
# swift.dev.recipe.preference.run_dpo (the offline-preference family: dpo/kto/cpo/orpo/simpo/rm share one entry point).
#
# What RM training does: instead of updating a policy, it trains a SCALAR REWARD MODEL -- the reward function
# that online RL (GRPO/PPO/RFT) later scores rollouts with. The model grows a single-logit sequence-
# classification head on top of the LM (task_type='seq_cls', num_labels=1, derived automatically from
# --rlhf_type rm -- you do NOT pass --task_type/--num_labels yourself). Each prompt+completion is pooled to one
# scalar score. The pairwise RewardLoss then asks for the CHOSEN completion to score higher than the REJECTED
# one:  loss = -log sigmoid(r_chosen - r_rejected). It reads no reference and no policy logps -- it is a
# regression on the preference labels, which is why it rides the same chosen/rejected data pipeline as DPO but
# with a seq_cls head instead of a causal-LM loss.
#
# Reference-free, and no ref_model: RM has no reference term. Do NOT pass --ref_model (rejected for rm). There
# is no --beta either (RM has no KL-to-reference weight); --center_rewards_coefficient is the only RM-specific
# knob -- an optional L2-style penalty that keeps the scalar rewards centred near zero (off by default; set it,
# e.g. 0.01, if the reward scale drifts and destabilises the downstream RL).
#
# HOW TO READ THIS RUN: RM is a seq_cls regression, so there is NO DPO reward-margin / accuracy in the head --
# those come from the paired-policy DPOMetric, which does not apply to a reward head. Judge by the LOSS falling
# and grad_norm staying finite:
#   - console head signal:  `step N  loss=...  grad_norm=...`;
#   - tensorboard scalars:  `loss` and `grad_norm` (bare tags, no `train/` prefix; no rewards/*, no logps/*).
#     A falling RewardLoss means the head is separating chosen from
#     rejected; the trained checkpoint is then handed to online RL as a frozen scorer via `--orm <this_ckpt>`.
#
# Data / NO periodic eval / placement: --dataset is a chosen/rejected preference set fed INTERLEAVED (the head
# scores both, the loss compares them); --per_device_train_batch_size counts preference ROWS (pairs). No eval
# loop, so no --split_dataset_ratio / --eval_steps. mode='ray' forced, no vLLM sampler; --nproc_per_node
# required. `#10000` takes a 10000-row slice of the real hub dataset.
# NOTE: --padding_free / --packing are REJECTED for rm (the seq_cls head pools one score per sequence, and a
# flattened packed row would not split back into one score per sub-sequence) -- do not enable them here.
#
# Attention kernel: --attn_impl flash_attention_2 is PINNED because bf16 training on the default SDPA path can
# select a cuDNN attention kernel whose BACKWARD returns NaN on some torch/cuDNN builds (reproduced on torch
# 2.11 / cuDNN 9.19: loss goes NaN a few steps in, during warmup, while the lr is still ~0 -- a kernel NaN, not
# divergence). flash-attn needs the flash-attn package and an Ampere+ GPU; drop the flag only if your stack does
# not hit the SDPA NaN.
#
# DEVICES: pick FREE cards -- this box keeps GPU 0/2 busy, so the default below uses 1,3. Global batch is
# per_device * dp * grad_accum; --nproc_per_node 1 halves dp, so double --gradient_accumulation_steps to 16.
#
# This is a real training scaffold, not a 2-step demo. For a fast smoke run, shrink everything at once: a
# smaller policy (Qwen/Qwen2.5-0.5B-Instruct), a tiny slice (`#512`), and cap `--max_steps 60`.
CUDA_VISIBLE_DEVICES=1,3 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type rm \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset 'hjh0119/shareAI-Llama3-DPO-zh-en-emoji#10000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --nproc_per_node 2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 2 \
    --gradient_checkpointing true \
    --output_dir output \
    --report_to tensorboard
