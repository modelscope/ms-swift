# v5 Simple Preference Optimization (SimPO) with LoRA, on 2 GPUs through Ray.
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI. `--rlhf_type simpo` dispatches to
# swift.dev.recipe.run_dpo (the offline-preference family: dpo/kto/cpo/orpo/simpo/rm share one entry point).
#
# What SimPO does: a reference-free preference loss whose implicit reward is the LENGTH-NORMALIZED log-prob of
# a completion -- the average per-token logp, beta * (1/|y|) * sum_t logp(y_t) -- rather than DPO's summed
# log-ratio against a reference. Normalizing by length removes SimPO's two big pitfalls: no reference model to
# load or forward, and no reward hacking toward long completions (a long chosen answer cannot win just by
# accumulating logp). The loss is a logistic on the margin between the chosen and rejected normalized rewards,
# shifted by a TARGET MARGIN --simpo_gamma:  loss = -log sigmoid( beta*(r_chosen - r_rejected) - gamma ).
# Requiring the chosen reward to beat the rejected one by at least gamma is what makes SimPO align better with
# generation (which is also length-normalized) than plain DPO.
#
# beta and gamma: --beta scales the normalized reward (SimPO's default is 2.0, notably larger than DPO's 0.1
# because the reward is a mean, not a sum); --simpo_gamma is the target margin (default 1.0). Tune the two
# together -- a larger gamma demands a wider separation and can destabilise training if beta is small.
#
# Reference-free: no second model, no adapter-disabled forward. Do NOT pass --ref_model (rejected for simpo) or
# --reference_free (a DPO knob; SimPO has no reference to free). The paper trains SimPO full-parameter; LoRA is
# used here to keep the example light and single-node smoke-able -- switch to `--tuner full` with a lower
# learning_rate (e.g. 1e-5) for the full-parameter recipe.
#
# HOW TO READ THIS RUN: SimPO is reference-free, so like CPO/ORPO it has NO rewards/margins and NO
# rewards/accuracies -- those need reference logps, which SimPO never computes (DPOMetric emits them only when
# a reference exists). The record holds just logps/chosen|rejected (scalar tags carry NO `train/` prefix):
#   - console head signal:  `step N  loss=...  grad_norm=...` (no margins/accuracies);
#   - tensorboard scalars:  `logps/chosen` and `logps/rejected` -- note these are SEQUENCE-SUMMED logps (the
#                           metric sums per-token logp), NOT the length-normalized reward the loss uses, so
#                           their absolute scale differs from the reward. The preference signal is the GAP
#                           logps/chosen - logps/rejected WIDENING (in practice SimPO drives it mostly by
#                           pushing logps/rejected DOWN), plus `loss` (falls) and `grad_norm`.
#                           No `rewards/*`, no `logps/ref_*`.
#
# Data / NO periodic eval / placement: identical to DPO. --dataset is a chosen/rejected preference set fed
# INTERLEAVED; --per_device_train_batch_size counts preference ROWS (pairs). No eval loop, so no
# --split_dataset_ratio / --eval_steps. mode='ray' forced, no vLLM sampler; --nproc_per_node required.
# `#10000` takes a 10000-row slice. --padding_free true is wired for simpo.
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
    --rlhf_type simpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset 'hjh0119/shareAI-Llama3-DPO-zh-en-emoji#10000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --beta 2.0 \
    --simpo_gamma 1.0 \
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
