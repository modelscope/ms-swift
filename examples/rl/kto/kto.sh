# v5 Kahneman-Tversky Optimization (KTO) with LoRA, on 2 GPUs through Ray.
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI. `--rlhf_type kto` dispatches to
# swift.dev.recipe.run_dpo (the offline-preference family: dpo/kto/cpo/orpo/simpo/rm share one entry point).
#
# What KTO does: it is the UNPAIRED cousin of DPO. Where DPO needs a (chosen, rejected) pair per prompt, KTO
# needs only a SINGLE completion per row carrying a BINARY label -- desirable (good) or undesirable (bad). The
# loss is a Kahneman-Tversky value function: it pushes UP the implicit reward of desirable completions and
# pushes DOWN that of undesirable ones, each measured against a reference point z_KL. The implicit reward is
# the same as DPO's, beta * (logp_policy - logp_reference), so KTO DOES need a reference. z_KL is a detached
# KL estimate the loop computes over a MISMATCHED "KL batch" (each prompt paired with an unrelated completion,
# built by rotating the completions by one within the batch) -- it is the "status quo" anchor from prospect
# theory. Being unpaired, KTO trains on the far more abundant thumbs-up/thumbs-down data instead of scarce
# side-by-side preferences.
#
# The `label` column: each dataset row is a conversation whose LAST assistant turn is the completion, plus a
# boolean `label` (true = desirable, false = undesirable). The template runs in mode='kto' and reads exactly
# that; there is no rejected_response. A batch that is entirely one-sided still trains, but exercises only half
# the value function -- a real dataset mixes both.
#
# Reference (and why KTO cannot be reference-free): KTO's implicit reward IS the policy-vs-reference log-ratio
# and z_KL is a KL against that same reference, so both terms need reference logps. --reference_free is
# therefore REJECTED by the validator for kto. Under LoRA the reference is the policy's OWN base weights with
# the adapter DISABLED (built automatically, no extra GPU); a full-parameter run loads a separate frozen
# ref_model. Do NOT pass --ref_model to a LoRA run.
#
# Class-imbalance weights: --desirable_weight and --undesirable_weight (both default 1.0) rescale the two
# branches when the dataset is lopsided (e.g. many more undesirable than desirable rows). Set them so the
# rarer class is not drowned out; the KTO paper suggests weighting toward a roughly balanced desirable:
# undesirable ratio.
#
# HOW TO READ THIS RUN: KTO is unpaired, so there is NO reward margin / accuracy -- those are DPO-family
# paired metrics. swift registers no KTO-specific metric (only dpo/cpo/orpo/simpo get DPOMetric), so the
# record holds just the bare loss + grad_norm and you judge by the LOSS falling and grad_norm staying finite:
#   - console head signal:  `step N  loss=...  grad_norm=...`;
#   - tensorboard scalars:  `loss` and `grad_norm` (no `train/` prefix; no rewards/*, no kl term). The loss
#                           starts near 0.5 and drifts down slowly -- with a single label per row there is no
#                           margin to widen, so a steady finite loss + finite grad_norm IS the healthy signal.
#
# Data / NO periodic eval / placement: --dataset is the unpaired kto set. Offline preference training has no
# eval loop, so there is no --split_dataset_ratio / --eval_steps (an eval split raises NotImplementedError).
# mode='ray' forced, no vLLM sampler; --nproc_per_node required. `#10000` takes a 10000-row slice of the real
# hub dataset AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto. --padding_free true is wired for kto.
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
    --rlhf_type kto \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset 'AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto#10000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --beta 0.1 \
    --desirable_weight 1.0 \
    --undesirable_weight 1.0 \
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
