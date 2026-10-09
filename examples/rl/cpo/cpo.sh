# v5 Contrastive Preference Optimization (CPO) with LoRA, on 2 GPUs through Ray.
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI. `--rlhf_type cpo` dispatches to
# swift.dev.recipe.run_dpo (the offline-preference family: dpo/kto/cpo/orpo/simpo/rm share one entry point).
#
# What CPO does: it is DPO's preference margin WITHOUT the reference model, plus an explicit behaviour-
# cloning term. The margin still pushes the chosen completion's logp above the rejected one's, but there is
# no logp_reference subtracted -- CPO assumes the reference is uniform and drops it, which removes the second
# forward pass entirely. On top of the margin it adds an NLL (cross-entropy) loss on the CHOSEN completion,
# weighted by --cpo_alpha (twinkle names it bc_coef), so the policy is also directly supervised toward the
# preferred answer. That BC anchor is what replaces the reference's role of keeping the policy sensible.
#
# Reference-free: because there is no reference, a CPO run loads NO second model and runs NO adapter-disabled
# forward -- lighter than DPO. Do NOT pass --ref_model (the validator rejects it for cpo); do NOT pass
# --reference_free (that is a DPO knob, CPO has no reference to free).
#
# HOW TO READ THIS RUN: CPO is reference-free, so unlike DPO it has NO rewards/margins and NO
# rewards/accuracies -- those need reference logps (the implicit reward is beta*(logp_policy - logp_reference)),
# which CPO never computes. twinkle.metric's DPOMetric emits them only when a reference is present, so the
# record here holds just logps/chosen|rejected (the scalar tags carry NO `train/` prefix):
#   - console head signal:  `step N  loss=...  grad_norm=...` (no margins/accuracies);
#   - tensorboard scalars:  `logps/chosen` (should RISE), `logps/rejected` (should FALL), so the GAP
#                           logps/chosen - logps/rejected WIDENS -- that widening is the preference being
#                           learned -- plus `loss` (falls) and `grad_norm`. No `rewards/*`, no `logps/ref_*`.
#
# Data / NO periodic eval / placement: identical to DPO. --dataset is a chosen/rejected preference set fed
# INTERLEAVED; --per_device_train_batch_size counts preference ROWS (pairs). Offline preference training has
# no eval loop, so there is no --split_dataset_ratio / --eval_steps (an eval split with --eval_steps raises
# NotImplementedError). mode='ray' is forced, no vLLM sampler; --nproc_per_node is required (the driver holds
# no GPU and sizes the data-parallel actor group). `#10000` takes a 10000-row slice of the real hub dataset.
# --padding_free true is wired for cpo and improves throughput on length-varied pairs.
#
# Attention kernel: --attn_impl flash_attention_2 is PINNED because bf16 training on the default SDPA path can
# select a cuDNN attention kernel whose BACKWARD returns NaN on some torch/cuDNN builds (reproduced on torch
# 2.11 / cuDNN 9.19: loss goes NaN a few steps in, during warmup, while the lr is still ~0 -- a kernel NaN, not
# divergence). flash-attn needs the flash-attn package and an Ampere+ GPU; drop the flag only if your stack does
# not hit the SDPA NaN.
#
# DEVICES: pick FREE cards -- this box keeps GPU 0/2 busy, so the default below uses 1,3. The global batch is
# per_device * dp * grad_accum; going to --nproc_per_node 1 halves dp, so double --gradient_accumulation_steps
# to 16 to hold the effective batch.
#
# This is a real training scaffold, not a 2-step demo. For a fast smoke run, shrink everything at once: a
# smaller policy (Qwen/Qwen2.5-0.5B-Instruct), a tiny slice (`#512`), and cap `--max_steps 60`.
CUDA_VISIBLE_DEVICES=1,3 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type cpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset 'hjh0119/shareAI-Llama3-DPO-zh-en-emoji#10000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --beta 0.1 \
    --cpo_alpha 1.0 \
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
