# v5 Odds Ratio Preference Optimization (ORPO) with LoRA, on 2 GPUs through Ray.
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI. `--rlhf_type orpo` dispatches to
# swift.dev.recipe.run_dpo (the offline-preference family: dpo/kto/cpo/orpo/simpo/rm share one entry point).
#
# What ORPO does: it fuses supervised fine-tuning and preference alignment into a SINGLE stage with NO
# reference model. The loss is the plain NLL (cross-entropy) on the CHOSEN completion -- the SFT term -- plus
# an ODDS-RATIO penalty that pushes the odds of the chosen completion above the odds of the rejected one. The
# odds ratio (not the log-prob ratio DPO uses) is what makes the reference unnecessary: the SFT term already
# anchors the policy to the preferred answer, and the odds-ratio term separates chosen from rejected. The two
# are combined as  loss = NLL(chosen) + lambda_orpo * odds_ratio_penalty.
#
# The beta slot carries lambda_orpo: ORPO has no reference and no KL, so it has no beta of its own. Its single
# weight is lambda_orpo, which legacy/TRL carry in the beta slot -- so `--beta 0.1` below IS lambda_orpo (the
# odds-ratio penalty weight), mapped onto twinkle's lambda_orpo by the loss configurator. The default is 0.1;
# a larger value leans harder on preference separation relative to the SFT term.
#
# Reference-free: no second model, no adapter-disabled forward. Do NOT pass --ref_model (rejected for orpo) or
# --reference_free (a DPO knob; ORPO has no reference to free).
#
# HOW TO READ THIS RUN: ORPO is reference-free, so like CPO it has NO rewards/margins and NO rewards/accuracies
# (those need reference logps, which ORPO never computes -- DPOMetric emits them only when a reference exists).
# The record holds just logps/chosen|rejected (scalar tags carry NO `train/` prefix):
#   - console head signal:  `step N  loss=...  grad_norm=...` (no margins/accuracies);
#   - tensorboard scalars:  `logps/chosen` (should RISE), `logps/rejected` (should FALL or rise less), so the
#                           GAP logps/chosen - logps/rejected WIDENS -- the preference being learned -- plus
#                           `loss` (falls) and `grad_norm`. No `rewards/*`, no `logps/ref_*`.
# The loss here includes the SFT NLL term on chosen, so it starts higher than pure-DPO and its fall mixes the
# behaviour-cloning fit with the odds-ratio separation; read the logps GAP for the preference signal itself.
#
# Data / NO periodic eval / placement: identical to DPO. --dataset is a chosen/rejected preference set fed
# INTERLEAVED; --per_device_train_batch_size counts preference ROWS (pairs). No eval loop, so no
# --split_dataset_ratio / --eval_steps. mode='ray' forced, no vLLM sampler; --nproc_per_node required.
# `#10000` takes a 10000-row slice. --padding_free true is wired for orpo.
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
    --rlhf_type orpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset 'hjh0119/shareAI-Llama3-DPO-zh-en-emoji#10000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --beta 0.1 \
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
