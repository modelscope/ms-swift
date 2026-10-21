# v5 Direct Preference Optimization (DPO) with LoRA, on 2 GPUs through Ray.
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI. `--rlhf_type dpo` dispatches to
# swift.dev.recipe.preference.run_dpo (the offline-preference family: dpo/kto/cpo/orpo/simpo/rm share one entry point).
#
# What DPO does: it turns a preference dataset into a policy update WITHOUT a reward model and WITHOUT
# rollout. Each row is a prompt plus a CHOSEN and a REJECTED completion. The loss maximises the margin
# between the policy's implicit reward on the chosen vs the rejected completion, where the implicit reward
# is beta * (logp_policy - logp_reference). The frozen reference keeps the policy from drifting arbitrarily
# far to widen the margin; --beta weights that anchor (larger beta = tighter leash to the reference).
#
# Reference under LoRA: DPO needs a frozen reference, but a LoRA run does NOT load a second model -- the
# reference is the policy's OWN base weights with the adapter DISABLED (run_dpo builds it automatically via
# forward_only(disable_lora=True)), so it costs no extra GPU. A FULL-parameter run instead loads a separate
# frozen ref_model (defaulted to the policy's init) on its own 'ref' DeviceGroup -- pass --ref_model only to
# point at a different checkpoint. Do NOT pass --ref_model to a LoRA run: the validator rejects it.
#
# HOW TO READ THIS RUN: judge convergence by the REWARD MARGIN and ACCURACY, not the raw loss. Every step's
# record is assembled from twinkle.metric's DPOMetric, so watch (the scalar tags carry NO `train/` prefix):
#   - console head signal:  `step N  loss=...  rewards/margins=<chosen_r - rejected_r>  rewards/accuracies=<% chosen wins>`;
#   - tensorboard scalars:  `rewards/margins` (should RISE), `rewards/accuracies` (should climb toward 100%),
#                           `rewards/chosen`, `rewards/rejected`, `logps/chosen` (rises), `logps/rejected`
#                           (falls), `logps/ref_chosen`, `logps/ref_rejected`, plus `loss` and `grad_norm`.
# A rising margin with an accuracy approaching 100% is the preference signal being learned; the loss alone
# (a softplus of the margin) hides the trend. The margin/accuracy are present BECAUSE DPO has a reference:
# the implicit reward is beta*(logp_policy - logp_reference), so DPOMetric needs the reference logps to form
# them (the reference-free cpo/orpo/simpo log only logps/chosen|rejected -- see those examples).
#
# Data: --dataset is a chosen/rejected preference set. The template encodes each row into chosen_*/rejected_*
# features and PreferenceLoop feeds them INTERLEAVED ([chosen_1, rejected_1, chosen_2, ...]), the layout the
# DPO loss's even/odd split expects, so every micro-batch stays even and gradient accumulation stays correct.
# --per_device_train_batch_size counts preference ROWS (pairs); each row becomes two sequences internally.
# This example uses the real hub dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji; `#10000` takes a 10000-row slice.
#
# NO periodic eval: offline preference training does not implement an eval loop -- passing an eval split with
# --eval_steps raises NotImplementedError. So there is no --split_dataset_ratio / --eval_steps here (the v4
# legacy scripts carried both; they are dead on the v5 path and removed). Judge progress on the train metrics.
#
# Variants (all off by default, so the run below is canonical sigmoid DPO):
#   --rpo_alpha 0.1        add an NLL/SFT term on the chosen response (RPO) to curb likelihood displacement;
#   --loss_type ipo|hinge  a different DPO objective; --loss_type sigmoid bco_pair sft + --loss_weights
#                          0.8 0.2 1.0 combines several as a weighted sum (MPO);
#   --reference_free true  drop the reference log-ratio (no anchor); --f_divergence_type reshapes the margin.
# --padding_free true is wired for the paired family and improves throughput on length-varied pairs.
#
# Placement: the offline preference family runs mode='ray' like every RL path -- the policy is a Ray actor on
# the 'model' DeviceGroup -- but it rolls out NOTHING, so there is no vLLM sampler and no vllm_mode flag. The
# dev CLI forces mode=ray for --rlhf_type dpo; --nproc_per_node IS required because in Ray mode the driver
# orchestrates and holds no GPU (it sizes the data-parallel actor group).
#
# Attention kernel: --attn_impl flash_attention_2 is PINNED because bf16 training on the default SDPA path can
# select a cuDNN attention kernel whose BACKWARD returns NaN on some torch/cuDNN builds (reproduced on torch
# 2.11 / cuDNN 9.19: loss is a correct ln2=0.6932 at step 1, then goes NaN a few steps in -- during warmup,
# while the lr is still ~0, so it is a kernel NaN, not divergence). flash-attn is the stable, fast path; it
# needs the flash-attn package and an Ampere+ GPU. If your stack does not hit the SDPA NaN you may drop the
# flag (or pass --attn_impl sdpa/eager), but flash_attention_2 is the safe default for bf16 preference training.
#
# DEVICES: pick FREE cards -- this box keeps GPU 0/2 busy, so the default below uses 1,3. Adjust
# CUDA_VISIBLE_DEVICES to the cards you actually have; a LoRA DPO on a 7B fits a single ~24GB card, so
# --nproc_per_node 1 with one device also works (the global batch is per_device * dp * grad_accum, so double
# --gradient_accumulation_steps to 16 to keep the same effective batch when dp drops from 2 to 1).
#
# This is a real training scaffold (a full pass over a 10000-row slice), not a 2-step demo, so it downloads a
# 7B policy and runs a while. For a fast smoke run, shrink everything at once: a smaller policy
# (Qwen/Qwen2.5-0.5B-Instruct), a tiny slice (`#512`), and cap `--max_steps 60`.
CUDA_VISIBLE_DEVICES=1,3 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type dpo \
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
