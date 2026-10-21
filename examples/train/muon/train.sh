# v5 SFT with the Muon optimizer (causal_lm, full-parameter, single GPU).
#
# `--optim muon` resolves to twinkle's MuonClip, which orthogonalises the update for the parameter groups
# it selects (the 2D hidden weights) and runs an AdamW step for the rest; a muon_config is assembled from
# TrainConfig's muon_* fields (all left at their defaults here). Muon trains FULL parameters, and lr 1e-4
# is stable for it (measured: loss stays finite, grad_norm ~10-20). On the Megatron backend the equivalent
# is `--optimizer muon` / `dist_muon` (the sharded variant). Note plain `muon` reads whole parameters, so
# the DistributedConfig overlap_grad_reduce/overlap_param_gather knobs must stay off (use `dist_muon` with
# those overlaps).
#
# QK-Clip IS OFF BY DEFAULT -- and you almost certainly want to keep it off for SFT. MuonClip can also run
# QK-Clip, a from-scratch-pretraining brake (Kimi K2) that rescales the Q/K WEIGHTS by sqrt(tau / peak
# attention logit) on every step whose peak exceeds `--qk_clip_tau`. dev defaults `--qk_clip_enabled false`
# so `--optim muon` is plain Muon. Do NOT flip it on for a pretrained model without raising tau: an
# instruct Qwen2.5 already runs attention logits ~1e3, so the default-scale ceiling trips EVERY step and
# geometrically collapses Q/K (measured: weight norm 60 -> 0.001 in 6 steps, loss 1.8 -> 8, and it does so
# at lr 1e-5 exactly as at 1e-4 because the rescale never multiplies by lr). Opt in only for from-scratch
# runs, e.g. `--qk_clip_enabled true --qk_clip_tau 10000` (tau above the model's natural logit scale).
#
# DATASET CHOICE = VISIBLE CONVERGENCE. `swift/self-cognition` (not alpaca-gpt4): an already-instruct
# Qwen2.5-0.5B plateaus at ~1.6 loss on alpaca (no headroom), so the run looks flat; self-cognition is new
# knowledge it must fit, so the loss falls clearly. Keep --num_train_epochs 3 so the descent completes.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'swift/self-cognition#600' \
    --tuner full \
    --optim muon \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --logging_steps 5 \
    --save_steps 100 \
    --output_dir output \
    --report_to tensorboard
