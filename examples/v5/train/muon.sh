# v5 SFT with the Muon optimizer (causal_lm, single GPU).
#
# `--optim muon` resolves to twinkle's MuonClip, which orthogonalises the update for the parameter groups
# it selects (the 2D hidden weights) and runs an AdamW step for the rest; a muon_config is assembled from
# TrainConfig's muon_* fields (all left at their defaults here). Muon trains FULL parameters. On the
# Megatron backend the equivalent is `--optimizer muon` / `dist_muon` (the sharded variant). Note plain
# `muon` reads whole parameters, so the DistributedConfig overlap_grad_reduce/overlap_param_gather knobs
# must stay off (use `dist_muon` with those overlaps).
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --tuner full \
    --optim muon \
    --num_train_epochs 1 \
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
