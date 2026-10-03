# v5 Multi-teacher On-Policy Distillation (MOPD) with LoRA on a single GPU.
# Paper: https://arxiv.org/abs/2606.30406 (Open-MOPD)
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI. `--rlhf_type mopd` dispatches to
# swift.dev.recipe.run_mopd.
#
# What MOPD does: the student generates ONE on-policy trajectory; K frozen DOMAIN teachers each re-score
# that SAME trajectory token-by-token; their per-token distributions are fused into a single target by
# WEIGHTED PROBABILITY MIXTURE (log sum_k w_k p_k), and the student is pulled toward it with the same
# sampled-token k3 surrogate OPSD uses (twinkle.loss.mopd.MOPDLoss). Compared with single-teacher
# OPSD/GKD, MOPD reduces the cross-domain degradation a student suffers when trained on one teacher alone.
# With a single teacher the fused target is exactly that teacher's log-probs, i.e. MOPD degenerates to OPSD.
#
# --teacher_model takes a LIST (one model id per domain expert). --teacher_weights is optional and must
# have exactly one entry per teacher; unset means uniform. Weights are normalized to sum 1 inside MOPDLoss.
# EVERY teacher must share the student's tokenizer, since each re-scores the student's own response token
# ids. This example uses two small Qwen2.5 checkpoints (the Instruct model and its base) to stay light; a
# real run passes K distinct domain experts (e.g. a math model and a code model) of the same tokenizer
# family. Teacher and student see the SAME on-policy prompt -- MOPD has no privileged `teacher_prompt`
# view (that is OPSD's single-teacher trick), so it needs no plugin and reads a plain prompts dataset.
#
# Placement: like OPSD, MOPD generates and scores in-process on the driver (mode='local', the default) and
# needs no vLLM/Ray in this stage; the K teachers are built as resident frozen models. Multiple resident
# teachers cannot be offloaded here, so --offload_teacher_model is rejected for --rlhf_type mopd.
# --lmbda 1.0 samples on-policy every step.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type mopd \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_model Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-0.5B \
    --teacher_weights 0.6 0.4 \
    --dataset examples/v5/rl/data/prompts.jsonl \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --lmbda 1.0 \
    --max_completion_length 256 \
    --max_length 1024 \
    --max_steps 20 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --save_steps 20 \
    --save_total_limit 2 \
    --gradient_checkpointing true \
    --output_dir output \
    --report_to tensorboard
