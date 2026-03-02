# GKD on GSM8K: Teacher Server Mode with Top-K Logits
#
# This script validates GKD effectiveness on mathematical reasoning using GSM8K.
# Student: Qwen2.5-1.5B-Instruct, Teacher: Qwen2.5-7B-Instruct (via vllm serve)
#
# Expected outcome: GSM8K accuracy should improve after GKD training, as the student
# learns the teacher's reasoning distribution on math problems.
#
# ===================== Step 1: Start Teacher Server =====================
# Run in a separate terminal / GPU:
#
#   CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
#       --port 8000 \
#       --max-logprobs 64 \
#       --gpu-memory-utilization 0.9
#
# Wait until the server is ready, then verify:
#   curl http://localhost:8000/v1/models
# ========================================================================
#
# ===================== Step 2: Prepare GSM8K Dataset =====================
# The dataset uses the standard GSM8K train split from Hugging Face:
#   openai/gsm8k (7473 training samples)
# Swift will auto-download it via the HuggingFace dataset name.
# ========================================================================
#
# ===================== Step 3: Evaluation =====================
# After training, evaluate on GSM8K test set:
#
#   CUDA_VISIBLE_DEVICES=0 swift eval \
#       --model <output_dir>/checkpoint-xxx \
#       --eval_backend OpenCompass \
#       --infer_backend vllm \
#       --eval_dataset gsm8k
#
# Compare with the base model to verify improvement:
#   CUDA_VISIBLE_DEVICES=0 swift eval \
#       --model Qwen/Qwen2.5-1.5B-Instruct \
#       --eval_backend OpenCompass \
#       --infer_backend vllm \
#       --eval_dataset gsm8k
# ========================================================================

TEACHER_SERVER_URL=${TEACHER_SERVER_URL:-"http://localhost:8000"}
GKD_LOGITS_TOPK=${GKD_LOGITS_TOPK:-64}

CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_model_server $TEACHER_SERVER_URL \
    --gkd_logits_topk $GKD_LOGITS_TOPK \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --dataset 'openai/gsm8k#train' \
    --seq_kd false \
    --lmbda 0 \
    --beta 0.5 \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 1024 \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --attn_impl flash_attn
