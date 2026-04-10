# GKD Training with External Teacher Model Server (vLLM) - Multimodal
# ===================== Step 1: Start Teacher Server =====================
# Run in a separate terminal / GPU:
#
#   CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
#       --port 8000 \
#       --max-logprobs 64 \
#       --gpu-memory-utilization 0.9 \
#       --max-model-len 4096 \
#       --limit-mm-per-prompt '{"image": 5}'
#
# ========================================================================

NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --teacher_model_server http://localhost:8000 \
    --gkd_logits_topk 20 \
    --lmbda 0 \
    --seq_kd false \
    --beta 0.5 \
    --torch_dtype bfloat16 \
    --dataset 'modelscope/coco_2014_caption:train#200' \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --max_length 2048 \
    --max_completion_length 512 \
    --warmup_ratio 0.1 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --report_to tensorboard \
    --num_train_epochs 1
