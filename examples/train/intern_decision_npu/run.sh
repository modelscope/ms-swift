#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export MASTER_PORT=29631
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/workspace:${PYTHONPATH:-}
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
stage=${1:-smoke}
micro_batch=1
accumulation=4
eval_batch=1
fsdp_config=/workspace/fsdp2.json
export DECISION_PAD_MULTIPLE=0
case "$stage" in
  smoke) output=/workspace/outputs/swift-smoke; extra=(--max_steps 2 --save_steps 1 --eval_strategy no) ;;
  resume) output=/workspace/outputs/swift-resume; extra=(--max_steps 3 --save_steps 1 --eval_strategy no --resume_from_checkpoint /workspace/outputs/swift-smoke/checkpoint-2) ;;
  train) output=/workspace/outputs/swift-full; micro_batch=4; accumulation=1; eval_batch=4; fsdp_config=/workspace/fsdp2-full.json; export DECISION_PAD_MULTIPLE=128; extra=(--num_train_epochs 2 --save_steps 150 --eval_strategy steps --eval_steps 150 --load_best_model_at_end true --metric_for_best_model loss --greater_is_better false) ;;
  *) exit 2 ;;
esac
swift sft \
  --model /models/Qwen3.5-4B --model_type qwen3_5 \
  --external_plugins /workspace/decision_plugin.py /workspace/checkpoint_fence.py \
  --template intern_decision_training --new_special_tokens '<decision>' \
  --tuner_type full --freeze_vit true --freeze_aligner true --freeze_llm false \
  --dataset /workspace/decision_data/train.jsonl \
  --val_dataset /workspace/decision_data/validation.jsonl \
  --remove_unused_columns false --strict true --split_dataset_ratio 0 \
  --enable_thinking false --max_length 8192 --truncation_strategy delete \
  --packing false --padding_free false --attn_impl sdpa \
  --torch_dtype bfloat16 --bf16 true --fsdp "$fsdp_config" \
  --gradient_checkpointing false --use_logits_to_keep true \
  --per_device_train_batch_size "$micro_batch" --gradient_accumulation_steps "$accumulation" \
  --per_device_eval_batch_size "$eval_batch" --learning_rate 2e-6 \
  --weight_decay 0.01 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --save_only_model false --save_total_limit 2 \
  --logging_steps 1 --report_to none --seed 42 --data_seed 42 \
  --dataset_num_proc 1 --dataloader_num_workers 0 \
  --output_dir "$output" --add_version false "${extra[@]}"
