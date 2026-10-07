#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin=${PYTHON_BIN:-/workspace/.venv/bin/python}
export PATH="$(dirname "$python_bin"):$PATH"
export PYTHONPATH="$PWD:/workspace/framework:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29651 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DECISION_PAD_MULTIPLE=128
stage=${1:-smoke}
output=${OUTPUT_ROOT:-/workspace/outputs}
case "$stage" in
  smoke) output="$output/smoke"; extra=(--max_steps 2 --save_steps 2 --save_only_model false) ;;
  resume) checkpoint="$output/smoke/checkpoint-2"; output="$output/resume"; extra=(--max_steps 3 --save_steps 3 --save_only_model false --resume_from_checkpoint "$checkpoint") ;;
  train) output="$output/train"; extra=(--num_train_epochs 2 --save_steps 120 --save_only_model true) ;;
  *) exit 2 ;;
esac
test ! -e "$output"
"$python_bin" -m torch.distributed.run --nproc-per-node 4 \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" -m swift.cli.sft \
  --model /models/Qwen3.5-4B --model_type qwen3_5 \
  --external_plugins "$PWD/decision_plugin.py" "$PWD/checkpoint_fence.py" \
  --template intern_decision_training --new_special_tokens '<decision>' \
  --tuner_type full --freeze_vit true --freeze_aligner true --freeze_llm false \
  --dataset /data/joint/train.jsonl --val_dataset /data/joint/validation.jsonl \
  --remove_unused_columns false --strict true --split_dataset_ratio 0 \
  --enable_thinking false --max_length 8192 --truncation_strategy delete \
  --packing false --padding_free false --attn_impl sdpa \
  --torch_dtype bfloat16 --bf16 true --fsdp "$PWD/fsdp2-full.json" \
  --gradient_checkpointing false --use_logits_to_keep true \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 1 --learning_rate 2e-6 --weight_decay 0.01 \
  --warmup_ratio 0.03 --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr":2e-7}' --eval_strategy no \
  --save_total_limit 1 --logging_steps 1 --report_to none --seed 42 --data_seed 42 \
  --dataset_num_proc 1 --dataloader_num_workers 0 --dataloader_pin_memory false \
  --output_dir "$output" --add_version false "${extra[@]}"
