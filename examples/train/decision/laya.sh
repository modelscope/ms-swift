LAYA_MODEL="${LAYA_MODEL:-convaiinnovations/laya}"
# e.g. LAYA_MODEL=convaiinnovations/laya LAYA_CHECKPOINT="$HOME/.cache/modelscope/hub/models/convaiinnovations/laya/typed-decisions"

USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --external_plugins examples/train/decision/dataset.py \
    --model "$LAYA_MODEL" \
    --model_type laya \
    --task_type decision \
    --tuner_type full \
    --loss_type ce_rps \
    --dataset laya_decision \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 512 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
