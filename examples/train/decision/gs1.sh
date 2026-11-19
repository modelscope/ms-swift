# Single-stage full-parameter fine-tuning of a native GS1 checkpoint.
# The objective is distribution NLL + GS1_BRIER_WEIGHT * expected multiclass Brier.
# Replace the demo dataset with a sampled, registered GS1 training dataset.
GS1_BRIER_WEIGHT=${GS1_BRIER_WEIGHT:-1.0} \
USE_HF=1 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --external_plugins examples/train/decision/dataset.py \
    --model bigai/GS1-0.8B \
    --model_type gs1 \
    --task_type decision \
    --loss_type gs1 \
    --dataset gs1_demo \
    --tuner_type full \
    --freeze_vision_tower true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 16640 \
    --split_dataset_ratio 0 \
    --output_dir output/gs1-0.8b-decision \
    --dataloader_num_workers 1
