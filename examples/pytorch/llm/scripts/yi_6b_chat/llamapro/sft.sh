# Experimental environment: A100
# 25GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_sft.py \
    --model_type  yi-6b-chat \
    --sft_type  llamapro \
    --tuner_backend  swift \
    --dtype  AUTO \
    --output_dir  output \
    --dataset  ms-agent \
    --llamapro_num_new_blocks  4 \
    --use_loss_scale  true \
    --train_dataset_sample  50000 \
    --train_dataset_mix_ratio  2.0 \
    --num_train_epochs 2 \
    --max_length  2048 \
    --check_dataset_strategy  warning \
    --gradient_checkpointing  true \
    --batch_size  1 \
    --weight_decay  0.1 \
    --learning_rate  1e-4 \
    --gradient_accumulation_steps  16 \
    --max_grad_norm  0.5 \
    --warmup_ratio  0.03 \
    --eval_steps  500 \
    --save_steps  500 \
    --save_total_limit  2 \
    --logging_steps  10 \
