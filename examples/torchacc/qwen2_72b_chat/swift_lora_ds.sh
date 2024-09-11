# Experimental environment: 4 * A100
# 80GB GPU memory
# Note: TorchAcc is currently only available internally.

export USE_TORCH_XLA=0

# PYTHONPATH=../../.. \
# NPROC_PER_NODE=4 \

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
  --model_type qwen2-72b-instruct \
  --dataset codefuse-python-en \
  --sft_type lora \
  --dtype AUTO \
  --output_dir output \
  --num_train_epochs 1 \
  --max_length 1024 \
  --batch_size  4 \
  --use_flash_attn true \
  --gradient_accumulation_steps 1 \
  --dataset_test_ratio 0 \
  --save_strategy no \
  --eval_steps 2000000 \
  --save_steps 2000000 \
  --logging_steps 100 \
  --acc_steps 100 \
  --preprocess_num_proc 1 \
  --metric_warmup_step 0.1 \
  --deepspeed default-zero3 \
