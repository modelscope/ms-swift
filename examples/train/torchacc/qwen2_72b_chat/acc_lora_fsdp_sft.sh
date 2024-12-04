# Experimental environment: 4 * A100
# 80GB GPU memory
# Note: TorchAcc is currently only available internally.
export USE_TORCHACC=1
# export TORCHACC_TRIM_GRAPH=1
export PJRT_ALLOCATOR_FRACTION=0.96
export XLA_EXPERIMENTAL=nonzero:masked_select
export TORCHACC_DATA_BUCKETS=512,1024,1536,2048

export TORCHACC_CACHE_PATH=./output/compiled_cache/qwen2-72b-instruct-0724
export XLA_FLAGS="--xla_gpu_enable_cudnn_fmha=false --xla_gpu_enable_priority_fusion=false --xla_gpu_normalize_layouts=false --xla_gpu_memory_limit_slop_factor=400"
mkdir -p $TORCHACC_CACHE_PATH

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MASTER_PORT=27829 \
swift sft \
--model_type qwen2-72b-instruct \
  --model_layer_cls_name Qwen2DecoderLayer \
  --dataset codefuse-python-en \
  --sft_type lora \
  --output_dir output \
  --num_train_epochs 1 \
  --max_length 2048 \
  --batch_size 6 \
  --use_flash_attn true \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing no \
  --tuner_backend 'peft' \
  --dataset_test_ratio 0 \
  --save_strategy no \
  --eval_steps 2000000 \
  --save_steps 2000000 \
  --logging_steps 100 \
  --acc_steps 100 \
  --preprocess_num_proc 1 \
  --metric_warmup_step 0.1 \
  --fsdp_num 4 \
  --report_to 'none'
