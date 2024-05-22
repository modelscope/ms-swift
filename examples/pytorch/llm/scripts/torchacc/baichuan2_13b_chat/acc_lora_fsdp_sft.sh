# Experimental environment: 2 * A100
# 80GB GPU memory
# Note: TorchAcc is currently only available internally.
# torchacc fsdp
export USE_TORCHACC=1
export XLA_FLAGS='--xla_gpu_force_compilation_parallelism=32 --xla_multiheap_size_constraint_per_heap=4831838208 --xla_disable_hlo_passes=all-gather-combiner,all-reduce-combiner,reduce-scatter-combiner,gpu-convert-async-collectives-to-sync,rematerialization'
export XLA_IR_SHAPE_CACHE_SIZE=100000000
export XLA_ALLOCATOR_FRACTION=0.95
export XLA_EXPERIMENTAL=nonzero:masked_select

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
  --model_id_or_path baichuan-inc/Baichuan2-13B-Chat \
  --model_layer_cls_name BaichuanLayer \
  --dataset codefuse-python-en \
  --sft_type lora \
  --output_dir output \
  --num_train_epochs 1 \
  --max_length 2048 \
  --batch_size 16 \
  --use_flash_attn true \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing no \
  --tuner_backend 'peft' \
  --dataset_test_ratio 0 \
  --save_strategy no \
  --eval_steps 2000000 \
  --save_steps 2000000 \
  --logging_steps 100 \
  --preprocess_num_proc 1 \
  --metric_warmup_step 0.1 \
  --fsdp_num 2 \
  --report_to 'none'
