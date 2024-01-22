export USE_TORCHACC=1
export XLA_FLAGS='--xla_multiheap_size_constraint_per_heap=4294967296 --xla_disable_hlo_passes=rematerialization,all-gather-combiner,all-reduce-combiner,reduce-scatter-combiner'
export XLA_IR_SHAPE_CACHE_SIZE=100000000
export XLA_ALLOCATOR_FRACTION=0.97

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
	--model_type qwen-72b-chat \
  --model_layer_cls_name QWenBlock \
	--dataset codefuse-python-en \
	--sft_type lora \
  --output_dir output \
  --num_train_epochs 1 \
  --max_length 1024 \
  --batch_size 1 \
  --use_flash_attn true \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing no \
  --tuner_backend 'peft' \
  --eval_steps 2000000 \
  --save_steps 2000000 \
  --logging_steps 10 \
  --preprocess_num_proc 1 \
  --dataloader_num_workers 0
