# Experimental environment: 2 * A100
# 80GB GPU memory
# Note: TorchAcc is currently only available internally.

export USE_TORCHACC=1
export TORCHACC_TRIM_GRAPH=1
export XLA_EXPERIMENTAL=nonzero:masked_select

export XLA_PERSISTENT_CACHE_PATH=./output/compiled_cache/Llama-2-13b-chat-ms
mkdir -p $XLA_PERSISTENT_CACHE_PATH

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
  --model_id_or_path modelscope/Llama-2-13b-chat-ms \
  --model_layer_cls_name LlamaDecoderLayer \
  --dataset codefuse-python-en \
  --template_type llama \
  --sft_type lora \
  --output_dir output \
  --num_train_epochs 1 \
  --max_length 2048 \
  --batch_size 14 \
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
  --report_to 'none'
