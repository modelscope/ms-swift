# v5 export: bitsandbytes load-time quantization (4-bit NF4).
#
# bnb is a LOAD-TIME scheme: its scales are fitted by transformers WHILE it materializes the weights,
# so there is nothing to rewrite on an already-loaded model. dev therefore loads the model WITH the bnb
# quantization_config and persists it with save_pretrained + save_checkpoint -- the output is a full,
# directly loadable model directory (weights + config.json carrying quantization_config + tokenizer),
# exactly what legacy `swift export --quant_method bnb` writes. It is NOT just a quantization_config.json.
#
# --quant_bits is required for every method except fp8. --safe_serialization / --max_shard_size are the
# CheckpointConfig save knobs and DO apply here (e.g. --max_shard_size 200MB shards the weights).
# Needs bitsandbytes installed. Single GPU is enough for a model this size.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift export \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_use_double_quant true \
    --output_dir Qwen2.5-1.5B-Instruct-BNB-NF4
