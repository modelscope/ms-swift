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
# MEASURED: peak VRAM ~3 GiB/card (Qwen3.5-0.8B, export smoke, 1 card(s), 2026-11-24)
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen3.5-4B \
    --attn_impl flash_attention_2 \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_use_double_quant true \
    --output_dir Qwen3.5-4B-BNB-NF4
