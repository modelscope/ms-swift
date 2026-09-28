# v5 merge: fold a LoRA adapter into its base weights and save one plain model.
#
# `swift merge` is dev's DEDICATED merge command (route swift.dev.cli.merge); `swift export
# --merge_lora true` is the chained form that merges first, then runs a follow-up quantize / push_to_hub
# step. Use this one for a standalone merge.
#
# Because a swift-trained `output/vx-xxx/checkpoint-xxx` carries an args.json, --model / --template / etc.
# are restored automatically (load_args defaults on for this command), so --adapters alone is enough. The
# base is loaded FULL-PRECISION by construction: a quantized base cannot absorb LoRA deltas correctly
# (huggingface/peft#2321), and dev reads no load-time quantization off this path, so no explicit clearing
# is needed. The output defaults to `{adapter}-merged`; --output_dir overrides it and an existing directory
# is left untouched unless --replace_if_exists. --safe_serialization / --max_shard_size control the write.
USE_SWIFT_V5=1 \
swift merge \
    --adapters output/vx-xxx/checkpoint-xxx \
    --replace_if_exists false
