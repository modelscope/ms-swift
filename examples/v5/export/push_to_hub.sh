# v5 export: push a model / adapter / export result to the hub.
#
# push_to_hub is a CHAINED step: it uploads whatever the preceding export step produced (merge_lora,
# quantize, ...). With no other step -- just --adapters here -- it uploads the adapter itself.
# The flag surface is split by ownership: the repo target (--hub_model_id / --hub_private_repo /
# --hub_revision) and --push_to_hub are CheckpointConfig; --commit_message is ConvertConfig; the hub
# credentials (--hub_token) and --use_hf (ModelScope vs HuggingFace) are DatasetConfig, the same place
# training reads them, so export and training push identically.
#
# Replace <model-id> / <sdk-token> before running. --use_hf false targets ModelScope (the default).
USE_SWIFT_V5=1 \
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<model-id>' \
    --hub_token '<sdk-token>' \
    --commit_message 'update files' \
    --use_hf false
