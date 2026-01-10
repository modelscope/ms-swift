# test env: 1 * A10
# Using use_cce: 2.62GB
# Not using use_cce: 16.24G

# Install CCE dependency
pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@f643b88"

# Run ms-swift (example)
swift sft \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k#1024 \
  --train_type lora \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --use_hf true \
  --use_cce true \
  "$@"
