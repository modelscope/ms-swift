export USE_SWIFT_V5=0
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    examples/v4/yaml/megatron/infer.yaml \
    --model megatron_output/Qwen3.5-35B-A3B/vx-xxx/checkpoint-xxx-merged \
