export USE_SWIFT_V5=0
# Mixed usage
CUDA_VISIBLE_DEVICES=0 \
swift infer examples/v4/yaml/deepspeed/infer.yaml \
    --adapters output/vx-xxx/checkpoint-xxx
