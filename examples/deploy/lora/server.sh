CUDA_VISIBLE_DEVICES=0 swift deploy \
    --host 0.0.0.0 \
    --port 8000 \
    --adapters swift-lora=swift/test_lora \
    --infer_backend vllm
