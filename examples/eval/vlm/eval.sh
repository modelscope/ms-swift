CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift eval \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --infer_backend vllm \
  --eval_limit 100 \
  --eval_dataset realWorldQA \
  --eval_backend VLMEvalKit
