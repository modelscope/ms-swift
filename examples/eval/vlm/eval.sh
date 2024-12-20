CUDA_VISIBLE_DEVICES=0 \
swift eval \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --infer_backend pt \
  --eval_limit 100 \
  --eval_dataset realWorldQA
