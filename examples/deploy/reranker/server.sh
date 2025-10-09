# GME/GTE models or your checkpoints are also supported
# pt/vllm/sglang supported
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --host 0.0.0.0 \
    --port 8000 \
    --model BAAI/bge-reranker-v2-m3 \
    --infer_backend vllm \
    --task_type reranker \
    --vllm_enforce_eager true \
