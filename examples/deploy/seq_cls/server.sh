# GME/GTE models or your checkpoints are also supported
# pt/vllm/sglang supported
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --host 0.0.0.0 \
    --port 8000 \
    --model /your/seq_cls/checkpoint-xxx \
    --infer_backend vllm \
    --task_type seq_cls \
    --num_labels 2 \
