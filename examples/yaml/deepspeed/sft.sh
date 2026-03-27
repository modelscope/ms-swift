NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft examples/yaml/deepspeed/sft.yaml

# NPROC_PER_NODE=2 \
# CUDA_VISIBLE_DEVICES=0,1 \
# swift sft examples/yaml/deepspeed/sft.json
