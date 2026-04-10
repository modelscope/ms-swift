CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/mnt/nas2/anaconda3/envs/vllm_main/bin/python \
    -m swift.ray.megatron.pipeline \
    --config examples/megatron/rlhf/dpo/ray.yaml
