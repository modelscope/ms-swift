# see rm_plugin example in swift/plugin/rm_plugin.py
# register customized plugin in external_plugins file

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs format \
    --reward_model Qwen/Qwen2.5-3B-Instruct Shanghai_AI_Laboratory/internlm2-7b-reward \
    --reward_model_plugin genrm my_rmplugin \
    --reward_weights 0.1 1 1 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --log_completions true \
    --deepspeed zero2
