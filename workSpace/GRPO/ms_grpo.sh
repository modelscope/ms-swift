# GRPO（组相对策略优化）训练框架支持 vLLM 等高性能推理引擎，以加速采样过程。
#`External Mode` 允许您连接到外部 vLLM 推理服务器，将推理服务与训练过程分离。此模式适用于您希望将推理卸载到专用硬件或服务器的场景，从而提高资源利用率和可扩展性。
#此文件夹包含运行 GRPO 在 `External Mode` 的脚本和说明，以实现与外部 vLLM 服务器的集成。



## 1.启动服务
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /mnt/cfs/ssw/ljc/ms-swift/output/v2-20250714-145544/checkpoint-20 --served-model-name "Qwen3-14B-test" --reasoning-parser deepseek_r1 --tool-call-parser hermes --enable-auto-tool-choice --trust-remote-code --tensor-parallel-size 8 --port 8000
## 2.调用vllm服务进行 外部推理

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs format \
    # 生成式奖励模型
    --reward_model Qwen/Qwen2.5-3B-Instruct Shanghai_AI_Laboratory/internlm2-7b-reward \ 
    --reward_model_plugin genrm my_rmplugin \
    --reward_weights 0.1 1 1 \
    --vllm_gpu_memory_utilization 0.5 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --log_completions true \
    --deepspeed zero2


# --use_vllm true \
# --vllm_mode server \
# --vllm_server_host <server ip> \
# --vllm_server_port <server port> \
# --vllm_server_timeout <Timeout duration> \
--use_vllm true \
--vllm_mode server \
--vllm_server_host 10.205.10.105 \
--vllm_server_port 8000 \
--vllm_server_timeout 60 \

## 3.多机推理，调用主节点
# 在每个节点上，执行原始的单节点训练脚本，使用环境变量 `NNODES` 和 `NODE_RANK`，并确保所有节点上配置参数的一致性。

