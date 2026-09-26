# README: GRPO External(Async) Mode Execution Scripts

---

> **Note**: External mode requires

1. vLLM version 0.8.3 or higher.
2. trl version 0.17.0 or higher
3. ms-swift source code version

```
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

## **Introduction**

The GRPO (Group Relative Policy Optimization) training framework supports high-performance inference engines like vLLM to accelerate the sampling process. The **External Mode** allows you to connect to an external vLLM inference server, separating the inference service from the training process. This mode is ideal for scenarios where you want to offload inference to dedicated hardware or servers, improving resource utilization and scalability.

This folder contains scripts and instructions for running GRPO in **External Mode**, enabling integration with an external vLLM server.

Before running the scripts, ensure the following:

1. **vLLM Server Deployment**:
   - An external vLLM server must be deployed and accessible.
   - Use the `swift rollout` command to deploy the vLLM server.

2. **Network Connectivity**:
   - Ensure the training nodes can communicate with the vLLM server over the network.

## **Deploying the vLLM Server**

To deploy an external vLLM server, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 \
swift rollout \
  --model Qwen/Qwen3-8B

# tp
CUDA_VISIBLE_DEVICES=0,1 \
swift rollout \
  --model Qwen/Qwen3-8B \
  --tensor_parallel_size 2

# dp
CUDA_VISIBLE_DEVICES=0,1 \
swift rollout \
  --model Qwen/Qwen3-8B \
  --data_parallel_size 2

# tp + dp
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rollout \
  --model Qwen/Qwen3-8B \
  --tensor_parallel_size 2 \
  --data_parallel_size 2
```

## Training with External vLLM Server
Configuration Parameters

```bash
--use_vllm true \
--vllm_mode server \
--vllm_server_host <server ip> \
--vllm_server_port <server port> \
--vllm_server_timeout <Timeout duration> \
```

## Multi-Node Training
On each node, execute the original single-node training script, using the environment variables `NNODES` and `NODE_RANK`, and ensure consistent use of configuration parameters across all nodes.
