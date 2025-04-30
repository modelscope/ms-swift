# README: GRPO External Mode Execution Scripts

---

> **Note**: External mode requires vLLM version 0.8.3 or higher.


## **Introduction**

The GRPO (Gradient-based Reinforcement Policy Optimization) training framework supports high-performance inference engines like vLLM to accelerate the sampling process. The **External Mode** allows you to connect to an external vLLM inference server, separating the inference service from the training process. This mode is ideal for scenarios where you want to offload inference to dedicated hardware or servers, improving resource utilization and scalability.

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
```

## Training with External vLLM Server
```bash
--vllm_server_host <server ip> \
--vllm_server_port <server port> \
--vllm_server_timeout <Timeout duration> \
```
Configuration Parameters
When using an external vLLM server, configure the following parameters:
