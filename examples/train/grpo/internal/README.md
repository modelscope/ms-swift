# README: GRPO Internal(Colocate) Mode Execution Scripts

---
**NOTE**
The scripts in this folder require the source code version of ms-swift.

```
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

## **Introduction**

The GRPO (Group Relative Policy Optimization) training framework supports high-performance inference engines like vLLM to accelerate the sampling process. The **Internal Mode** allows you to deploy vLLM and perform training using the same GPU resources.

This folder contains scripts and instructions for running GRPO in **Internal Mode**

## Training with Internal mode
```bash
--use_vllm true \
--vllm_mode colocate \
--vllm_gpu_memory_utilization [ut_ratio] \
```

## Multi-Node Training
On each node, execute the original single-node training script, using the environment variables `NNODES` and `NODE_RANK`, and ensure consistent use of configuration parameters across all nodes.
