# README: GRPO Internal Mode Execution Scripts

---

## **Introduction**

The GRPO (Group Relative Policy Optimization) training framework supports integrating high-performance inference engines like vLLM to accelerate the sampling process. The **Internal Mode** allows the inference service to be directly launched within the Trainer, reducing external dependencies and simplifying deployment.

This folder contains scripts and instructions for running GRPO in **Internal Mode**, where the model training and inference are tightly integrated with flexible resource allocation strategies.


## **Resource Allocation Strategies**

GRPO provides two resource allocation strategies under the Internal mode:

### 1. **Colocate Mode**

- **Description**: Training and inference share GPU resources.
- **Recommended Setting**:
  - Set `sleep_level=1` to release vLLM memory during training steps.
- **Resource Allocation Rules**:
  ```plaintext
  NPROC_PER_NODE = Total number of GPUs
  num_infer_workers = Total number of GPUs
  ```

### 2. **Async Mode**

- **Description**: Training and inference use independent GPU resources.
- **Recommended Setting**:
  - Set `sleep_level=1` to release vLLM memory during training steps.
- **Resource Allocation Rules**:
  ```plaintext
    NPROC_PER_NODE = Number of training GPUs
    num_infer_workers = Number of inference GPUs
    Must satisfy: Number of training GPUs + Number of inference GPUs = Total GPU count
  ```
