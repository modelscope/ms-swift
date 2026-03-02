# Elastic


## Installing Dependencies

Deploy a K8S cluster and deploy [DLRover](https://github.com/intelligent-machine-learning/dlrover) in the cluster, and install the required packages using `pip install dlrover && pip install tornado && pip install kubernetes && pip install ms-swift`

Other dependencies and versions verified through repeated testing in the training image:
deepspeed 0.16.5 (refer to this [PR](https://github.com/deepspeedai/DeepSpeed/pull/7585/files) to fix issues related to universal checkpoint)
pytorch 2.6.0


## How to Start

Enable elastic training by adding the `deepspeed_elastic` callback (optionally `graceful_exit`) in `--callbacks`, and configure DeepSpeed elasticity settings.

The command format is dlrover-run + DLrover command parameters + Swift startup command + Swift parameters.dlrover-run behaves like torchrun for most arguments, except for its custom parameters.

The dlrover-run arguments are as follows:

```
usage: dlrover-run [-h] [--nnodes NNODES] [--nproc-per-node NPROC_PER_NODE]
                   [--rdzv-backend RDZV_BACKEND] [--rdzv-endpoint RDZV_ENDPOINT] [--rdzv-id RDZV_ID]
                   [--rdzv-conf RDZV_CONF] [--standalone] [--max-restarts MAX_RESTARTS]
                   [--monitor-interval MONITOR_INTERVAL] [--start-method {spawn,fork,forkserver}]
                   [--role ROLE] [-m] [--no-python] [--run-path] [--log-dir LOG_DIR] [-r REDIRECTS]
                   [-t TEE] [--local-ranks-filter LOCAL_RANKS_FILTER] [--node-rank NODE_RANK]
                   [--master-addr MASTER_ADDR] [--master-port MASTER_PORT] [--local-addr LOCAL_ADDR]
                   [--logs-specs LOGS_SPECS] [--precheck {0,1,2}] [--node_unit NODE_UNIT]
                   [--auto_config] [--auto_tunning] [--exclude-straggler] [--save_at_breakpoint]
                   [--accelerator {nvidia.com/gpu,ascend-npu}] [--training_port TRAINING_PORT]
                   [--switchbox-check] [--box-pairs PAIR [PAIR ...]] [--min-bandwidth MIN_BANDWIDTH]
                   [--min-channels MIN_CHANNELS] [--numa-affinity] [--network-check]
                   [--comm-perf-test] [--ucp_device_type UCP_DEVICE_TYPE]
                   training_script

```
In elastic training, the parameters  you may pay attention to focus on are:

--nnodes NNODES
Number of nodes, or the range of nodes in the form <minimum_nodes>:<maximum_nodes>.

--nproc-per-node NPROC_PER_NODE
Number of processes per node.

Example:

```bash
model=your model path
dataset=your dataset
output= your output dir
export CUDA_VISIBLE_DEVICES=0 # Set according to the actual GPU usage
deepspeed_config_or_type=deepspeed type or configuration file path, e.g., zero1 or /xxx/ms-swift/swift/llm/ds_config/zero1.json

dlrover-run --nnodes 1:$NODE_NUM --nproc_per_node=1  \
/opt/conda/lib/python3.10/site-packages/swift/cli/sft.py --model $model \
--model_type qwen3 \
--train_type lora  \
--torch_dtype bfloat16 \
--dataset $dataset \
--num_train_epochs 4 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--learning_rate 5e-7 \
--gradient_accumulation_steps 8 \
--eval_steps 500 \
--save_steps 10 \
--save_total_limit 20 \
--logging_steps 1 \
--output_dir $output \
--warmup_ratio 0.01 \
--dataloader_num_workers 4 \
--temperature 1.0 \
--system You\ are\ a\ helpful\ assistant. \
--lora_rank 8 \
--lora_alpha 32 \
--target_modules all-linear \
--dataset_num_proc 1 \
--use_flash_ckpt true \
--callbacks deepspeed_elastic graceful_exit \
--deepspeed $deepspeed_config_or_type \
```

## Configuration
By default, the zero1 configuration is as follows:

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "bf16": {
        "enabled": "auto"
    },

    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": false,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false,
    "elasticity": {
        "ignore_non_elastic_batch_info": true,
        "enabled": true,
        "max_train_batch_size": 8,
        "micro_batch_sizes": [
          4,
          2
        ],
        "min_gpus": 1,
        "max_gpus": 4,
        "min_time": 20,
        "version": 0.1
      }
}
```

If users need custom configurations, they can specify the path to the custom zero1.json file in the deepspeed_config_or_type parameter. The elasticity-related configuration is as follows:
```json
...

  "elasticity": {
        "ignore_non_elastic_batch_info": true,
        "enabled": true,
        "max_train_batch_size": 8,
        "micro_batch_sizes": [
          4,
          2
        ],
        "min_gpus": 1,
        "max_gpus": 4,
        "min_time": 20,
        "version": 0.1
      }
```

- ignore_non_elastic_batch_info：Indicates that the batch size configurations outside the elasticity settings will be ignored. During training, the batch size and related parameters will be dynamically adjusted based on the number of training processes.
Calculation principle：
 global-training-batch-size = micro-batch-size * gradient-accumulation-steps * world-size
- max_train_batch_size： Maximum batch size
- micro_batch_sizes：List of allowed per-GPU micro-batch sizes under elasticity; candidates for train_micro_batch_size_per_gpu.
- min_gpus：Minimum number of GPUs.
- max_gpus：Maximum number of GPUs.
For more details, see: [Deepspeed](https://www.deepspeed.ai/docs/config-json/#elastic-training-config-v01-and-v02)

## Starting Training

```yaml
---
apiVersion: elastic.iml.github.io/v1alpha1
kind: ElasticJob
metadata:
  name: deepspeed-elastic-swift
  namespace: dlrover
spec:
  distributionStrategy: AllreduceStrategy
  optimizeMode: single-job
  replicaSpecs:
    worker:
      replicas: 1 # This should match the maximum value of --nnodes NNODES in the startup command
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: main
              image: #【Training image, needs to have deepspeed, dlrover, and swift installed】
              imagePullPolicy: IfNotPresent
              command:
                - /bin/bash
                - -c
                - sh start.sh # Startup script
              resources:
                limits:
                  cpu: '8'
                  memory: 16Gi
                  nvidia.com/gpu: '1'
              volumeMounts:
                - mountPath: /model
                  name: volume-model
                - mountPath: /dev/shm
                  name: volume-shm
          restartPolicy: Never
          volumes:
            - hostPath:
                path: /model
                type: Directory
              name: volume-model
            - emptyDir:
                medium: Memory
                sizeLimit: 200Gi
              name: volume-shm

```
