# Elastic



## 安装依赖

集群部署K8S,并在集群中部署DLrover,[DLRover](https://github.com/intelligent-machine-learning/dlrover),
`pip install dlrover && pip install tornado && pip install kubernetes && pip install ms-swift`

经过反复测试验证的训练镜像中的其它依赖以及版本：
deepspeed 0.16.5（需参考https://github.com/deepspeedai/DeepSpeed/pull/7585/files 修复universal checkpoint 相关问题）
pytorch 2.6.0


## 如何启动

通过在`--callbacks`中添加`deepspeed_elastic`（可选`graceful_exit`）启用弹性训练，并配置DeepSpeed弹性参数。
命令组成=dlrover-run +dlrover 命令参数+swift 启动命令 +swift参数,dlrover-run除自定义的参数外，其他参数与torchrun一致；
dlrover-run 参数如下:
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
在弹性训练中我们需要关注的参数为：

--nnodes NNODES       Number of nodes, or the range of nodes in form
                        <minimum_nodes>:<maximum_nodes>.

--nproc-per-node NPROC_PER_NODE   Number of processes per node.
示例:

```bash
model=your model path
dataset=your dataset
output= your output dir
export CUDA_VISIBLE_DEVICES=0 根据实际使用的GPU情况设置
deepspeed_config_or_type=deepspeed类型或者配置文件的路径，如 zero1 或者/xxx/ms-swift/swift/llm/ds_config/zero1.json

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

## 配置文件示例
默认情况下的zero1为以下示例配置,

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

如果用户需要自定义，可以在启动命令中deepspeed_config_or_type指定自定义的zero1.json的存放路径,其中弹性相关的配置为：
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

- ignore_non_elastic_batch_info：代表在elasticity里的配置会忽略外层的batch_size相关的配置，训练过程中会根据实际的训练进程个数实时修改batch_size等相关的参数
计算原则为：
 global-training-batch-size = micro-batch-size * gradient-accumulation-steps * world-size
- max_train_batch_size：最大batch_size数
- micro_batch_sizes：elasticity下允许的每卡micro-batch size列表，相当于train_micro_batch_size_per_gpu的候选值
- min_gpus：最小gpu数目
- max_gpus：最大gpu数目
更详细的内容见：[Deepspeed](https://www.deepspeed.ai/docs/config-json/#elastic-training-config-v01-and-v02)


## 启动训练

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
      replicas: 1 #【这里需要与启动命令中的--nnodes NNODES的最大值一致】
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: main
              image: #【训练镜像，需要安装deepspeed,dlrover 和swift 】
              imagePullPolicy: IfNotPresent
              command:
                - /bin/bash
                - -c
                - sh start.sh # 启动脚本
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
