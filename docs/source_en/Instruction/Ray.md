# Ray Support

## Megatron Ray

The Megatron backend supports GRPO and GKD training via Ray:

| Feature | Example | Assignable Roles |
|---------|---------|------------------|
| megatron grpo | https://github.com/modelscope/ms-swift/tree/main/examples/ray/grpo | train/rollout |
| megatron gkd  | https://github.com/modelscope/ms-swift/tree/main/examples/ray/gkd  | train/rollout/teacher |

### When to Use

Non-Ray Megatron (`megatron rlhf`) and Ray Megatron (`megatron rlhf --use_ray true`) have identical training functionality.

The core difference is in **deployment**:

- **Non-Ray Megatron**: Launched via torchrun. Inference can use colocate (same process) or server (manually started vLLM server) mode. Multi-node requires manually configuring `MASTER_ADDR/PORT` on each node and separately launching torchrun and vLLM server.
- **Ray Megatron**: A single YAML declares the GPU count for each role (`train.gpus`, `rollout.gpus`, `teacher.gpus`). Ray automatically handles process creation, GPU allocation, and cross-node scheduling — no manual multi-process management needed.

Both support GPU isolation between training and inference (non-Ray via `vllm_mode=server`, Ray via YAML separate mode), making them functionally equivalent. Ray's advantage is automating multi-process orchestration — in multi-node scenarios, it eliminates the operational burden of manually launching torchrun and vLLM server on each node.

**Selection guide:**

| Scenario | Recommendation |
|----------|---------------|
| Single-node training | **Non-Ray** — simpler |
| Multi-node cluster | **Ray** — automatic cross-node scheduling, one YAML to launch |

### Quick Start

```bash
# 1. Start Ray cluster (optional for single node)
ray start --head                        # head node
ray start --address=<head_ip>:6379      # worker nodes

# 2. Submit training
megatron rlhf --use_ray true --config examples/ray/grpo/ray_grpo_colocate.yaml
```

### GPU Allocation Modes

**Colocate (shared GPU)** — Training and inference share the same GPUs, alternating usage via sleep/wake to free memory:

```yaml
colocate_groups: [[train, rollout]]
offload_model: true
offload_optimizer: true
sleep_level: 1

train:
  gpus: 4
rollout:
  gpus: 4    # must match train
```

**Separate (dedicated GPU)** — Training and inference occupy separate GPUs with no memory contention:

```yaml
# do not set colocate_groups
train:
  gpus: 4
rollout:
  gpus: 4    # dedicated 4 GPUs
```

### GKD Teacher Modes

| Mode | Configuration | top-k | full-vocab |
|------|--------------|:-----:|:----------:|
| Colocated teacher | Set `teacher_model` + `offload_teacher_model: true` | ✅ | ✅ |
| Standalone teacher GPU group | Add `teacher:` group with `gpus`, `model` | ✅ | ❌ |

- **Colocated teacher**: The teacher is a Megatron model sharing the same GPUs and parallel parameters as the student, with memory alternated via offload.
- **Standalone teacher GPU group**: The teacher is an independent vLLM inference engine running on separate GPUs, with independently configured parallel parameters (`vllm_tensor_parallel_size`).
- **top-k**: The distillation loss is computed only over the teacher's top-k highest-probability tokens (set via `gkd_logits_topk`). Lower memory usage, but discards long-tail distribution information.
- **full-vocab**: The distillation loss is computed over the entire vocabulary, preserving the full distribution. Higher memory usage.

For detailed configuration and examples, see [examples](https://github.com/modelscope/ms-swift/tree/main/examples/ray).

## Swift Ray

SWIFT's HF Trainer also supports using Ray for multi-GPU or multi-node training:

| Feature  | Ray Support | Example                                                                        | Assignable Roles |
|----------|-------------|--------------------------------------------------------------------------------|------------------|
| pt/sft   | ✅           | https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node/ray | default          |
| dpo      | ❎           |                                                                                |                  |
| grpo     | ❎           |                                                                                |                  |
| ppo      | ❎           |                                                                                |                  |
| sampling | ✅           | https://github.com/modelscope/ms-swift/tree/main/examples/sampler/distill      | sampler/prm/orm  |
| distill  | ✅           | https://github.com/modelscope/ms-swift/tree/main/examples/sampler/sample       | sampler/prm/orm  |

### Technical Details

Before describing parameter settings, it's necessary to first explain the technical details. Since SWIFT currently uses many existing implementations from transformers and trl internally, decomposing into different Ray roles like veRL or ROLL is impractical, and decomposition would center around Ray, resulting in poor support for non-Ray scenarios.

Therefore, SWIFT adopts a decorator-based technical approach, defining different roles at the function level. These roles can be defined in parameters to specify how they are used. See the example below:

```python
from swift.ray_utils import RayHelper

@RayHelper.worker(group=['model1', 'model2'])
class MyTrainer:

    def __init__(self, args):
        self._prepare_model1()
        self._prepare_model2()
        self._prepare_datasets()

    @RayHelper.function(group='model1')
    def _prepare_model1(self):
        ...

    @RayHelper.function(group='model2')
    def _prepare_model2(self):
        ...

    @RayHelper.function(group='model1')
    def rollout(self, inputs):
        return self.model1.generate(inputs)

    @RayHelper.function(group='model2')
    def forward_model2(self, inputs):
        loss = self.model2.forward(inputs)
        loss.backward()

    def _prepare_datasets(self):
        self.dataset = ...

    def train(self):
        for batch in DataLoader(self.dataset):
            generated = self.rollout(batch)
            self.forward_model2(generated)
            ...


if __name__ == '__main__':
    ...
    MyTrainer(args).train()
```

RayHelper distributes decorated methods to different hardware clusters, and local calls are smoothly converted to remote calls in the Ray cluster. You can also partition centered around classes:

```python

@RayHelper.worker(group=['model1'])
class Model1:
    ...

    @RayHelper.function(group='model1')
    def rollout(self):
        ...

@RayHelper.worker(group=['model2'])
class Model2:
    ...

    @RayHelper.function(group='model2')
    def forward_and_optimize(self):
        ...


class Trainer:
    ...
```

SWIFT's support for Ray essentially uses a combination of @worker and @function annotations. The worker specifies Ray cluster roles, and function specifies how to distribute data.

The function annotation has several additional parameters:
```python
    @staticmethod
    def function(group: str,
                 dispatch: Union[Literal['slice', 'all'], Callable] = 'all',
                 execute: Literal['first', 'all'] = 'all',
                 collect: Union[Literal['none', 'flatten'], Callable] = 'none'):
```

- dispatch: How to distribute call input parameters
  - slice: Split the input parameters, meaning workers execute with load balancing
  - all: All workers receive identical input parameters
  - Custom slicing method, format:
    ```python
        def my_custom_slice(n, i, data):
            # n is the number of workers, i is the current worker index, data is the original input parameters
            # Return the input parameters for the i-th worker
    ```
- execute: How to execute
  - first: Execute on rank0; slice and Callable slicing methods are invalid in this case
  - all: Execute on all

- collect: How to collect returned data
  - none: Return as-is, format is a list of return values from each worker
  - flatten: Flatten the results returned by workers, supports tuple flattening
  - Callable: Custom collect method, format:
    ```python
        def my_custom_collect(result):
            # result is a list returned by each worker
            # Return in your desired format
    ```

### Parameter Settings

After explaining the technical details, we can configure the parameters. Developers can set different hardware configurations according to the role list in different processes. For example, in the sampling function, there are three roles: sampler, prm, orm. You can configure them like this:

```yaml
device_groups:
  nproc_per_node: 4
  sample_group:
    device: GPU
    ranks: list(range(0, 2))
    workers:
      - sampler
  rm_group:
    device: GPU
    ranks: list(range(2, 4))
    workers:
      - prm
      - orm
```

- nproc_per_node: The minimum number of GPUs per node required in the Ray cluster.
- xxx_group: The name of each Ray group, can be specified arbitrarily
  - device: Device type, currently supports GPU/CPU, etc.
  - ranks: Which ranks are allocated to the current group. If CPU, ranks can only be an integer representing the total number of processes needed. If GPU, can be in formats like `[0,1,2,3]`, `4`, `list(range(0, 4))`, etc.
  - workers: Which roles are allocated to the current group.

All available roles can be found in the table at the top of this document.

If using the command line, device_groups can also be passed as `--device_groups xxx`, where xxx is a JSON string. For configuration simplicity, we strongly recommend using YAML format in conjunction with Ray.
