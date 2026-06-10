# ray的支持

## Megatron Ray

Megatron 后端支持通过 Ray 进行 GRPO 和 GKD 训练：

| 功能 | 例子 | 可分配角色 |
|------|------|-----------|
| megatron grpo | https://github.com/modelscope/ms-swift/tree/main/examples/ray/grpo | train/rollout |
| megatron gkd  | https://github.com/modelscope/ms-swift/tree/main/examples/ray/gkd  | train/rollout/teacher |

### 何时使用

非 Ray Megatron（`megatron rlhf`）和 Ray Megatron（`megatron rlhf --use_ray true`）训练功能相同

核心区别在于**部署方式**：

- **非 Ray Megatron**：通过 torchrun 启动。推理可选 colocate（同进程）或 server（手动启动 vLLM server）模式。多节点需要在每个节点手动配置 `MASTER_ADDR/PORT` 并分别启动 torchrun 和 vLLM server。
- **Ray Megatron**：通过一份 YAML 声明各角色的 GPU 数量（`train.gpus`、`rollout.gpus`、`teacher.gpus`），Ray 自动完成进程创建、GPU 分配和跨节点调度，无需手动管理多个进程。

两者都支持训练和推理的 GPU 隔离（非 Ray 通过 `vllm_mode=server`，Ray 通过 YAML 配置 separate 模式），功能上等价。Ray 的优势是将多进程的编排自动化——在多节点场景下，免去逐节点手动启动 torchrun 和 vLLM server 的运维负担。

**选择建议：**

| 场景 | 建议 |
|------|------|
| 单机训练 | **非 Ray** — 更简单 |
| 多节点集群 | **Ray** — 自动跨节点调度，一份 YAML 一键启动 |

### 快速开始

```bash
# 1. 启动 Ray 集群（单节点可省略）
ray start --head                        # 主节点
ray start --address=<head_ip>:6379      # 其他节点

# 2. 提交训练
megatron rlhf --use_ray true --config examples/ray/grpo/ray_grpo_colocate.yaml
```

### GPU 分配模式

**Colocate（共享 GPU）**— 训练和推理共享同一组 GPU，交替使用，通过 sleep/wake 释放显存：

```yaml
colocate_groups: [[train, rollout]]
offload_model: true
offload_optimizer: true
sleep_level: 1

train:
  gpus: 4
rollout:
  gpus: 4    # 必须和 train 相同
```

**Separate（独立 GPU）**— 训练和推理各占独立 GPU，无显存竞争：

```yaml
# 不设置 colocate_groups
train:
  gpus: 4
rollout:
  gpus: 4    # 独立的 4 张卡
```

### GKD Teacher 模式

| 模式 | 配置方式 | top-k | full-vocab |
|------|---------|:-----:|:----------:|
| Colocated teacher | 设置 `teacher_model` + `offload_teacher_model: true` | ✅ | ✅ |
| 独立 teacher GPU 组 | 添加 `teacher:` 组并设置 `gpus`、`model` | ✅ | ❌ |

- **Colocated teacher**：teacher 是 Megatron 模型，与 student 共享同一组 GPU 和相同的并行参数，通过 offload 交替释放显存。
- **独立 teacher GPU 组**：teacher 是独立的 vLLM 推理引擎，运行在单独 GPU 上，并行参数独立配置（`vllm_tensor_parallel_size`）。
- **top-k**：蒸馏损失仅在 teacher 概率最高的 k 个 token 上计算（通过 `gkd_logits_topk` 设置），显存占用更低，但会丢弃长尾分布信息。
- **full-vocab**：蒸馏损失在完整词表上计算，保留完整分布信息，但显存占用较高。

### 相关文档

更多文档请参考

- **GRPO 训练**：[Megatron GRPO 文档](../Megatron-SWIFT/GRPO.md)
- **GKD 训练**：[GKD 文档](../Megatron-SWIFT//GKD.md)
- **Megatron 训练参数**：[命令行参数文档](../Megatron-SWIFT/Command-line-parameters.md)
- **Megatron 快速开始**：[Quick Start](../Megatron-SWIFT/Quick-start.md)

详细配置说明和示例见 [examples](https://github.com/modelscope/ms-swift/tree/main/examples/ray)。

## Swift Ray

SWIFT 的 HF Trainer 侧也支持使用 ray 来进行多卡或多节点训练：

| 功能       | 支持ray | 例子                                                                             | 可分配角色           |
|----------|-------|--------------------------------------------------------------------------------|-----------------|
| pt/sft   | ✅     | https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node/ray | default         |
| dpo      | ❎     |                                                                                |                 |
| grpo     | ❎     |                                                                                |                 |
| ppo      | ❎     |                                                                                |                 |
| sampling | ✅     | https://github.com/modelscope/ms-swift/tree/main/examples/sampler/distill      | sampler/prm/orm |
| distill  | ✅     | https://github.com/modelscope/ms-swift/tree/main/examples/sampler/sample       | sampler/prm/orm |

### 技术细节

在叙述参数设置之前，我们有必要先行讲一下技术细节。由于SWIFT的内部当前使用了大量transformers和trl的已有实现，像veRL或ROLL一样拆解为不同的ray角色是不现实的，而且拆解后会以ray为中心，对非ray的场景的支持会不良。
因此SWIFT采取了装饰器为主的技术方案，以函数级别定义了不同角色，这些角色可以在参数中被定义如何使用。看下面的例子：

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

RayHelper会将被装饰的方法分配到不同的硬件集群中，本地调用会被平滑转换到ray集群中进行远程调用。也可以以类为中心进行划分：

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

SWIFT对ray的支持本质上是使用@worker和@function两个注解的组合使用，worker指定ray集群的角色，function指定如何分配数据。

function注解有额外的几个参数：
```python
    @staticmethod
    def function(group: str,
                 dispatch: Union[Literal['slice', 'all'], Callable] = 'all',
                 execute: Literal['first', 'all'] = 'all',
                 collect: Union[Literal['none', 'flatten'], Callable] = 'none'):
```

- dispatch: 如何分配调用入参
  - slice：对入参切分，也就是worker负载均衡执行
  - all：各个worker入参完全相同
  - 自定义切分方式，格式为：
    ```python
        def my_custom_slice(n, i, data):
            # n是worker数量，i是当前worker索引，data是原始入参
            # 返回第i个的入参
    ```
- execute: 如何执行
  - first: rank0执行，此时slice和Callable方式切分无效
  - all: 全部执行

- collect: 如何收集返回数据
  - none：原样返回，格式为各个worker返回值的列表
  - flatten: 将worker返回的结果进行拉平，支持tuple的拉平
  - Callable: 自定义collect方式，格式为：
    ```python
        def my_custom_collect(result):
            # result是各个worker返回的列表
            # 输入你想要的格式
    ```

### 参数设置

讲完技术细节后，可以将参数配置了。开发者可以根据不同的流程中的角色列表，设置不同的硬件搭配方式，例如采样功能中，共有三个角色，sampler、prm、orm，可以这样配置：

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

- nproc_per_node: ray集群中需要的每个node的最小卡数。
xxx_group: 每个ray组的名称，可以随意指定
  - device: 设备类型，当前支持GPU/CPU等。
  - ranks: 当前组分配到哪些ranks上。如果是CPU，ranks只能为整数，代表共需要多少进程，如果是GPU，可以为`[0,1,2,3]`, `4`, `list(range(0, 4))`等格式。
  - workers: 哪些角色分配到当前组中。

所有可用的角色可以见本文最上面的表。

如果使用命令行，device_groups也可以以`--device_groups xxx`方式传入，xxx为jsonstring。为了配置的简便，我们强烈推荐使用yaml方式搭配ray使用。
