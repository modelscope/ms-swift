# ray的支持

SWIFT已经支持使用ray来进行多卡或多节点训练。已有功能中对ray的支持情况如下：

| 功能       | 支持ray | 例子                                                                             | 可分配角色           |
|----------|-------|--------------------------------------------------------------------------------|-----------------|
| pt/sft   | ✅     | https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node/ray | default         |
| dpo      | ❎     |                                                                                |                 |
| grpo     | ❎     |                                                                                |                 |
| ppo      | ❎     |                                                                                |                 |
| megatron | ❎     |                                                                                |                 |
| sampling | ✅     | https://github.com/modelscope/ms-swift/tree/main/examples/sampler/distill      | sampler/prm/orm |
| distill  | ✅     | https://github.com/modelscope/ms-swift/tree/main/examples/sampler/sample       | sampler/prm/orm |

## 技术细节

在叙述参数设置之前，我们有必要先行讲一下技术细节。由于SWIFT的内部当前使用了大量transformers和trl的已有实现，像veRL或ROLL一样拆解为不同的ray角色是不现实的，而且拆解后会以ray为中心，对非ray的场景的支持会不良。
因此SWIFT采取了装饰器为主的技术方案，以函数级别定义了不同角色，这些角色可以在参数中被定义如何使用。看下面的例子：

```python
from swift.ray import RayHelper

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

## 参数设置

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
