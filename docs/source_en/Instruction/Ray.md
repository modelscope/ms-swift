# Ray Support

SWIFT already supports using Ray for multi-GPU or multi-node training. The support status for Ray in existing features is as follows:

| Feature  | Ray Support | Example                                                                        | Assignable Roles |
|----------|-------------|--------------------------------------------------------------------------------|------------------|
| pt/sft   | ✅           | https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node/ray | default          |
| dpo      | ❎           |                                                                                |                  |
| grpo     | ❎           |                                                                                |                  |
| ppo      | ❎           |                                                                                |                  |
| megatron | ❎           |                                                                                |                  |
| sampling | ✅           | https://github.com/modelscope/ms-swift/tree/main/examples/sampler/distill      | sampler/prm/orm  |
| distill  | ✅           | https://github.com/modelscope/ms-swift/tree/main/examples/sampler/sample       | sampler/prm/orm  |

## Technical Details

Before describing parameter settings, it's necessary to first explain the technical details. Since SWIFT currently uses many existing implementations from transformers and trl internally, decomposing into different Ray roles like veRL or ROLL is impractical, and decomposition would center around Ray, resulting in poor support for non-Ray scenarios.

Therefore, SWIFT adopts a decorator-based technical approach, defining different roles at the function level. These roles can be defined in parameters to specify how they are used. See the example below:

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

## Parameter Settings

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
