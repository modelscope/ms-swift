# Ascend NPU

For environment preparation of Megatron-SWIFT on Ascend NPU, please refer to [NPU Best Practices](../BestPractices/NPU-support.md).

## NPU Performance Data Collection

NPU performance collection is conducted through the `torch_npu.profiler.profile` interface. To begin, create an instance of `torch_npu.profiler.profile`, then use the `start` and `stop` methods to control the performance data collection process. During this process, modifications to the dependent Megatron source code are required, specifically altering the `train` function in the `Megatron-LM/megatron/training/training.py` file. Below is an example of the collection process:

```python
import torch_npu
...

experimental_config = torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
)

prof = torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
        ],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=6),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    profile_memory=False, # Close the collection of memory information
    with_stack=False,    # Close the collection of stack information
    experimental_config=experimental_config)
prof.start()
# megatron code
while iteration < args.train_iters:
  ...
  (
       loss_dict,
        skipped_iter,
        should_checkpoint,
        should_exit,
        exit_code,
        grad_norm,
        num_zeros_in_grad,
  ) = train_step(
            forward_step_func, train_data_iterator, model, optimizer, opt_param_scheduler, config, forward_backward_func)
  # collect performance data
  prof.step()
  ...
prof.stop()
```

# NPU Accuracy Data Collection
### Installing msprobe
```shell
pip install mindstudio-probe
```

### Code Modification
To support accuracy debugging with the msprobe tool, we need to modify the `_patch_word_embeddings` function in the `swift/megatron/model/mm_gpt_model.py` file. The main changes are to adjust the function parameters and internal implementation logic so that it can correctly patch the embedding layer.

The specific modification content is as follows:

Before modification:
```python
def _patch_word_embeddings(self, kwargs):
    origin_forward = VocabParallelEmbedding.forward

    def forward(_self, input_):
        args = get_args()
        reduce_scatter_embeddings = _self.reduce_scatter_embeddings
        _self.reduce_scatter_embeddings = False
        input_ = torch.masked_fill(input_, input_ < 0, 0)
        res = origin_forward(_self, input_)
        _self.reduce_scatter_embeddings = reduce_scatter_embeddings
        packed_seq_params = kwargs.get('packed_seq_params')
        # ...other logic...
        return res
    VocabParallelEmbedding.forward = forward
    try:
        yield
    finally:
        VocabParallelEmbedding.forward = origin_forward

def forward(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    decoder_input: torch.Tensor = None,
    labels: torch.Tensor = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
    **kwargs,
) -> torch.Tensor:
    if decoder_input is not None:
        pass
    elif self.pre_process:
        kwargs.update({'input_ids': input_ids, 'packed_seq_params': packed_seq_params})
        with self._patch_word_embeddings(kwargs):
            decoder_input = self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)

    # ...other logic...
```

After modification:
```python
def _patch_word_embeddings(self, kwargs, emb):          # Modification 1
    origin_forward = emb.word_embeddings.forward        # Modification 2

    def forward(input_):                                # Modification 3
        args = get_args()
        _self = emb.word_embeddings                     # Modification 4
        reduce_scatter_embeddings = _self.reduce_scatter_embeddings
        _self.reduce_scatter_embeddings = False
        input_ = torch.masked_fill(input_, input_ < 0, 0)
        res = origin_forward(input_)                    # Modification 5
        _self.reduce_scatter_embeddings = reduce_scatter_embeddings
        packed_seq_params = kwargs.get('packed_seq_params')
        # ...other logic...
        return res

    emb.word_embeddings.forward = forward               # Modification 6
    try:
        yield
    finally:
        emb.word_embeddings.forward = origin_forward    # Modification 7

def forward(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    decoder_input: torch.Tensor = None,
    labels: torch.Tensor = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
    **kwargs,
) -> torch.Tensor:
    if decoder_input is not None:
        pass
    elif self.pre_process:
        kwargs.update({'input_ids': input_ids, 'packed_seq_params': packed_seq_params})
        with self._patch_word_embeddings(kwargs, self.language_model.embedding):                # Modification 8
            decoder_input = self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)

    # ...other logic...
```

Major changes include:
1. The `_patch_word_embeddings` method adds an `emb` parameter to receive the embedding module instance
2. Directly obtain `emb.word_embeddings.forward` instead of `VocabParallelEmbedding.forward`
3. The internal `forward` function signature changed from `(_self, input_)` to `(input_)`
4. Get `_self` through `emb.word_embeddings` inside the function
5. Pass `input_` directly when calling the original forward
6. Use `emb.word_embeddings.forward` for replacement and recovery operations (Modifications 6, 7)
7. Pass the `self.language_model.embedding` instance when calling `_patch_word_embeddings`


Modify the train_step function in the file swift/megatron/trainers/base.py

Before modification:
```python
def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, *args,
               **kwargs):
    new_data_iterator = self._replace_data_iterator(data_iterator, model)
    return self._origin_train_step(forward_step_func, new_data_iterator, model, optimizer, opt_param_scheduler,
                                   config, *args, **kwargs)

```

After modification:
```python
def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, *args,
               **kwargs):
    new_data_iterator = self._replace_data_iterator(data_iterator, model)
    from msprobe.pytorch import PrecisionDebugger
    debugger = PrecisionDebugger(dump_path='./dump_path', level='mix', model=model)
    debugger.start()
    try:
        origin_train_step_out = self._origin_train_step(
            forward_step_func, new_data_iterator, model, optimizer, opt_param_scheduler,config, *args, **kwargs)
    finally:
        debugger.stop()
        debugger.step()
    return origin_train_step_out

```

### Enable

Additionally, since msprobe does not support fusion computation, you need to add `--bias_dropout_fusion false`, `--bias_swiglu_fusion false`, `--cross_entropy_loss_fusion false` to the launch script.

#### Example
```shell
PYTORCH_NPU_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --mcore_model Qwen2.5-7B-Instruct-mcore \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --tensor_model_parallel_size 2 \
    ...
    --bias_dropout_fusion false \
    --bias_swiglu_fusion false \
    --cross_entropy_loss_fusion false
```
