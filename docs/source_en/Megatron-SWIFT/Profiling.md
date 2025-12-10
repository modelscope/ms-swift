# NPU Performance Data Collection

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
