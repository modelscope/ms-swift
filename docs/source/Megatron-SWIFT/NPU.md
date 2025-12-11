# NPU

## NPU 性能数据采集

NPU性能采集通过`torch_npu.profiler.profile`接口进行采集，创建torch_npu.profiler.profile对象，通过start和stop接口控制采集性能数据，采集过程需要修改依赖的megatron源码，修改Megatron-LM/megatron/training/training.py文件中的train函数，采集示例如下：
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
    profile_memory=False, # 关闭采集内存信息
    with_stack=False,    # 关闭采集堆栈信息
    experimental_config=experimental_config)
prof.start()
# megatron 逻辑
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
  # 性能数据采集
  prof.step()
  ...
prof.stop()
```
