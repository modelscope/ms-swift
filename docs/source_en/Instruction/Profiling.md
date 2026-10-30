# Training Profiling

ms-swift provides the built-in `profiler` callback to collect performance data at selected training steps. The default Transformers backend and Megatron-SWIFT use the same callback name and environment variables. Enable it with `--callbacks profiler`; no external plugin script is required.

| Training backend | CUDA GPU | Ascend NPU |
| --- | --- | --- |
| Default Transformers backend (`swift sft/pt/rlhf`) | `torch.profiler` | `torch_npu.profiler` |
| Megatron-SWIFT (`megatron sft/pt/rlhf`) | `torch.profiler` | `torch_npu.profiler` |

The callback profiles only the training process that enables it. Separate rollout services, such as vLLM in RLHF, require their own profiling configuration. Custom launchers, including Ray setups, must forward the callback selection and environment variables to the actual training processes.

## Prerequisites

Start with a working training environment. See [SWIFT Installation](../GetStarted/SWIFT-installation.md), [Megatron-SWIFT Quick Start](../Megatron-SWIFT/Quick-start.md), and [NPU Support](../BestPractices/NPU-support.md).

- CUDA GPU: PyTorch with profiler support, matched to your CUDA environment.
- Ascend NPU: compatible PyTorch, `torch_npu`, CANN, and device drivers.
- The default training backend does not require Megatron. The CUDA profiling path does not require installing or importing `torch_npu`.

Install an ms-swift version that includes the built-in `profiler` callback on every training node. The callback is included in the standard pip package. The Bash examples below can run from any working directory; copying a script or checking out the source repository is unnecessary.

If upgrading from the external profiler example, remove `--external_plugins examples/custom/profiler.py` and keep `--callbacks profiler` and the `SWIFT_PROFILER_*` settings. Preserve any unrelated external plugins.

## Quick Start

### Enable profiling in an existing training script

Configure the collection window:

```bash
export SWIFT_PROFILER_DEVICE=auto
export SWIFT_PROFILER_START_STEP=1000
export SWIFT_PROFILER_WARMUP_STEPS=1
export SWIFT_PROFILER_ACTIVE_STEPS=3
export SWIFT_PROFILER_OUTPUT_DIR=output/profiling
export SWIFT_PROFILER_RANKS=all
```

Add this argument to your existing `swift sft/pt/rlhf` or `megatron sft/pt/rlhf` command. Keep your model, dataset, parallelism, and training arguments:

```bash
--callbacks profiler
```

This is a command argument, not a standalone shell command. If you already use other callbacks, append `profiler` to that list. Keep any external plugins needed by those callbacks. For example:

```bash
--external_plugins /path/to/my_plugin.py \
--callbacks my_callback profiler
```

This configuration warms up at step 999, records steps 1000 through 1002, and exports and stops the profiler at the end of step 1002. Training continues afterward. Each process collects one window only.

### Short run with the default backend

First verify loading, collection, and export on your target machine using a short run:

```bash
export SWIFT_PROFILER_START_STEP=3
export SWIFT_PROFILER_WARMUP_STEPS=1
export SWIFT_PROFILER_ACTIVE_STEPS=2

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset swift/self-cognition \
    --tuner_type lora \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_steps 6 \
    --max_length 512 \
    --split_dataset_ratio 0 \
    --save_strategy no \
    --output_dir output/profiling-sft \
    --callbacks profiler
```

Step 2 warms up, steps 3 and 4 are recorded, and training continues through step 6. The model and dataset need to be downloaded; you can substitute local paths. The example requires a device that supports `bfloat16`; otherwise, adjust precision to match your working training setup.

On NPU, use the same environment variables and callback argument. Set `ASCEND_RT_VISIBLE_DEVICES` as in your existing NPU training script and remove `CUDA_VISIBLE_DEVICES`. You can also set `SWIFT_PROFILER_DEVICE=npu` explicitly. For Megatron, retain the model loading, parallelism settings, and dependencies described in [Megatron-SWIFT Quick Start](../Megatron-SWIFT/Quick-start.md), and append `--callbacks profiler`.

## Configuration and Defaults

All settings use environment variables, so neither backend requires additional CLI parameters. Unset variables use their defaults.

| Environment variable | Default | Description |
| --- | --- | --- |
| `SWIFT_PROFILER_DEVICE` | `auto` | `auto`, `cuda`, or `npu`; use `cpu` for local debugging |
| `SWIFT_PROFILER_START_STEP` | `1000` | First recorded training step, numbered from 1; must be at least 1 |
| `SWIFT_PROFILER_WARMUP_STEPS` | `0` | Warmup steps before recording; nonnegative and less than START_STEP |
| `SWIFT_PROFILER_ACTIVE_STEPS` | `1` | Consecutive recorded steps; must be at least 1 |
| `SWIFT_PROFILER_OUTPUT_DIR` | `<training output_dir>/profiling` | Output root; an empty string also uses this default |
| `SWIFT_PROFILER_RANKS` | `all` | All processes, or comma-separated global ranks such as `0` or `0,1,8,9` |
| `SWIFT_PROFILER_RECORD_SHAPES` | `true` | Record operator input shapes |
| `SWIFT_PROFILER_PROFILE_MEMORY` | `true` | Record memory allocation events |
| `SWIFT_PROFILER_WITH_STACK` | `false` | Record call stacks |
| `SWIFT_PROFILER_NPU_LEVEL` | `Level1` | NPU only: `Level0`, `Level1`, or `Level2` |
| `SWIFT_PROFILER_NPU_ASYNC_MODE` | `false` | NPU only: use asynchronous analysis when automatic analysis is enabled |
| `SWIFT_PROFILER_NPU_ANALYSE_FLAG` | `true` | NPU only: enable automatic analysis; set to `false` to retain raw data for offline analysis |

`auto` follows the device in the training arguments when available. Otherwise, it detects an available NPU or CUDA device. If neither is available, it raises an error; set `cpu` explicitly for local debugging. The callback does not change the device used for training.

Boolean settings accept `true/false` and `1/0`, without case sensitivity. Integer settings reject `1.0`, `1.5`, `abc`, and empty strings. Invalid configuration stops callback initialization instead of silently reverting to a default:

```text
ValueError: SWIFT_PROFILER_WARMUP_STEPS must be an integer >= 0; got 'abc'
ValueError: SWIFT_PROFILER_ACTIVE_STEPS must be an integer >= 1; got 0
ValueError: SWIFT_PROFILER_WARMUP_STEPS (10) must be smaller than SWIFT_PROFILER_START_STEP (10)
```

## Step Semantics and Checkpoint Resumption

The collection step matches the training step in your logs. It is not a dataloader batch or a gradient accumulation microbatch:

- At `on_step_begin`, the default backend reads `state.global_step`, and Megatron reads `state.iteration`. Both represent the number of completed training steps at that point.
- `START_STEP=1000` records training step 1000, starting when 999 steps have completed. If your previous callback started at `global_step == 1000`, use `START_STEP=1001` to preserve that collection window.
- Each profiling step covers all gradient accumulation microbatches in that training update or iteration. Counting follows each backend's own behavior when updates are skipped, for example because of gradient overflow.
- Including warmup, the profiler starts when `START_STEP - WARMUP_STEPS - 1` steps have completed.
- Resumed runs use absolute training steps. If the start of the window, including warmup, has already passed, the callback logs `skipped` and skips the entire window. Set a future START_STEP to collect a new window.
- If training ends before the window, the callback logs `training ended before requested collection window`. Ensure that training lasts through the warmup and recording interval.

If training ends early, `on_train_end` closes the profiler. Once recording has started, the underlying profiler attempts to export the collected data. A trace is not guaranteed if only warmup has run. An `atexit` handler provides best-effort cleanup during normal Python interpreter shutdown; forced termination, SIGKILL, or device failures cannot guarantee export.

## Distributed Collection

For multi-device and multi-node training, the callback runs independently on each selected global rank. It adds no distributed barriers.

```bash
# Communication analysis usually needs data from all ranks.
export SWIFT_PROFILER_RANKS=all

# To inspect selected processes instead:
# export SWIFT_PROFILER_RANKS=0,1,8,9
```

Enable the callback with the same settings on every node, and ensure your launcher forwards the environment variables to all training processes. `RANKS` uses global ranks, not local ranks repeated on each machine. You can use shared storage or collect local files from each node afterward.

## Output and Analysis

```text
<OUTPUT_DIR>/
  profile_step_1000/
    rank_0/
      <hostname>_<pid>_<timestamp>/
        config.json
        ... profiler-generated files
    rank_1/
      ...
```

The global rank, hostname, process ID, and timestamp prevent collisions across machines and repeated runs. `config.json` records the configuration, training backend, device, global rank, PyTorch version, and planned collection window. Check the actual trace and `completed_window_steps` in the `stopped` log to determine whether the window completed. This count includes warmup steps.

- GPU: `torch.profiler.tensorboard_trace_handler` exports the trace. Inspect `.pt.trace.json` files using TensorBoard's PyTorch Profiler plugin or Perfetto.
- NPU: `torch_npu.profiler.tensorboard_trace_handler` exports Ascend profiler results. Use MindStudio/Ascend analysis tools matched to your CANN and torch_npu versions. Level1 includes data used for HCCL communication analysis; the exact files depend on the runtime version.

GPU and NPU outputs and analysis tools differ. The callback unifies configuration and the callback lifecycle. See the [PyTorch profiler API](https://docs.pytorch.org/docs/stable/profiler.html) and [Ascend profiler API](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000289.html) for details.

### NPU Analysis Modes

By default, NPU traces are analysed synchronously after the collection window, preserving the `torch_npu` default. To run automatic analysis asynchronously, so the analysis process does not block the training flow, set these variables before training:

```bash
export SWIFT_PROFILER_NPU_ANALYSE_FLAG=true
export SWIFT_PROFILER_NPU_ASYNC_MODE=true
```

For large distributed runs, you can also disable automatic analysis and analyse the raw data separately after collection:

```bash
export SWIFT_PROFILER_NPU_ANALYSE_FLAG=false
```

With `NPU_ANALYSE_FLAG=false`, automatic analysis does not start, even if `NPU_ASYNC_MODE=true`. The asynchronous option controls analysis only; collection, writing data, and stopping the profiler can still add overhead. These two parameters are passed only to the NPU `tensorboard_trace_handler`; CUDA and CPU handler calls do not use them.

NPU raw data is stored in a `<worker_name>_<timestamp>_ascend_pt/` subdirectory of each run directory shown above. For offline analysis, pass the run directory containing that subdirectory to `analyse`. Save and run the following Python script, replacing the example path with the actual run directory:

```python
from torch_npu.profiler.profiler import analyse

if __name__ == '__main__':
    analyse(
        profiler_path='/path/to/profiling/profile_step_1000/rank_0/run_name',
        max_process_number=4,
    )
```

`max_process_number` limits the number of processes used by this analysis task; choose a value based on available CPU and memory. When analysing data from multiple ranks, also limit the number of concurrent analysis tasks. With automatic analysis disabled, complete offline analysis before inspecting the parsed results. See the [Ascend Profiler parameter reference](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0033.html) and [offline analysis guide](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0034.html) for parameter and output details.

## Troubleshooting

- **No trace:** Check that the training command includes `--callbacks profiler`. Inspect `[SwiftProfiler]` logs, rank selection, the resumed step, the output directory, and whether training reached the collection window. Recording must start; warmup alone is insufficient.
- **No-warmup warning:** PyTorch warns when the default `WARMUP_STEPS=0` is used. This is expected. To enable warmup, set a positive value smaller than START_STEP.
- **High overhead:** Shorten ACTIVE_STEPS or disable RECORD_SHAPES, PROFILE_MEMORY, or WITH_STACK. Memory event collection does not generate a full memory timeline; the callback does not separately export one.
- **Evaluation or checkpoint saving appears in the trace:** A continuous window may include logging, evaluation, or saving between training steps. Adjust their intervals to keep them outside the collection window.
- **Another profiler is already active:** Disable the existing profiling callback or session to avoid running multiple profilers in one process.
- **Custom Python launcher:** Include `profiler` in the training arguments' `callbacks` list and forward the environment settings to each worker. Both backends register the callback through their built-in callback modules.
