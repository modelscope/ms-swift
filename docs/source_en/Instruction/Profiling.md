# Training Profiling

ms-swift provides an external callback plugin, [profiler.py](https://github.com/modelscope/ms-swift/blob/main/examples/custom/profiler.py), to collect performance data at selected training steps. The default Transformers backend and Megatron-SWIFT use the same plugin path, callback name, and environment variables, without modifying trainer source code.

| Training backend | CUDA GPU | Ascend NPU |
| --- | --- | --- |
| Default Transformers backend (`swift sft/pt/rlhf`) | `torch.profiler` | `torch_npu.profiler` |
| Megatron-SWIFT (`megatron sft/pt/rlhf`) | `torch.profiler` | `torch_npu.profiler` |

The plugin profiles only the training process that loads it. Separate rollout services, such as vLLM in RLHF, require their own profiling configuration. Custom launchers, including Ray setups, must forward the plugin path and environment variables to the actual training processes.

## Prerequisites

Start with a working training environment. See [SWIFT Installation](../GetStarted/SWIFT-installation.md), [Megatron-SWIFT Quick Start](../Megatron-SWIFT/Quick-start.md), and [NPU Support](../BestPractices/NPU-support.md).

- CUDA GPU: PyTorch with profiler support, matched to your CUDA environment.
- Ascend NPU: compatible PyTorch, `torch_npu`, CANN, and device drivers.
- The default training backend does not require Megatron. The CUDA profiling path does not require installing or importing `torch_npu`.

The examples use Bash and run from the root of an ms-swift checkout containing this plugin. Alternatively, copy `examples/custom/profiler.py` to your training machine and load it using an absolute path. No other plugin files are required.

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

Add these two arguments to your existing `swift sft/pt/rlhf` or `megatron sft/pt/rlhf` command. Keep your model, dataset, parallelism, and training arguments:

```bash
--external_plugins examples/custom/profiler.py \
--callbacks profiler
```

These lines are command arguments, not a standalone shell command. If you already use other plugins or callbacks, append this plugin and `profiler` to the corresponding lists. For example:

```bash
--external_plugins /path/to/my_plugin.py examples/custom/profiler.py \
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
    --external_plugins examples/custom/profiler.py \
    --callbacks profiler
```

Step 2 warms up, steps 3 and 4 are recorded, and training continues through step 6. The model and dataset need to be downloaded; you can substitute local paths. The example requires a device that supports `bfloat16`; otherwise, adjust precision to match your working training setup.

On NPU, use the same environment variables and plugin arguments. Set `ASCEND_RT_VISIBLE_DEVICES` as in your existing NPU training script and remove `CUDA_VISIBLE_DEVICES`. You can also set `SWIFT_PROFILER_DEVICE=npu` explicitly. For Megatron, retain the model loading, parallelism settings, and dependencies described in [Megatron-SWIFT Quick Start](../Megatron-SWIFT/Quick-start.md), and append the same two plugin arguments.

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

`auto` follows the device in the training arguments when available. Otherwise, it detects an available NPU or CUDA device. If neither is available, it raises an error; set `cpu` explicitly for local debugging. The plugin does not change the device used for training.

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
- Resumed runs use absolute training steps. If the start of the window, including warmup, has already passed, the plugin logs `skipped` and skips the entire window. Set a future START_STEP to collect a new window.
- If training ends before the window, the plugin logs `training ended before requested collection window`. Ensure that training lasts through the warmup and recording interval.

If training ends early, `on_train_end` closes the profiler. Once recording has started, the underlying profiler attempts to export the collected data. A trace is not guaranteed if only warmup has run. An `atexit` handler provides best-effort cleanup during normal Python interpreter shutdown; forced termination, SIGKILL, or device failures cannot guarantee export.

## Distributed Collection

For multi-device and multi-node training, the plugin runs independently on each selected global rank. It adds no distributed barriers.

```bash
# Communication analysis usually needs data from all ranks.
export SWIFT_PROFILER_RANKS=all

# To inspect selected processes instead:
# export SWIFT_PROFILER_RANKS=0,1,8,9
```

Provide the same plugin path and settings on every node, and ensure your launcher forwards the environment variables to all training processes. `RANKS` uses global ranks, not local ranks repeated on each machine. You can use shared storage or collect local files from each node afterward.

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

GPU and NPU outputs and analysis tools differ. The plugin unifies configuration and the callback lifecycle. See the [PyTorch profiler API](https://docs.pytorch.org/docs/stable/profiler.html) and [Ascend profiler API](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000289.html) for details.

## Troubleshooting

- **No trace:** Check that the training command includes both `--external_plugins` and `--callbacks profiler`. Inspect `[SwiftProfiler]` logs, rank selection, the resumed step, the output directory, and whether training reached the collection window. Recording must start; warmup alone is insufficient.
- **No-warmup warning:** PyTorch warns when the default `WARMUP_STEPS=0` is used. This is expected. To enable warmup, set a positive value smaller than START_STEP.
- **High overhead:** Shorten ACTIVE_STEPS or disable RECORD_SHAPES, PROFILE_MEMORY, or WITH_STACK. Memory event collection does not generate a full memory timeline; the plugin does not separately export one.
- **Evaluation or checkpoint saving appears in the trace:** A continuous window may include logging, evaluation, or saving between training steps. Adjust their intervals to keep them outside the collection window.
- **Another profiler is already active:** Disable the existing profiling callback or session to avoid running multiple profilers in one process.
- **Custom Python launcher:** Megatron registration assumes that the plugin is loaded after the normal Megatron entry point initializes the backend. If you import the plugin first and switch to Megatron later, call its `register_callbacks()` again after backend initialization.
