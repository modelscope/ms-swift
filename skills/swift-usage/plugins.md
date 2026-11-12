# Plugin development: every extension point, with contracts

## The one loading mechanism

All external code loads through a single flag:

```shell
--external_plugins path/to/plugin.py        # a .py file
--external_plugins path/to/plugin_dir/      # a directory
--external_plugins hf://or/ms://hub-id      # a hub id
```

Loading happens **before anything is built** — the file is imported so its registrations take
effect, then the names you registered are selected by the corresponding flag. The same works
in code form: `PluginConfig(external_plugins=['my_plugin.py'])`.

swift declares its own extension points (kinds with base classes + a name->class registry);
twinkle owns the loading itself. Two levels exist, so a third party can even add a whole new
*kind* without editing swift (see the last section).

## Complete inventory of extension points

### swift plugin kinds (register with `@PluginRegistry.register(kind, name)`)

| Kind | Base | Select with | What it does |
| --- | --- | --- | --- |
| `reward` | `RewardPlugin` (`swift.dev.utils.plugin`) | `--orm name` | score completions against dataset columns; sync or async |
| `prm_scorer` | `PRMScorer` (`swift.dev.rewards.prm`) | `--prm_scorer name` | process-reward scoring over reasoning steps |
| `tool` | `ToolPlugin` | `--tools name` (+ `--max_turns`) | provide the tools a multi-turn rollout may call |
| `callback` | `TrainCallback` (`swift.dev.callbacks`) | `--callbacks name` | observe/steer the training loop at Trainer-equivalent events |
| `param_groups` | `ParamGroupsPlugin` (`swift.dev.builders.optimizer_plugins`) | `--param_groups_plugins name` | per-tower / LoRA-vs-base learning rates, per-group weight decay — serializable grouping rules applied worker-side on both backends |
| `sampler` | twinkle `Sampler` | `--sampler name-or-path` | a custom sampling engine; keep it in its **own** .py file (importing `twinkle.sampler` pulls vllm+torch — keep it out of shared plugin files) |

### Registry-based extension points (not PluginRegistry kinds, but plugins all the same)

| Surface | How to extend | Select with |
| --- | --- | --- |
| **dataset** | subclass `DatasetLoader` and `@register_dataset` (`swift.dev.dataset.loader`; keyed by `dataset_type`) | `--dataset your_name` |
| **model** | swift's family loaders live in `swift.dev.model.loader.MODEL_MAPPING`; a custom model that fulfills `ModelLoaderProtocol` (build_config/build_processor/build_model/...) plugs into twinkle-server via `model_loader: module:callable` in the server YAML | server YAML `model_loader:` |
| **template** | legacy `get_template` registry (swift-side template selection) | template name in configs |
| **loss_scale** | named per-token loss-weighting strategies in legacy `get_template`'s own registry | `--loss_scale name` |

Not plugins (closed choices, not registries): `InferConfig.metric` is a fixed `acc`/`rouge`
enum computed by run_infer itself.

### twinkle kernel classes by NAME (no registration needed — just subclass and pass the name)

`set_loss` / `set_optimizer` / `set_lr_scheduler` / `set_template` / `set_processor` /
`add_metric` / `apply_patch` accept a class **name string**, resolved in twinkle's kernel
namespaces. Write a subclass of `Loss` / `Metric` / `Template` / `InputProcessor` /
`Preprocessor` / `DataFilter` / `LRScheduler` / optimizer / `Patch` in an
`--external_plugins` file and pass its name:

```python
model.set_loss('MyCustomLoss', my_kwarg=1.0)   # class MyCustomLoss(Loss) in a loaded plugin
```

These eight kernel bases are the twinkle-side roster (`Patch` is applied by name through
`model.apply_patch`); the swift kinds above are the product-semantic ones. When deciding
where an extension belongs: if it fits a kernel base's contract, it's a name-string kernel
class; if it needs swift semantics (dataset columns, CLI flags, registries), it's a swift kind.

## Contracts and examples, kind by kind

### reward — the most-used kind

```python
from typing import List
from swift.dev.utils.plugin import PluginRegistry, RewardPlugin

@PluginRegistry.register('reward', 'length_bonus')
class LengthBonus(RewardPlugin):
    def __call__(self, completions, **columns) -> List[float]:
        # `completions`: the generated texts; `**columns`: every dataset column of the
        # batch (e.g. `solution`, `ground_truth`) — match names to your dataset.
        return [1.0 if len(c) > 100 else 0.0 for c in completions]
```

- Select: `--orm length_bonus`; several channels `--orm accuracy length_bonus` weighted by
  `--orm_weights`.
- Async rewards just declare `async def __call__` — I/O-bound scoring (API calls, DB queries)
  then overlaps. Sync and async rewards mix freely in one run.
- A frozen reward **model** (instead of a rule) is `--orm <model_id>`; it lands on its own
  GPUs (`orm_parallel_spec` shapes its DeviceGroup).
- The legacy dict idiom writes to the same registry:
  `from swift.rewards import orms; orms['my'] = MyReward`.
- Example: `examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py`.

### tool — agentic rollouts

```python
from typing import List
from swift.dev.utils.plugin import PluginRegistry, ToolPlugin

@PluginRegistry.register('tool', 'my_tools')
class MyTools(ToolPlugin):
    def build(self):                      # called once per run; the rollout copies the tools per episode
        return [MyTool()]                 # [] = contributes nothing

class MyTool:
    def tool_info(self):                  # OpenAI-shaped function schema
        return {'type': 'function',
                'function': {'name': 'calculator', 'description': '...',
                             'parameters': {'type': 'object', 'properties': {...}}}}
    def __call__(self, name, arguments) -> str:   # the observation fed back to the model
        ...
```

- Select: `--tools my_tools` — requires `--max_turns` (tools are injected per turn).
- Built-in `--tools sandbox` builds the sandbox env; the rollout binds its run_command /
  write_file / read_file to each episode's session — no plugin code needed.
- Env choice: default `LocalEnv` (in-process); `--sandbox_template <id>` switches to an
  E2B microVM `AgentEnv`. `--sandbox_num_envs` caps the episodes holding a session at once,
  `--sandbox_num_workers` sets the env replicas under Ray.
- Avoid eval/exec in tools — the sandbox caps memory/time, keep tools narrow.

### callback — training-loop hooks

```python
from swift.dev.callbacks import CallbackControl, CallbackState, TrainCallback
from swift.dev.utils.plugin import PluginRegistry

@PluginRegistry.register('callback', 'progress_printer')
class ProgressPrinter(TrainCallback):
    def on_step_end(self, state: CallbackState, control: CallbackControl) -> None:
        if state.global_step % 10 == 0:
            ...  # state: global_step / epoch / loss / metrics / best_metric
                 # control: set should_save / should_evaluate / should_log /
                 #          should_training_stop to steer the loop
```

Select: `--callbacks progress_printer` (built-ins their own knobs warrant are appended
automatically). Built-in registered callbacks: `early_stop`, `perf_log`, `graceful_exit`.
References: `examples/train/plugins/my_plugin.py` (a full callback plugin, runnable) and
`examples/infer/code/custom_plugins.py` (reward — sync and async — plus a tool, for the
infer/rollout side).

### dataset — self-contained datasets

```python
from swift.dev.dataset.loader import DatasetLoader, register_dataset

@register_dataset
class MyLoader(DatasetLoader):
    dataset_type = 'my_math'
    # build the rows in code (stream from a hub dataset, derive columns, ...) — see
    # examples/rl/opsd/opsd_plugin.py for the full pattern

# register_dataset is keyed by cls.dataset_type: use it bare (@register_dataset)
# or register_dataset(MyLoader) — a first name argument is NOT part of the signature
```

Select: `--dataset my_math`. This is the pattern when your data needs derivation logic
(a column computed from another column, an in-code corpus for offline smoke runs, ...).
For the data-only path (ids/column renames via JSON, local files with zero registration)
see [custom-data.md](custom-data.md).

### sampler — custom sampling engine

Subclass twinkle's `Sampler`. Selection accepts a **registered name** (built-in engines
`vllm`/`sglang`/`transformers` are seeded names in the same registry your
`@PluginRegistry.register('sampler', 'my_engine')` writes into) or an **external source path**
(`--sampler path/to/file.py` — twinkle's loader imports it and finds the `Sampler` subclass).
Keep the custom engine in its own .py file: importing `twinkle.sampler` has heavy import side
effects (vllm + torch), which you don't want polluting a shared plugin file.

## Declaring a whole new kind (third-party extension points)

```python
from swift.dev.utils.plugin import PluginRegistry
from swift.dev.utils.plugin import SwiftPlugin

class MyExtension(SwiftPlugin): ...

kind = PluginRegistry.register_kind('my_ext', MyExtension, config_field='my_ext')
```

A kind may adopt a pre-existing dict as its `entries`, so legacy-style direct writes and
`@register` write to the same place. Users then register implementations against your kind and
select them via the Config field you named. (swift's own kinds are declared this way in
`_BUILTIN_KIND_MODULES`; if you contribute a kind to swift itself, add your module there so it
is declared before user files load.)

## Gotchas

- **Kind declaration precedes loading**: registrations against a kind that was never declared
  fail with "plugin kind ... is not registered". All built-in kinds are handled for you;
  only new kinds need care.
- **Duplicate names**: registering the same kind+name twice errors (exist_ok=False by default).
- **`prm_scorer`'s flag is `--prm_scorer`** (the Config field), not `--prm`.
- `SwiftPlugin.__init__` receives `args=` (the running Config) — read your own hyperparameters
  off it instead of adding plumbing.
