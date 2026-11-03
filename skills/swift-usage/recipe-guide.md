# Code-first development: from CLI flags to twinkle components

Three code forms with increasing freedom — and increasing responsibility. Rule of thumb:
**if the CLI suffices, don't write code**; use a recipe for multi-stage flows; compose twinkle
components only for fully custom training loops.

## Level 1: swift.dev recipes (same chain as the CLI)

The entry functions in `swift.dev.recipe` take **Config dataclass objects** — the very classes
CLI flags land in — assemble the objects, and drive the whole flow. The CLI is merely a shell
("parse argv -> build Configs -> call run_*"), so this path can do everything the CLI can,
plus what only code gives you: conditionals, multi-stage pipelines, sweep loops.

Minimal LoRA SFT:

```python
from swift.dev.config import (
    CheckpointConfig, DatasetConfig, DistributedConfig,
    ModelConfig, TemplateConfig, TrainConfig, TunerConfig,
)
from swift.dev.recipe.sft import run_sft

run_sft(
    model_config=ModelConfig(model='Qwen/Qwen2.5-0.5B-Instruct'),
    template_config=TemplateConfig(max_length=2048),
    dataset_config=DatasetConfig(dataset=['AI-ModelScope/alpaca-gpt4-data-zh']),
    train_config=TrainConfig(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
    ),
    distributed_config=DistributedConfig(),
    checkpoint_config=CheckpointConfig(save_steps=500, save_total_limit=2),
    tuner_config=TunerConfig(tuner='lora'),
)
```

Essentials:

- Pass **Config objects by keyword**; do not explode fields into dozens of kwargs.
- Sibling entries: `run_pt / run_embedding / run_reranker / run_seq_cls`
  (`swift.dev.recipe.sft`); `run_grpo / run_ppo / run_rft / run_dpo / run_opsd ...`
  (`recipe/grpo|ppo|rft|distill|rlhf`); `run_infer`; the export and eval entries.
- **Plugins work here too**: pass `PluginConfig(external_plugins=['my_plugin.py'])` and
  registered names resolve as usual. A real example: `examples/rl/code/opsd.py`, which loads
  a custom dataset plugin and calls the RL entry in code form.
- The authoritative list of every Config knob: read `swift/dev/config/<name>_config.py`
  field comments.

The common auto-research shape — sweeps and pipelines:

```python
for lr in [1e-4, 5e-5]:
    history = run_sft(
        model_config=ModelConfig(model=BASE),
        ...,
        train_config=TrainConfig(learning_rate=lr, ...),
        output_dir=f'output/lr-{lr}',
    )
    # `history` is the per-step loss/grad_norm record — feed it straight into analysis
```

## Level 2: twinkle components directly (the cookbook style)

Step outside swift's command surface and compose twinkle components yourself — custom training
loops, RL experiments, anything the recipe shapes don't fit. `twinkle/cookbook/` holds the
reference scripts; the GRPO one shows the full skeleton in ~200 lines:

```python
import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel          # or twinkle.model.megatron.MegatronModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.metric import CompletionRewardMetric
from peft import LoraConfig

MODEL_GPUS, SAMPLER_GPUS = 4, 4

# 1. Placement: name your device groups, then initialize the cluster
device_groups = [
    DeviceGroup(name='model',   ranks=list(range(MODEL_GPUS)),              device_type='GPU'),
    DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, MODEL_GPUS + SAMPLER_GPUS)), device_type='GPU'),
]
model_mesh   = DeviceMesh.from_sizes(world_size=MODEL_GPUS,   dp_size=MODEL_GPUS)
sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
twinkle.initialize(mode='ray', nproc_per_node=MODEL_GPUS + SAMPLER_GPUS, groups=device_groups)

# 2. Data: dataset -> template -> preprocess -> encode -> dataloader
dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
dataset.set_template('Qwen3_5Template', model_id='ms://Qwen/Qwen3.5-4B', max_length=400)
dataset.map(GSM8KProcessor())
dataset.encode(add_generation_prompt=True)
dataloader = DataLoader(dataset=dataset, batch_size=8)

# 3. Model + sampler as separate residents (LoRA adapter by name)
model = TransformersModel(model_id='ms://Qwen/Qwen3.5-4B', model_cls=Qwen3_5ForConditionalGeneration,
                          device_mesh=model_mesh, remote_group='model')
model.add_adapter_to_model('default', LoraConfig(target_modules=[...], r=32, lora_alpha=64))
model.set_optimizer('AdamW', lr=1e-5)
model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
model.set_loss('GRPOLoss', epsilon=0.2)          # any twinkle Loss by name (components.md)
model.set_processor(InputProcessor)
model.set_template('Qwen3_5Template', model_id='ms://Qwen/Qwen3.5-4B')
model.add_metric(CompletionRewardMetric)

sampler = vLLMSampler(model_id='ms://Qwen/Qwen3.5-4B', device_mesh=sampler_mesh,
                      remote_group='sampler')

# 4. The loop is yours: sample -> reward -> advantage -> train step
```

Read `twinkle/cookbook/rl/grpo/grpo.py` for the complete loop (reward functions, GRPO
advantage, checkpointing); `twinkle/cookbook/` covers many more shapes.

What you take over at this level: template/tuner semantics that swift's command surface would
normally handle, weight sync between sampler and trainer (CheckpointEngineManager), and
validation of your own invariants.

## Cookbook index (`twinkle/cookbook/`)

| Directory | Shows |
| --- | --- |
| `rl/grpo`, `rl/ppo`, `rl/dpo`, `rl/gkd`, `rl/rft` | one script per algorithm: full custom loops over components |
| `rl/multi_turn`, `rl/envs` | multi-turn rollout with tool calls / sandbox envs |
| `rl/async_rl` | fully-async RL (generation overlapped with training) |
| `transformers/` | FSDP2 / EP / SP / spectral-hybrid-LoRA training shapes |
| `megatron/` | Megatron-backend equivalents |
| `client/twinkle/` | driving a REMOTE twinkle-server: self-cognition, dpo, sample, embedding, multi_turn_rollout, upload_to_hub |
| `client/tinker/` | same flows through the Tinker-compatible surface |
| `client/server/` | launching the server side |
| `sample/` | plain sampling / embedding / rag_recall |
| `mm/`, `rsi/`, `observability/`, `client/swift_model/` | multimodal, RSI loop, observability, swift-loader models |

## When to use which level

| Need | Level |
| --- | --- |
| One training/inference/deploy run, standard knobs | CLI flags |
| Sweep, multi-stage, conditional pipeline, still standard semantics | dev recipe (`run_*`) |
| Custom training loop, custom RL wiring, non-standard objective composition | twinkle components (cookbook) |
| Same experiment against a resident server, no local weights | server-client mode (server-client.md) |
