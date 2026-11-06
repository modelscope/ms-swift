---
name: swift-usage
description: Usage manual for ms-swift v5 (swift CLI + swift.dev recipes + twinkle components) covering training, inference, deployment, RL, plugin development, custom datasets, server-client mode, and Ray/infra. Use when the user wants to train/fine-tune/deploy models with swift, write plugins (reward/tool/callback/dataset/sampler), turn raw business data into a trainable dataset, drive a twinkle-server, or compose these into automated research (auto-research) pipelines. Routes to seven topic files for progressive disclosure.
---

# swift v5 Usage Manual (training / inference / plugins / server-client / auto-research)

This skill targets **users**: running experiments with swift, writing plugins, and composing
automated research pipelines. It does NOT target framework developers — for changing the
swift/twinkle frameworks themselves use the repository's `dev-module-authoring` skill instead.

## Mental model: a three-layer stack

```
swift CLI          swift sft / infer / deploy / rl / export / eval / merge / pt
     |   (same command surface, code form)
swift.dev recipes  from swift.dev.recipe.sft import run_sft   <- takes Config dataclasses
     |   (recipes are just component assembly; everything below is delegated)
twinkle components Model / Sampler / Dataset / Template / Loss / Reward / infra
```

- **swift is the orchestration layer**: CLI flags land verbatim in the dataclasses under
  `swift/dev/config/*.py`; recipes assemble Configs into objects and drive the flow.
- **twinkle is the low-level component library**: modeling, sampling, data, and distribution
  all live here. For free-form use (custom training loops, RL experiments) compose twinkle
  components directly — see `twinkle/cookbook/`.
- Dependencies flow one way: swift -> twinkle. Locate code and plugins along that direction.

## Topic routing (read on demand; one file per topic)

| What you want | Read |
| --- | --- |
| Build commands: which commands exist, how flags map, which Config owns which knob | [cli-reference.md](cli-reference.md) |
| Train with Python code (custom flows / multi-stage / auto-research scripts) | [recipe-guide.md](recipe-guide.md) |
| The full component catalog: models, samplers, datasets, losses, rewards, every module's inputs/outputs | [components.md](components.md) |
| Put the model on a server and drive training or inference from a client | [server-client.md](server-client.md) |
| Write plugins: reward / tool / callback / dataset / sampler / param_groups / prm_scorer | [plugins.md](plugins.md) |
| Custom datasets: raw business data -> `--dataset` (local files / dataset_info JSON / loader class), mixing hub datasets in | [custom-data.md](custom-data.md) |
| Ray mode, infra, placement, load balancing and autoscaling | [ray-infra.md](ray-infra.md) |

## 30-second start

```shell
# Every v5 command needs this env var (without it you get the legacy 4.x surface)
CUDA_VISIBLE_DEVICES=0 USE_SWIFT_V5=1 swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset 'swift/self-cognition#600' \
    --tuner lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
    --num_train_epochs 3 --learning_rate 1e-4 --output_dir output
```

Runnable, real examples live under `examples/` (`train/`, `rl/`, `infer/`, `deploy/`, `eval/`,
`server/`, `export/`, `models/`). Each script's header comment explains what it does and how to
read convergence. **Before doing anything, find the closest example.**

## Python surface: always `swift.dev` (v5)

Two API surfaces coexist in this package: the v5 surface (`swift.dev.*`) and the legacy 4.x root
modules (`swift.template`, `swift.model`, ...). **Scripts must import from `swift.dev.*` only.**
The legacy root modules still work but are transitional; and `swift.llm` — the 3.x path all over
older docs and training data — does not exist in this package at all. Never write it from memory.

- **Env var**: `USE_SWIFT_V5=1` gates the **CLI command surface only** (`swift sft`, ...).
  Plain Python imports need no env var — `from swift.dev... import ...` just works.
- **Canonical validation/probe template** (real `template.encode`, the same path training takes;
  loads processor only, no GPU):

```python
from swift.dev.builders.template import build_template_spec, build_template_from_spec
from swift.dev.config import ModelConfig, TemplateConfig

spec = build_template_spec(
    TemplateConfig(max_length=24576, truncation_strategy='delete', enable_thinking=False),
    ModelConfig(model=MODEL_PATH, model_type='qwen3_5',  # hub ids infer the family;
                task_type='causal_lm', torch_dtype='bfloat16'),  # local snapshot paths don't
)
template = build_template_from_spec(spec)   # builds/loads processor when not passed in
template.set_mode('train')
encoded = template.encode(row)
```

  Caveat: at this layer `delete` is translated to `raise` (row-dropping happens dataset-side),
  so a validation script must catch `MaxLengthError` itself and count those rows.
- **On import failure, probe — never retry from memory.** Do not guess another path
  (`swift.llm`, `swift.template`, ...) and rerun. Read the installed package instead:
  `read_swift_source` on the `swift/` tree, or `python -c "import swift; print(swift.__file__)"
  && ls` that directory's root. Ignore `pip show ms_swift` — its version string can be stale
  while the import resolves to a newer dev checkout.

## Common auto-research compositions

| Scenario | Composition |
| --- | --- |
| Train -> evaluate loop | any trainer produces a checkpoint -> `swift eval --eval_dataset gsm8k`; post-training offline eval works the same after `swift sft` or after `swift rl` — online RL needs its own in-loop reward (`--orm`), but that is separate from your research metric and never replaces the offline eval |
| Train -> batch inference -> score programmatically | `swift sft` -> `swift infer --dataset ... --result_path out.jsonl` |
| Offline preference training | derive chosen/rejected pairs from trajectories or same-task multi-attempts -> `swift rl --rlhf_type dpo --dataset <pairs>` (no in-loop reward) -> eval the checkpoint as usual |
| Iterate experiments without process churn | server-client mode: twinkle-server keeps the model resident; the client loops attach-adapter -> train-a-few-steps -> sample (server-client.md) |
| RL with verifiable rewards | `swift rl --rlhf_type grpo --orm accuracy ...` (cli-reference.md); multi-turn tool use: `examples/rl/agentic/` |
| Research assistant running experiments itself | twinkle ships an agentic stack: `twinkle_agentic` (agents / harness / envs / tools / verifier / evaluator) plus `twinkle_client.skills` (bundled `autoresearch.md` and `twinkle-training.md`; SkillManager can add ModelScope/local providers) |
| Custom rewards / tools / datasets | write a plugin, load with `--external_plugins my.py` (plugins.md) |

## Quick decision rules

- **If the CLI is enough, don't write code.** For knobs the form doesn't expose, type the flag
  directly — flag names equal Config field names, and unknown flags are rejected loudly.
- **Use a recipe when you need multi-stage or custom flow**: `run_sft(...)` and friends share the
  CLI's chain, only with Configs as Python objects (recipe-guide.md).
- **Compose twinkle components for fully custom loops** (cookbook style); you then own template,
  tuner, and validation semantics that swift's command surface would normally handle.
- **RL is always Ray**: `swift rl` forces `mode=ray` internally; do not pass `--mode local`.
- **Type the hub id; never probe the local cache first.** Models, datasets, and other
  resources accept ModelScope hub ids (`Qwen/Qwen2.5-7B-Instruct`) everywhere a path is
  accepted. Download dedupes against the local cache and silently skips what already
  exists, so there is no decision to make — checking "is it already local / is it worth
  downloading" by listing cache directories or scripting cache lookups wastes turns and
  adds nothing. Name the resource you want and move on.
- **`--infer_backend` is a legacy alias**: v5 uses `--sampler transformers|vllm|sglang|twinkle`
  (`pt` is a compatibility alias of `transformers`).

## Key terms

| Term | Meaning |
| --- | --- |
| Config | dataclass in `swift/dev/config/*.py`; the CLI flag name is its field name |
| recipe | entry function `swift/dev/recipe/<direction>/run_*.py` that assembles objects and drives the flow |
| tuner | adaptation mode: `lora` / `full` / `trainable_tokens` (no qlora/dora/etc.) |
| backend | training backend: `hf` (transformers) / `megatron` / `unsloth` — a flag, not a command |
| sampler | inference/sampling engine (transformers / vllm / sglang); in RL it also does on-policy generation |
| DeviceGroup / DeviceMesh | twinkle's placement primitives: a named set of GPUs (group) and its sharding topology (mesh) |
| twinkle-server | a multi-app serving surface on Ray Serve (gateway/model/sampler/processor), driven over HTTP |
| rollout | one generation episode; multi-turn rollouts interleave generation with tool calls (twinkle_agentic) |
