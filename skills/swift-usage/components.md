# Component catalog: what swift + twinkle give you, module by module

Three packages: `twinkle` (the component kernel), `twinkle_client` (HTTP client handles to a
running twinkle-server), and `twinkle_agentic` (agentic RL / auto-research building blocks).
`swift.dev` orchestrates them; its subpackages are listed at the end.

## Package map

```
twinkle/            the kernel: model, sampler, dataset, template, loss, reward, infra, ...
twinkle_client/     client-side handles: remote model/sampler/dataset/skills over HTTP
twinkle_agentic/    agentic stack: rollout, envs, tools, verifier, evaluator, agents, harness
swift/dev/          orchestration: cli, config, recipe, builders, plugins glue
```

## twinkle modules (complete)

| Module | What it gives you |
| --- | --- |
| `twinkle.model` | `TrainableModel` ABC with `TransformersModel` / `MegatronModel` backends (see method table below); multi-LoRA residency (`MultiLora`, `LoraTenant`), target-parameter LoRA (`trainable_tokens`), optimizer grouping (`BaseOptimizerGroup`), model loading contract (`ModelLoaderProtocol`), CPU offload (`offload_to_cpu`/`reload_to_gpu`) |
| `twinkle.sampler` | `Sampler` ABC + engines: `vLLMSampler` (vLLM), `SGLangSampler` (SGLang), `TransformersSampler` (plain HF `generate`). Inputs: a list of `InputFeature`/prompt objects + `SamplingParams`; output: `SampleResponse` (token ids, texts, logprobs) / `Trajectory` for multi-turn. `PartialRolloutMixin` = resumable partial generations |
| `twinkle.dataset` | `Dataset` + `DatasetMeta` (hub id `ms://`/`hf://`, subset, split, `data_slice`), `PackingDataset`, `IterableDataset` (streaming). Pipeline: `set_template` -> `map(Preprocessor)` -> `encode()` |
| `twinkle.dataloader` | `DataLoader` (map-style + streaming), samplers (`EpochSampler`, `ShardedBatchSampler`, `SkipSampler` for fast-forward resume), deterministic seeding (`derive_seed`, `isolated_random_state`) |
| `twinkle.template` | chat templates: `Template` (renders through the checkpoint's own jinja chat_template — correct for most models), `Qwen3_5Template`, `DeepseekV4Template` (dedicated classes for families needing special token handling). Selected by class name string |
| `twinkle.processor` | `InputProcessor`: the collation side — turns encoded rows into model-ready batches (padding, position_ids, mrope...). Attached to a model via `set_processor` |
| `twinkle.data_format` | the shared value types crossing every seam: `Message`/`Tool`/`ToolCall`, `InputFeature` (encoded row), `ModelOutput`, `SamplingParams`/`PoolingParams`, `SampleResponse`/`PoolingResponse`, `Trajectory`, `LossOutput`, `is_encoded` |
| `twinkle.loss` | losses, all selectable by name: `CrossEntropyLoss`, `ChannelLoss`, `ChunkedCrossEntropyLoss`, Liger fused CE/GRPO, `DPOLoss`/`KTOLoss`/`SimPOLoss`/`CPOLoss`/`ORPOLoss` (preference family), `GRPOLoss`, `GKDLoss`, `MOPDLoss`, `OPSPLoss` (distillation family), `DPPOLoss`/`DPPOKLLoss`, `DROLoss`, `GeoMeanLoss`, `ClipCovLoss`, `ValueLoss`, `SeqClsLoss`, `RerankerLoss`, `InfoNCELoss`, `MSELoss`, `RewardLoss`, `SDARLoss` |
| `twinkle.reward` | rule rewards (score completions): `GSM8KAccuracyReward`, `GSM8KFormatReward`, `GSM8KBrevityReward`, `MathVerifyAccuracyReward` (math_verify-backed `accuracy`), `BoxedMathAccuracyReward`, `DAPOMathReward`, `FormatReward`, `MathReward`, `MultiModalAccuracyReward`, `OlympiadBenchAccuracyReward`. Note: by-NAME selection of rewards is a swift-side feature (`--orm`); in pure twinkle you instantiate them yourself |
| `twinkle.advantage` | advantage estimators for RL: `GRPOAdvantage`, `GAEAdvantage`, `GPGAdvantage`, `ReinforcePlusPlusAdvantage`, `ReMaxAdvantage`, `OPOAdvantage`, pass@k and baseline variants, plus group admission policy |
| `twinkle.metric` | `Metric` ABC + built-ins: `Accuracy`, `ExactMatch`, `LossMetric`, `CompletionRewardMetric` (driver-side reward curve), `GRPOMetric` (clip_ratio/approx_kl/policy stats), `DPOMetric`, `EmbeddingMetric`, `MetricBuffer` |
| `twinkle.preprocessor` | dataset-row transforms: `AlpacaProcessor`, `GSM8KProcessor`, `CompetitionMathProcessor`, `CountdownProcessor`, `CLEVRProcessor`, DPO variants, ... plus the generic `Mapper`/`Filter`/`DataFilter` bases for your own row-level logic |
| `twinkle.checkpoint_engine` | weight transport between trainer and sampler: `CheckpointEngineManager` with direct/Broadcast/IPC/NCCL/HCCL engines; also the cross-actor device-to-device checkpoint channel |
| `twinkle.infra` | distribution: `initialize`, `remote_class`/`remote_function` decorators, `DeviceGroup`/`DeviceMesh` handling, worker pools, dispatch — see ray-infra.md |
| `twinkle.quantizer` | quantization: `AwqQuantizer`, `GptqQuantizer`/`GptqV2Quantizer`, `BnbQuantizer`, `HqqQuantizer`, `EetqQuantizer`, `QuantoQuantizer`, `Fp8Quantizer` |
| `twinkle.patch` | `Patch` mechanism: model/backend compatibility fixes applied by name via `model.apply_patch` (FSDP no-split modules, Megatron fixes, padding-free gates...) |
| `twinkle.hub` | `MSHub`/`HFHub`/`HubOperation`: hub download, model alias resolution (weights default to ModelScope; `USE_HF`/`--use_hf` switches) |
| `twinkle.kernel` | kernel registry: Liger ops etc., availability probes (`is_liger_available`, `is_npu_available`) |
| `twinkle.perf` | observability: `Perf` timing spans, `ModuleProfiler`, `CounterRegistry`, chrome trace / DAG reports — low-intrusion, safe to leave on |
| `twinkle.notifier` | run notifications: `Notifier` base, `DingNotifier`, `notify_exception` |
| `twinkle.protocol` | the wire contract shared by server and client (routing headers, object serialization) — you rarely touch this directly |
| `twinkle.module` | module-level registration helpers used by the kernel namespaces |
| `twinkle.utils` | `DeviceGroup`, `DeviceMesh`, `Platform`, `Plugin`, `get_logger`, `trust_remote_code`, torch utils |
| `twinkle.exceptions` / `twinkle.version` | shared exception types; release metadata |
| `twinkle.server` | the twinkle-server app surface (gateway / model / sampler / processor apps) — see server-client.md |

## TrainableModel: the training-side interface

One interface for both backends (`TransformersModel`, `MegatronModel`); the only asymmetry is
that `generate` exists on Transformers in-place only. All `set_*` methods accept an instance,
a class, or a **name string** (resolved in twinkle's kernel namespace — this is the plugin-ish
by-name surface; see plugins.md).

| Group | Methods |
| --- | --- |
| Forward/backward | `forward`, `forward_only(inputs) -> ModelOutput`, `calculate_loss() -> float`, `backward`, `forward_backward(inputs)`, `clip_grad_norm`, `step`, `zero_grad`, `lr_step`, `clip_grad_and_step` |
| Configuration | `set_loss`, `set_optimizer`, `set_lr_scheduler`, `set_template`, `set_processor`, `add_metric`, `apply_patch`, `freeze_parameters` (ratio / names / regex) |
| Adapters | `add_adapter_to_model(name, config_or_dir, gradient_accumulation_steps=...)` — named LoRA adapters, several co-resident on one base |
| Metrics | `calculate_metric(is_training) -> dict`, `get_train_configs()` |
| Checkpoints | `save(name, output_dir=None) -> path/URI`, `load(name)`, `resume_from_checkpoint`, `get_state_dict`, `upload_to_hub` |
| Memory | `empty_cache`, `offload_to_cpu`, `reload_to_gpu`, async save (`submit_async_write` / `wait_async_saves`) |

Typical in-process loop (what every cookbook script boils down to):

```python
for epoch in range(EPOCHS):
    for batch in dataloader:
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        metric = model.calculate_metric(is_training=True)
```

## Sampler: the generation-side interface

`vLLMSampler` / `SGLangSampler` / `TransformersSampler` share the `Sampler` contract:
feed a list of prompts/`InputFeature`s plus `SamplingParams` (temperature, top_p, max_tokens,
n, stop, logprobs...), get `SampleResponse`s back. Samplers are also the RL rollout engine —
they accept adapters/weights updates between rollouts. Constructors take `model_id`,
`device_mesh`, `remote_group`, engine args (`gpu_memory_utilization`, `tensor_parallel_size`,
`max_loras`, ...).

## twinkle_client: handles to a running server

| Import | What it is |
| --- | --- |
| `twinkle.init_twinkle_client(base_url, api_key)` | bind the process to a twinkle-server gateway (call once before creating client handles) |
| `twinkle_client.model.MultiLoraTransformersModel` | a remote trainable model: same method names as `TrainableModel`, but arguments must be serializable (class names + scalar kwargs; no live objects) |
| `twinkle_client.sampler.vLLMSampler` | a remote sampling engine handle |
| `twinkle_client.manager` | ClientManager: sessions, heartbeats, server capabilities, training-run and checkpoint bookkeeping over the wire |
| `twinkle_client.dataset` / `twinkle_client.dataloader` | client-side dataset/batching (data stays local; encoding can be local or delegated to the server's processor app) |
| `twinkle_client.rollout` | multi-turn rollout over the remote sampler |
| `twinkle_client.auto` | Twinkle Auto — the terminal auto-research surface: a minimal chat loop with streaming output, a background TrainingMonitor for automatic error detection and script repair (`twinkle_client.auto` + its `agent/` subpackage) |
| `twinkle_client.skills` | SkillManager + providers (bundled `autoresearch.md`, `twinkle-training.md`; ModelScope and local providers) for agent prompt injection |
| `twinkle_client.data_plane` | token-in-token-out data plane client for RL rollouts (server-side feature refs instead of shipped tensors) |

See server-client.md for the full usage pattern.

## twinkle_agentic: the agentic RL stack

| Module | What it gives you |
| --- | --- |
| `rollout` | `MultiTurnRollout` (turn-by-turn generation with tool calls), `Rollout`/`RolloutSample`/`Round` (the training-side sample contract), `PolicyEndpoint` (an OpenAI-compatible `/v1/chat/completions` server in front of the live policy — zero off-policy drift), `ExternalRollout` (wrap an external CLI agent into trainable trajectories), `LedgerBook` (per-trajectory token bookkeeping) |
| `envs` | sandbox environments: `Env` ABC (reset/step/run_script/tool_manager/evaluate), `LocalEnv` (in-process subprocess), `AgentEnv` (E2B microVM), `OpenEnv`/`EnvPool` (embedded/server/distributed pools), `EnvLeases` (concurrency = leased envs) |
| `tools` | `Tool` ABC (`__call__(name, arguments) -> str` observation + `tool_info()` OpenAI-shaped schema) and `ToolManager`; `EnvTool.from_env(env)` wraps an env's built-in tools |
| `verifier` | programmatic grading inside the episode's env: `Check`/`Criterion` (file_*/shell/python/answer_*), weighted or all-or-nothing; `rubric_score` for LLM-judge PASS/FAIL |
| `evaluator` | `Evaluator` + EvalScope adapters: run benchmarks against an in-process sampler or a protocol API — this is what `swift eval` uses |
| `harness` | `AgentHarness` lifecycle hooks (tool schemas before generation, post-generation/post-tool append-only token chains), `MsAgentHarness` |
| `agents` | `CliAgent` ABC and `MsAgent` (ms-agent connector) for external agent programs |
| `challenger` / `summarizer` | auto task generation (RSI) and run summarization |
| `async_rl` | the fully-async RL data plane and pipeline pieces |

## swift.dev subpackages (orchestration)

| Package | Role |
| --- | --- |
| `swift.dev.cli` | the eight command parsers (`sft/pt/rl/infer/deploy/export/merge/eval`) + `parser.py` (Config parsing), `legacy_coverage.py` (4.x alias map), `_megatron_compat.py` |
| `swift.dev.config` | all Config dataclasses + `process.py` (assembly) + `validate.py` (early failure with guidance) |
| `swift.dev.recipe` | `run_*` entry functions per direction: `sft/`, `grpo/`, `ppo/`, `rft/`, `distill/`, `rlhf/`, `infer/`, `export/` |
| `swift.dev.builders` | reusable Config -> object glue (model, sampler, template, optimizer, adapter, remote, sandbox) |
| `swift.dev.model` | `MODEL_MAPPING` + `build_model_loader`: swift's per-family model loading (model_cls, remote code, quant corrections) |
| `swift.dev.dataset` | `DatasetLoader` + `register_dataset` + `DATASET_MAPPING` — the dataset plugin surface |
| `swift.dev.rewards` | reward plugins: `orm.py` (REWARD kind, the `orms` registry) and `prm.py` (PRM_SCORER kind) |
| `swift.dev.callbacks` | `TrainCallback` + built-ins (`EarlyStopCallback`) — the callback plugin surface |
| `swift.dev.loss` | Config -> twinkle loss wiring (loss selection, Liger fused CE) |
| `swift.dev.utils` | `plugin.py` (PluginRegistry — see plugins.md) and `naming.py` (name resolution into twinkle kernel namespaces) |
| `swift.dev.template` / `template_contract` | swift-side template selection + `TemplateSpec` (serializable template description for remote mode) |
| `swift.dev.patch` | dev-side compatibility patches (Accelerate offload hook, torch conv3d / dist-save, transformers hub kernels, unsloth distributed, ...) |
| `swift.dev.data_format` | re-exports the shared value types (`InputFeature`, `ModelOutput`, `LossOutput`) for dev-side code |
| `swift.dev.components` | the infer component assembly |
| `swift.dev.webui` | webui parameter registry |
| `swift.dev.weight_sync` | trainer->sampler weight sync orchestration |
