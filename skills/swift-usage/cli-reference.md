# CLI Reference: the v5 command surface and its parameter system

## The gate: USE_SWIFT_V5=1

The v5 command surface is routed by the environment variable `USE_SWIFT_V5=1` to
`swift.dev.cli.*`. Without it you get the legacy 4.x surface (both coexist; old scripts keep
running as-is). This manual covers v5 only.

Weights download from ModelScope by default; add `--use_hf true` to use HuggingFace.

## The eight commands

| Command | Purpose | Dispatches to |
| --- | --- | --- |
| `swift pt` | continued pre-training | `swift.dev.recipe.sft.run_pt` |
| `swift sft` | supervised fine-tuning (all task types / tuners / backends) | `run_sft` / `run_embedding` / `run_reranker` / `run_seq_cls` |
| `swift rl` | RLHF/RL (`rlhf` is a kept alias), 12 `--rlhf_type` algorithms | `recipe/grpo\|ppo\|rft\|distill\|rlhf` `run_*` entries |
| `swift infer` | offline batch inference, interactive REPL, rollout + sampling (old `sample` folded in) | `recipe/infer/run_infer` |
| `swift deploy` | serve as an OpenAI endpoint (twinkle-server on Ray Serve) | `recipe/deploy` (see server-client.md) |
| `swift export` | quantize / push_to_hub / merge_lora / cached_dataset / mcore convert | `recipe/export` |
| `swift merge` | standalone LoRA merge (`export --merge_lora true` is the chained form) | `recipe/export` |
| `swift eval` | EvalScope benchmarks: in-process sampler, or `--eval_url` against a deployed endpoint | `recipe/run_eval` |

Key differences from 4.x: **no** `app` command, no lmdeploy sampler, **no separate `megatron`
command** (Megatron is the `--backend megatron` flag), and tuners are exactly
`lora / trainable_tokens / full`.

## The parameter system: flag name = Config field name

CLI flags land directly in the dataclasses under `swift/dev/config/*.py` — there is no bridging
layer. The authoritative answer to "does this flag exist / what is its default" is the field
definition in the Config file (each field carries a `#:` comment explaining its semantics).

| Config | Owns |
| --- | --- |
| `ModelConfig` | model id, task_type, torch_dtype, revision, quant method |
| `TemplateConfig` | max_length, system prompt, packing-related, template choice |
| `DatasetConfig` | dataset / val_dataset, streaming, split ratio, preprocessing procs |
| `TrainConfig` | epochs, batch size, grad accumulation, LR + scheduler, loss type, GaLore/Liger/Muon knobs |
| `TunerConfig` | tuner type and LoRA knobs (rank/alpha/target_modules/...) |
| `DistributedConfig` | mode (local/ray/remote), backend (hf/megatron/unsloth), parallel spec (TP/PP/CP/EP, parallel_spec), remote server address |
| `CheckpointConfig` | save_steps, save_total_limit, resume, output-dir behavior |
| `LoggingConfig` | logging_steps, report_to (tensorboard, ...) |
| `GenerationConfig` | inference generation params (max_new_tokens / temperature / num_samples) |
| `RolloutConfig` | multi-turn rollout: max_turns, tools, sandbox family |
| `RLConfig` | rlhf_type, num_generations, beta/epsilon, orm reward channels, KL/distillation knobs |
| `RewardConfig` | reward models (RM/PRM): which model, which parallel spec |
| `InferConfig` | infer: best-of-n, result_path, metric |
| `EvalConfig` | eval_dataset, eval_limit, eval_url and other EvalScope knobs |
| `DeployConfig` | host/port/route_prefix/api_key/served_model_name |
| `PluginConfig` | external_plugins (plugin loading, see plugins.md) |
| `QuantizeConfig` / `MegatronConfig` / `MoEConfig` / `ConvertConfig` | quantization / Megatron-only / MoE-only / mcore conversion |

### Parsing rules (four things to know before writing flags)

1. **Unknown flags are rejected** (`Unrecognized arguments`), never silently dropped — a typo
   surfaces immediately.
2. `List[str]` fields take space-separated values: `--dataset a b`.
3. `dict` fields take a JSON string or a JSON file path:
   `--eval_generation_config '{"max_tokens": 1024}'`.
4. Scalar-or-list union fields (e.g. `Union[float, List[float]]`) fold a single value back to a
   scalar and keep multiple values as a list: `--moe_aux_loss_coeff 1e-3` vs `1e-3 2e-3`.

### Legacy aliases

Most legacy 4.x flag **names** are rewritten automatically (`--tuner_type` -> `--tuner`,
`--infer_backend` -> `--sampler`, `pt` -> the `transformers` engine alias), but **values** are
not mapped: `--attn_impl flash_attn` is rejected by HF — v5 only accepts
`flash_attention_2 / sdpa / eager`. When porting an old script, check value semantics flag by
flag; do not worry about names.

## Frequently used flags

```text
--model <hub id or local path>        # base model; under remote mode: the server-side served name
--task_type causal_lm|seq_cls|embedding|reranker|generative_reranker
--dataset 'id#600'                    # '#N' takes an N-row slice; space-separate several; local paths as-is
--tuner lora|full|trainable_tokens
--backend hf|megatron                 # training backend; Megatron-only knobs live in MegatronConfig
--mode local|ray|remote               # local=torchrun (default); ray: ray-infra.md; remote: server-client.md
--nproc_per_node N                    # processes/GPUs; REQUIRED under ray/remote (the driver holds no GPU)
--sampler transformers|vllm|sglang|twinkle   # engine for infer/eval/deploy; 'pt' aliases transformers
--adapters <path or name=path>        # attach LoRA for inference/resume; base model + template restored from the adapter's args.json
--orm accuracy format                 # RL reward channels (registered plugin name or model id); weight via --orm_weights
--external_plugins my.py              # plugin loading (see plugins.md)
--output_dir output                   # artifacts directory
```

## Typical commands

```shell
# LoRA SFT (watch the loss curve; self-cognition gives a 0.5B model a visible descent)
USE_SWIFT_V5=1 swift sft --model Qwen/Qwen2.5-0.5B-Instruct --tuner lora \
    --dataset 'swift/self-cognition#600' --lora_rank 8 --learning_rate 1e-4 \
    --num_train_epochs 3 --output_dir output

# Interactive REPL (no --dataset => REPL opens, stream auto-on)
USE_SWIFT_V5=1 swift infer --model Qwen/Qwen2.5-0.5B-Instruct --sampler transformers

# Batch inference: one row in, one row out, appended to --result_path
USE_SWIFT_V5=1 swift infer --model Qwen/Qwen2.5-0.5B-Instruct --sampler vllm \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#64' --result_path out.jsonl --max_new_tokens 512

# GRPO: online RL with verifiable rewards (Ray + colocated vLLM are forced; do not pass --mode)
USE_SWIFT_V5=1 CUDA_VISIBLE_DEVICES=0,1 swift rl --rlhf_type grpo \
    --model Qwen/Qwen3-1.7B --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --orm accuracy --num_generations 8 --nproc_per_node 2 \
    --tuner lora --learning_rate 1e-5 --output_dir output
#   Judge convergence by the REWARD, not the loss (a policy-gradient loss oscillates around
#   zero by construction): console `step N loss= reward=` and tensorboard train/total_reward.

# Evaluation: in-process sampler (or --eval_url http://... against a deployed endpoint)
USE_SWIFT_V5=1 swift eval --model Qwen/Qwen2.5-0.5B-Instruct --sampler vllm \
    --eval_dataset gsm8k --eval_limit 10 --eval_num_proc 16 \
    --eval_generation_config '{"max_tokens": 1024, "temperature": 0.0}' \
    --eval_output_dir ./eval_output

# Megatron-backend SFT: same command, swap backend + parallel spec
USE_SWIFT_V5=1 swift sft --model Qwen/Qwen3.5-4B --backend megatron \
    --tensor_model_parallel_size 2 --nproc_per_node 4 ...

# Offline preference family (dpo etc.): same command shape as sft
USE_SWIFT_V5=1 swift rl --rlhf_type dpo --model ... --dataset <preference dataset> ...
```

The full `rlhf_type` set: `dpo / orpo / simpo / kto / cpo / rm / ppo / grpo / gkd / opsd / mopd / rft`.
The online family (`grpo / ppo / rft / gkd / opsd / mopd`) is forced onto Ray + vLLM.

## Known gotchas

- **vLLM startup memory floor**: vLLM defaults to `gpu_memory_utilization=0.9`; if the target
  GPU lacks that much free memory it raises ValueError during engine allocation — that is not
  "model unsupported". Use a free GPU or lower `--vllm_gpu_memory_utilization`.
- **Generation params are not deploy flags**: temperature/top_p/max_new_tokens ride on each
  request; a server-side default would be silently ignored (setting one triggers a warning).
  Only engine args (`--vllm_gpu_memory_utilization`) belong to deploy.
- **Multi-GPU local training** uses torchrun semantics: `--nproc_per_node N`
  (or the NPROC_PER_NODE env var).
- **Tools require multi-turn**: `--tools` demands `--max_turns` — a single-turn rollout has no
  turn to inject tools into, and validation rejects the combination up front.
