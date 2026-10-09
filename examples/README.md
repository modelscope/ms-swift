# SWIFT v5 Examples

Runnable, smoke-tested scaffolds for the SWIFT **v5** command surface. Every script here is a real
scaffold (a genuine training / inference / export run you can shrink for a smoke test), not a placeholder.

## The v5 command surface

v5 is gated by the `USE_SWIFT_V5=1` environment variable, which routes `swift <cmd>` to the dev CLI
(`swift.dev.cli.*`). Every script in this tree sets it in its own header, so you can run them as-is:

```shell
bash examples/train/lora/train.sh
```

The v5 commands are:

| command | purpose |
| --- | --- |
| `swift pt` | continued pre-training |
| `swift sft` | supervised fine-tuning (all task types, all tuners, all backends) |
| `swift rl` | RLHF / RL (`rlhf` is a kept alias): dpo, kto, cpo, orpo, simpo, rm, ppo, grpo, gkd, opsd, mopd, rft |
| `swift infer` | offline batch inference, interactive TUI, rollout + sampling (the old `sample` is folded in) |
| `swift deploy` | serve a model as an OpenAI endpoint (twinkle-server on Ray Serve) |
| `swift export` | quantize / push_to_hub / cached_dataset / mcore convert / chained merge |
| `swift merge` | standalone LoRA merge (`export --merge_lora true` is the chained form) |
| `swift eval` | EvalScope benchmarks (in-process sampler, or a deployed URL) |

Notes on the v5 surface that differ from 4.x:

- **No `app` command** (removed) and **no `lmdeploy`** sampler.
- **No separate `megatron` command** — Megatron is a backend flag: `swift sft --backend megatron`.
- **Tuners** are exactly `lora`, `adalora`, `trainable_tokens` (plus `full`). There is **no qlora / dora /
  rslora / boft / ...**. GaLore, Liger and Muon are optimizer / kernel knobs on the training config, not tuners.
- **Task types** are `causal_lm`, `seq_cls`, `embedding`, `reranker`, `generative_reranker`.
- **Quant methods** are `awq`, `gptq`, `gptq_v2`, `bnb`, `hqq`, `eetq`, `quanto`, `fp8`.
- **`rlhf_type`** set is `dpo / orpo / simpo / kto / cpo / rm / ppo / grpo / gkd / opsd / mopd / rft`.
  The online family (`grpo / ppo / rft / gkd / opsd / mopd`) is Ray + vLLM only.

By default models are downloaded from ModelScope. To use HuggingFace instead add `--use_hf true` to any command.

## Directory layout

```
examples/
  infer/     vllm/ sglang/ transformers/ agentic/ dataset/ sample/ embedding/ grounding/
             reranker/ reward_model/ code/
  deploy/    vllm/ sglang/ embedding/ reranker/ reward_model/ seq_cls/ lora/ code/
             (each server dir ships a deploy.sh + client.py)
  eval/      eval.sh  eval_lora.sh  eval_url/  vlm/  train_eval/
  export/    awq/ gptq/ gptq_v2/ bnb/ hqq/ eetq/ quanto/ fp8/  merge/  push_to_hub/
             convert/ (mcore <-> hf)  cached_dataset/
  train/     lora/ full/ adalora/ trainable_tokens/ pretrain/ embedding/ reranker/ seq_cls/
             reward_model/ multi-modal/ multimodal_embedding/ liger/ muon/ galore/ packing/
             padding_free/ flash_attention_3/ early_stop/ think_model/ base_to_chat/ streaming/
             agentic/ sequence_parallel/ plugins/ custom/ cached_dataset/ megatron/ code/ yaml/
             multi-gpu/(torchrun,ray)  multi-node/(torchrun,ray)  data/
  rl/        grpo/ ppo/ dpo/ kto/ orpo/ cpo/ simpo/ rm/ rft/ gkd/ opsd/ mopd/ agentic/ code/  data/
  models/    per-model train.sh + infer.sh (model_type verified against the loader registry)
  server/    twinkle server-client multi-LoRA cookbook
  notebook/  Jupyter walkthroughs (kept as-is)
  ascend/    NPU examples — kept outside the v5 tree, blocked on kernel support (see below)
```

## Refactor status

The v5 target tree is now fully authored. Every directory below ships real, header-documented scaffolds;
the ones marked *(smoke-verified)* were run end-to-end on this box, the rest are *reference-only* with the
exact reason (large / gated / uncached model, missing engine, or hardware) stated in their script header.

- **train** *(32 v5 scripts)*: `lora`, `full`, `adalora`, `trainable_tokens`, `pretrain`, `embedding`,
  `reranker`, `seq_cls`, `reward_model`, `multi-modal`, `multimodal_embedding`, `liger`, `muon`, `galore`,
  `packing`, `padding_free`, `flash_attention_3`, `early_stop`, `think_model`, `base_to_chat`, `streaming`,
  `agentic`, `sequence_parallel`, `plugins`, `custom`, `cached_dataset`, `megatron/{tp_pp,moe}`,
  `multi-gpu/{torchrun,ray}`, `multi-node/{torchrun,ray}`, `code`, `yaml`.
- **rl** *(19 v5 scripts)*: `grpo` (+ `external`/`internal` multi-controller), `ppo`, `dpo`, `kto`, `orpo`,
  `cpo`, `simpo`, `rm`, `rft`, `gkd`, `opsd`, `mopd`, `agentic`, `code`.
- **infer** *(11 v5 scripts)*: `transformers`, `vllm`, `sglang`, `agentic`, `dataset`, `sample`,
  `embedding`, `grounding`, `reranker`, `reward_model`, `code`.
- **deploy** *(8 server dirs, each `deploy.sh` + `client.py`)*: `vllm`, `sglang`, `embedding`, `reranker`,
  `reward_model`, `seq_cls`, `lora`, `code`. See [deploy/README.md](deploy/README.md).
- **export** *(12 v5 scripts)*: `awq`, `gptq`, `gptq_v2`, `bnb`, `hqq`, `eetq`, `quanto`, `fp8`, `merge`,
  `push_to_hub`, `convert` (mcore ↔ hf), `cached_dataset`.
- **eval**: `eval.sh`, `eval_lora.sh`, `eval_url/`, `vlm/`, `train_eval/`.
- **models** *(6 v5 target families)*: `qwen3_5` *(smoke-verified multimodal SFT)*, `qwen3_next`,
  `deepseek`, `qwen3_tts`, `gemma4`, `gpt_oss` — each a `train.sh` + `infer.sh` with `model_type`
  verified against the loader registry. See [models/README.md](models/README.md).
- **server**: the twinkle server-client multi-LoRA cookbook *(e2e-verified: two named adapters trained and
  sampled off one base)*. See [server/README.md](server/README.md).

Legacy directories that predate v5 (`custom/`, `megatron/`, `ray/`, `sampler/`, the 12 extra multimodal
families under `models/`, and the doc-referenced `qwen3_5/{transformers,packing,fp8}.sh`) are still on disk.
They are removed only in coordination with the bilingual docs that link them, so no doc reference is left
dangling; each is superseded by a v5 equivalent in the layout above.

### device (NPU / XPU)

`device/npu` and `device/xpu` have no production code path in `swift/dev` today (only test skips), so they
are **blocked on kernel support** and intentionally not part of this tree yet. The existing `examples/ascend/`
scripts are preserved unchanged, outside the v5 layout, until the kernel work lands.
