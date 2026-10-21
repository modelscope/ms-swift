# v5 model-family examples

Per-family `train.sh` + `infer.sh` scaffolds for the SWIFT **v5** command surface. Each family's
`model_type` is resolved by swift from `--model` (no explicit `--model_type`/`--template` needed) and was
verified against the loader registry in `swift/dev/model/loader/`. Every script sets `USE_SWIFT_V5=1`.

The six v5 target families:

| dir | model_type (loader) | shape | train / infer | status |
| --- | --- | --- | --- | --- |
| `qwen3_5/` | `qwen3_5` (qwen.py) | multimodal (vision + audio), Qwen3.5-4B | LoRA SFT, `--freeze_vit/--freeze_aligner` + vLLM infer | **smoke-verified** (loss 2.69 → 0.89) |
| `qwen3_next/` | `qwen3_next` (qwen.py) | 80B-A3B MoE | Megatron backend (ep8) + vLLM expert-parallel infer | reference-only (80B, uncached) |
| `deepseek/` | `deepseek_v4` (deepseek.py) | Megatron-primary MoE (DeepSeek-V4-Flash) | Megatron backend (ep8 + pp2) + vLLM TP8 infer | reference-only (multi-node cluster) |
| `qwen3_tts/` | `qwen3_tts` (qwen.py) | TTS (audio out), Qwen3-TTS-12Hz-1.7B | `--tuner full` SFT; `prepare_data.py` → `train.sh` → `infer.py` | reference-only (`qwen-tts` not installed) |
| `gemma4/` | `gemma4` (gemma.py) | multimodal (vision + audio), gemma-4-E2B-it | LoRA SFT + DeepSpeed ZeRO2, vLLM infer | reference-only (gated, uncached) |
| `gpt_oss/` | `gpt_oss` (llm.py) | ~20B sparse MoE (harmony chat + reasoning) | LoRA SFT (`--router_aux_loss_coef`), vLLM infer | reference-only (~42GB, uncached) |

Conventions used across these scripts:

- **Backend.** The dense / smaller families run on the default transformers backend; the large MoE families
  (`qwen3_next`, `deepseek`) use `--backend megatron` with expert / pipeline parallelism, which swaps the
  flag vocabulary (`--lr`/`--min_lr`/`--lr_warmup_fraction`/`--save_steps`/`--attention_backend flash`
  instead of `--learning_rate`/`--lr_scheduler`/`--warmup_ratio`/`--save_interval`).
- **Attention.** The transformers-backend bf16 recipes pin `--attn_impl flash_attention_2`: the default SDPA
  path lands on a cuDNN attention kernel whose bf16 backward returns NaN on longer rows (clean for ~18 steps,
  then loss becomes nan). The token is passed straight through to HF, so it must be `flash_attention_2`, not
  legacy swift's `flash_attn`. The Megatron recipes express the same via `--attention_backend flash`.
- **Inference.** `infer.sh` uses `--sampler vllm` by default (swap in `--sampler transformers`/`sglang`);
  with no `--dataset` it opens an interactive REPL. To infer a fine-tune, replace `--model` with
  `--adapters output/vx-xxx/checkpoint-xxx` — base model, template and system are read from `args.json`.
- **"reference-only"** means the exact flag set was parsed and its `model_type` verified against the loader,
  but the run could not be smoke-executed here for the stated reason (model too large, gated, uncached, or a
  missing engine package). The header of each such script states the reason and the resource needed.

## Legacy scripts co-resident in these dirs

The six target dirs also still hold pre-v5 scripts (`qwen3_5/{transformers,packing,fp8,mcore,mcore_grpo_moe}.sh`,
`qwen3_next/{transformers,mcore,mtp}.sh`, `gemma4/{diffusion_gemma,gemma4_unified,mcore}.sh`,
`gpt_oss/{internvl3_5_gpt,mcore}.sh`), and `models/` keeps 12 further legacy-only multimodal families
(`deepseek_ocr`, `deepseek_vl2`, `glm-4.6v`, `hunyuan_ocr`, `internvl3`, `keye`, `llama4`, `minicpmv`,
`ovis2`, `paddle_ocr`, `qwen3_omni`, `qwen3_vl`). Several are referenced by the bilingual docs, so they are
removed only in coordination with a docs update; each is superseded by the v5 `train.sh`/`infer.sh` above.
