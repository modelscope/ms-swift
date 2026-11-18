# Hy4-flash

Training examples for [Tencent-Hunyuan/Hy4-flash](https://modelscope.cn/models/Tencent-Hunyuan/Hy4-flash).

These scripts and the plugin (`hy_v4_flash_swift_patches.py`) come from the official `finetune/ms_swift_support`
directory released by the Tencent Hunyuan team (Day-0 support). They are kept close to the vendor
implementation and will be replaced by built-in `hy_v4_flash` support in a later release.

## Requirements

- 4 nodes × 8 GPUs, ≥96 GB VRAM per GPU, ≥2 TB host RAM (the vendor's minimum for both full and LoRA training).
- A transformers build that ships `hy_v4_flash` (not in any release yet):

```bash
pip install "transformers[torch,vision] @ git+https://github.com/huggingface/transformers.git@refs/pull/50903/head"
```

- The checkpoint, plus the data / DeepSpeed configs / plugin shipped next to it:

```bash
modelscope download Tencent-Hunyuan/Hy4-flash --local-dir Tencent-Hunyuan/Hy4-flash
```

## Scripts

| script | model | tuner | distributed |
| --- | --- | --- | --- |
| `sft_train.sh` | LLM | full | DeepSpeed ZeRO-3 + offload |
| `sft_train_lora.sh` | LLM | LoRA | FSDP |
| `sft_train_vl.sh` | VL | full | DeepSpeed ZeRO-3 + offload |
| `sft_train_vl_lora.sh` | VL | LoRA | FSDP |

Each script:

- loads the plugin with `--custom_register_path "${PATCHES_FILE}"` (registers `hy_v4_flash_vl` and the `hy_v4_flash` template);
- sets `--attn_impl eager`: the text backbone declares neither flash-attention nor sdpa support;
- reads `HY4_CKPT_DIR` (default `./Tencent-Hunyuan/Hy4-flash`) for the checkpoint, the example dataset and the
  DeepSpeed config, and dispatches to other nodes with `pdsh` when `IP_LIST` contains several addresses.

```bash
# single node, 8 GPUs
HY4_CKPT_DIR=/data/Tencent-Hunyuan/Hy4-flash bash sft_train_lora.sh

# 4 nodes, run once on any node that can ssh to the others
IP_LIST="10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4" \
HY4_CKPT_DIR=/data/Tencent-Hunyuan/Hy4-flash bash sft_train_lora.sh
```

## Data

JSONL, one sample per line. The vendor's data format is LLaMA-Factory style; under ms-swift only the
`messages` / `images` / `videos` fields reach the template:

```jsonl
{"messages": [{"role": "user", "content": "<image>Describe this image."}, {"role": "assistant", "content": "A cat by the window."}], "images": ["/data/images/cat.jpg"]}
```

`<image>` / `<video>` markers must match the number and order of `images` / `videos`. Text, image and video
samples can be mixed in one file.

## Notes

- Use `--model_type hy_v4_flash_vl`. The plugin also registers a text-only `hy_v4_flash` type mapped to
  `HYV4FlashForCausalLM`, which the released checkpoint does not contain.
- The plugin's template hardcodes `reasoning_effort:no_think` in `default_system`, and ms-swift's template
  layer does not read a per-sample `reasoning_effort` or an assistant `reasoning_content` field. Training
  therefore runs in no-think mode only: thinking content must be inlined into `content` (between
  `<think:opensource>` and `</think:opensource>`) and `default_system` changed to
  `reasoning_effort:high` if a thinking-mode run is needed.
- The plugin also carries vendor workarounds (ZeRO-3 grad-norm skip, meta-device loading, FSDP dtype guard,
  `_compute_acc` opt-out). They are applied at import time and affect the whole process, so keep this plugin
  dedicated to Hy4-flash runs.
- LoRA: rank 64, alpha 128, dropout 0.05, learning rate 2e-4, targets
  `q_a_proj q_b_proj kv_a_proj_with_mqa kv_b_proj o_proj`.
  Full: learning rate 1e-5.
