# Independent FSDPTurbo SFT backend

`swift fsdpturbo sft` reuses Swift's arguments, tokenizer, templates and datasets. The optional `fsdp_turbo` package owns model parallelism, parameter wrapping, optimizer steps and clipping through its existing training lifecycle. This entry does not use HF Trainer or Megatron parallel wrapping.

## Source pair

Install the paired FSDPTurbo at [`2d74804`](https://gitcode.com/hazeldxq/FSDPTurbo/commit/2d748047993b8701c1293afcb1898e4b71fc49f1). Swift uses its native shard materialization, frozen expert gradients and chunked context-parallel attention.

## Native DeepSeek V4.1

Prepare a dequantized BF16 Hugging Face checkpoint, its tokenizer, and a matching flat native `ModelArgs` JSON. A composite HF `config.json` and the native architecture JSON can be supplied separately. Training requires `dtype=bf16`, `expert_dtype=null`, `n_mtp_layers=0` and `dspark_block_size=0` in the native JSON. A reduced-depth experiment must retain the original widths/expert counts and declare only its instantiated Engram layers, preserving the corresponding hash buckets and table sizes.

For four available accelerator processes, the following illustrates the CLI. Choose device count and model size according to the available memory; this is an interface example, not a full 40-layer capacity result.

```bash
NPROC_PER_NODE=4 MASTER_PORT=29501 swift fsdpturbo sft \
  --model /path/to/v41-bf16 --model_type deepseek_v41 --template deepseek_v41 \
  --model_kwargs '{"config_path":"/path/to/native_model_args.json"}' \
  --dataset /path/to/train.jsonl --output_dir /path/to/output \
  --use_chat_template false --lazy_tokenize false --truncation_strategy split \
  --tuner_type full --torch_dtype bfloat16 --attn_impl eager \
  --fsdp_size 4 --tp_size 1 --cp_size 1 --ep_size 4 --efsdp_size 1 \
  --fsdp_implementation native --ep_dispatcher eager --offload_params true \
  --forward_prefetch 0 --backward_prefetch 0 \
  --max_steps 20 --max_length 1024 --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing false --vit_gradient_checkpointing false \
  --save_strategy no --eval_strategy no --split_dataset_ratio 0 \
  --add_version false --report_to none --enable_npu_model_patch false
```

The CLI starts one torchrun launcher. Set device visibility in the caller's CUDA or CANN environment. `enable_npu_model_patch=false` must take effect before Swift model imports because this backend owns parallel wrapping.

Swift prepares the native model on meta, applies EP/FSDP, allocates the local parameter shards on the offload or compute device, and loads the BF16 checkpoint with `strict=True`. Before allocating pinned Host Engram storage, it warms the multi-rank communication groups. The head computes in FP32 by default. Trainable Host Engram weights and their active-row FP32 moments use the model's existing optimizer and Swift's decay/no-decay parameter groups.

## Parallel interfaces

The independent meshes overlap. `WORLD_SIZE` must be divisible by `fsdp_size * tp_size`, `ep_size * efsdp_size`, and `cp_size * tp_size`; multiplying all five sizes is incorrect. EFSDP requires EP greater than one, and TP requires native composable FSDP.

| Interface | Configuration |
| --- | --- |
| FSDP | `fsdp_size`; model-plan units are checked against actual modules |
| TP | `tp_size`; V4.1 uses shared-expert `w1/w3` column shards and `w2` row shards |
| EP / EFSDP | `ep_size` / `efsdp_size`; only expert containers enter these plans |
| CP | `cp_size` and `cp_implementation=auto/ulysses/kv_allgather/kcp`; the selected mode must match the model plan |

`auto` selects native KV-AllGather for V4.1 and Ulysses for the existing Qwen3.5 MoE plan. Native V4.1 CP delegates to `set_context_parallel_group` and model-owned loss, with no legacy attention/loss replacement. Its inputs are right-padded to CP/compression alignment, then token IDs, masks and global positions are split once; labels stay complete and unshifted. Text-only metadata takes this path even though HF metadata contains a vision tower.

V4.1 KV-AllGather CP requires TP=Ulysses=KCP=1 and eager operators. Block recomputation can be enabled using both checkpointing switches. CP=1 uses stateless recomputation so native shared attention caches are not retained between checkpoint passes. V4.1 attention-head TP is not implemented upstream; the TP plan above covers the shared experts.

The registered HF model path keeps its loader-selected FP32 stability parameters. Those parameters are excluded individually from FSDP, remain on the accelerator with CPU offload, and have gradients averaged before the optimizer step. An unused local branch still participates in synchronization; globally unused parameters retain `grad=None`.

## Data and lifecycle

The entry supports full causal-LM SFT and vanilla PEFT LoRA with map-style datasets, positive `max_steps`, AdamW and eager attention. Native V4.1 accepts text-only Swift batches; real image training is not validated. The previous V4 model plan is absent. Native LoRA accepts explicit attention targets such as `--target_regex 'model\.layers\.\d+\.attn\.(wq_a|wq_b|wkv|wo_b)'`; base weights remain frozen and adapter gradients are replicated and averaged before the optimizer update.

Native V4.1 LoRA supports gradient accumulation with CPU parameter offload. Full SFT and non-native LoRA still require accumulation 1 when parameters are offloaded. Native Full SFT on Ascend additionally requires `pin_memory=true`; native LoRA can use unpinned parameter shards. `host_pin_memory` controls Host Engram separately and otherwise inherits `pin_memory`.

Evaluation and saving accept `no`, `steps` or `epoch`. Evaluation weights by valid target tokens and excludes examples duplicated by distributed sampling. Native LoRA saves standard PEFT adapters. With `save_only_model=false`, it also saves optimizer, scheduler, per-rank RNG and consumed-sample state. Keep the dataset, topology, sequence configuration and trainable parameters fixed when resuming. Use `--adapters` for a model-only warm start. HF Full SFT retains its DCP save/resume lifecycle.

Pipeline parallelism, packing, padding-free input, streaming, DoRA, LoRA+, `modules_to_save` and arbitrary Swift plugins remain unsupported. Explicit rejection at startup does not mean these features passed usability acceptance.

For long native sequences, `model_kwargs` accepts `chunk_loss`, `chunk_loss_size`, `head_compute_dtype`, `attention_query_chunk_size` and `index_query_chunk_size`. Loss chunks count total tokens. The attention and indexer settings bound their query windows. Set `offload_activations=true` to keep saved recomputation inputs on the host.

For HF models, only rank zero loads original CPU weights; other ranks build on meta and load owner-local shards through DCP. Native V4.1 directly uses its native sharded initial-load API. Its raw FP8/FP4 checkpoint storage must be decoded separately before using this BF16 path.

## Validation

Run with the real Swift and optional FSDPTurbo dependencies:

```bash
python -m pytest -q tests/fsdpturbo
python -m torch.distributed.run --standalone --nproc_per_node=4 tests/fsdpturbo/distributed_loss.py
python -m torch.distributed.run --standalone --nproc_per_node=4 tests/fsdpturbo/distributed_replicated_gradients.py
```

The distributed commands use CPU/Gloo tensors. They establish metric and replica-gradient contracts, not accelerator model training. Actual model runs must separately verify initial weights, input/label hashes, loss/gradient curves, optimizer updates, topology, peak memory and clean exits. Results from earlier source pairs do not certify this pair.
