# Ray-based Megatron RLHF examples (GKD & GRPO)

On-policy RL/distillation on top of Megatron, orchestrated by Ray. The student is
trained with Megatron, generates completions with vLLM, and — for GKD — is distilled with a teacher model.

## How to run

```bash
# via the helper scripts
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/ray/gkd/run.sh

# or directly
megatron rlhf --use_ray true --config examples/ray/gkd/rollout_colocate_teacher_colocate.yaml
```

The YAML is split into a top-level section (shared args) and per-role groups
(`train`, `rollout`, and optionally `teacher`). Each group's `gpus:` field sets how
many GPUs that role uses; `CUDA_VISIBLE_DEVICES` must expose at least the total
number of GPUs the chosen placement needs (see below).

The `gkd/` folder ships three ready-to-run configs. The file name encodes the two
independent choices — **rollout placement** and **teacher mode**:

| file | rollout | teacher |
|------|---------|---------|
| `rollout_colocate_teacher_colocate.yaml` | colocate (shares train GPUs) | colocated (shares train GPUs) |
| `rollout_separate_teacher_colocate.yaml` | separate (own GPUs) | colocated (shares train GPUs) |
| `rollout_colocate_teacher_standalone.yaml` | colocate (shares train GPUs) | standalone vLLM replicas (own GPUs) |

---

## 1. GPU placement: colocate vs separate

This is controlled by `colocate_groups` plus each role's `gpus`.

| Placement | `colocate_groups` | GPUs needed | When to use |
|-----------|-------------------|-------------|-------------|
| **colocate** | `[[train, rollout]]` | `train.gpus` — all roles in the group **must** set the same `gpus` (one shared set) | default; fewer GPUs, train and rollout time-share the same devices |
| **separate** | *omit* | `train.gpus + rollout.gpus` (disjoint sets) | more GPUs, rollout overlaps with training |

- **colocate** — train and rollout live on the *same* devices and take turns.
  Set `offload_model`/`offload_optimizer`
  (+ `offload_teacher_model` for GKD) and `sleep_level: 1` so the idle role releases
  GPU memory to the active one.
  Example: `train.gpus=4`, `rollout.gpus=4`, `colocate_groups: [[train, rollout]]`
  → 4 GPUs total, with TP2 giving **DP2**.

- **separate** — train and rollout occupy *disjoint* GPU sets; weights are pushed to
  the rollout engine every step.

---

## 2. Teacher modes (GKD only)

Pick exactly one. `gkd_logits_topk: K` selects top-k distillation;
omit it for full-vocab distillation.

| Mode | How to configure | top-k | full-vocab | Status |
|------|------------------|:-----:|:----------:|--------|
| **Colocated `teacher_model`** | set top-level `teacher_model:` (+ `offload_teacher_model: true`) | ✅ | ✅ | supported |
| **Standalone teacher replicas** | add a `teacher:` group with `gpus`, `model`, and `vllm_engine_kwargs.max_logprobs` | ✅ | ❌ | supported |
| **External teacher API server** (`teacher_model_server`) | — | – | – | ❌ not supported — the Ray pipeline raises `NotImplementedError`; no plan to add it, no example shipped |

### 2a. Colocated teacher (`rollout_colocate_teacher_colocate.yaml`, `rollout_separate_teacher_colocate.yaml`)
The teacher shares the **train** GPUs and is offloaded to CPU between teacher
forwards. It is the only mode that supports full-vocab distillation, and it works
with both colocate and separate rollout placements.

```yaml
teacher_model: Qwen/Qwen3.5-4B
offload_teacher_model: true
gkd_logits_topk: 64      # omit for full-vocab
```

### 2b. Standalone teacher replicas (`rollout_colocate_teacher_standalone.yaml`)
The teacher runs as its own set of Ray-managed vLLM replicas on **separate** GPUs and
returns prompt top-k logprobs; the driver fetches them per step.

```yaml
gkd_logits_topk: 64                 # REQUIRED — replicas are top-k only
# do NOT set top-level teacher_model here (that would also load a colocated teacher)
teacher:
  gpus: 4
  model: Qwen/Qwen3.5-4B            # the teacher checkpoint these replicas serve
  vllm_engine_kwargs: {"max_logprobs": 64}   # MUST be >= gkd_logits_topk
```
- `max_logprobs` must be `>= gkd_logits_topk`, or vLLM rejects the `prompt_logprobs`
  request.
- GPUs needed = colocated train+rollout set **+** `teacher.gpus`.

---

## 3. top-k vs full-vocab distillation

- **top-k** (`gkd_logits_topk: K`): the teacher exposes only the top-K logprobs per
  position. Much lower memory, works for every teacher mode.
- **full-vocab** (omit `gkd_logits_topk`): distill the full vocabulary distribution.
  Colocated teacher only, and **memory-heavy at TP>1** (caches per-rank vocab-sharded
  teacher logits). If you OOM: switch to top-k, lower `micro_batch_size`, or lower
  `vllm_gpu_memory_utilization`.

---

## 4. OPSD (On-Policy / privileged Distillation)

OPSD lets the teacher see a *different* (privileged) prompt than the student while
scoring the **same** on-policy response — e.g. the teacher sees the problem + a
reference solution. A dataset preprocessor (loaded via `external_plugins`) emits a
per-row `teacher_prompt`; the loss aligns the shared response tokens by mask.

```yaml
external_plugins: examples/train/rlhf/opsd/opsd_plugin.py   # registers teacher_prompt
teacher_model: Qwen/Qwen3.5-4B
gkd_logits_topk: 64
```
- Supported in Ray for **colocated teacher + top-k** only (full-vocab OPSD and
  standalone-teacher OPSD are not supported yet).
- No extra flag is needed: OPSD activates automatically when rows carry a non-empty
  `teacher_prompt`; otherwise training falls back to plain GKD.

---

## 5. Things to know (common knobs & pitfalls)

- **Sequence length**: the encoder budget is `max_length + max_completion_length`
  (prompt is capped at `max_length`, the on-policy completion adds up to
  `max_completion_length`). Size `vllm_max_model_len` accordingly.
- **`padding_free: true`** packs a micro-batch into one sequence; pair with
  `sequence_parallel: true` when `tensor_model_parallel_size > 1`.
- **Parallelism / DP**: data parallel size = `gpus / (TP * PP * CP)`. e.g. 4 GPUs
  with `tensor_model_parallel_size: 2` → DP2.
- **Memory release (colocate)**: `offload_model`, `offload_optimizer`,
  `offload_teacher_model`, and `sleep_level: 1` are what let colocated roles fit.
- **vLLM memory**: colocate rollout `~0.4`; separate rollout `~0.6` (weight-sync
  headroom); standalone teacher replicas can go high (`~0.9`).
- **GRPO specifics**: rewards via `reward_funcs` + `external_plugins`; sampling via
  `num_generations` / `steps_per_generation`; no `teacher_*` settings.

## Validated reference
The GKD examples were validated with student **Qwen3.5-2B** / teacher **Qwen3.5-4B**
on 4–8×H20, train **TP2 → DP2**, across all three placement/teacher combinations.
