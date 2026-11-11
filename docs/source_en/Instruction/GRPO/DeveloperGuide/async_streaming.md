# Batch and publication contracts for v5 asynchronous rollout

This page covers streaming GRPO, PPO, and the `lmbda=1` GKD/OPSD/MOPD paths on `dev-swift-v5`.

## Responsibilities

- Swift recipes choose trajectories per optimizer update, publication cadence, and static or adaptive placement.
- Twinkle's `StreamingDriver` owns admission, version windows, completion queues, and training handoffs. It receives a publication callback; it does not own model weights or the training algorithm.
- Twinkle samplers implement resumable pause/resume; checkpoint engines transfer weights.
- `swift.dev.recipe.streaming` centralizes recipe-side capacity planning and publication orchestration. `batching` only balances an already assembled training batch across DP ranks.

## Capacity and pull size are separate constraints

Let `B = train_batch_size * gradient_accumulation_steps` be trajectories per update, `G = num_generations`, `S = parameter_sync_step`, and `L = max_staleness`:

```text
Q = lcm(B, G)
C = ceil(B * S / Q) * Q
groups_per_partition = C / G
buffer_depth = (L + 1) * C
max_rows_per_pull = Q (GRPO, whole groups) or B (PPO/distillation, per sample)
```

`plan_streaming_batches` returns these three driver settings together as a `StreamingBatchPlan`, so callers do not calculate capacity and pull limits independently. Partitions count prompt groups; capacity and pull limits count trajectories.

For B=64, G=8, S=1, `one_step_off` has capacity 128 but hands off 64 trajectories at a time, allowing the next batch to generate during training. `none` uses L=0 and capacity 64; `fully_async` uses its configured L. Capacity does not relax the driver's version validation.

Increasing capacity alone is insufficient: draining 128 completed trajectories in one handoff combines two batches and moves the recipe's publication boundary. When B and G do not divide evenly, GRPO uses Q as its minimum complete handoff rather than splitting groups to force a B-sized pull.

## In-place publication and failure semantics

Static placement uses this sequence for `in_place` publication:

```python
sampler.pause_generation()
engine.sync_weights()
sampler.resume_generation()
```

Pause closes the retry gate and waits for in-flight requests to leave decode. Abort alone lets partial rollouts retry on old weights before the new weights arrive. Pause or transfer failures propagate; generation must not resume unconditionally in a `finally` block. Recipe exception handling tears down the driver and actors.

Adaptive placement delegates to `ClusterAdapter.publish()`, which already owns pause, resource changes, transfer, and resume. It is not wrapped in another static publication transaction. `adapter_snapshot` retains the driver's immutable snapshot path.

Twinkle abort/pause/resume/status methods must be control RPCs that bypass generation-drain barriers. Otherwise an interruption waits behind the work it needs to interrupt. Ordinary lifecycle and weight operations retain their synchronization constraints.

## Compatibility and validation

Swift requires a matching Twinkle version with `StreamingDriver(max_rows_per_pull=...)` and sampler control RPCs; merge and upgrade Twinkle first. Omitting the pull limit retains Twinkle's existing behavior. A custom `assembly_ready` owns its batching and cannot be combined with this limit.

Regression coverage includes real driver handoffs for five algorithms, synchronous/asynchronous modes, sparse publication, completed prefetch, partial rollout resume, and failures in each publication phase. Measure performance using consumed tokens and distinguish generation wait, training compute, and publication. Asynchronous timer intervals may overlap and must not be summed as exclusive costs.
