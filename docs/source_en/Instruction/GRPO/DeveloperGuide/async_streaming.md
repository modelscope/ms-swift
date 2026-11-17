# Batches and weight sync for v5 streaming rollout

This page covers streaming GRPO, PPO, and the on-policy (`lmbda>0`) GKD/OPSD/MOPD paths on `dev-swift-v5`. RFT builds one driver per round; infer uses the same driver without a checkpoint manager.

## Responsibilities

- Swift recipes choose the trajectories per batch, when to call `checkpoint_manager.sync_weights()`, when to put a colocated sampler to sleep, and how advantages are computed.
- Twinkle's `StreamingDriver` draws prompts in order, admits a group of `num_generations` episodes per prompt, advances them with `rollout.step`, scores finished episodes with the orms/prms, queues each group once complete, and hands out `(trajs, samples)` batches. It reads the checkpoint manager's version to pace admission and drop groups that fell behind; it holds no weights and implements no training algorithm.
- `MultiTurnRollout.step` advances one generation (and that turn's tool calls); the last step returns the trajectory with its token record (`input_ids` / `labels` / `logprobs` ...).
- `CheckpointEngineManager` is a single shared instance; every completed `sync_weights()` adds one to its version.
- Samplers provide `pause_generation` / `resume_generation` for the weight transfer.

## Batch size

Let `B = train_batch_size * gradient_accumulation_steps` be the trajectories of one update and `G = num_generations`:

```text
disaggregated: batch_size = lcm(B, G)
colocate:      batch_size = ceil(generation_batch_size * G / lcm(B, G)) * lcm(B, G)
```

A batch is always whole groups and whole optimizer updates (`stream_batch_size`). Colocate pays a sleep, a wake-up and a weight sync per batch, so its batch grows to one generation batch to amortize them. The driver's `max_steps` counts batches; recipes convert optimizer steps to batches before passing it.

## Admission and staleness

Let `K = parameter_sync_step` and `L = max_staleness` (0 for `none`, 1 for `one_step_off`, the configured value for `fully_async`):

```text
capacity = (version - base_version + L + 1) * K * batch_size
```

The driver keeps in-flight plus ready trajectories within `capacity`, where `base_version` is the version when the driver started; each new version lets K more batches in. A ready group with `version - (earliest admission version in the group) > L` is dropped and counted in `dropped_stale`.

Without a checkpoint manager the version is "batches handed out // K" and only paces admission; the weights never change, so no group is dropped.

For B=64, G=8, K=1, `one_step_off` has batch size 64 and capacity 128: the next batch generates while the current one trains. `none` has capacity 64, so a batch is fully generated before it trains.

## Weight sync

Recipes call `sync_weights()` once every K batches, before pulling the next batch. In standalone mode it runs:

```python
sampler.pause_generation()   # the engine freezes running requests in place; failing that, waits for them
# transfer the weights
sampler.resume_generation()  # running requests continue on the new weights
```

An episode may therefore span versions. `sample.policy_version` is the version it was admitted under, and `version_span` is the version at hand-out minus `policy_version`.

In colocate mode the recipe puts the sampler to sleep once it has a batch and trains; the sync wakes the sampler and transfers the weights.

Adaptive placement (`ClusterAdapter`) lends the trainer's idle devices to the sampler as extra replicas while the trainer waits for a batch, and reclaims them before the batch is handed out; a new replica loads the current weights through `CheckpointEngineManager.warm` before it takes requests.

## Metrics

- `policy_version`: completed syncs; it should keep advancing in an async run.
- `version_span_mean` / `max_partial_span`: the batch's version skew, bounded by `max_staleness`.
- `partial_ratio`: the share of the batch with a version skew above 0.
- `dropped_stale`: samples dropped for falling behind.
- `trainer_idle_ratio`: the share of time the trainer waited for a batch.
