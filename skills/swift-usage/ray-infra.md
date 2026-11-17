# Ray mode and infra: placement, distribution, load balancing

## The three modes (DistributedConfig.mode)

| mode | What runs where | Typical use |
| --- | --- | --- |
| `local` (default) | torchrun spawns N processes; each runs the same recipe with its own `twinkle.initialize(mode='local')`; world size from env | single/multi-GPU SFT, Megatron TP/PP groups |
| `ray` | one GPU-less driver orchestrates Ray actors; components placed by DeviceGroup | RL (always), multi-resident topologies (model + sampler + reward models) |
| `remote` | this process is an HTTP client; training/sampling live in a twinkle-server | server-client mode (server-client.md) |

Facts that prevent misconfiguration:

- **RL is always Ray**: `swift rl` forces `mode=ray` for all 12 `rlhf_type`s (the online family
  additionally forces `use_vllm` + `vllm_mode=colocate`). Passing `--mode local` is silently
  overridden — don't fight it.
- **`--nproc_per_node` is REQUIRED under ray** (and remote): the driver holds no GPU and has no
  WORLD_SIZE to infer from.
- Under `ray`, `--model`'s parallel layout comes from `parallel_spec` / TP/PP/CP/EP flags, not
  from the launcher.

## Placement: the only infra surface you touch in code

```python
import twinkle
from twinkle import DeviceGroup, DeviceMesh

device_groups = [
    DeviceGroup(name='model',   ranks=[0, 1, 2, 3], device_type='GPU'),
    DeviceGroup(name='sampler', ranks=[4, 5, 6, 7], device_type='GPU'),
]
mesh = DeviceMesh.from_sizes(world_size=4, dp_size=4)     # or from_spec('tp2dp2')
twinkle.initialize(mode='ray', nproc_per_node=8, groups=device_groups)

model   = TransformersModel(model_id=..., device_mesh=mesh, remote_group='model')
sampler = vLLMSampler(model_id=..., device_mesh=sampler_mesh, remote_group='sampler')
```

`DeviceGroup` = a named set of GPUs (what a resident component claims); `DeviceMesh` = how
those ranks shard (TP/PP/DP/EP...). Groups are disjoint by design — that is how a run keeps
the trainer, the sampler, and each frozen auxiliary model (reference/teacher/reward/critic) on
separate GPUs without hand-placing anything.

The design rule that keeps your code readable: **the only legitimate infra touchpoints are
`twinkle.initialize` (+ device mesh setup) and the `@remote_class` / `@remote_function`
decorators.** Read driver/recipe code as if it were a single-machine program — call methods,
get results, pass them downstream. Ray, pools, routing, and replicas stay behind the
decorators. Never call `ray.get` / `ray.kill` / worker-pool APIs directly from business code.

## Cross-process calls: @remote_class / @remote_function

- `@remote_class()` on a class (or a factory returning one): its methods become callable from
  the driver; arguments and returns must be serializable — pass names/strings/scalars, not
  live objects.
- `@remote_function` on free functions: same, per-method wrappers.
- **The driver-side handle is a PROXY**: it has no real modules. Anything that needs the
  model's actual structure (module names, weight shapes) must run worker-side; from the driver
  you only ever pass serializable candidates and read serializable results.
- In `mode='local'` the same decorated calls run in-process — the decorators are what make
  code portable between local and Ray without branches.

## Load balancing and autoscaling

Three planes, all declarative — you configure, infra routes:

**Training-batch plane (`--dp_balance true`, a `TrainConfig` flag; default off).** For RL
training batches, every per-sample training signal rides the trajectories, so reordering
the trajectory list is the whole permutation: with the flag on, the recipe reorders each
optimizer batch so the DP ranks get even token workloads (metrics land under `dp_balance/*`).
Off by default — the reordering is a pure wall-clock optimization, gradients are unchanged.
One exception to remember: the PPO value model always trains in the original order (its
rank-local token mean is not permutation-invariant).

**In-process / actor plane (WorkerPool, inside twinkle.infra).** Every resident component
built through the standard path gets this for free, no code of yours involved:

- Requests route **least-in-flight**: the worker with the fewest outstanding requests takes
  the next one; per-request sharding hashes `request_id` deterministically (same request ->
  same worker), batches split evenly across workers.
- Only workers in `ACTIVE` state (and not suspended) receive work.
- **Elastic replicas**: pools grow/shrink through their declared configuration; scale-out
  adds replicas, scale-in retires them and drains their in-flight work.

**Server plane (Ray Serve, in the twinkle-server YAML).** Per application `deployments`:

```yaml
deployments:
  - name: TinkerCompatServer
    max_ongoing_requests: 50                 # per-replica in-flight cap
    autoscaling_config:
      min_replicas: 1
      max_replicas: 4
      target_ongoing_requests: 128           # scale signal
    ray_actor_options:
      num_cpus: 0.1
```

- `proxy_location: EveryNode` puts a gateway replica on every node (front-of-node routing for
  multi-node clusters).
- Sampler apps with `dp_size > 1` shard batches across their own workers (the even-split /
  request-hash rules above).
- `route_prefix` is the external mount point — everything clients see is decided there.

## RL topology specifics

- `vllm_mode=colocate` (default for the online family): sampler shares the trainer's GPUs;
  weight sync is CUDA-IPC with a sleep/wake device hand-over each step.
  `--vllm_gpu_memory_utilization` must leave room for training.
- `vllm_mode=disaggregated`: the sampler gets its OWN GPUs (NCCL weight sync); required for
  `--async_mode` (one_step_off / fully_async), which overlaps rollout with training.
- Weight transport between trainer and sampler goes through the checkpoint engine
  (`CheckpointEngineManager`: direct / Broadcast / IPC / NCCL / HCCL), never through the
  driver — peak memory stays independent of model size.

## Gotchas

- `--nproc_per_node` required under `ray`/`remote` (driver holds no GPU).
- Colocated vLLM + training on the same GPUs: budget `gpu_memory_utilization` for both.
- A component's `remote_group` must name a group you declared in `initialize` — mismatched
  names fail at placement, not silently.
- Don't reach for `ray.get`/`ray.kill`/pool APIs in experiment code; if a capability seems to
  need them, it belongs in a component behind `@remote_class` (or already exists — check
  components.md first).
