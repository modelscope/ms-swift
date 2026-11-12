# v5 流式 rollout 的批次与权重同步

本文适用于 `dev-swift-v5` 的流式 GRPO、PPO，以及 `lmbda>0` 的 GKD/OPSD/MOPD on-policy 路径。RFT 每一轮建一个 driver；infer 也用同一个 driver（不传 checkpoint manager）。

## 职责边界

- Swift 配方决定每批多少条轨迹、何时调用 `checkpoint_manager.sync_weights()`、colocate 下何时让 sampler 休眠，以及 advantage 的计算。
- Twinkle `StreamingDriver` 按顺序抽取 prompt，每个 prompt 准入一组 `num_generations` 个 episode，用 `rollout.step` 逐轮推进，结束后调用 orms/prms 打分；整组完成后进入就绪队列，按批交出 `(trajs, samples)`。它读取 checkpoint manager 的 version 来控制准入和丢弃过旧的组，不持有模型权重，也不实现训练算法。
- `MultiTurnRollout` 每次 `step` 推进一轮生成（含这一轮的工具调用）；最后一轮返回带 token 记录（`input_ids` / `labels` / `logprobs` 等）的轨迹。
- `CheckpointEngineManager` 是全局唯一的实例：每完成一次 `sync_weights()`，version 加一。
- sampler 提供 `pause_generation` / `resume_generation`，供权重传输期间暂停生成。

## 批大小

设 `B = train_batch_size * gradient_accumulation_steps` 为一次更新的轨迹数，`G = num_generations`：

```text
disaggregated: batch_size = lcm(B, G)
colocate:      batch_size = ceil(generation_batch_size * G / lcm(B, G)) * lcm(B, G)
```

一批总是由完整的组和完整的 optimizer update 组成（`stream_batch_size`）。colocate 每批都要做一次休眠、唤醒和权重同步，所以把批放大到一个 generation batch 来摊薄这部分开销。driver 的 `max_steps` 按批计数，配方把 optimizer 步数换算成批数后再传入。

## 准入与过期

设 `K = parameter_sync_step`，`L = max_staleness`（`none` 为 0，`one_step_off` 为 1，`fully_async` 取配置值）：

```text
capacity = (version - base_version + L + 1) * K * batch_size
```

driver 保证在途和就绪的轨迹总数不超过 `capacity`，其中 `base_version` 是 driver 启动时的 version。version 每加一，就多放行 K 批。就绪组如果满足 `version - 组内最早准入版本 > L`，就会被丢弃，并计入 `dropped_stale`。

没有 checkpoint manager 时，version 取「已交出批数 // K」，只用来控制准入节奏；这时权重不会变化，所以不丢弃任何组。

例如 B=64、G=8、K=1 时，`one_step_off` 的批大小是 64、容量是 128：训练当前批次的同时，可以继续生成下一批。`none` 的容量是 64，即先生成完一批再训练。

## 权重同步

配方每交出 K 批后调用一次 `sync_weights()`，并且在拉取下一批之前完成。standalone 模式下的顺序是：

```python
sampler.pause_generation()   # 优先让引擎把运行中的请求原地冻结；不支持时等待它们结束
# 传输权重
sampler.resume_generation()  # 运行中的请求在新权重上继续
```

因此一个 episode 可能跨越多个版本。`sample.policy_version` 记录的是准入时的版本，`version_span` 等于交出时的 version 减去 `policy_version`。

colocate 模式下，配方拿到一批后让 sampler 休眠、开始训练，到同步时由 manager 唤醒 sampler 并传输权重。

自适应部署（`ClusterAdapter`）在 trainer 等待批次期间，把空闲的卡借给 sampler 增加副本，交出批次前收回；新副本先由 `CheckpointEngineManager.warm` 加载当前权重，再开始接收请求。

## 指标

- `policy_version`：已完成的同步次数，异步训练时应逐步增长。
- `version_span_mean` / `max_partial_span`：本批的版本差，上界为 `max_staleness`。
- `partial_ratio`：本批中版本差大于 0 的样本占比。
- `dropped_stale`：因过期被丢弃的样本数。
- `trainer_idle_ratio`：trainer 等待批次的时间占比。
