# v5 异步 rollout 的批次与发布契约

本文适用于 `dev-swift-v5` 的流式 GRPO、PPO，以及 `lmbda=1` 的 GKD/OPSD/MOPD 路径。

## 职责边界

- Swift 配方决定一次 optimizer update 消耗多少轨迹、何时发布权重，以及使用静态还是自适应部署。
- Twinkle `StreamingDriver` 负责准入、版本窗口、完成队列和训练交接；它接收发布回调，不拥有模型权重或训练算法。
- Twinkle sampler 负责可恢复的暂停与继续，checkpoint engine 负责权重传输。
- `swift.dev.recipe.streaming` 统一配方侧的容量规划和发布编排；`batching` 只负责已经组成的训练批次的 DP 负载均衡。

## 容量和每次取数是两个约束

设 `B = train_batch_size * gradient_accumulation_steps` 为一次更新的轨迹数，`G = num_generations`，`S = parameter_sync_step`，`L = max_staleness`：

```text
Q = lcm(B, G)
C = ceil(B * S / Q) * Q
groups_per_partition = C / G
buffer_depth = (L + 1) * C
max_rows_per_pull = Q（GRPO，完整组）或 B（PPO/蒸馏，逐样本）
```

`plan_streaming_batches` 返回一个包含上述三个 driver 参数的 `StreamingBatchPlan`，调用方不再分别计算容量与取数上限。分区按 prompt group 计数，容量和取数上限按轨迹计数。

例如 B=64、G=8、S=1 时，`one_step_off` 的容量为 128，每次交接仍为 64：训练当前批次期间可以继续生成下一批。`none` 对应 L=0、容量 64；`fully_async` 使用配置的 L。改变容量不会放宽 driver 的版本校验。

只扩大容量仍不够：如果一次取走全部已完成的 128 条，就会把两批合并到同一次交接中，推迟配方的权重发布。GRPO 在 B/G 不整除时以 Q 为最小完整交接单位，不能拆开组来强行凑成 B。

## 原地发布的顺序与失败语义

静态部署的 `in_place` 路径执行：

```python
sampler.pause_generation()
engine.sync_weights()
sampler.resume_generation()
```

暂停必须关闭重试入口，并等待在途请求离开 decode；只调用 abort 会让 partial rollout 在新权重到达前用旧权重重试。暂停或传输异常会向上传播，不能在 `finally` 中无条件恢复生成。配方异常路径负责关闭 driver 和 actors。

自适应部署委托 `ClusterAdapter.publish()` 完成其已有的暂停、资源调整、传输和恢复事务，不再外套一次静态发布。`adapter_snapshot` 仍使用 driver 的不可变快照路径。

Twinkle 的 abort/pause/resume/状态查询必须声明为控制 RPC，能够越过等待生成完成的屏障；否则中断指令本身会排在需要中断的生成后面。普通生命周期与权重操作仍遵循同步约束。

## 兼容性与验证

Swift 需要支持 `StreamingDriver(max_rows_per_pull=...)` 和 sampler 控制 RPC 的配套 Twinkle 版本，建议先合入并升级 Twinkle。Twinkle 未指定取数上限时保留原有行为；自定义 `assembly_ready` 必须自行管理批次，不能同时传入该上限。

回归测试覆盖五种算法的真实 driver 交接、同步/异步模式、稀疏发布、已完成的预取批次、partial rollout 恢复，以及发布各阶段失败。性能应按实际消费 token 计算，并分别查看生成等待、训练计算与权重发布；异步阶段计时可能重叠，不能直接相加。
