# NPU GRPO vLLM Group Lifecycle 根修演进方案

## 0. 当前结论

当前基线：

```text
worktree: /data/zyh/code/ms-swift-npu-grpo-dp-subgroup
branch:   npu-grpo-vllm-dp-subgroup-clean
commit:   92363b57e Fix NPU GRPO vLLM DP subgroup initialization
```

这版已经从旧的：

```text
default world collective + dp_ranks projection
```

演进为：

```text
真实预创建 vLLM DP HCCL subgroup
+ vLLM GroupCoordinator 复用该 DP subgroup
+ rollout object/control 走 Gloo
+ tensor/payload 走 HCCL
```

因此后续不应回退到 world reconstruction fallback。后续目标是继续收敛 **vLLM group lifecycle**：

```text
Megatron 已有 group 优先复用
缺失的 vLLM group 全 rank 同序预创建
GroupCoordinator cache-only
CPU/control group 正规化
```

## 1. 设计目标

最终目标：

```text
Megatron 初始化完成
  ↓
进入通信静默区
  ↓
收集 Megatron 已有 HCCL group candidates
  ↓
构建 vLLM 完整 group specs
  ↓
校验所有 ranks 看到的 specs 完全一致
  ↓
为每个 spec 生成全 rank 一致的 execution plan：
    - reuse Megatron group
    - or precreate new vLLM group
  ↓
全 rank 按 execution plan 同序执行
  ↓
group warmup + cache
  ↓
GroupCoordinatorPatch.__init__ 只 lookup cache
  ↓
CPU/control group 恢复真实 Gloo 或独立 control plane
```

核心不变量：

```text
1. GroupCoordinatorPatch.__init__ 不应承担 group 创建职责。
2. 所有 new_group 必须集中在 group factory 中。
3. 所有 ranks 必须看到相同的 desired specs。
4. 所有 ranks 必须看到相同的 execution plan。
5. 所有 ranks 必须按同一顺序处理每个 group spec。
6. 非成员 rank 也必须参与 precreate 型 new_group 调用。
7. cache miss 必须 fail loud，不能在 GroupCoordinator 内临时补建。
8. metadata/control 和 tensor/payload 通信路径必须分离。
```

## 2. 当前基线定位

`92363b57e` 是新的安全基线，不是最终根修。

它已经解决或规避了：

```text
1. vLLM DP HCCL subgroup 后置动态创建不稳定。
2. rollout object collective 在 HCCL TP group 上 metadata 损坏。
3. vLLM DP CPU tensor sync 误跑 HCCL 的问题。
4. mcore-bridge metadata/control 和 payload 混跑 HCCL 的问题。
5. non-quant MoE routing custom op 异步失败问题。
```

但仍有剩余技术债：

```text
1. 非 DP 的部分 vLLM NPU-only groups 仍可能动态 new_group。
2. 部分 cpu_group 仍然 alias 到 HCCL device_group。
3. 尚未优先复用 Megatron 已存在的 HCCL groups。
4. 尚未形成完整 group factory。
5. GroupCoordinator 还没有完全 cache-only。
6. CPU/control group 还没有完全恢复真实 Gloo 或独立 control plane。
```

因此后续演进原则是：

```text
不推倒当前基线。
不回退 world fallback。
不一次性大改 CPU/control plane。
先 audit，再 registry，再 factory，再 cache-only，再 CPU 正规化。
```

## 3. 阶段拆分

推荐拆成 5 个阶段。

```text
P0: group creation audit
P1: Megatron HCCL group registry
P2: vLLM device group factory
P3: GroupCoordinator device cache-only
P4: CPU/control group 正规化
```

每个阶段必须满足：

```text
1. 可以独立开关。
2. 可以单独验证。
3. 失败时能回退到 92363b57e。
4. 不改变旧基线的已验证行为，除非显式开启新开关。
```

建议环境变量：

```bash
SWIFT_VLLM_ASCEND_GROUP_AUDIT=0/1
SWIFT_VLLM_ASCEND_REUSE_MEGATRON_GROUPS=0/1
SWIFT_VLLM_ASCEND_GROUP_FACTORY=0/1
SWIFT_VLLM_ASCEND_GROUP_CACHE_ONLY=0/1
SWIFT_VLLM_ASCEND_PRECREATE_GLOO_GROUPS=0/1
SWIFT_VLLM_ASCEND_ALLOW_CPU_GROUP_ALIAS=0/1
```

默认策略：

```text
92363b57e 行为保持默认。
新 root-fix 功能先显式开启。
验证稳定后再逐步改默认值。
```

## 4. P0：group creation audit

### 4.1 目的

先确认当前还剩哪些动态 `new_group`。

不要先重构。先用 audit 回答三个问题：

```text
1. 哪些 group 仍在 GroupCoordinatorPatch.__init__ 里动态创建？
2. 哪些 group 可以 exact-order 复用 Megatron 已有 HCCL group？
3. 哪些 group 即使所有 ranks 同序进入也可能卡在 backend/store？
```

### 4.2 日志字段

建议统一日志格式：

```text
[swift-group-create-enter]
seq=...
rank=...
world_size=...
backend=hccl/gloo
kind=device/cpu/control
group_name=world/tp/dp/pp/ep/cp/pcp/mc2/bridge_pp/bridge_ep_pp
ranks=[...]
source=megatron_reuse/vllm_precreate/groupcoordinator_dynamic/bridge_control
phase=megatron_init/vllm_preinit/vllm_groupcoordinator/rollout/bridge
```

退出日志：

```text
[swift-group-create-exit]
seq=...
rank=...
backend=...
group_name=...
ranks=[...]
elapsed_ms=...
```

失败日志：

```text
[swift-group-create-error]
seq=...
rank=...
backend=...
group_name=...
ranks=[...]
error=...
```

### 4.3 判定

```text
类型 1：某些 rank 没有 enter 同一个 seq
=> 初始化路径分叉 / 创建顺序不一致。

类型 2：所有 rank enter 了，但 group_name/backend/ranks 不一致
=> group spec 计算不一致。

类型 3：所有 rank enter 完全一致，但卡在 new_group 内
=> backend/store/Gloo-HCCL 混合实现问题。

类型 4：enter/exit 成对，但后续 collective hang
=> group 创建已过，问题在运行期 collective sequence 或 phase 交错。
```

### 4.4 注意

audit 必须只在显式开启时生效：

```bash
SWIFT_VLLM_ASCEND_GROUP_AUDIT=1
```

不要默认向主日志灌大量 group 创建信息。

## 5. P1：Megatron HCCL group registry

### 5.1 目的

优先复用 Megatron 已经创建过的 HCCL groups，减少 vLLM 在 external_launcher 里重复创建相同 subgroup。

但注意：

```text
Megatron group 和 vLLM group 不一定语义一致。
只有 exact rank order match 才允许复用。
```

例如：

```text
Megatron: TP=2 PP=2 CP=2 EP=4
vLLM:     TP=2 DP=4
world=8

vLLM DP:
[0,2,4,6]
[1,3,5,7]
```

Megatron 不一定有 rank 顺序完全一致的 DP group。因此复用必须是机会型优化，而不是假设。

### 5.2 registry 结构

建议 registry 以 ranks 为主键，axis 作为 provenance metadata。

```python
@dataclass(frozen=True)
class MegatronGroupRecord:
    backend: str              # "hccl"
    kind: str                 # "device"
    axis: str                 # "world" / "tp" / "dp" / "pp" / "cp" / "ep" / ...
    ranks: tuple[int, ...]    # exact rank order
    group: object             # ProcessGroup or WORLD
    source: str               # megatron_mpu / default_world / bridge / etc.
```

主 lookup key：

```python
(
    backend,
    kind,
    tuple(ranks),
)
```

`axis` 不建议作为强匹配条件，否则可能错过可安全复用的 same-rank group。但日志必须保留 axis/source，方便 reviewer 和 debug。

### 5.3 全 rank 一致的 reuse 决策

这里要特别小心：**不能让某些 rank 判断 reuse，另一些 rank 判断 precreate**。

对每个 vLLM desired group spec，生成 reuse decision：

```python
local_reuse_ok = (
    rank not in spec.ranks
    or has_exact_order_megatron_group(spec.ranks)
)

flag = torch.tensor(
    [1 if local_reuse_ok else 0],
    device=npu,
    dtype=torch.int32,
)
dist.all_reduce(flag, op=dist.ReduceOp.MIN)

can_reuse = bool(flag.item())
```

含义：

```text
非成员 rank 对 reuse decision 置 neutral 1。
所有成员 rank 都有 exact-order Megatron group，才允许 reuse。
只要任意成员 rank 没有 exact match，就全 world 决定 precreate。
```

这样所有 ranks 会得到相同的：

```text
reuse or precreate
```

避免 execution plan 分叉。

### 5.4 复用时的 phase boundary

复用 Megatron group 比新建 group 更敏感，因为它共享已有 communicator。

必须保证：

```text
Megatron 不在同一个 group 上同时发起 collective。
vLLM 不在 Megatron collective 未完成时复用该 group。
```

进入 vLLM group init / rollout 前：

```python
# 伪代码
wait_all_known_megatron_async_work()
torch.npu.synchronize()
dist.barrier()
```

退出 rollout 回到训练前：

```python
torch.npu.synchronize()
dist.barrier()
```

如果无法证明 phase 隔离，优先不要开启 Megatron group reuse。

### 5.5 开关

```bash
SWIFT_VLLM_ASCEND_REUSE_MEGATRON_GROUPS=1
```

建议初期只对：

```text
world group
TP group
```

开放复用；DP/EP/CP 等复杂组合后续再扩大。

## 6. P2：vLLM device group factory

### 6.1 目的

把 vLLM 需要的 device groups 在 LLMEngine 初始化前统一处理：

```text
能 reuse 的 reuse。
不能 reuse 的全 rank 同序 precreate。
```

GroupCoordinator 不再负责 device group 创建。

### 6.2 VLLMGroupSpec

建议定义：

```python
@dataclass(frozen=True)
class VLLMGroupSpec:
    name: str                 # "world" / "tp" / "dp" / "pp" / "ep" / "pcp" / "mc2"
    index: int
    backend: str              # "hccl" / "gloo"
    kind: str                 # "device" / "cpu" / "control"
    ranks: tuple[int, ...]
    required: bool = True
```

不要把 `source="reuse/precreate"` 放进 desired spec。原因是：

```text
desired spec 描述 vLLM 需要什么 group；
source 描述这个 group 如何获得。
```

应分两层：

```python
desired_specs: list[VLLMGroupSpec]
execution_plan: list[VLLMGroupExecution]
```

```python
@dataclass(frozen=True)
class VLLMGroupExecution:
    spec: VLLMGroupSpec
    action: str               # "reuse" / "precreate"
    reuse_source: str | None
```

### 6.3 specs hash

在任何 `new_group` 前，先校验 desired specs。

```text
hash(desired_specs) 在所有 ranks 上必须一致。
```

校验内容：

```text
name
index
backend
kind
ranks exact order
required
```

不一致直接 fail loud，并打印本 rank specs。

之后再校验 execution plan：

```text
hash(execution_plan) 在所有 ranks 上必须一致。
```

这样可以分别定位：

```text
desired specs 不一致
```

还是：

```text
reuse/precreate 决策不一致
```

### 6.4 precreate 规则

错误模式：

```python
if rank in ranks:
    dist.new_group(ranks, backend="hccl")
```

正确模式：

```python
for execution in execution_plan:
    spec = execution.spec

    if execution.action == "reuse":
        group = lookup_megatron_group(spec)
        if rank in spec.ranks:
            cache.put(spec, group)

    elif execution.action == "precreate":
        group = dist.new_group(
            ranks=list(spec.ranks),
            backend=spec.backend,
            timeout=GROUP_CREATE_TIMEOUT,
        )
        if rank in spec.ranks:
            cache.put(spec, group)

    if rank in spec.ranks:
        warmup_group(spec, cache.get(spec))

    dist.barrier()
```

这里 `dist.barrier()` 很重要。原因是：

```text
成员 ranks 可能正在 warmup subgroup。
非成员 ranks 如果直接进入下一个 new_group，会再次造成跨 process group 顺序交错。
```

init 阶段开销可以接受，安全性优先。

### 6.5 warmup

HCCL device group warmup：

```text
all_reduce int32 scalar
all_gather small int32 tensor
```

Gloo CPU group warmup：

```text
all_reduce CPU int32 scalar
可选 broadcast_object_list 小 object
```

warmup 失败直接报错，不进入 vLLM runtime。

### 6.6 group cache key

cache key 建议包含：

```python
(
    kind,             # device / cpu / control
    backend,          # hccl / gloo
    name,             # dp / tp / world / ...
    tuple(ranks),     # exact rank order
)
```

lookup 时必须 exact match。

## 7. P3：GroupCoordinator cache-only

### 7.1 目标

`GroupCoordinatorPatch.__init__` 不再调用：

```python
dist.new_group(...)
```

只允许：

```python
group_cache.lookup(...)
```

### 7.2 device cache-only 逻辑

```python
device_group = group_cache.lookup(
    kind="device",
    backend=torch_distributed_backend,
    name=group_name,
    ranks=tuple(ranks),
)

if device_group is None:
    raise RuntimeError(
        "vLLM device group was not precreated: "
        f"group_name={group_name}, backend={torch_distributed_backend}, ranks={ranks}"
    )
```

如果当前阶段只做 device cache-only，CPU group 可以暂时保留旧逻辑，但必须 audit：

```text
source=groupcoordinator_dynamic_cpu
```

### 7.3 cache-only 准入标准

只有满足以下条件才开启：

```text
1. group audit 已确认所有 device groups 均进入 factory。
2. desired specs hash 一致。
3. execution plan hash 一致。
4. precreate/reuse enter-exit 成对。
5. warmup 通过。
6. 原 10/10 smoke 通过。
```

开启开关：

```bash
SWIFT_VLLM_ASCEND_GROUP_CACHE_ONLY=1
```

初期建议：

```text
cache-only 只管 device groups。
CPU groups 暂不强制 cache-only。
```

等 P4 完成后再：

```text
device + CPU/control 全部 cache-only。
```

## 8. P4：CPU/control group 正规化

当前最大抽象债务：

```text
部分 cpu_group = HCCL device_group
```

短期可用，但不能作为最终形态。

### 8.1 首选：真实 Gloo CPU group 进入 factory

把 CPU/Gloo group 也纳入 `desired_specs`：

```text
world cpu/gloo
tp cpu/gloo
dp cpu/gloo
必要的 pp/ep/pcp/mc2 cpu/gloo
```

同样执行：

```text
desired specs hash
execution plan hash
全 rank 同序 precreate
warmup
cache
GroupCoordinator lookup
```

如果通过：

```text
去掉 cpu_group = device_group
逐步删除 CPU tensor -> NPU tensor 的兼容 patch
```

### 8.2 如果真实 Gloo 仍卡

如果所有 ranks 全序进入 Gloo `new_group` 仍然卡，说明问题更接近：

```text
Gloo/default store/Ascend colocate 混合实现问题
```

此时不要再 alias 到 HCCL 作为长期解，改为显式独立 control plane：

候选：

```text
独立 TCPStore
独立 socket control channel
vLLM StatelessProcessGroup
SWIFT 专用 swift_cpu_control_group
```

注意：

```text
StatelessProcessGroup 不能无脑替代所有 torch.distributed CPU group 用法。
```

必须逐个审计：

```text
broadcast_object_list
send_object / recv_object
CPU tensor metadata
MessageQueue / broadcaster
```

原则：

```text
object / metadata / scalar control -> Gloo 或 independent control plane
NPU tensor / weight payload -> HCCL
```

## 9. group factory 执行顺序建议

建议顺序：

```text
1. world device
2. tp device
3. dp device
4. pp / pcp device
5. ep / mc2 device
6. world cpu
7. tp cpu
8. dp cpu
9. pp / pcp cpu
10. ep / mc2 cpu
```

同类 group 内按：

```text
index 升序
ranks tuple 字典序
```

排序。

不要依赖 Python dict 遍历顺序。所有 specs 必须显式排序。

## 10. 失败策略

### 10.1 audit 阶段失败

如果 audit 发现动态 new_group 顺序不一致：

```text
保留 92363b57e。
不推进 cache-only。
先修 specs 构建或初始化路径分叉。
```

### 10.2 Megatron reuse 失败

如果 exact-order reuse 不命中：

```text
不报错。
走 precreate。
```

如果 reuse 命中但 warmup 失败：

```text
禁用 reuse。
回退 precreate。
记录该 group blacklist。
```

### 10.3 precreate 失败

如果所有 ranks 同序进入仍卡：

```text
保留 92363b57e DP-only 预创建逻辑。
不要启用完整 factory。
保留 audit 日志定位具体 group。
```

### 10.4 cache miss

cache-only 模式下 cache miss 必须报错：

```text
不允许在 GroupCoordinator 内临时 new_group。
```

错误信息必须包含：

```text
rank
group_name
backend
kind
ranks
当前 cache keys 摘要
是否启用 reuse/factory/cache-only
```

### 10.5 CPU/Gloo 失败

如果真实 Gloo CPU group 失败：

```text
不要阻塞 device group cache-only。
CPU/control 正规化另开阶段处理。
```

也就是说：

```text
device root fix 和 CPU root fix 解耦。
```

## 11. 验证矩阵

### 11.1 每个 commit 必跑

```bash
python -m py_compile \
  swift/model/npu_patch/vllm_ascend.py \
  swift/megatron/init.py \
  swift/megatron/trainers/rollout_mixin.py \
  swift/megatron/utils/megatron_lm_utils.py
```

主 smoke：

```text
/data/zyh/tmp/run_ms_swift_grpo_dp_subgroup_clean.sh
```

成功标准：

```text
10/10 step
loss printed
grad_norm printed
End time printed
无残留 process
无 PrefixStore/new_group hang
无 1EB all_gather_object
无 MoeInitRoutingCustom binary error
```

### 11.2 group lifecycle 验证

必须检查：

```text
desired specs hash 一致
execution plan hash 一致
new_group enter/exit 成对
reuse/precreate action 全 rank 一致
warmup 通过
GroupCoordinator device cache hit
cache-only 下无 dynamic device new_group
```

### 11.3 rank lattice 验证

最低矩阵：

```text
vLLM TP=2 DP=4    当前主路径
vLLM TP=1 DP>1    只测 DP
vLLM TP>1 DP=1    只测 TP/control
vLLM TP=4 DP=2    更复杂 vLLM rank lattice
```

Megatron 侧扩展：

```text
TP/PP/CP/EP 当前组合
去掉 CP
去掉 EP
不同 PP
dense 模型
MoE 模型
```

### 11.4 负向验证

建议加至少三个 fail-fast 测试：

```text
1. 故意让某 rank 的 specs hash 不同，确认报错而不是 hang。
2. 故意删除一个 cache entry，确认 GroupCoordinator cache miss 报错。
3. 故意禁用某个 member rank 的 Megatron reuse，确认全 world 决策变成 precreate。
```

## 12. PR / commit 拆分

建议拆成：

```text
Commit 1: Add opt-in NPU vLLM group creation audit
Commit 2: Add Megatron HCCL group registry and exact-order lookup
Commit 3: Build vLLM desired group specs and execution plan hash
Commit 4: Precreate/reuse complete vLLM device group cache
Commit 5: Make GroupCoordinator device cache-only
Commit 6: Experimentally precreate vLLM Gloo CPU/control groups
Commit 7: Remove or gate remaining cpu_group=device_group alias
```

每个 commit 描述必须包含：

```text
symptom
root cause hypothesis
fix
verification
fallback
known remaining risks
```

## 13. 对原方案的关键修改点

主要调整如下。

### 13.1 `source` 不放进 desired spec

原方案里：

```python
source: str  # "reuse" / "precreate"
```

建议移到 execution plan。

原因：

```text
spec 描述 vLLM 需要什么；
execution 描述这个 group 怎么获得。
```

这能避免 specs hash 和 reuse 决策混在一起。

### 13.2 reuse 决策必须全 rank 一致

不能只在当前 rank lookup Megatron group 后自己决定。

必须做：

```text
所有 member ranks exact-order match
=> 全 world reuse
否则
=> 全 world precreate
```

### 13.3 warmup 后必须 world barrier

否则非成员 rank 可能提前进入下一个 `new_group`，成员 rank 还在 subgroup warmup，重新引入 process group 顺序交错。

### 13.4 device root fix 和 CPU root fix 解耦

不要让 Gloo CPU group 是否能修好阻塞 device group cache-only。

合理状态可以是：

```text
device groups 已 factory/cache-only
CPU groups 仍走当前受控兼容逻辑
```

然后单独推进 CPU/control 正规化。

### 13.5 cache-only 先做 device，后做 CPU

一步到位 device+CPU cache-only 风险偏高。

推荐：

```text
device cache-only
  ↓
验证稳定
  ↓
CPU/control cache-only
```

## 14. 当前落地状态

当前工作区已经把 P0-P4 的主线方案落到 opt-in runtime patch：

```text
worktree: /data/zyh/code/ms-swift-npu-grpo-dp-subgroup
branch:   npu-grpo-vllm-dp-subgroup-clean
```

已落地：

```text
1. P0 group creation audit：
   SWIFT_VLLM_ASCEND_GROUP_AUDIT=1 时记录 enter/exit/error/reuse。

2. P1 Megatron HCCL group registry：
   注册 world/tp/pp/cp/dp/dp_cp/partial_dp/ep/etp/edp 等可检查 group。
   复用要求 exact rank-order match。
   成员 rank 还会校验 reuse provenance 一致；不一致则回退 precreate。

3. P2 vLLM group factory：
   在 vLLM LLMEngine 初始化前构建 desired specs 和 execution plan。
   specs hash 与 execution plan hash 必须全 rank 一致。
   reuse 不命中时由所有 ranks 全局同序 precreate。

4. P3 GroupCoordinator cache-only：
   cache-only 开启后，GroupCoordinatorPatch.__init__ 不再动态 new_group。
   device group cache miss 直接报错，不做隐式兜底。

5. P4 Gloo CPU group 预创建：
   对本 patch 接管的 NPU-only vLLM groups，预创建匹配的 Gloo CPU group。
   在 SWIFT_VLLM_ASCEND_ALLOW_CPU_GROUP_ALIAS=0 下验证通过，说明当前验证路径不再需要 cpu_group=HCCL alias。

6. bridge / rollout control group audit：
   mcore-bridge metadata Gloo groups 和 rollout TP Gloo control group 也接入 audit wrapper。
```

当前没有实现独立 stateless control plane。原因是本地验证中真实 Gloo CPU group 已经可以通过全 rank 同序预创建和 warmup 稳定工作；stateless control plane 只保留为后备路线：

```text
如果后续某个版本/拓扑下 Gloo CPU group 在同序预创建后仍卡住，再单独实现 independent control plane。
```

## 15. 当前验证结果

正向 smoke：

```text
TP=2 DP=4: 10/10, bad_count=0
  /data/zyh/tmp/dp-subgroup-p1-consensus-tp2dp4-v1-20260528_145946.log

TP=4 DP=2: 10/10, bad_count=0
  /data/zyh/tmp/dp-subgroup-p1-consensus-tp4dp2-v1-20260528_150558.log

TP=1 DP=8: 10/10, bad_count=0
  /data/zyh/tmp/dp-subgroup-p1-consensus-tp1dp8-v1-20260528_151205.log

TP=2 DP=1: 10/10, bad_count=0
  /data/zyh/tmp/dp-subgroup-p1-consensus-tp2dp1-v1-20260528_151758.log
```

`bad_count` 覆盖：

```text
Traceback
RuntimeError
[ERROR]
ERR00100
ERR02200
PrefixStore
1EB
MoeInitRoutingCustom
cache miss
groupcoordinator_dynamic
mismatch across ranks
provenance differs
```

负向验证：

```text
1. 故意让 specs hash 按 rank 不一致：
   结果：所有 rank 报 expected hash mismatch detected。

2. 故意让相同 ranks 的 Megatron reuse provenance 在成员 rank 间不一致：
   结果：所有 rank 回退 precreate，未复用错误 communicator。

3. cache-only 模式下清空 group cache 后构造 GroupCoordinator：
   结果：所有 rank 报 expected cache miss detected。
```

静态检查：

```bash
python -m py_compile \
  swift/model/npu_patch/vllm_ascend.py \
  swift/megatron/init.py \
  swift/megatron/trainers/rollout_mixin.py \
  swift/megatron/utils/megatron_lm_utils.py

git diff --check
```

均通过。

## 16. 剩余风险

当前仍不默认承诺：

```text
1. 多机。
2. 量化 MoE。
3. NZ。
4. MC2 / DeepEP 的复杂生产拓扑。
5. vLLM / vLLM-Ascend 0.13、0.14 在同一套 group factory 下的完整回归。
```

这些路径需要单独 smoke 或版本回归后再改默认开关。
