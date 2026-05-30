# NPU GRPO vLLM-Ascend root-cause notes

本文记录当前 worktree 为跑通 NPU GRPO 保留的最小修复，以及已经通过日志/消融排除掉的历史 workaround。

```text
worktree: /data/zyh/code/ms-swift-npu-grpo-dp-subgroup
branch:   npu-grpo-vllm-dp-subgroup-clean
env:      /data/zyh/miniconda3/envs/swift_dev_v18_swiftpatch
CANN:     /data/cann/cann900/cann-9.0.0/set_env.sh
vLLM:     /data/zyh/code/vllm-018-src-swiftpatch/vllm
Ascend:   /data/zyh/code/vllm-018-src-swiftpatch/vllm-ascend
```

## 1. 最终结论

之前 vLLM-Ascend 初始化 hang 的主因不是 vLLM DP subgroup 必须由 SWIFT 自己预创建，也不是 Megatron group 必须复用；真实根因是 SWIFT 在 `patch_npu_vllm()` 里把所有 `torch.distributed.new_group()` 都无差别改成了：

```python
torch.distributed.new_group = partial(original_new_group, use_local_synchronization=True)
```

这个改动同时影响 Gloo 和 HCCL：

1. Gloo world group 创建会卡在 vLLM-Ascend `GroupCoordinator` 初始化阶段。
2. 如果只让 Gloo 不带 local sync，但 HCCL 仍带 local sync，后续会卡在 MoE token dispatcher 的 HCCL comm name 初始化。
3. 完全移除这个 `new_group(..., use_local_synchronization=True)` 注入后，vLLM-Ascend 自己创建 Gloo/HCCL groups 可以跑通，MoE token dispatcher 也能拿到 comm name，GRPO 能稳定跑到 10 step loss。

因此当前不再保留这些历史补丁：

- vLLM DP group factory / cache-only patch。
- Megatron HCCL group registry / reuse patch。
- GroupCoordinator runtime monkey patch。
- vLLM TP Gloo control group patch。
- mcore-bridge PP/EP control-plane patch。

## 2. 卡住点逐个定位

### 2.1 第一处：Gloo world group 创建 hang

root-cause worktree:

```text
/data/zyh/code/ms-swift-npu-grpo-rootcause-debug
```

诊断方式：

- 给 `torch.distributed.new_group` 打 enter/exit 日志。
- 给 vLLM-Ascend `GroupCoordinatorPatch.__init__` 打 group_name、backend、ranks、flags。
- 先关闭 SWIFT 的 group factory/cache-only patch，只保留诊断。

卡住日志：

```text
/data/zyh/tmp/grpo-rootcause-no-group-patch-20260529_195046.log
```

关键现象：

```text
groupcoord-enter group_name=world backend=hccl
new-group-enter seq=1 backend=hccl ranks=[0,1,2,3,4,5,6,7] use_local_synchronization=True
new-group-exit  seq=1 backend=hccl
new-group-enter seq=2 backend=gloo ranks=[0,1,2,3,4,5,6,7] use_local_synchronization=True
```

`seq=2` 没有 exit。也就是说所有 rank 并不是卡在 Megatron 初始化，也不是 rank 顺序明显分叉，而是 vLLM-Ascend 在初始化自己的 Gloo world group 时，被 `use_local_synchronization=True` 这层 SWIFT monkey patch 卡住。

### 2.2 第二处：MoE token dispatcher HCCL comm name 初始化 hang

修掉第一处后，做了第二组消融：只让 Gloo 不注入 `use_local_synchronization=True`，HCCL 仍然注入。

日志：

```text
/data/zyh/tmp/grpo-rootcause-gloo-no-local-sync-20260529_205242.log
```

现象：

- Gloo world / dp / ep / mc2 groups 都能创建完成。
- 训练继续推进到 vLLM-Ascend MoE token dispatcher。
- rank 栈卡在：

```text
vllm_ascend/ops/fused_moe/token_dispatcher.py:424
self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
```

同时 token dispatcher 日志只有 enter，没有 exit：

```text
token-dispatcher-enter class=TokenDispatcherWithAll2AllV ...
```

这说明 HCCL group 创建阶段即使看起来过了，`use_local_synchronization=True` 仍会影响后续 communicator name 初始化。它不是另一个 MoE layout 问题，也不是 routing custom op 问题。

### 2.3 第三组：完全移除 local sync 注入

第三组消融把 `patch_npu_vllm()` 里的 `new_group` monkey patch 完全去掉，Gloo/HCCL 都走 PyTorch 默认行为。

日志：

```text
/data/zyh/tmp/grpo-rootcause-no-local-sync-all-20260529_215601.log
```

关键证据：

```text
new-group-enter ... use_local_synchronization=None
new-group-exit  ...
token-dispatcher-enter class=TokenDispatcherWithAll2AllV ...
token-dispatcher-exit  class=TokenDispatcherWithAll2AllV ep_size=8 ep_rank=0 comm_name=group_name_156
iteration: 10/10
End time of running main: 2026-05-29 22:01:28.759000
```

结论：Gloo hang 和 MoE token dispatcher HCCL comm name hang 是同一个 SWIFT monkey patch 的两种表现。正向修复是删除这层全局 `new_group` 改写，而不是再用更大的 group lifecycle monkey patch 绕开它。

## 3. 当前保留的代码改动

### 3.1 不再给 `new_group` 注入 local sync

文件：

```text
swift/infer_engine/utils.py
```

当前 `patch_npu_vllm()` 只做两件事：

- NPU 下调用 `patch_vllm_ascend_runtime()`。
- 在 context 内把 `torch.npu.mem_get_info` 绑定到当前 vLLM device。

保留代码形态：

```python
def patch_npu_vllm(vllm_device: str, *, colocate: bool = False):
    if isinstance(vllm_device, int):
        vllm_device = get_device(vllm_device)
    device_type = vllm_device.split(':')[0]
    if device_type == 'npu':
        from swift.model.npu_patch.vllm_ascend import patch_vllm_ascend_runtime
        patch_vllm_ascend_runtime(colocate=colocate)

    @contextmanager
    def npu_vllm_context():
        torch.npu.mem_get_info = partial(torch.npu.mem_get_info, device=vllm_device)
        yield

    return npu_vllm_context() if device_type == 'npu' else nullcontext()
```

已删除的旧逻辑：

```python
torch.distributed.new_group = partial(original_new_group, use_local_synchronization=True)
```

### 3.2 `mem_get_info` 绑定兼容

文件：

```text
swift/model/npu_patch/vllm_ascend_memory.py
```

问题：

```text
TypeError: mem_get_info() got multiple values for argument 'device'
```

根因：vLLM-Ascend 调 `current_platform.mem_get_info(device)`，不同 torch-npu / vLLM-Ascend 版本对方法绑定方式不一致。

当前只修 API call surface，不改上游 memory profiling 策略：

```python
@classmethod
def mem_get_info(cls, device=None):
    if device is None:
        return torch.npu.mem_get_info()
    try:
        return torch.npu.mem_get_info(device=device)
    except TypeError:
        return torch.npu.mem_get_info()

NPUPlatform.mem_get_info = mem_get_info
```

### 3.3 非量化 MoE routing custom op fallback

文件：

```text
swift/model/npu_patch/vllm_ascend_moe.py
```

问题：某些 vLLM-Ascend 实现会把 non-quant MoE routing 派发到：

```text
npu_moe_init_routing_custom / aclnnMoeInitRoutingCustom
```

在当前 CANN / torch-npu 组合下，这个 custom op 缺 binary 或异步失败时可能污染 NPU stream。不能先调用 custom op 再 fallback，因为失败 launch 本身可能导致后续同步点 hang。

修复：检测上游实现。如果 non-quant branch 已经走 `torch_npu.npu_moe_init_routing_v2`，不 patch；否则只把 non-quant branch 改到 v2，量化和其他分支继续走上游。

### 3.4 vLLM-Ascend MoE expert runtime weight sync layout

文件：

```text
swift/rlhf_trainers/utils.py
swift/model/npu_patch/vllm_ascend_moe.py
swift/rlhf_trainers/rollout_mixin.py
```

背景：

GRPO 会把训练端权重同步到 rollout 端 vLLM。vLLM-Ascend 非量化 MoE 初始加载后会把 expert 权重转成 grouped-matmul 需要的 processed 3D layout：

```text
w13_weight: [local_experts, hidden, 2 * intermediate_per_tp]
w2_weight : [local_experts, intermediate_per_tp, hidden]
```

但训练端同步过来的权重可能是 HF/Megatron 2D：

```text
gate_proj/up_proj: [intermediate, hidden]
down_proj:         [hidden, intermediate]
```

FSDP2 Qwen MoE 还可能是 fused 3D：

```text
gate_up_proj: [experts, 2 * intermediate, hidden]
down_proj:    [experts, hidden, intermediate]
```

修复：

- 通用 vLLM MoE weight_loader 逻辑仍留在 `swift/rlhf_trainers/utils.py`。
- 只有 NPU + `quant_method` 来自 `vllm_ascend` 时，才给 `w13_weight` / `w2_weight` 的 runtime loader 加 Ascend layout 处理。
- `swift/rlhf_trainers/rollout_mixin.py` 把 FSDP2 fused MoE 参数名展开成 vLLM checkpoint-style 名字，例如：

```text
mlp.experts.gate_up_proj
  -> mlp.experts.0.gate_proj.weight
  -> mlp.experts.0.up_proj.weight

mlp.experts.down_proj
  -> mlp.experts.0.down_proj.weight
```

注意：这里的 `expert 0` 只是名字锚点，tensor 仍然保留 `[experts, ...]` 维度，由 patched loader 按全 expert tensor 写入本地 experts。

## 4. 已删除的历史 workaround

### 4.1 vLLM group factory / registry / runtime patch

已删除文件：

```text
swift/model/npu_patch/vllm_ascend_group_factory.py
swift/model/npu_patch/vllm_ascend_group_registry.py
swift/model/npu_patch/vllm_ascend_group_runtime.py
swift/model/npu_patch/vllm_ascend_groups.py
```

删除原因：

- root-cause 消融证明原始卡住点来自 SWIFT 自己的 `use_local_synchronization=True` 注入。
- 去掉该注入后，上游 vLLM-Ascend group 创建可以跑通 10 step。
- 因此不需要再维护 group factory/cache-only 这套大 patch。

### 4.2 vLLM TP Gloo control group

已删除文件：

```text
swift/model/npu_patch/vllm_ascend_group_control.py
```

同时删除 Megatron rollout 里的：

```python
self.vllm_tp_control_group = get_or_create_vllm_tp_gloo_group(...)
object_group = self.vllm_tp_control_group or self.vllm_tp_group
```

消融验证：

```text
log: /data/zyh/tmp/grpo-ablate-no-tp-gloo-vtp4-vdp2-20260529_221315.log
case: vLLM TP=4 DP=2, Megatron TP=2 PP=2 CP=2 EP=4
result: 10/10 step, no all_gather_object 1EB error, no hang
```

结论：在去掉 `use_local_synchronization=True` 后，这层 TP Gloo control patch 对当前 GRPO 路径不是必要条件。

## 5. 验证矩阵

所有验证均在：

```text
env: /data/zyh/miniconda3/envs/swift_dev_v18_swiftpatch
code: /data/zyh/code/ms-swift-npu-grpo-dp-subgroup
model(MoE): /data/model/Qwen3-30B-A3B-Instruct-2507-8layers
model(Dense): /data/model/Qwen3-4B
dataset: /data/dataset/NuminaMath-TIR/data/train-00000-of-00001.parquet#1000
reward: /data/zyh/code/ms-swift/examples/ascend/grpo/manual_reward_plugin.py
cards: ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 5.1 Megatron backend

| vLLM TP | vLLM DP | Megatron TP | PP | CP | EP | 结果 | 日志 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4 | 2 | 2 | 2 | 4 | 10/10 | `/data/zyh/tmp/grpo-current-megatron-base-205841-20260529_205841.log` |
| 1 | 8 | 2 | 2 | 2 | 4 | 10/10 | `/data/zyh/tmp/grpo-current-megatron-vtp1-vdp8-20260529_210447.log` |
| 4 | 2 | 2 | 2 | 2 | 4 | 10/10 | `/data/zyh/tmp/grpo-current-megatron-vtp4-vdp2-20260529_211033.log` |
| 2 | 4 | 2 | 1 | 2 | 2 | 10/10 | `/data/zyh/tmp/grpo-current-megatron-pp1-ep2-20260529_211635.log` |
| 2 | 4 | 1 | 2 | 2 | 2 | 10/10 | `/data/zyh/tmp/grpo-current-megatron-mtp1-pp2-ep2-20260529_220444.log` |
| 4 | 2 | 2 | 2 | 2 | 4 | 10/10 without TP Gloo control | `/data/zyh/tmp/grpo-ablate-no-tp-gloo-vtp4-vdp2-20260529_221315.log` |

### 5.2 FSDP2 backend

| Model | vLLM TP | vLLM DP | 结果 | 日志 |
| --- | --- | --- | --- | --- |
| Qwen3-4B dense | 2 | 4 | 10/10 | `/data/zyh/tmp/grpo-current-fsdp2-dense-tp2-dp4-20260529_212340.log` |
| Qwen3-4B dense | 1 | 8 | 10/10 | `/data/zyh/tmp/grpo-current-fsdp2-dense-tp1-dp8-20260529_213017.log` |
| Qwen3-4B dense | 4 | 2 | 10/10 | `/data/zyh/tmp/grpo-current-fsdp2-dense-tp4-dp2-20260529_213603.log` |
| Qwen3 MoE reduce | 2 | 4 | 10/10 | `/data/zyh/tmp/grpo-current-fsdp2-moe-tp2-dp4-20260529_214743.log` |

### 5.3 静态检查

```bash
python -m py_compile \
  swift/model/npu_patch/vllm_ascend.py \
  swift/model/npu_patch/vllm_ascend_moe.py \
  swift/model/npu_patch/vllm_ascend_memory.py \
  swift/infer_engine/utils.py \
  swift/megatron/utils/megatron_lm_utils.py \
  swift/megatron/trainers/rollout_mixin.py \
  swift/rlhf_trainers/rollout_mixin.py
```

## 6. 仍未覆盖的范围

当前结论只覆盖本地 8 卡、vLLM/vLLM-Ascend 0.18、CANN 9.0.0、Qwen3 dense/MoE、FSDP2/Megatron GRPO。

未默认承诺：

- 多机。
- 量化 MoE。
- Qwen3.5 模型支持。
- NZ / MC2 / DeepEP 全组合。
- 旧版 vLLM-Ascend 0.13 / 0.14 的完整回归。

## 7. 当前最小保留原则

当前代码应保持以下边界：

1. 不再 patch `torch.distributed.new_group`。
2. 不再接管 vLLM-Ascend `GroupCoordinator`。
3. 不再预创建或 cache vLLM groups。
4. 只保留实际验证需要的窄补丁：
   - `mem_get_info` API 绑定兼容。
   - non-quant MoE routing 避开不稳定 custom op。
   - vLLM-Ascend MoE runtime weight sync layout 兼容。
   - FSDP2 fused MoE 权重名展开。
