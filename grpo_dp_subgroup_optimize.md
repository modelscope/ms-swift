# vLLM DP subgroup factory 方案状态：已废弃

这个文件原本记录的是一条更重的修复路线：

```text
Megatron group registry
vLLM group factory
GroupCoordinator cache-only
CPU/control group 正规化
```

后续 root-cause 消融证明，当前 vLLM-Ascend hang 的直接根因不是 vLLM DP subgroup 必须由 SWIFT 预创建，而是 SWIFT 在 `patch_npu_vllm()` 中无差别给所有 `torch.distributed.new_group()` 注入了：

```python
use_local_synchronization=True
```

因此这条 DP subgroup factory 方案不再作为当前主线。当前代码已经删除：

```text
swift/model/npu_patch/vllm_ascend_group_factory.py
swift/model/npu_patch/vllm_ascend_group_registry.py
swift/model/npu_patch/vllm_ascend_group_runtime.py
swift/model/npu_patch/vllm_ascend_groups.py
swift/model/npu_patch/vllm_ascend_group_control.py
```

当前有效结论和验证矩阵见：

```text
npu_grpo_runtime_patches.md
```

保留本文仅用于说明：不要继续沿着旧的 group factory/cache-only 方案推进，除非未来出现新的、独立于 `use_local_synchronization=True` 的 vLLM-Ascend group lifecycle 问题，并且有新的日志证据证明需要恢复这条路线。
