# Ring Attention 进展总结

## 来源

- 参考会话：`/home/zyh/.codex/sessions/2026/03/17/rollout-2026-03-17T01-55-03-019cf980-f131-7531-b0ab-8b70a95f1a4b.jsonl`
- 这条线从 `2026-03-17` 的 HF 路线 `Ulysses SP + zigzag ring attention` 分析开始，后面逐步演进成 NPU 适配、真机验证、多卡 backward 校正。

## 一页结论

这条线不是只做了“给 NPU 接个后端”这么简单，而是分了四个阶段逐步收敛：

1. 先确认仓库里的 HF `sequence_parallel` 是“两层并行”：
   `Ulysses` 负责 `all_to_all` 重排，`ring attention` 负责 `rp_world_size > 1` 时的 zigzag KV 环形交换。
2. 然后把 `swift/sequence_parallel/zigzag_ring_attn.py` 从“硬绑 CUDA `flash_attn`”改成“GPU/NPU 可切换后端”。
3. 接着在真实 Ascend 机器上定位到第一批真 bug：
   `_reshape_npu_lse()` 错把 `npu_fusion_attention` 返回的 `(T, N, 8)` 中间量当成普通三维布局处理，导致前向直接崩。
4. 最后继续做 2 卡、4 卡 HCCL smoke，发现并修正多卡 backward 里 `dq/dk` 的 full-normalization 语义问题，最终把多卡结果收敛到 fp16 正常误差范围。

按当前工作树和 `tasks/todo.md` 的复盘来看，这条线已经不只是“能跑通前向”，而是已经走到了：

- NPU block forward/backward 已接通
- `tests/utils/test_zigzag_ring_attn.py` 已补上关键契约测试
- 远端单卡与多卡 smoke 都做过
- 2 卡、4 卡 packed case 的 `out/dq/dk/dv` 已对齐到可接受误差

## 当前进展

### 1. 设计与边界已经看清

- HF 路线入口是 `swift sft/rlhf --sequence_parallel_size N`，不是 Megatron/MindSpeed 那套 `--sequence_parallel true`。
- `swift/sequence_parallel/ulysses.py` 负责 mesh 初始化、attention patch、输入切分和 gather。
- `swift/sequence_parallel/zigzag_ring_attn.py` 负责真正的 zigzag ring 前后向。
- 初期分析时的关键边界是：
  “仓库整体支持 NPU” 不等于 “HF 这套 Ulysses + ring 已经在 NPU 上被官方完整验证”。

### 2. NPU 后端已经接到 ring attention

- 当前 `zigzag_ring_attn.py` 里已经有 NPU helper：
  `_call_npu_fusion_attention()`、`_npu_forward()`、`_npu_backward()`。
- 这层做的事情包括：
  - 把 `cu_seqlens` 转成 Ascend 需要的 `actual_seq_qlen/actual_seq_kvlen`
  - 把 `causal/window_size` 转成 `sparse_mode + atten_mask + pre_tockens/next_tockens`
  - 从 `softmax_max/softmax_sum` 重建 ring 逻辑需要的 `block_lse`
- GPU 路径和 `ulysses.py` 外部调用协议保持不变，改动面集中在 `zigzag_ring_attn.py`。

### 3. 真机上先后定位出两层问题

- 第一层问题是 NPU wrapper 的形状假设错误。
  `npu_fusion_attention` 在 `TND` 路径下返回的中间量可能是 `(T, N, 8)`，最后那一维是 Ascend 的重复槽位，不是新的序列维。旧逻辑把它错误 reshape，触发了 `shape '[2, 8]' is invalid for input of size 128`。
- 第二层问题是多卡 backward 语义不完整。
  早期 2 卡、4 卡 smoke 里，`out` 和 `dv` 基本对齐，但 `dq/dk` 明显偏大，说明问题不在通信入口，而在 block backward 和 full `softmax_lse/out` 的配合语义。

### 4. 当前收敛状态

根据现有 `tasks/todo.md` 复盘，最终收敛结果是：

- `2` 卡：`lengths=[8, 12]`，`out_diff=0.0009322166442871094`，`dq_diff=0.0009765625`，`dk_diff=0.0009765625`，`dv_diff=0.001953125`
- `4` 卡：`lengths=[16, 24]`，`out_diff=0.0007905960083007812`，`dq_diff=0.0009765625`，`dk_diff=0.0009765625`，`dv_diff=0.00390625`

这些量级已经落在 fp16 多卡通信和 attention 计算的正常误差范围内，所以这条线当前可以认为已经从“功能打洞”推进到“多卡 correctness 验收通过”。

## 阶段时间线

### 阶段 1：只读分析 HF 路线

目标是回答四件事：

- HF 路线怎么开
- Ulysses 和 ring attention 如何组合
- 代码调用链如何落到训练 step
- HF 这条线对 NPU 的真实支持边界是什么

阶段产出是把 `sequence_parallel.prepare()`、`DistributedAttention.forward()`、`zigzag_ring_flash_attn_varlen_func()` 这些关键路径串起来，并确认一个重要边界：
NPU 在仓库里主要成熟的是 Megatron/MindSpeed 路线，HF 路线的 ring attention 当时还没有完整的 NPU 落地证据。

### 阶段 2：把 NPU backend 接到 `zigzag_ring_attn.py`

设计原则非常明确：

- 不动 `ulysses.py` 编排
- 不改公开接口
- 只替换 block 级 attention 内核

中间过程里尝试过两条思路：

- 用 `torch_npu.npu_fusion_attention` 接前向
- 再尝试用 `torch_npu.npu_fusion_attention_grad` 接反向

但会话后段已经证明：

- `npu_fusion_attention` 前向这条路是成立的
- `npu_fusion_attention_grad` 在当前需要的 `TND varlen` 路径上并不可靠，不能直接拿来做最终方案

所以后续代码演进成“前向用 fused attention，反向按当前 ring 语义做显式/手写校正”。

### 阶段 3：本地 `swift_next` 验证与环境分流

本地验证分成两层：

- Python 契约层
  - `py_compile`
  - `python -m unittest tests.utils.test_zigzag_ring_attn`
- 真机最小 smoke
  - 直接调 `_npu_forward/_npu_backward`
  - 再直接调原生 `torch_npu.npu_fusion_attention`

这一步有一个很重要的分流：

- 在 `swift_next` 上，曾经出现过 `npu_fusion_attention` 自身卡住，最终从 `~/ascend/log/debug/plog` 里定位到本机 Ascend 运行时初始化异常，不是 ring 代码映射本身的问题。
- 这使得后续验证转移到你提供的远端机器 `115.190.166.102:443` 上进行。

### 阶段 4：远端 `swift_dev` 真机验证

远端验证先做环境确认，再做最小算子 smoke，再做 ring 相关验证。

关键结论依次是：

- 远端基础 NPU 算子正常
- 原生 `torch_npu.npu_fusion_attention()` 正常
- 真正崩的是 wrapper 对 LSE 形状的理解
- 修完 `_reshape_npu_lse()` 后，单卡 `_npu_forward/_npu_backward` 通过
- 再往上走到公开入口和 HCCL smoke，前向已成立，但早期多卡 `dq/dk` 仍不对

### 阶段 5：多卡 backward 语义校正

2 卡和 4 卡 smoke 暴露出一个非常典型的问题模式：

- `out` 基本对
- `dv` 也基本对
- 只有 `dq/dk` 系统性偏离

这个模式说明：

- 问题不在 gather/split 脚本
- 不在通信拓扑入口
- 不在纯前向
- 而在 backward 里对 block 概率和 merged `softmax_lse` 的使用语义

最终修正方向是让 NPU 路径的 block backward 对齐 GPU `flash_attn` 的 full-normalization 语义，而不是把每个 ring block 当成“局部独立 attention”去反传。

## 操作流程

下面这套流程就是这次会话里反复打磨后的可复用流程。

### 1. 先做只读定位

先看这几类文件：

- `swift/sequence_parallel/ulysses.py`
- `swift/sequence_parallel/zigzag_ring_attn.py`
- `swift/sequence_parallel/utils.py`
- `examples/train/sequence_parallel/*.sh`
- `docs/source/BestPractices/NPU-support.md`

目的不是立刻改代码，而是先确认：

- 哪一层是编排
- 哪一层是算子后端
- 哪些结论只是文档声明
- 哪些结论已经被示例或真机覆盖

### 2. 修改时只动最小层

这条线的经验是：
如果问题看起来出在 NPU 适配，不要先改 `ulysses.py` 或示例脚本，优先把问题收敛到 `zigzag_ring_attn.py` 的 block backend。

原因很简单：

- `ulysses.py` 是并行编排层，改它风险大
- `zigzag_ring_attn.py` 才是 GPU/NPU 分叉点
- 把改动压在这层，最容易保持 GPU 路径零回归

### 3. 本地先跑契约测试

优先做这几类本地验证：

- `python -m py_compile swift/sequence_parallel/zigzag_ring_attn.py tests/utils/test_zigzag_ring_attn.py`
- `conda run --no-capture-output -n swift_next python -m unittest tests.utils.test_zigzag_ring_attn`

目的不是证明真机可用，而是先锁住：

- 参数映射
- LSE 形状整理
- block backward buffer 写回语义

### 4. 真机验证先检查环境，再检查算子，再检查 ring

远端推荐顺序：

1. `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
2. `source /usr/local/Ascend/nnal/atb/set_env.sh`
3. `source ~/miniconda3/etc/profile.d/conda.sh`
4. `conda activate swift_dev`
5. 做最小 NPU 张量算子 smoke
6. 直接调 `torch_npu.npu_fusion_attention`
7. 再调 `_npu_forward/_npu_backward`
8. 最后才做 2 卡、4 卡 HCCL smoke

这个顺序能把“环境问题”和“ring 语义问题”拆开，不会一上来就把所有错误混在一起。

### 5. 多卡 smoke 要严格复刻真实切分逻辑

不要按总序列粗暴 chunk。
必须按 `ulysses._split_packed()` 的逐序列 zigzag 规则切分和回收，否则你测到的是 smoke 脚本误差，不是 ring attention 真问题。

### 6. 验证结束后清理远端工作树

这次会话里有一个好习惯要保留：

- 同步过去的临时文件验证完就恢复
- 删除远端 smoke 脚本
- 确保 `/home/zyh/code/ms-swift` 不残留验证痕迹

这样不会把临时调试结果污染远端仓库状态。

## 本轮沉淀下来的关键判断

### 判断 1

如果单卡 `_npu_forward/_npu_backward` 能跑，而多卡只剩 `dq/dk` 偏差，优先怀疑 backward 语义，不要继续先怀疑 HCCL 或 `send_recv_kv`。

### 判断 2

`npu_fusion_attention` 的中间量布局不能想当然。Ascend 返回值里像 `(T, N, 8)` 这种重复槽位语义，必须先看真实 shape，再做 reshape。

### 判断 3

`dv` 对、`dq/dk` 不对，通常意味着问题在 softmax normalization 或 `delta=sum(out * dout)` 这一层，而不是值向量聚合本身。

### 判断 4

验证 NPU 问题时，一定先分清是：

- Python 层映射错误
- Ascend 运行时初始化错误
- 单块 attention kernel 语义错误
- 多卡 ring backward 语义错误

这四层不拆开，debug 会非常慢。

## 相关文件

- `swift/sequence_parallel/zigzag_ring_attn.py`
- `tests/utils/test_zigzag_ring_attn.py`
- `examples/train/sequence_parallel/sequence_parallel_512k.sh`
- `tasks/todo.md`

## 当前建议

如果后面继续沿这条线工作，优先顺序建议是：

1. 先以 `tests/utils/test_zigzag_ring_attn.py` 和远端 2 卡/4 卡 smoke 作为回归基线。
2. 如果还要继续优化 NPU 性能，再单独评估 `npu_fusion_attention_grad` 是否有新的可用姿势；但 correctness 基线不要再退回去。
3. 如果要扩到更多样例，仍然优先把框架性质的逻辑留在 `swift/sequence_parallel`，不要把 patch 塞回示例脚本。
