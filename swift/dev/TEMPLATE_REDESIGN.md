# Swift / Twinkle Template 重构设计 v2.3.2

> 本文是阶段 A 的正式执行文档。目标是先建立唯一标准实现，再进入阶段 B 的唯一类终态。

## 1. 目标

### 阶段 A：唯一标准实现

- Swift dev Template 是语义事实源，覆盖全部 family、agent 和 loss scale。
- Twinkle 持有 Protocol、共享类型和纯函数，不依赖 Swift。
- Swift Template 与冻结的 Twinkle Jinja Template 暂时并存；Twinkle 旧 `encode` 输出登记为例外。

### 阶段 B：唯一公开 Template 类

- 将 `swift/dev/template` 迁入 Twinkle，Swift 只做 re-export。
- Twinkle 旧类成为统一门面下的 `JinjaTemplate`。
- 保留 remote class、server/client 按名构造、外部 family 插件和旧名字弃用别名。
- 第三方媒体依赖在 Twinkle 中作为可选 extras。

阶段 B 仅在阶段 A 的 golden、三路径 token、数值、吞吐和依赖边界门禁全部通过后开始。

## 2. 已确认决策

1. 阶段 A 的实现与契约内核均位于 Swift；Twinkle 仅从 `swift.dev.template_contract` re-export，并以能力检查保证匹配；形成提交检查点后再固定 Git SHA。
2. family 保留继承，不改为组合，不改变业务逻辑。
3. ENGINE 由真实 ENGINE encode 产生，不再依赖全局媒体 token 合并启发式。
4. 暂时无法接线的模型能力显式声明不支持，调用时 fail-fast。
5. `ignore_empty_think` 只对 `is_thinking` 且存在 `non_thinking_prefix` 的混合思考模板自动启用。
6. `encode` 只编码单个样本；Preference/KTO 使用运行时可区分的 frozen dataclass。
7. Template 边界统一为 list；labels、loss_scale、completion_mask 使用输出顺序；position_ids 在 prepare 阶段产生。
8. episode 状态只属于 `RolloutState`；feature 不保存 messages、tools、原始媒体、engine ids 或 render cache。
9. 所有 state 操作 copy-on-write；Template 不持有 episode cache。
10. Twinkle 旧类阶段 A 不增加模型能力，原有 `encode` 输出保持兼容；新增 Protocol 方法返回标准类型。
11. 不引入 CollateSpec 式第二事实源。
12. 不 commit、tag 或 push；每个任务必须保持工作区可 import、可测试。

## 3. 目标结构

```text
swift/dev/template_contract/
├── types.py      样本、行、state、spec、response 类型
├── encoding.py   输入行转换、续写拼接与 delta render
├── output.py     reasoning / tool / schema 输出纯函数
└── protocol.py   消费者角色接口

swift/dev/template/
├── base.py       标准 Template 门面
├── families/     保留继承的 family 实现
├── agent/        agent 格式与解析
├── loss_scale/   loss-scale 注册
└── register.py / meta.py / media.py / runtime.py

twinkle/template/
├── contract.py   从 Swift 叶子契约包 re-export
└── base.py       冻结的 Jinja fallback
```

`swift/dev/template_contract` 只依赖标准库与 typing；`swift/dev/template` 只允许依赖自身、契约叶子包和第三方库。外部模型能力通过 `model_factory`、`multimodal_graph_adapter` 等构造注入。

## 4. 类型和用户接口

### 4.1 样本

```python
class ConversationSample(TypedDict, total=False):
    messages: list[Message]
    tools: list[Tool]
    images: list
    videos: list
    audios: list
    objects: dict

@dataclass(frozen=True)
class PreferenceSample:
    chosen: ConversationSample
    rejected: ConversationSample
    margin: float | None = None

@dataclass(frozen=True)
class KTOSample:
    conversation: ConversationSample
    label: bool
```

- 原始 dict 永远按 ConversationSample 编码，即使含 `rejected_*` 列。
- 裸 `list[dict]` 仅在每项都有 role 时解释为 messages。
- DPO/KTO recipe 在数据边界显式构造 PreferenceSample/KTOSample。
- Template 不提供 batch encode；消费方循环单样本 encode。

### 4.2 公开白名单

```python
class Template:
    @classmethod
    def from_pretrained(cls, model, **config) -> Template: ...

    @classmethod
    def from_spec(cls, spec: TemplateSpec, processor=None) -> Template: ...

    @property
    def spec(self) -> TemplateSpec: ...

    @property
    def tokenizer(self): ...

    @property
    def processor(self): ...

    def encode(self, sample, intent=TRAIN) -> EncodedRow | PreferencePair: ...
    def decode(self, token_ids, *, state: RolloutState | None = None) -> DecodedResponse: ...
    def parse_response(self, text, *, state: RolloutState | None = None) -> DecodedResponse: ...
```

配置变更通过 `Template.from_spec(template.spec.replace(...), processor=template.processor)` 创建新实例，不修改共享实例。

## 5. 行顺序契约

```python
InputOrderLabels = NewType('InputOrderLabels', list[int])
OutputOrderLabels = NewType('OutputOrderLabels', list[int])
InputOrderLossScale = NewType('InputOrderLossScale', list[float])
OutputOrderLossScale = NewType('OutputOrderLossScale', list[float])
InputOrderCompletionMask = NewType('InputOrderCompletionMask', list[int])
OutputOrderCompletionMask = NewType('OutputOrderCompletionMask', list[int])
```

- family `_encode` 与模型 hook 输入使用 `InputOrderRow`。
- Template.encode 与 RolloutState.feature 使用 `EncodedRow`。
- 只能通过 `twinkle.template.sequence.shift/unshift` 转换。
- 唯一移位字段表：labels→-100、loss_scale→0.0、completion_mask→0。
- 新 token 级字段必须登记为移位或明确登记为不移位。
- `InputOrderRow` 与 `EncodedRow` 只通过显式 shift/unshift 转换，不使用运行时标记。

## 6. RolloutState

```python
@dataclass(frozen=True)
class RolloutState:
    feature: EncodedRow
    engine_input_ids: tuple[int, ...]
    messages: tuple[Message, ...]
    tools: tuple[Tool, ...]
    reasoning_started: bool
    media: MediaBundle
    render_cache: Any | None = None

@dataclass(frozen=True)
class PromptEncoding:
    state: RolloutState

@dataclass(frozen=True)
class CompletionEncoding:
    state: RolloutState
    response: DecodedResponse
```

- feature 仅保存模型 token 字段和处理后的模型 payload。
- messages/tools/raw media/engine ids/cache 只保存在 state。
- state 操作必须返回新对象；修改嵌套 dict 前先复制。
- cache 只能由框架 append 操作传播；阶段 4 完成前不公开其字段布局。

### 6.1 两份 ids

`sequence.splice` 同时接收 feature tokens 与 engine tokens；二者允许不同。

- generated tokens 原样追加到两份 ids。
- append_context 默认分别执行 INFER/ENGINE 渲染。
- 单渲染快速路径只能由 golden 证明后进入内部清单；清单不是 family API。

> TODO(TEMPLATE-BRIDGE-MODES)：L3 分别对拍 legacy 对应 mode；两种 bridge 不要求相同。
>
> TODO(TEMPLATE-QWEN3-FOLLOWUP)：qwen3 使用 `loss_scale=last_round` 时，追加 user follow-up 会重写上一条 assistant 的渲染；thinking 开关均不满足增量账本的前缀约束。可训练 follow-up 暂不支持并保持 fail-fast；tool observe 不受影响。

### 6.2 Reasoning 状态机

- start：renderer 与 prompt 一起返回 reasoning_started，不从 capability 或 token 扫描推断。
- append_generated：共享 reasoning 状态转移按生成文本中 open/close 的出现顺序推进状态。
- append_context：桥接生成下一轮 prompt 时重新返回状态。
- follow-up、tools 撤回、thinking 切换：重新渲染，不继承状态。
- reasoning/tool parsing 使用 `skip_special_tokens=False`，完成拆分和 tool parse 后再清理 special tokens。
> TODO(TEMPLATE-RENDER-OPTIONS)：per-sample `chat_template_kwargs` 未随 `RolloutState` 保存，第二轮起的
> `render_delta` 会退回模板默认的 thinking 配置。修复前多轮只支持与模板级一致的渲染选项。
>
> TODO(TEMPLATE-SPECIAL-SPANS)：当前渲染结果只有文本，控制词只能靠 tokenizer 的 added-vocabulary 匹配。
> 用户内容中出现的控制词拼写会被提升为 control token。目标形态参考 deepseek-recipe 的
> `Prompt { text, special_ranges }`：渲染记录控制 token 的 span，纯文本区域用不带 added-vocabulary 的
> tokenizer 编码。它同时可以让 reasoning 起态不再依赖 token 扫描（E8）。

### 6.3 采样与远程

- 进程内/Ray 使用 RolloutState；Ray 序列化时可丢弃 render cache。
- `SampledSequence` 新增 `new_state`，保留 `new_input_feature == new_state.feature` 一个弃用周期。
- HTTP schema 暂不传完整 state；单轮仍返回 new_input_feature。

> TODO(TEMPLATE-MM-FEATURE)：任务 3.5 后，仅纯文本允许只传 feature；多模态必须传 Trajectory 或 RolloutState，因为 raw media 不属于 EncodedRow。
>
> TODO(TEMPLATE-HTTP-STATE)：HTTP 多轮仅在当前 state 无媒体且每轮 `feature.input_ids == engine_input_ids` 时允许重提交 feature；模板本身可具备多模态能力。实现服务端 state_id 后移除该限制。

## 7. TemplateSpec

```python
@dataclass(frozen=True)
class ModelRef:
    model_id_or_dir: str
    revision: str | None
    trust_remote_code: bool
    processor_kwargs: tuple

@dataclass(frozen=True)
class ModelInfo:
    torch_dtype_name: str
    task_type: str
    max_model_len: int | None
    model_type: str
    is_multimodal: bool

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.torch_dtype_name)

@dataclass(frozen=True)
class TemplateSpec:
    model: ModelRef
    model_info: ModelInfo
    template_type: str
    # 其余字段与 TemplateConfig 一一对应，均为强类型字段。

    def replace(self, **changes) -> TemplateSpec:
        return dataclasses.replace(self, **changes)
```

model_dir/config 在目标进程由 ModelRef 重建，不序列化源机器缓存路径。TemplateConfig→TemplateSpec 只在 build_template 一处完成。

## 8. 框架角色对象

```python
template.rollout.start(sample) -> PromptEncoding
template.rollout.append_generated(state, tokens, tool_calls=None) -> CompletionEncoding
template.rollout.append_context(state, messages) -> RolloutState | None

template.batching.prepare_row(row) -> PreparedRow
template.batching.collate(rows) -> dict

template.pre_forward_hook(model, args, kwargs)
```

契约由四层保证：pyright、signature 测试、两实现行为测试、set_template capability validator。

## 9. 实施顺序

### 阶段 0：基线与门禁

1. 提交准备：提供人工 commit 分组和排除清单，不执行 commit。
2. 设计文档：本文替换旧设计并维护例外表。
3. Golden：fixture/test 入仓，分 L0 注册、L1 tokenizer、L2 唯一实现、L3 特殊 hook、L4 行为、L5 契约。
4. 测试环境：登记可运行多卡/MM/ZeRO-3/CP 的环境。
5. 依赖边界：AST 门禁禁止 template runtime import Swift 其他模块。

### 阶段 1：数据基础与已知缺陷

1. 最小类型：样本、两种 row、state、Prompt/CompletionEncoding。
2. EncodingOps：唯一 shift/unshift/splice/refresh/delta-render，替换两仓重复实现。
3. labels hook：临时 InputOrderRow 视图，返回后重新 shift，正式 feature 顺序不变。
4. unsupported capabilities：以 `TemplateCapability` 枚举登记 12 个未接线 family 能力，并在训练/采样入口 fail-fast；system、multi-round、task type 与 multimodal padding-free 是额外通用门禁。
5. ignore_empty_think 对齐 legacy。
6. 修复 Twinkle render_delta 历史 tool_calls 规范化。
7. ENGINE 使用独立 encode；删除 feature collapse 路径。

### 阶段 2：输出与配置内核

1. OutputCodec：reasoning→tool parse→schema coercion→special-token cleanup，coercion 仅一个调用点。
2. 完整 types/TemplateSpec/ModelInfo，由 builders 通过 dev ModelLoader 重建 processor。
3. 删除 `_reasoning_started` feature 字段与 `_labels_shifted` 临时标记。

### 阶段 3：接口与消费者

1. RolloutCodec + ENGINE/INFER 双编码；同一份只读规范输入，各自深拷贝。
2. 用户白名单与 batching 角色。
3. Protocol 与所有消费者迁移，包括 sampler、ledger、token_utils、processor、model、client、server、mock、agentic、cookbook。
4. TemplateSpec 部署重建，删除 resolve_twinkle_template。
5. 删除 engine 合并与旧公开方法，消费者只读 RolloutState.engine_input_ids。
6. 机械 hook 改名、typing.final 与公开面守护测试。

### 阶段 4：批处理、媒体和性能

1. episode-local RenderCache；默认双渲染，golden 清单才允许单渲染。
2. Twinkle processor 开放 media hooks/grouping，删除 S1/S2/S3。
3. MediaAdapter 声明化，dummy 注入前移到 packing/CP 前。
4. ContextVar 显式化可延后到阶段 B。

### 阶段 5：验收

- 全矩阵 golden、packing、CP/SP、DDP/FSDP/ZeRO-3、LoRA/Multi-LoRA、vLLM/Megatron MM。
- 确定性 SFT：固定环境 50 步 loss/grad norm/lr 逐步一致。
- GRPO 固定轨迹：token/logps/loss 逐位一致。
- GRPO 在线采样：reward/logps/长度/轮数分布落入冻结 fixture 的基线包络。
- encode/collate/rollout 吞吐不超过基线 +5%。

## 10. Golden 分层

| 层级 | 范围 | Oracle |
|---|---|---|
| L0 | 231 个 template type 注册/import/分类 | legacy 集合 |
| L1 | 可获得 tokenizer 的类型，TRAIN/INFER/Preference/KTO | legacy + shift |
| L2 | 每个唯一 `(template_cls, meta shape)` 一个真实 tokenizer；禁止 dev 自快照哈希 | legacy 逐字段/逐 token |
| L3 | 特殊 hook、多模态、collate、position_ids、ENGINE 与 bridge 双 mode | legacy 对应 mode |
| L4 | 32 agent、thinking、rollout/SFT 一致性 | agent/人工/自身一致性 |
| L5 | 两实现的 Protocol 签名和行为 | 契约 |

L3 现状：`test_template_golden_l3.py` 使用真实 processor 逐 token 对拍 legacy 对应 mode，覆盖 6 family 的 api_turn、3 family 的 tool observe，以及 Qwen2.5-VL 的 train/infer/engine ids；`test_vl_mm.py` 另对拍 VL payload、collate 与 mRoPE position_ids。116/117 指 representative 覆盖门禁，不表示全部 chat-template parity 通过；当前 slow parity 仍有 123 个既有失败。其余多模态 family 和完整 thinking/tool decode 矩阵仍待补齐。

不能为覆盖率制造与真实 processor 不一致的 mock token；所有 skip 必须登记原因。

## 11. 例外与 Shim

| ID | 例外 | 移除条件 |
|---|---|---|
| E1 | 已移除：row 顺序由类型与显式 shift/unshift 表达，不再写运行时标记 | 完成 |
| E2 | 两个 Template 类并存 | 阶段 B |
| E3 | 已移除：engine ids 仅由 PromptEncoding/RolloutState 携带，feature-only 只允许纯文本 | 完成 |
| E4 | ContextVar mode | 4.4 或阶段 B |
| E5 | Twinkle 旧 encode 输出 ndarray/position_ids 等 | 阶段 B |
| E6 | HTTP schema 暂无 state_id；纯文本等价 ids 路径保留，其他路径 fail-fast | 服务端 state_id 上线并经过弃用周期 |
| E7 | 已移除：PromptEncoding/CompletionEncoding 均以 RolloutState 为唯一载体 | 完成 |
| E8 | reasoning_started 初值由 token 扫描推导而非渲染器返回 | 3.1：渲染器显式返回 |
| E9 | 已移除：生产 RolloutState.feature 不再携带 messages/tools | 完成 |
| E10 | 渲染产物不含 special-token span；纯文本区按 added-vocabulary 编码 | 渲染器返回 (text, spans)，纯文本区单独编码 |
| S1 | processor 复制 micro-batch grouping | 4.2 |
| S2 | `_media` 旁路槽 | 4.2 |
| S3 | pad 值覆盖 | 4.2 |
| S4 | 已移除：deploy 通过 TemplateSpec 重建 dev Template | 完成 |
| S8 | legacy processor/model_info 依赖 | 2.2 |
| T1 | `Template.from_spec` 以 import spec 引用 `swift.dev.builders.template`，是 AST import 门禁的登记例外 | 阶段 B：builder 由注入的工厂提供 |

## 12. 红线

1. 不 commit/tag/push。
2. 每个任务保持工作区可 import、可测试。
3. 不改 family 业务逻辑，除明确登记的机械修复。
4. 状态只放 RolloutState；feature 不保存 rollout 元数据或派生状态。
5. 拼接、移位、reasoning、schema coercion 各只有一个实现。
6. runtime 不 import legacy `swift.template`；`swift.dev.template` 不 import Swift 其他模块；Twinkle 仅可 import `swift.dev.template_contract`。
7. Golden 不一致不得进入下一任务；禁止用重新生成 dev 自快照替代 legacy 差异审核。
8. 每轮交付报告必须列出 golden 结果和失败列表相对基线的变化。

## 13. 当前提交准备

可以独立人工提交：

1. Swift LoRA 冻结分区 targeting（adapter、model builder、assembly 对应 hunks及测试）。
2. Swift HF5 nested multimodal attrs（loader/base 对应 hunks及测试）。
3. Twinkle shared types 与纯 helper（不包含 Template 新控制流）。

暂不能提交：

- 当前 Swift staged 138 文件整体：缺少未跟踪核心文件与 unstaged helpers，不是可导入闭包。
- Template 主迁移：需先完成阶段 0 golden/契约门禁并整理完整依赖闭包。
- Graph adapter：必须与 Template 接口分开，并在真实多卡 uneven-rank 验证后成对发布。

禁止提交：`._____temp/`、`_smoke_reasoning_forms.py`、根目录 `swift_template_refactor_CR.md`。

## 14. 测试环境

- CPU/接口与 vLLM 0.27 专属测试：`/mnt/nas2/anaconda3/envs/vllm027`，vLLM 0.26.1rc1.dev、Transformers 5.15、PyTorch 2.13/cu130。
- 当前主机：8×NVIDIA H20，驱动 535.183.06。该驱动不能初始化 CUDA 13，因此 vllm027 的 GPU 专属测试登记为环境阻塞。
- 通用 GPU/MM/多卡门禁临时使用 `/mnt/nas2/anaconda3/envs/vllm023`（PyTorch 2.10/cu128，8 卡均可用）；这不能替代 vLLM 0.27 专属门禁。
- 运行 Swift/Twinkle 测试必须显式设置两个仓库的 PYTHONPATH；`qwen_vl_utils` 使用工作区源码路径。

## 15. 后续顺序

1. 扩充多模态与 thinking/tool decode 的 legacy oracle。
2. 修复 QWEN3-FOLLOWUP 的历史重写能力倒退；修复前继续显式 fail-fast。
3. 阶段 4：收敛 processor media hooks、batch grouping 和 dummy 注入。
