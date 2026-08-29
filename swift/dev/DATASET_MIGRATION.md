# 数据集迁移结果表（legacy `swift/dataset/` → `swift/dev/dataset/`）

> 记录数据集从 legacy 迁到 dev 的进度、结论与依据。与 `MODEL_MIGRATION.md` 同体例，但有一处结构差异：模型侧只有「一个 loader 层」，数据集侧是**四层基础设施 + 数据集声明**，所以本文件先记录各层状态，再记录数据集批次——一个数据集能不能迁，往往取决于它依赖的层有没有到位。

> **本文只记「已经搬了什么」。接下来怎么改形状，见 [`DATASET_REDESIGN.md`](./DATASET_REDESIGN.md)**——那份含完整的猫腻清单（带代码位置）、目标类层次设计、`cache_encoded`（文本落盘 + 媒体运行时）、下游硬约束清单与分阶段计划。两者分工：**先有本文的逐字 parity，才有资格谈那份的重排。**

> **2026-08 legacy 解耦（dev 构建路径不再依赖 `swift.dataset`/`swift.pipelines`）**：`swift/dev/builders/dataset.py` 与 recipe（`cached_dataset.py`/`quantize.py`/`run_infer.py`）的 `load_dataset`、`DatasetLoader.concat_datasets`、`EncodePreprocessor`、`LazyLLMDataset`、`PackingDataset`、`IterablePackingDataset` 全部指向 `swift.dev.dataset`；legacy `AddLengthPreprocessor` 换成 dev 的 `MeasurePreprocessor`（不可编码行保留并置空 `lengths`，是 dev 既定语义）。legacy `swift.pipelines.utils.get_cached_dataset` 在 dev 重写为 `DatasetLoader.load_cached_datasets`（staticmethod：逐 path 解析 `#N` 采样、`length`→`lengths` 重命名、`truncation_strategy=='delete'` 的 `max_length` 过滤），`builders/dataset.py` 直接读 config 字段传入，去掉了 legacy 版的 `SimpleNamespace` shim。数据集层目前仅剩 `loader/base.py:482` 的 `from swift.hub import get_hub` 一处 legacy 引用（复杂件，保留）。

## 判定规则
- **迁移（migrated）**：在 dev 侧新建 `DatasetLoader` 子类并 `@register_dataset`，且与 legacy 做过逐用例 parity 比对。
- **待迁（pending）**：无真障碍，只是还没轮到。数据集与模型不同——绝大多数数据集是**纯声明**（无自定义逻辑），迁移成本近乎为零，所以 pending 才是常态，不是问题。
- **阻塞（blocked）**：依赖的层或组件还没迁。逐个登记卡在什么上，不允许悬空。
- **不适用（n/a）**：legacy 里的某些「preprocessor」其实不是输入格式而是任务后处理（cls / grounding / embedding 多塔），归属在别的层，不在数据集声明层解决。

## 列含义
`legacy ms id`：legacy 注册的 ModelScope id ｜ `dev loader`：dev 侧类名（未迁留空）｜ `声明档位` ｜ `结论` ｜ `依据 / 备注`。

## 声明档位（dev 侧新增的组织维度）

dev 的 `llm.py` 按**声明成本**分三档（另加完全不进 Python 的档 0），这也是读它和加新数据集的顺序。模型侧没有这个维度，因为模型不存在「零声明即可用」的情况；数据集有——格式层能自动探测行形状。

| 档位 | 含义 | 典型 |
|---|---|---|
| 0 | **JSON 声明**：`dataset_info.json` 一条记录，零 Python | `path-vqa`、`medical_zh` |
| 1 | 纯声明：ids / subsets / split，**无 preprocessor** | `gsm8k`、`sharegpt` |
| 2 | 声明 + `Preprocessor.columns`：行形状标准、字段名奇怪 | `sql-create-context` |
| 3 | `Preprocessor` 子类：字段合并 / 行过滤 / 文本清洗 | `dolly-15k`、`firefly-zh` |

> 档 0 与档 1-3 的分界不是「简单/复杂」，而是**数据还是代码**：能用 JSON 表达的一律留在 JSON，不写 Python 类。判据是这条数据集是否只需要 ids / subsets / split / columns / tags —— 一旦需要一行 Python，就升到档 2 或 3。

> 约定：preprocessor 一律声明为**类**，绝不是实例。loader 的 `preprocessor` 是所有加载共享的类属性，而 preprocessor 带 per-load 可变状态（已探测的 converter、traceback 计数）——legacy 写 `preprocess_func=SomePreprocessor(...)` 正是在共享这些状态。

---

# 一、基础设施分层

数据集侧的迁移主体是这四层，而不是数据集本身。层到位后，数据集迁移就退化成填声明。

| 层 | dev 位置 | legacy 对应 | 状态 | 关键差异 / 修复 |
|---|---|---|---|---|
| 名称解析 + 加载编排 | `loader/base.py` | `loader.py` + `register.py` + `dataset_syntax.py` | migrated | 修 legacy 两个 bug：① `_dataset_meta_mapping` 反查缓存**无失效钩子**，导致 `--custom_dataset_info` 里后注册的条目永远查不到（dev 用 `_ID_MAPPING.clear()`）；② registry miss 语义——dev 明确返回基类而非 raise（任意 hub id / 本地 jsonl 都是合法数据集）。另含 JSON 声明入口 `register_dataset_info` + `DatasetLoader.from_dict`（合成 loader 子类），以及 `columns` 的**两级声明位**（loader 级 + `SubsetMeta` 级） |
| 输入格式归一 | `format_converter/` | `preprocessor/core.py` | migrated | 拆成 `FormatConverter` + `get_converter` 工厂 + `register_format`；`priority` 显式声明探测顺序（legacy 藏在 if-chain 的书写顺序里）；`apply_aliases` 改为**第一个赢**（legacy `_to_std_key` 是最后一个赢，结果依赖字面量顺序）。alias 表已与 legacy 逐项对账（见批次 4 发现的缺失别名）。批次 5 补 `FormatConverter.MEDIA_ALIASES`（`image`/`audio`/`video` → 复数），与 `aliases` 分开放，避免子类声明自己的表时把它覆盖掉——legacy 是在 `RowPreprocessor.__init__` 里给所有 preprocessor 硬塞这三条 |
| 行变换执行 | `preprocessor/` | `RowPreprocessor` | migrated | `Preprocessor` 只管 map 编排 / 校验 / 丢坏行；列改名下沉到 converter 的 aliases。批次 4 补齐：`converter` 惰构属性、`standardise()`（复用 converter 自己的 alias 表让子类能像 legacy 一样读标准列名）、`random_state`（seeded，取代 legacy 的全局 numpy）、`converter_kwargs`、`MessagesRepairPreprocessor`。批次 5 补齐：`prepare_dataset()` 钩子（map 前一次性取媒体归档）、`pin_features()`（scoped 钉 Arrow 列类型，详见批次 5「一处自我纠正」） |
| 多模态资源下载 | `mm_download/` | `media.py` | migrated | `MediaResource._safe_download` 的三分支 → 三个策略子类 + 注册表工厂。**修非原子下载 bug**：legacy 直接解压进 `final_folder`，中途崩溃留下不全目录，此后永远被当成完整缓存；dev 落 `.tmp` 后原子 `rename` |

## 待建层 —— 已清零

上一版列的五层，批次 7/8 全部落地：

| 层 | 落地位置 | 批次 |
|---|---|---|
| syntax DSL 解析 | `loader/base.py:parse_legacy_syntax`（仅向后兼容，非一等语法） | 8 |
| cls → 生成式改写 | `loader/llm.py:ClsGenerationPreprocessor` | 7 |
| prompt 模板 | `loader/llm.py:AdvertiseGenPreprocessor` | 7 |
| 多塔样本构造 | 不需要新层——`MESSAGE_COLUMNS` 早已含 `positive_messages`/`negative_messages` | 7 |
| self-cognition 注入 | `loader/llm.py:SelfCognitionLoader.build_preprocessor` | 7 |

> 一个修正：更早的版本把「`ClsPreprocessor`（label→int）」也列为待建层。批次 4 迁 HC3 时实测发现：`label` 只需一个 `int` 穿透，`Preprocessor` 基类已经能做（`hc3_cls` 子集已与 legacy parity），**不需要单独一层**。真正缺的是 `ClsGenerationPreprocessor` 那种把标签重写成选项文本的改写。

---

# 二、数据集批次

## 批次 1：pilot（3）

打通端到端流程用的最小集，故意覆盖三种接入姿势。

| legacy ms id | dev loader | 声明档位 | 结论 | 依据 / 备注 |
|---|---|---|---|---|
| AI-ModelScope/alpaca-gpt4-data-zh | `AlpacaZhLoader` | 3 | migrated | 微调字段后 `super()`：剥掉 `input` 的 `'输入：'` 前缀 |
| AI-ModelScope/LongAlpaca-12k | `LongAlpacaLoader` | 3 | migrated | 同上：剥掉 `output` 的 `'Answer: '` 前缀 |
| AI-ModelScope/ruozhiba | `RuozhibaLoader` | 3 | migrated | 多 subset（3）+ **全自建行**，不经 converter；pretrain 型，单条 assistant turn |

## 批次 2：纯文本扩充（20）

| legacy ms id | dev loader | 声明档位 | 结论 | 依据 / 备注 |
|---|---|---|---|---|
| swift/ToolBench | `ToolBenchLoader` | 1 | migrated | |
| swift/sharegpt | `ShareGptLoader` | 1 | migrated | 3 subset |
| modelscope/gsm8k | `Gsm8kLoader` | 1 | migrated | subset `main` |
| modelscope/MathR | `MathRLoader` | 1 | migrated | 2 subset |
| modelscope/MathR-32B-Distill | `MathRDistillLoader` | 1 | migrated | subset `data` |
| tastelikefeet/competition_math | `CompetitionMathLoader` | 1 | migrated | **per-subset `split`**：`SubsetMeta('default', split=['train','test'])`，test 也拿来训 |
| AI-ModelScope/ultrafeedback-...-kto | `UltrafeedbackKtoLoader` | 1 | migrated | kto |
| OmniData/Zhihu-KOL | `ZhihuKolLoader` | 1 | migrated | `huge_dataset=True` |
| OmniData/Zhihu-KOL-More-Than-100-Upvotes | `ZhihuKolFilteredLoader` | 1 | migrated | |
| AI-ModelScope/sql-create-context | `SqlCreateContextLoader` | 2 | migrated | pin `format_name='alpaca'` + 3 列改名 |
| codefuse-ai/CodeExercise-Python-27k | `CodeExerciseLoader` | 2 | migrated | `chat_rounds`→`messages` |
| AI-ModelScope/math-trn-format | `MathTrnLoader` | 3 | migrated | 只取 query/response，忽略记账列 |
| AI-ModelScope/firefly-train-1.1M | `FireflyLoader` | 3 | migrated | **行过滤**：`kind` 不在 24 个白名单内的丢弃 |
| AI-ModelScope/blossom-math-v2 | `BlossomMathLoader` | 3 | migrated | 先 pop `answer`（它也是 response 的 alias，不 pop 会被误当答案），转换后再把数值答案追加到解题过程末尾 |
| AI-ModelScope/synthetic_text_to_sql | `SyntheticText2SqlLoader` | 3 | migrated | schema 折进问题、explanation 折进答案（CoT） |
| AI-ModelScope/leetcode-solutions-python | `LeetcodePythonLoader` | 3 | migrated | 单列拆成一轮的两半（```python 分界） |
| AI-ModelScope/tigerbot-law-plugin | `TigerBotLawLoader` | 3 | migrated | pretrain 型：法条标题 + chapter1-3 + 正文拼一段；正文列名用 `pop_first` 兜底 |
| AI-ModelScope/databricks-dolly-15k | `Dolly15kLoader` | 3 | migrated | 有 context 时前置为参考段落 |
| hjh0119/shareAI-Llama3-DPO-zh-en-emoji | `EmojiDpoLoader` | 3 | migrated | 扁平 DPO；清洗按**原始列名**做（改名发生在 converter 内，`preprocess` 还没走到那步） |
| AI-ModelScope/orpo-dpo-mix-40k | `OrpoDpoMix40kLoader` | 3 | migrated | 行过滤（丢 `toxic-dpo-v0.2`）+ `chosen`/`rejected`→`messages`/`rejected_messages` |

### 本批次的一个纠正：扁平 DPO 不在 dataset 层展开

parity 测出 `shareai-dpo-emoji` 差异后追查发现，我此前在 `ResponseConverter` 里把扁平 `rejected_response` **提前展开**成 `rejected_messages` 了。这是越界：

- 展开是 template `_compat_rejected_response` 的职责，且它顺带做两件校验——断言 rejected ≠ chosen、拒绝 rejected 里出现 `user` role；
- `template_inputs.py:195/210` 会因 `rejected_messages` 已存在而走不同分支。

提前展开会**静默跳过这些校验**。已改为保持扁平，并实测确认：template 展开结果结构正确（含多轮 history 前缀），且 `rejected == chosen` 的断言真的触发。

## 批次 3：`dataset_info.json` 声明式（111）

legacy `swift/dataset/data/dataset_info.json` 整份搬过来，**字节级相同**（字段全部兼容，无需转换）。不写 111 个 Python 类——它们本就是纯数据。

**新增的入口**：

| 件 | 作用 |
|---|---|
| `register_dataset_info(path_or_entries, *, exist_ok=False)` | 读 JSON（或直接收 entry 列表）→ 逐条合成 loader 子类 → 注册。也是未来 `--custom_dataset_info` 的入口 |
| `DatasetLoader.from_dict(entry, *, base_dir, exist_ok)` | 单条 entry → loader 子类。子类的是 `cls` 而非 `DatasetLoader`，所以某个 family 可以对自己调用此方法来暂开一个 JSON 可声明变体 |
| `SubsetMeta.columns` / `DatasetLoader.columns` | `columns` 的两级声明位。per-subset 是必需的：同一数据集的兄弟 subset 往往是不同源的 dump（`medical_zh` 的 zh 叫 `instruction`、en 叫 `input`） |

**关键设计决定**：

- `dataset_type` 取 id 的 **basename**。已实测 111 个 basename 全局唯一，且与已有 23 个 `dataset_type`、与已注册 id 均无碰撞。
- JSON 注册在 `__init__.py` 的**模块体末尾显式调用**，不依赖 import 顺序——无论 ruff 怎么重排 import，Python 手写 loader 永远先注册。
- 用默认 `exist_ok=False`：一个数据集若同时在 JSON 和 Python 里声明，**报错而不是静默覆盖**。
- 那 5 条**只有 `hf_dataset_id`** 的条目（`allenai/c4`、`HuggingFaceFW/fineweb`、`HuggingFaceTB/cosmopedia`、`tiiuae/falcon-refinedweb`、`cerebras/SlimPajama-627B`）已全部落地，已实测 hf 侧反查命中。

**已测的边界情况**：

| 情况 | 条目 | 结果 |
|---|---|---|
| per-subset `columns`（dict 型 subsets） | `medical_zh` 等 4 条 | zh/en 各自生效，与 legacy parity |
| entry 级 `columns` | 35 条 | 与 legacy `AutoPreprocessor` 逐行 parity |
| 调用方 `--columns` 与声明冲突 | — | 调用方胜（与 legacy 一致） |
| `help` 字段是 list 而非 str | 1 条 | 正常处理 |
| id 以数字开头（非法标识符） | `100PoisonMpts` | 类名加前缀 `Dataset_`，`dataset_type` 保留原名 |
| 纯 hf 条目的 ms 侧解析 | 5 条 | `resolve_id(use_hf=False)` 返回 None → 退回原始 id 交 hub 报错，与 legacy 同行为 |

## 批次 4：`llm.py` 收口（17）

legacy `llm.py` 剩下的全部非阻塞项。自此 **`llm.py` 待迁清零**，只剩 7 个阻塞项。

### 先补的四个基础设施

这批数据集靠现有抽象写不了，所以 `preprocessor/base.py` 先长了四样东西：

| 件 | 为何需要 |
|---|---|
| `Preprocessor.converter`（属性） | 把原来埋在 `preprocess` 里的惰构建提出来，供下面两项复用 |
| `Preprocessor.standardise(row)` | **契约差异的桥**。legacy 先全表改名再 `preprocess`，所以它的 `preprocess` 体写的是 `row['query']`；dev 的 `preprocess` 看到原始行。该方法复用 **converter 自己那张 alias 表**（不是第二份拷贝）把行归一，让 Sudoku/Countdown/HC3 这类数据集能忠实迁过来——也顺手消掉了上一版登记的「按假设的原始列名读字段」风险 |
| `Preprocessor.random_state` | HC3 / xlam-grpo 要在多个候选答案里选一个。seeded（默认 42）；legacy 的 xlam-grpo 用的是**全局未播种的 `np.random`**，两次跑结果不同，这里改成可复现 |
| `MessagesRepairPreprocessor` | 对话列得先修再转。legacy 是 `MessagesPreprocessor(repair_messages=某个函数)`，修复逻辑落在声明的参数表里；现在是 `repair()` 覆盖点，每个修复都有名字、带注释 |

### 数据集

| legacy ms id | dev loader | 声明档位 | 结论 | 依据 / 备注 |
|---|---|---|---|---|
| open-r1/DAPO-Math-17k-Processed | `DapoMath17kLoader` | 1 | migrated | subset `all` |
| iic/ms_bench | `MsBenchLoader` | 3 | migrated | 丢样样子 system；并丢掉泄露其他助手人格的行（`moss`/`human:`/`assistant:`/`user:`） |
| huangjintao/AgentInstruct_copy | `AgentInstructLoader` | 3 | migrated | 6 subset；修丢了逗号的 repr |
| AI-ModelScope/lmsys-chat-1m | `LmsysChat1mLoader` | 3 | migrated | 同上，四种缺口写法 |
| shenweizhou/alpha-umi-toolbench-processed-v2 | `AlphaUmiToolbenchLoader` | 3 | migrated | 4 subset，`huge_dataset`；`caller`/`conclusion` → `assistant` |
| damo/MSAgent-Bench | `MSAgentBenchLoader` | 3 | migrated | **per-subset preprocessor**：default 全收，`mini` 只留多 plugin 的行且为 weak subset |
| iic/MSAgent-MultiRole | `MultiRoleAgentLoader` | 3 | migrated | 群聊压成一轮：前面的发言进 system；带 `next_speakers:` 的不动 |
| AI-ModelScope/function-calling-chatml | `FunctionCallChatmlLoader` | 3 | migrated | 工具 schema 从单列拆成 `tools`，再丢 system（否则 prompt 重复一份） |
| AI-ModelScope/GuanacoDataset | `GuanacoLoader` | 3 | migrated | 转录在 `instruction` 里，前缀有 7 种拼写（含 `Assistenz:` 等错字）；破坏交替或缺答的行丢弃 |
| AI-ModelScope/hh-rlhf | `HHRLHFLoader` | 3 | migrated | 3 subset，`huge_dataset`；两份转录各自拆成 `messages`/`rejected_messages` |
| AI-ModelScope/hh_rlhf_cn | `HHRLHFCNLoader` | 3 | migrated | 5 subset；`columns={'context':'messages'}` + `converter_kwargs={'content_key':'text'}`；rejected 保持扁平 |
| simpleai/HC3 | `HC3Loader` | 3 | migrated | 2×2 subset（生成 + cls）；**一行扇出两行** |
| simpleai/HC3-Chinese | `HC3ChineseLoader` | 3 | migrated | 7×2 subset，同上 |
| modelscope/DuReader_robust-QG | `DureaderLoader` | 3 | migrated | 反向 QA：给段落+答案生成问题；`text1` 按 `[SEP]` 拆 |
| zouxuhong/Countdown-Tasks-3to4 | `CountdownTaskLoader` | 3 | migrated | RL 任务：故意**不给 assistant turn**，答案当 `target` 做奖励 |
| sapientinc/sudoku-extreme-1k | `SudokuLoader` | 3 | migrated | 81 字符单行折成 9 行 |
| LLM-Research/xlam-function-calling-60k | `XlamFunctionCallingLoader` | 3 | migrated | 2 subset：default 出 `tool_call` 消息，grpo 出文本 `Action:` |

### 本批次找到的两个真缺陷

**1. alias 表漏了 `answers` 和 `answer_key`。** xlam-grpo 读 `row['response']` 直接 `KeyError`。legacy 的 `response_keys` 有这两个，我的 `ResponseConverter.aliases` 没有——之前的 format parity 用例恰好没覆盖到。已补齐，并新增一个**表级对账测试**（把 legacy `ResponsePreprocessor().columns` 与我的表按 system/query/response 三类逆向比对），以后不依赖逐例运气：现为 3/3 OK。

**2. `standardise` + 转换的二次 alias 陷阱。** 转换会**再跑一遍** aliases。重读已标准化的行无害（标准名不被覆盖），但**把带别名的列重新放回行上就不是**：countdown 把 `target` 塞回去，第二遍 alias 把它升成 `response`，凭空多出一条 assistant turn（内容还是个 `int`）。legacy 因为只在开头改一次名而没这个问题。修法：这类列（`target`、`solution`）改成设在转换**返回的**行上，已写进 `standardise` 的 docstring 当告警。


---

## 批次 5：`mllm.py` 多模态首批（12）

第一批多模态数据集。分组依据不再是「要声明多少」，而是**媒体在哪里**：行内已有可用引用（URL / 绝对路径 / bytes），还是只有一个相对路径、需要先把归档拉下来。

### 先补的三个基础设施

| 件 | 位置 | 为何需要 |
|---|---|---|
| `Preprocessor.prepare_dataset()` | `preprocessor/base.py` | 媒体归档是**整个数据集一次**的事，不能每行做。钩子在 map 前跑，存在 `self` 上的路径会随 pickle 到 `num_proc>1` 的 worker（与 legacy 同机制） |
| `FormatConverter.MEDIA_ALIASES` | `format_converter/base.py` | legacy 在 `RowPreprocessor.__init__` 里给**所有** preprocessor 加了 `image`/`audio`/`video` → 复数；dev 的 alias 表按 format 分类，而媒体列是行级属性、与对话形状无关，所以放基类并与 `aliases` 分开合并 |
| `Preprocessor.pin_features()` | `preprocessor/base.py` | 钉住那几个推断不稳定的列的 Arrow 类型（详下） |

### 一处自我纠正：`_patch_arrow_writer` 不能只钉 images

上一版本文件（和一条记忆）写的是「实证只有 `images`/`rejected_images` 真需要强制 schema，其余 6 字段多余且有害」。本批次重测后确认：**这个结论的适用边界弄错了**。

上次只测了「单一数据集内跳 batch 边界」；没测「两个各自 preprocess 完的数据集 / subset 拼接」——而后者才是 ms-swift 真实的混训场景。实测（datasets 4.7.0）：

| 场景 | 不钉 | 钉住 |
|---|---|---|
| `messages` 一边带 `loss_scale` 一边不带，拼接 | **FAIL**（features can't be aligned） | OK |
| `images.bytes` 一边 `null` 一边 `binary`，拼接 | **FAIL** | OK |
| `objects.bbox` 一边 int 一边 float，拼接 | **FAIL** | OK |
| `images` 首行为 None、后行才有路径（map 内） | **FAIL**（cast 不过去） | OK |
| 缺列拼接、`objects` 键不同、`chat_template_kwargs` 键不同 | OK | OK |

所以钉的字段集合与 legacy 4.x 分支一致（messages 四件套 + images/rejected_images + objects/chat_template_kwargs）。legacy 真正的缺陷在另一处：它**无条件**给每个数据集塑入全部键，纯文本数据集也会凭空多出全 null 的 `images`/`objects`/`rejected_messages`。dev 只钉**输出里真存在的列**，已反证（纯文本数据集输出仍只有 `messages` 一列）。

> 一个未解释的观察：我两次用探针 spy `ArrowWriter.__init__` 都记录到 **0 次构造**，但 pin 确实生效（A/B 对照：开→`Json` + concat 通过，关→struct + FAIL）。机制层没查清，改为**按结果验收**（断言输出 features 已被钉）。

### 数据集

| # | legacy ms id | dev dataset_type | 声明档 | 要点 |
|---|---|---|---|---|
| 1 | swift/RLAIF-V-Dataset | `RLAIF-V-Dataset` | 档 2 | 零 preprocessor，只靠 loader 级 `columns`（含 `rejected_response`） |
| 2 | swift/gpt4v-dataset | `gpt4v-dataset` | 档 3 | 固定 query（`FixedQueryPreprocessor`） |
| 3 | tany0699/garbage265 | `garbage265` | 档 3 | 固定 query + `label`（无 assistant 轮） |
| 4 | modelscope/coco_2014_caption | `coco-en-mini` | 档 3 | `&&` 拆多 caption 取第一个；2 subset |
| 5 | speech_asr/speech_asr_aishell1_trainsets | `aishell1-zh` | 档 3 | 音频；转录按字空格分隔需去空；3 subset |
| 6 | Tongyi-DataEngine/SA1B-Paired-Captions-Images | `sa1b-paired-caption` | 档 3 | 随机中文 caption 提示（seeded） |
| 7 | Tongyi-DataEngine/SA1B-Dense-Caption | `sa1b-dense-caption` | 档 3 | `cap_seg` 字符串解成 dict 取 `global_caption` |
| 8 | swift/pixelprose | `pixelprose` | 档 3 | 随机英文 caption 提示；去 VLM 前缀 |
| 9 | AI-ModelScope/ShareGPT-4o | `sharegpt-4o-image` | 档 3 + 归档 | 图在归档里深层目录（上传者集群的绝对路径被打进包） |
| 10 | swift/Mantis-Instruct | `mantis-instruct` | 档 3 + 归档 | 17 subset，**每个 subset 一个归档**；多图行缺一张就丢整行 |
| 11 | swift/llava-data | `llava-data-instruct` | 档 3 + 归档 | 6 个公开图库，路径前缀定位归档 |
| 12 | AI-ModelScope/egoschema | `egoschema` | 档 3 + 归档 | 视频；default 出选项字母 / cls 出选项下标 |

两个结构上的决定：

- **`ArchiveMediaPreprocessor` / `ArchiveImagePreprocessor` 两层**。前者只管取归档，后者才带「解析行内图片路径 / 缺文件就丢行」。最初只写了一层，结果 EgoSchema（视频）不得不用 `super(ArchiveMediaPreprocessor, self)` 跳级——那就是分层切错了的信号。
- **per-subset preprocessor 用类工厂**。loader 把 preprocessor 声明为**类**，没有传参的调用点，所以 Mantis 用 `MantisPreprocessor.for_subset(name)` 生成 17 个绑定子类。

### 本批次找到的两个真缺陷

**1. 四个多模态数据集静默丢掉图像列**。legacy 的 `batched_preprocess` 不会把 `images` 补回返回行，而 `SA1BPairedCaption`、`SA1BDenseCaption`、`Ocrvqa`、`ScienceQA` 四个 preprocessor 都是 `return {'messages': ...}`。已用 legacy 代码实测确认：输出列只有 `messages`。于是这些标着 `multi-modal` 的数据集产出的是纯文本行，而且问题是「图片中展示了什么」——没图根本无法回答。旁边同作者的 `PixelProse` 却显式带上了 `images`，可以看出是遗漏而非设计。dev 保留图像列（本批已修 SA1B 两个，OCR-VQA / ScienceQA 随下批）。

**2. EgoSchema 只下载到了五分之一的视频**。legacy 用 `for i in range(1,6)` 循环调 `download(url_i, 'egoschema')`，但下载函数在目标目录已存在时直接返回——第一个 zip 建好目录后，剩下四个全被跳过。属于后四个分片的行随后被 `mp4_set` 过滤静默丢掉。dev 改用 `file_type='sharded'` 一次传五个 URL（这本来就是 `ShardedDownloader` 的用途，legacy 自己在 LLaVA-Video-178K 里就是这么用的），已断言验证。

## 批次 6：与 DataLoader 对接的五个组件

前五批迁的是「数据集条目」，这一批迁的是**机制**：把标准 messages 行变成 DataLoader 能吃的东西。数据集条目覆盖率一动不动（163 注册不变），但这五个组件缺一个，训练就跑不起来。

每个类一个文件，直接放在 `dataset/` 下（不进 `preprocessor/`）——它们的产物不是「标准行」，而是 `input_ids` 或 pack 好的样本组，属于管线的下一段。`PackingDataset` 与 `IterablePackingDataset` 同文件，因为后者依赖前者的 `calculate_matched_group`。

| 组件 | dev 文件 | legacy 位置 | 作用 |
|---|---|---|---|
| `PackingDataset` | `packing.py` | `packing.py` | 把若干短样本拼成一条定长序列。先跑一遍长度统计再规划分组，`__getitem__` 返回**一组行**（拼接由 collator 做） |
| `IterablePackingDataset` | `packing.py` | 同 | 流式版。长度未知，所以边取边攒；编码放在子进程（`mp` 队列），因为流不允许预处理趟 |
| `LazyLLMDataset` | `lazy_dataset.py` | `utils.py:145` | 取行时才编码。坏行不 crash 而是**随机换一行**（`n_try_fetch` 次），`strict=True` 时直接抛 |
| `EncodePreprocessor` | `encode_preprocessor.py` | `utils.py:115` | 预先把全表编码成 `input_ids`（与 lazy 相对的 eager 路径） |
| `AddLengthPreprocessor` | `add_length_preprocessor.py` | `utils.py:125` | 编码**只为量长度**，编码结果丢掉、只往原始行加 `lengths` 列——packing 规划分组要用 |

### 顺带修的一处 parity 差异

`EncodePreprocessor` 是唯一会在 `map` 里抛 `MaxLengthError` 的路径，接上后发现 dev 的 `batched_preprocess` 对它的处理与 legacy 不同：legacy 硬编码 `ignore_max_length_error=True`，超长行**静默丢弃**；dev 把它当普通坏行，打一条 traceback 并**吃掉 `traceback_limit` 配额**——结果是真正的数据错误可能因为配额被超长行用光而看不到日志。已改为单独识别 `MaxLengthError` 并静默 `continue`。实测：dev 3 行 == legacy 3 行、`input_ids` 相同、配额只被真错误消耗（counter=1）、`strict=True` 两边都抛。

### 一处已知未对齐（登记，不在本批解决）

legacy 的 `map` 外面还包了 `safe_ddp_context`（rank0 先跑、其余 rank 复用 cache）与显式 `cache_file_name`。两者 dev 都已补上（见批次 8），串行化改用 `twinkle.utils.processing_lock`。

### 两条编码路径的完整链路（与 twinkle 对比）

这五个组件怎么串起来的，以及为何 `AddLengthPreprocessor` 看着像多余的一步。以非流式 + `--packing` 为例：

```
swift：
  _encode_dataset()  sft.py:321-331
    AddLengthPreprocessor 经 map      【encode #1】编码全表，只取 lengths，return row
  _post_process_datasets()  sft.py:136-148
    LazyLLMDataset(ds, template.encode)  ds[int] 现编码；ds[str] 透传到底层 HF 列
    PackingDataset(template, lazy_ds)    self.dataset['lengths'] 靠上面那个透传拿到 #1 的产物
                                         master 装箱 → broadcast_object_list 给其他 rank
  训练：PackingDataset[i] → [lazy_ds[j] ...]  【encode #2，每 epoch 每行】返回 list，不合并
  template.data_collator  base.py:1668   packing_row(...) 在这里才合并

twinkle：
  PackingDataset(dataset_meta)          它本身就是 Dataset 子类（is-a）
  .encode()                            【encode #1，也是唯一一次】materialize input_ids/labels/length
                                       length 在 template/base.py:336 顺手写，无独立趟
  .pack_dataset()                      assert 'input_ids' in dataset[0]；lengths = dataset['length']
  训练：PackingDataset[i]                读已存的行，__getitem__ 里自己合并成单个 dict
```

| | swift | twinkle |
|---|---|---|
| 关系 | `PackingDataset(dataset)` **包装** | `PackingDataset(Dataset)` **继承** |
| encode 次数 | **2**（1 趟只取长度 + 每 epoch 每行一次） | **1** |
| `length` 来源 | 专门一趟 `AddLengthPreprocessor` | encode 内部顺手写 |
| pack 合并位置 | **collator**（`__getitem__` 返回 list） | **`__getitem__` 内部** |
| 分布式 | master 装箱 + `broadcast_object_list` | 各 actor 自己装箱 |
| 坏行 | 训练时随机替换（`n_try_fetch` 次） | encode 后 `.filter` 一次性丢掉 |
| lazy + packing | 显式 `raise ValueError` | `assert 'input_ids' in ...` 挡住 |

最后一行值得注意：**「lazy 不能配 packing」两边都成立**，不是 swift 特有的限制——bin-packing 必须在训练前知道全部长度，而长度只能靠 encode 得出，那一趟全表 encode 无论如何躲不掉。真正的分叉点只有一个：encode 完之后 `input_ids` 留不留。twinkle 留（一次到位），swift 不留（所以必须拿 `lengths` 列当备忘，并在 `__getitem__` 重新编码）。

swift 不留的理由只有两条站得住：① `template_mode` / `task_type` 切换时编码的**结构**会变（实测：sft 出 `['input_ids','labels']`，rlhf 出 `['chosen_input_ids','rejected_input_ids',...]`），存原始文本才能一份缓存喂 sft/dpo/kto/seq_cls/reranker；② 多模态体积，`template.encode` 会连 `pixel_values` 一起产出。**除此之外，同 tokenizer 同配置的纯文本场景下这一趟是净亏的**——多付一次全表编码，只换一列数字，而那列数字的配置绑定程度和 `input_ids` 一样。

### legacy 的一处设计隐患：陈旧 `lengths` 会让 pack 超长

上一节的最后一句引出一个真问题，与 dev 迁移无关但值得登记。`cached_dataset` 重载时 [`_select_dataset`](../../pipelines/utils.py) **不校验也不重算** `lengths`（只做了 3.x 的 `length`→`lengths` 改名兼容）。于是开 `--packing` 时：

1. `PackingDataset` 拿陈旧 `lengths` 规划分组，保证每组和 ≤ `packing_length`
2. `__getitem__` 又用**当前** template 重新编码那些行
3. 两者不一致时，一个 pack 的真实 token 数可能超过规划值，而链路上**没任何一处重校**

共同作用的还有 `LazyLLMDataset` 的坏行随机替换：装箱规划用的行集合与实际服务的行集合可以不同（twinkle 提前 filter 掉，不存在这个问题）。

另外，`load_from_cache_file` 默认是 `False`（`data_args.py:83`，文档建议实跑时设 `true` 但默认值没跟上），所以量长度那趟默认**每次启动都重跑**。

三项均为 legacy 行为，dev 已逐字 parity，因此**本次不改**——改它们属于设计变更，不是迁移。

### 目录重构的连带修复

本批开工时包是坏的：`base.py`/`llm.py`/`mllm.py` 已移入 `loader/`、`format/` 已改名 `format_converter/`，但接线没跟上（`loader/` 缺 `__init__.py`、顶层仍写 `from . import llm`、子模块的相对 import 层级失效）。已补 `loader/__init__.py` 并修 5 处相对 import，`import swift.dev.dataset` 恢复正常、163 注册不变。

---

## 批次 7：收口（34）—— 数据集条目清零

把剩下的 34 个一次迁完，**覆盖率 163 → 197，与 legacy 完全相等（差集为空）**。分三组落地：

| 组 | 数量 | 数据集 | 需要新建的能力 |
|---|---|---|---|
| 文本 / 分类 / 嵌入 | 8 | AdvertiseGen、clue(cmnli)、jd、stsb(4 子集)、MTEB ×2、self-cognition | `ClsGenerationPreprocessor`（标签集写进 prompt）、`model_name`/`model_author` 接线 |
| 图像 / 音频 | 21 | A-OKVQA、OK-VQA、lnqa、Midefics、OCR-VQA、ScienceQA、LaTeX_OCR、captcha、clevr、voc2007、geometry3k、llava-mix-vsft、TextCaps(3 子集)、coco、ShareGPT4V(2)、LLaVA-Instruct-150K、LLaVA-Pretrain、GQA、refcoco、refcocog、GRIT、Qwen3-TTS | `GroundingPreprocessor`（提示表 + 两种任务向） |
| 视频 / 特殊 | 5 | VideoChatGPT、MovieChat-1K、LLaVA-Video-178K(8 子集)、M3IT(49 子集)、Multimodal-Mind2Web | `FilesDownloader`（`file_type='files'`）、`prepare_dataset` 做数据集级折叠 |

### 之前登记为「阻塞」的 7 项，全部不需要新层

上一版把 self-cognition / jd / clue / AdvertiseGen / stsb / MTEB ×2 记作「层缺失」。实测后只有两项真的要动基础设施，其余五项是**已有机制的普通用法**：

| 原判定 | 实际 |
|---|---|
| 需要 embedding / reranker 多塔层 | 不需要。`Preprocessor.MESSAGE_COLUMNS` 早已含 `positive_messages`/`negative_messages`，Arrow 层直接就能存；preprocessor 自己造行即可 |
| 需要 prompt 模板层 | 不需要。就是子类里一个 `prompt.format(...)` |
| 需要 cls→生成式改写层 | 需要一个类，但不是「一层」：`ClsGenerationPreprocessor` 声明 `labels` / `task` / `sentence_keys` 三个类属性 |
| 需要 `set_name_author` 注入 seam | 需要，但改的是 loader 而非 preprocessor 层：`SelfCognitionLoader.build_preprocessor()` override 一次。legacy 反过来做——在**共享**加载路径里遍历所有数据集的 preprocess_func 去 isinstance 判断（`loader.py:212`） |

### 顺带修掉的 legacy 缺陷（6 处）

| 数据集 | legacy 行为 | dev |
|---|---|---|
| lmms-lab/GQA | `if os.path.join(...)`——拼出的路径是非空串，恒真，**图片不存在的行照样留下** | 改 `os.path.exists` |
| MovieChat-1K-test | ~150 个 mp4 逐个用 `file_type='file'` 下到**同一个别名目录**；目录一存在就跳过 → 只有第一个文件真的落地，其余行全被当缺媒体丢掉 | 新增 `file_type='files'`，一次资源、原子提升 |
| AI-ModelScope/LLaVA-Pretrain | `return {'images': path}`——整行只剩图片，**caption（messages）被丢掉** | 保留整行 |
| swift/ScienceQA | 从零造行，`images` 没接回去 → 标 multi-modal 却产出纯文本行 | 接回 `images` |
| swift/VideoChatGPT | `os.listdir` 写在 `preprocess` 里，**每行列一次目录** | 提到 `prepare_dataset` 一次 |
| swift/OCR-VQA、grounding 提示 | 用全局 `np.random` 抽题/抽提示，同一份数据两次运行结果不同 | 用 seeded `self.random_state` |

### 两处刻意不与 legacy 逐字相同

1. **`AI-ModelScope/captcha-images` 不再多留一个 `solution` 列。** legacy 的 `__#solution` 保护对**任何**含 `solution` 列的数据集都生效，而这个数据集里 `solution` 只是答案列名，复制出来是同一个字符串两遍。dev 只在真需要的地方显式写回（`clevr_cogen_a_train`，标了 grpo）。
2. **`AI-ModelScope/coco` 的 `objects` 不再带 `category`。** 索引在 preprocessor 里已被翻成 `ref`，用完即弃。legacy 是靠 `_patch_arrow_writer` 钉死 `objects` 的 struct 顺手丢掉的，不是有意为之。

### 本批次的验证

`swift/dev/tests/test_dataset_parity.py`（新建，**37/37 通过**，与 `test_swift_dataset.py` 合计 52/52）。口径：同一份合成行分别喂 legacy preprocessor 和 dev preprocessor，比对 `None` 值剔除后的整行。

> 对账时发现一处**口径陷阱**：legacy 的内建别名表（`question`→`query` 等）只在 `enable_auto_mapping=True` 时生效（`core.py:341`），而这个标志由 legacy 的 loader 传入。不传就只有显式声明的 rename 起作用——最初 6 个失败用例里有 4 个是这个原因，不是代码差异。dev 没有这个开关：alias 是 converter 自己的知识，恒定生效。

随机类不适用逐字比对的，改为断言式：grounding 提示表与 legacy `_grounding_prompts` **逐字相等**（防措辞漂移）+ 抽出的对必属于表；OCR-VQA 断言 question/answer 配对正确。Mind2Web 的折叠断言「2 episode × 2 action → 2 行而非 4 行」，且只有起始 action 被告知任务目标。

---

## 批次 8：机制收口 —— 条目之外的功能对账

数据集条目在批次 7 已清零，本批次处理的是「机制」：legacy `load_dataset` / `RowPreprocessor` 上那些不属于任何单个数据集、但影响整条管线的能力。

### 落地的（6 项）

| 机制 | dev 位置 | 说明 |
|---|---|---|
| 混训编排 5 个参数 | `loader/base.py:load_dataset` | `shuffle` / `interleave_prob` / `stopping_strategy` / `shuffle_buffer_size` / `hub_token`。`interleave_prob` 走 HF 官方 `interleave_datasets`（两种 dataset 类型都支持），`shuffle` 作用于合并后的结果 |
| 旧 DSL 向后兼容 | `loader/base.py:parse_legacy_syntax` | `hf::org/name:sub1/sub2#500` 拆成 `(dataset, subsets, sample_count, use_hf)`。**不作为 dev 的一等语法**：dev 的正道是传 `subsets=` / `use_hf=` 参数，这个 util 只为让已有脚本继续跑 |
| streaming 的 train/val 切分 | `loader/base.py:split_streaming` | 此前 dev 对 `IterableDataset` 调 `train_test_split` 会直接崩。现在 `#N` 有界时从头切，无界时只接受 ratio 0/1，其余明确报错 |
| bbox 校验与归一 | `preprocessor/base.py:check_objects` | 长度必须 2 或 4；角点颠倒（`x1>x2`）自动交换 |
| 临时缓存目录 | `loader/base.py:use_swift_cache_for_temp_files` | Arrow 临时文件从 `/tmp` 挪到 swift cache。**改了口径**：legacy 是 import 副作用，dev 由 `load_dataset` 显式调用 |
| map 缓存 | `preprocessor/base.py:map_cache_path` | 无 `cache_files` 的内存数据集（`from_list` 等）按 fingerprint 落盘，避免每次启动重算 |

### 判定为不需要迁的（4 项，均有实测背书）

| 机制 | 实测结论 |
|---|---|
| `_cast_pil_image` | 真实 `List(Image(decode=True))` 数据集过两边，输出 feature 与行内类型完全一致，**bytes 逐字节相同（152→152）**——没有发生 PIL 解码-重编码。dev 的 `pin_features` 已达成同一效果。结论绑定 datasets 4.7.0 |
| `_check_rejected_response` | `template_inputs.py:166-186` 的 `_compat_rejected_response` 已做同样三项：类型必须 str/list、rejected 里禁 `user` 角色、`assert rejected != chosen`。legacy 那份是重复校验 |
| `__@` iterable 前缀（datasets#6408） | 同一 iterable 数据集：dev 正常产出，**legacy 反而 `IndexError`**。该 workaround 在当前 datasets 上已失效 |
| `_inject_dataset_routing_tag` | 全仓 grep `['dataset']` 除 legacy 自身只命中 1 处（eval 报告字典，与数据列无关）。channel loss 读的是 `channel` 列。加一个无人消费的常量列只会污染 schema |

`get_dataset_list` 也不做——唯一调用方是 Web UI（`swift/ui/`），已确认废弃重写。

### `_check_objects` 只补了一半

legacy 那个函数干三件事，逐件核实后只有两件需要：

- **钉 `ref/bbox/bbox_type/image_id` 的 key 顺序 → 不需要**。dev 把 `objects` pin 成 `Json()`，实测两半 key 顺序不同、一半多带 `bbox_type`、bbox 一半 int 一半 float，`concatenate_datasets` 仍成功且逐行保真。legacy 需要钉顺序，是因为它只能让 Arrow struct 字段严格对齐
- `len(bbox) in {2,4}` 校验 → 需要（下游 `normalize_bbox` 用 `zip(bbox[::2], bbox[1::2])`，长度 3 时静默出错）
- `x1>x2` 交换 → 需要（全仓只有这一处做）

### 顺带修的一处真 bug：`apply_aliases` 会丢列

审计别名机制时发现的，与本批次的机制清单无关但同属 dataset 层。行里同时有 `response` 和 `text`（`text` 是无关的原文留档列）时：

```
修复前 dev : {'messages': [...]}                      ← text 整列消失
legacy     : {'text': '原文留档', 'messages': [...]}   ← 保留
```

原实现在别名撞上已有标准名时 `continue`，把落选列的数据一起扔了，且结果依赖 dict 迭代顺序。改为：**标准名在场则别名保留原名；仅别名之间竞争时按 `aliases` 声明顺序取先**，两条规则都只读输入的键集合，与行的列顺序无关，且不丢数据。

### 本批次的验证

`test_dataset_parity.py` + `test_swift_dataset.py` **52/52 通过**（改动未引入回归）。新机制用本地 jsonl 离线实测：concat 110 行、`interleave` first_exhausted 17 行 / all_exhausted 193 行、`shuffle` 同 seed 可复现、`#20 + ratio 0.25` → 15/5、streaming `#40 + ratio 0.25` → 30/10、streaming + shuffle buffer 正常；bbox `[30,40,10,20]` → `[10,20,30,40]`，长度 3 在 strict 下报错、非 strict 丢行。

> `test_dataset_api.py` 有 45 个失败，全部是 `ModuleNotFoundError: No module named 'swift.dev.configs'`（正确名是 `config`）——该文件的过期引用，与本批次无关，未处理。

### 补审：`BaseDatasetLoader.load` 那一层（首轮机制审计漏掉的）

前面几轮只对了 `load_dataset` 的参数和 `RowPreprocessor` 的行变换，漏了 legacy `loader.py:_load_repo_dataset` / `_load_dataset_path` 这一层。补审后落地：

| 差异 | legacy | dev 落地 |
|---|---|---|
| **hub 抽象层** | `get_hub(use_hf).load_dataset` → ModelScope 走 `MsDataset.load`（含 `try_login`、revision `main`↔`master` 归一、`trust_remote_code=True`、日志降噪） | `loader/base.py:load_from_hub` 复用 `swift.hub`（与 dev/model 用 `safe_snapshot_download` 同一口径），含 `_hf_ds` 解包与 streaming 转换 |
| `#N` 取样口径 | 默认 `shuffle=False` → 取**前 N 行** | `sample_dataset` 加 `shuffle` 参数，默认顺序取前 N |
| `ms_revision` / `hf_revision` | 按 hub 选用 | 此前是**死声明**（`DatasetInfo(...)` 不传 revision，`info.revision` 恒 None，2 个数据集受影响）。已按 hub 接上 |
| 本地目录 | `isdir` 分支 + 把 `dataset_infos.json` 改名 | `build_dataset` 目录分支 + `hide_dataset_infos` |
| csv `na_filter=False` | 传 | 传（否则空字段变 NaN，浮点 NaN 会流到 template） |

> **这一层的实测**：hub 路由用假 hub 打桩验证——`use_hf=False` 命中 MS 分支并带上 `revision`/`token`，`use_hf=True` 命中 HF 分支；LLaVA-Instruct-150K 的 `ms_revision` 现在能取到（`d5db3806...`），HF 侧为 `None`。本地侧：目录 100 行、csv 空字段为 `''` 而非 NaN、`#5` 默认得 `[a0..a4]`（顺序）、`shuffle=True` 得 `[a44,a70,...]`（随机）。

按判定**不做**的三项：多 subset 无 `default` 时 legacy 报错要求指定、dev 静默加载全部（保留 dev 的宽松行为）；本地文件 `cache_dir` 不指向 swift cache；不存在的绝对路径的报错措辞。

仍未迁：**下载重试 `retry=3`**。

### 批次 8 补：多卡串行化改用 `twinkle.utils.processing_lock`

legacy 在 dataset 里用 `safe_ddp_context` 共 6 处（`media.py:52`、`loader.py:56/86/95`、`preprocessor/core.py:334/352`、`dataset_meta.py:114/166`）。dev 全部接上，但用的不是 `swift.utils.safe_ddp_context` 而是 `twinkle.utils.processing_lock`：

| | `safe_ddp_context` | `processing_lock` |
|---|---|---|
| 排序机制 | `dist.barrier()` | 自带 TCPStore（global master → node masters → 其余），无 store 时退化 FileLock |
| 长耗时 | 数据集预处理落到 **NCCL collective watchdog 超时**之下 | 不经 NCCL，不受 watchdog 约束 |
| writer 崩溃 | 等待方死等 | flag 置 `0`，等待方收到 `LockPeerError` |
| 非对称进入 | barrier 数量不匹配 → 挂死 | `sticky=True` 时安全 |

选它还因为依赖方向已成立：`swift/dev` 非测试代码已 import twinkle 37 处（`processor/base.py:4` 是顶层 import），且 twinkle 自己的 dataset 层就是这么用的（`twinkle/dataset/base.py:154/169/201`）。

接入的 4 处：

| 位置 | key | sticky | 理由 |
|---|---|---|---|
| `loader/base.py:build_dataset` | `{dataset}/{subset}/{split}` | 是 | 内容寻址、幂等，晚到的 rank 直接读缓存 |
| `preprocessor/base.py:__call__`（map） | `dataset_preprocess` | 否 | 同一 key 每轮重复使用，需要各轮独立排序 |
| `loader/base.py` 的 `shuffle` / `train_test_split` | `dataset_shuffle` / `dataset_split` | 否 | 两者都会写 indices 缓存文件 |
| `mm_download/base.py:run` | `lock_key` | 是 | **顺带修一个隐患**：`run` 的早退快路径在锁外，各 rank 非对称进入，原先的 barrier 语义会挂死 |

> **实测**：两进程并发同一 key，非 sticky 下 body 都执行且执行区间不重叠（overlap = −0.02s）；只有一个进程进入 sticky 锁时，另一个不挂死。52/52 回归通过。

> ⚠️ **环境提示**：本机 `import twinkle` 解析到的是 `/mnt/workspace/yzhao/tastelikefeet/twinkle`（169 行，只有 FileLock，`processing_lock` 无 `sticky` 参数），不是仓库内的 `twinkle/src`（303 行，含 TCPStore 排序与 `sticky`）。代码按仓库内这份写，验证时用 `PYTHONPATH=twinkle/src:.`。装的那份需要更新，否则 `TypeError: processing_lock() got an unexpected keyword argument 'sticky'`。这里**没有加兼容降级**——把 `sticky` 静默降级会让媒体下载那处重新具备挂死条件，宁可报错。

---

## 批次 9：DataLoader 层切 twinkle —— `legacy_dataloader` 退役

批次 6 把「数据 → input_ids」接上了，最后一段「input_ids → 训练步」一直还跑在 `swift/dev/legacy_dataloader/`（包装 legacy `swift.dataloader` 的 `BatchSamplerShard` / `DataLoaderShard` / `DataLoaderDispatcher`）。本批次整体换成 **twinkle `DataLoader`**，`swift/dev/legacy_dataloader/` 随之退役（运行时已无任何调用者）。

> 该包曾被删除，后**按用户要求恢复保留**（493 行，待用户 review）。因此 `swift.dataloader` 尚未归零：`legacy_dataloader/factory.py` 的模块顶层仍 import 它（另有 1 处延迟 import `swift.megatron.trainers.utils`）。删掉这个包即可归零，详见 PATCH_INVENTORY.md 第 13 节。

### 换下来的对应关系

| 能力 | legacy_dataloader | twinkle `DataLoader` |
|---|---|---|
| 顺序/shuffle | `BatchSamplerShard`（seed+epoch） | `EpochSampler`（同为 seed+epoch，语义一致）|
| DP 分片 | `BatchSamplerShard` 按**全局 rank** stride | `DeviceMeshSampler` 按 **DeviceMesh 的 dp 坐标**切每个 batch |
| group_by_length | `BatchSamplerShard(group_by_length, lengths)` | `EpochSampler(group_by_length, lengths)`，同样调 transformers `get_length_grouped_indices` |
| resume | `ResumableDataLoaderWrapper`（dev 自写，数 batch） | 内置 `skip_consumed_samples` / `get_state`（数 sample）|
| iterable 分发 | rank0 `DataLoaderDispatcher` scatter | `DeviceMeshDataset` worker-fetcher（各 rank 自切片）|
| 失败重试 | 无 | `RetryDataset`（按 `TWINKLE_SEED` 跘 rank 一致的替换样本）|
| Megatron `data_sharding` | `_MegatronDPBatchSampler`（dev 自写） | **本批次移进 twinkle**（下节）|

### twinkle 侧补的唯一缺口：`data_sharding`

Megatron 的 `--data_sharding` 是「先分桶再桶内 shuffle」（`MegatronPretrainingRandomSampler`），twinkle `DeviceMeshSampler` 只有「全局 permutation 再逐 batch 切片」，两者不等价。按「只补缺失、组件化、默认不变」给 `DeviceMeshSampler` 加了 opt-in 的 `data_sharding` 模式（`twinkle/src/twinkle/dataloader/device_mesh_sampler.py`）：每个 dp rank 只看 `[rank*bucket, (rank+1)*bucket)`，桶内按 `data_seed+epoch` permute，尾巴不足一个 micro batch 丢弃（与 legacy 一致）；`emitted_batch_sizes` 记的是**全局宽度**，使 consumed 计数与非 data_sharding 路径同口径。`DataLoader` 同步加了 `data_sharding` 参数与 epoch 跟踪（`set_epoch` 转发给 batch_sampler）。

### 行为变化（需真实多卡环境确认）

| 项 | 变化 | 影响 |
|---|---|---|
| `batch_size` 口径 | legacy 收**每卡** batch，twinkle 收**全局** batch 再切 | `_twinkle_loader_layout` 传 `per_device * dp_world_size`；local 传 mesh，ray 传 `None`（DP scatter 由 `forward_backward(dispatch='slice_dp')` 做）|
| `nproc_per_node` | 以前可缺 | **多卡 local 必需**：它定 DP 布局；不传则 dp=1，不分片 |
| hf backend 的 DP 源 | 全局 rank stride | DeviceMesh dp 坐标（TP/PP/CP==1 时两者相等）|
| iterable 数据集 | rank0 scatter | 各 worker 自切片（各 rank 自行读流）|
| resume 粒度 | batch（`consumed_batches * batch_size`）| sample（`consumed_train_samples`，按真实宽度累加）|
| resume 读取方式 | `dl.consumed_samples` / `dl._resume_epoch` 属性 | `dl.get_state()['consumed_train_samples' \| 'resume_epoch']`。**必须走方法**：`DataLoader` 是 `remote_class`，ray 模式下 driver 只持 handle，属性读会静默得 0 而丢掉断点（`recipe/train_loop.py` 已改）|

### 附带修的一个真 bug

`_load_kwargs` 仍在转发 `remove_unused_columns` / `disable_auto_column_mapping`，而 dev 的 `load_dataset` 没有这两个参数——任何带 `dataset` 的 `build_dataset` 调用都会 `TypeError`。之前没暴露，是因为测试还 patch 在 legacy `swift.dataset.load_dataset` 上（round 1 改成 `swift.dev.dataset` 后失效的 mock）。现在：两个旋钮不再转发（dev 本就等价于它们的**默认值**：预处理器总是丢掉已消费的源列，列别名总是生效），非默认值直接**报错**而不是静默忽略；所有失效的 patch 目标修正到 `swift.dev.dataset.*`。

### 验证

`swift/dev/tests/test_dataset_api.py` **69 项通过**（仅剩 3 项 pre-existing 失败，LISA 旋钮已从 `TunerConfig` 移除而 `config/validate.py:428` 未同步，与本批次无关）。四个旧测试类已改指 twinkle：`TestDeviceMeshSampler`（两个真 DeviceMesh 验 DP 不重不漏）、`TestDataSharding`（桶内限定 + 桶内 shuffle + 默认仍为全局 permutation）、`TestDataLoaderResumeContract`（`get_state`/`skip_consumed_samples` 跨 epoch 分解）、`TestTwinkleLoaderLayout`（全局 batch 宽度与 mesh 传递）。**未验证**：真实多卡下的 DP 切片、iterable worker-fetcher、Megatron `data_sharding` 端到端（均需 GPU）。

---

# 三、全量对账（ground truth，按注册表差集自动统计）

> 数据来源：`swift.dataset.register.DATASET_MAPPING`（legacy，导入 llm+mllm 后）vs `swift.dev.dataset.DATASET_MAPPING`（dev）。以 ModelScope id 为对账主键。

## 指标 1：已迁移——**差集为空**

| 指标 | legacy | dev |
|---|---|---|
| 注册条目 | 197（键为 `(ms_id, hf_id, subset)` 元组） | **197** 个 `dataset_type`（= 86 Python 类 + 111 JSON 条目）|
| distinct ms id | 189 | **189** |
| 纯 hf 条目 | 8 | **8** |
| 覆盖率 | — | **189 / 189 = 100%**，`leg_ms - dev_ms == ∅`、`leg_hf_only - dev_hf == ∅` |

dev 专有名 = 0，与模型侧不同——数据集没有「按模板拆分」的需要。反查已实测：所有 id 在 ms/hf 两个 hub 上双向解析，无一错配。

> 对账脚本的一个坑（曾算错过一次）：**两边的字典键格式不同**。legacy 是 `(ms_id, hf_id, subset)` 元组 → `DatasetMeta`（且元组长度并非恒为 3），dev 是 `dataset_type` 字符串 → loader 类。直接做键集差集得到的 197 vs 163 “缺 34” 是巧合，不是同口径结果。正确做法是把 dev 侧 `cls.iter_ids()` 展开成 id 集合册比。

## 指标 2：未迁移 0

批次 7 后无未迁项，也无阻塞项。下表保留各源头的最终归属，供反查：

| 源头 | 数量 | 落地位置 |
|---|---|---|
| `dataset_info.json`（声明式） | 111 | `loader/dataset_info.json`（批次 3）|
| legacy `dataset/llm.py` | 47 | `loader/llm.py`（批次 1/2/4/7）|
| legacy `dataset/mllm.py` | 39 | `loader/mllm.py`（批次 5/7）|
| 合计 | 197 | — |

字段忠实度已逐条实测：ids / `columns` / `split` / `subsets` / `tags` / `huge_dataset` / `help`。

> 一条口径提醒：本文以 ms id 为主键，所以 8 条纯 hf 数据集不在 189 这个数字里，已单独核对（8/8）。

## 指标 3：验证口径

已迁部分的验证方式（不只是「能 import」）：

| 项 | 方法 | 结果 |
|---|---|---|
| 与 legacy 逐数据集 parity | 同一份合成数据分别喂 legacy preprocessor 和 dev preprocessor，比对 `messages`/`rejected_*`/`tools`/`label`/`target`/`solution` | 批次 2：11/11；批次 4：17/17；批次 5：4/4 逐字相同，另 3 个（SA1B×2、pixelprose）因随机 prompt 不适用逐字比对，已改为断言式验证 6/6 |
| DataLoader 对接层 parity（批次 6） | `calculate_matched_group` 9 例逐字、`packed_idx`/`packed_length`/`__getitem__` 与 legacy 相同、sequential 策略保序、lazy 逐项相同 + 坏行替换 + 超长行替换 + `strict=True` 抛出 + 字符串索引透传、流式 packing 样本数/pack 上限/空流、`template.packing` 置位 | 15/15 |
| `MaxLengthError` 处理 | eager 编码路径上超长行丢弃行数与 `input_ids` 与 legacy 比对；`traceback_limit` 配额是否被超长行吃掉；`strict=True` 两边行为 | 4/4 |
| MM 归档路径解析 | 桩掉下载器指向临时目录，建真文件，验证路径拼接 / 缺文件丢行 / 多图全或无 / 分片取数 | 6/6（ShareGPT-4o、Mantis、llava-data 三种前缀规则、egoschema default+cls+sharded） |
| Arrow 列类型钉住 | 构造拼接冲突的两个子集（`loss_scale` 有无 / `bytes` null-vs-binary / `bbox` int-vs-float），开关 pin 做 A/B；并反证纯文本无幽灵列、patch 退出后还原、`num_proc=2` 下生效 | 8/8 |
| alias 表对账 | 把 legacy `ResponsePreprocessor().columns` 与 `ResponseConverter.aliases` 按 system/query/response 三类逆向比对 | 3/3 OK（此测试因批次 4 发现的缺失别名而新增） |
| format 层 parity | 与 legacy 9 例、与 `origin/main` 的 provider 归一逐函数比对 | ALL PARITY OK |
| 注册层 | 163 个 dataset_type × 双 hub 反查；子集/split 解析；weak subset 语义；preprocessor 均为类而非实例 | 无错配 |
| JSON 声明层 | 111 条逐条字段忠实度 + entry/per-subset `columns` 与 legacy 逐行 parity + 6 类边界情况 | 全过 |
| 可复现性 | 带 `random_state` 的数据集（HC3、xlam-grpo）两个独立实例跑出相同结果 | 一致（legacy 的 xlam-grpo 做不到） |
| 端到端 | 本地 jsonl 经公开 `load_dataset`：格式自动探测、`#N` 采样、train/val 切分、`num_proc=2`（含 `random_state` 跟着 pickle） | 全过 |
| mm_download | 注册表 / 工厂 / 原子性（含崩溃后重试）/ 缓存命中 / `safe_save` 幂等 | 15 项全过 |
| lint | `ruff check`（项目 pre-commit 同规则，`select=["B","C","E","F","W","I"]`） | 仅剩 6 项 I001 import 排序（四个新模块文件已按 ruff 修成仓库扁平风格；剩下的是两个 `__init__.py`（ruff 建议把紧凑续行炸成畸形多行，不采纳）与四个手改过的既有文件） |

### 已知未验证项（诚实登记）

- **未跑真实 hub 下载**。`build_dataset` 只是转调 `hf_load_dataset`，与本轮改动正交；注册路由用 monkeypatch 证明。
- **JSON 那 111 条未逐条跑真实数据**。验证停在「声明忠实落地 + columns 行为与 legacy parity」；它们本身无自定义逻辑，风险主要在 hub 侧（subset 名 / split 名是否仍有效），而那是 legacy 同样承担的风险。
- **parity 用的是合成 schema，不是真实数据**。批次 4 的 `standardise()` 让子类不再猜原始列名（改为复用 converter 的 alias 表），上一版登记的「按假设列名读字段」风险因此消除；但仍有一层假设未验——**不经 alias 表直读原始列的那几个**（`text1`/`text2`、`nums`、`human_answers`、`conversations`、`function_description`、`chosen`/`rejected`，批次 5 新增 `global_caption`、`cap_seg`、`vlm_caption`、`Text:LABEL`、`video_idx`、`option`）：这些列名是从 legacy 代码里读出来的，与 legacy 一致，但两边如果都错则无法由 parity 发现。真实拉一次数据可消除。
- **MM 归档的真实目录结构未验**。批次 5 的路径解析测试是自己造目录造文件，验的是拼接逻辑与丢行规则。归档内部真实层级（尤其 ShareGPT-4o 那个 `mnt/petrelfs/...` 深层路径、以及 llava-data 六个图库的 `vg/` 写法）均抗自 legacy 代码，未实拉归档核对。
- **`pin_features` 的生效机制未查清**。结果层验收（A/B + features 断言）没问题，但探针两次都没抓到 `ArrowWriter.__init__` 被构造，说明它生效的路径与我的理解不符。若将来 datasets 换版后 MM 拼接报 schema 错，这里是第一个该查的地方。

---

# 四、机制完整性审计

> 批次 5 完成后做的系统对账，批次 6/7/8 后已按落地情况更新。问的是：不看数据集条目覆盖率（现为 100%），只看**「从 hub 拉数据 → 训练能跑」整条管线的功能模块**，dev 到底缺了什么。

## 已迁完的机制

| legacy 机制 | dev 位置 | 说明 |
|---|---|---|
| 名称解析 + hub 加载 | `loader/base.py` | `DatasetLoader` + registry + `load_dataset` |
| `DatasetSyntax` 的 `#N` 采样 | `loader/base.py:sample_dataset` | 含过采样重复 |
| `split_dataset_ratio` 切分 | `loader/base.py` | 同 |
| `interleave_datasets` | `loader/base.py` | 调 HF 的 `interleave_datasets` |
| `register_dataset` / `register_dataset_info` | `loader/base.py` + `loader/__init__.py` | JSON 声明 + Python 类双入口 |
| `AutoPreprocessor` 格式探测 | `format_converter/get_converter` | 按 priority + detect |
| `ResponsePreprocessor` / `MessagesPreprocessor` / `AlpacaPreprocessor` | `format_converter/` 四个 converter | 已逐项 parity |
| `_patch_arrow_writer`（列类型钉住） | `preprocessor/base.py:pin_features` | 只钉输出里真有的列 |
| `_check_messages` | `preprocessor/base.py:check_messages` | 同 |
| `cast_mm_data`（images str→struct、videos/audios→list） | `preprocessor/base.py:cast_mm_data` | 同 |
| `prepare_dataset` 钩子 | `preprocessor/base.py:prepare_dataset` | 批次 5 加的 |
| `MediaResource.download` | `mm_download/` | 三策略子类 + 原子 rename |
| `SubsetDataset` | `SubsetMeta` | 同 |
| `columns` 别名 | `FormatConverter.aliases` + `MEDIA_ALIASES` | 两级声明 |
| `PackingDataset` / `IterablePackingDataset` | `packing.py` | 批次 6 加的，parity 15/15 |
| `LazyLLMDataset` | `lazy_dataset.py` | 同 |
| `EncodePreprocessor` / `AddLengthPreprocessor` | `preprocessor/encode.py` / `preprocessor/measure.py` | 后者更名为 `MeasurePreprocessor` |
| `MaxLengthError` 静默丢弃 | `preprocessor/base.py:batched_preprocess` | 批次 6 对齐，parity 4/4 |
| `shuffle` / `shuffle_buffer_size` | `loader/base.py:shuffle_dataset` | 批次 8；materialised 全局打乱 / stream 走 buffer |
| `interleave_prob` / `stopping_strategy` | `loader/base.py:load_dataset` | 批次 8；调 HF `interleave_datasets` |
| `hub_token` | `loader/base.py:load_dataset` → `load()` | 批次 8；仅在有值时下传，不覆盖已缓存凭据 |
| `DatasetSyntax` 完整 DSL | `loader/base.py:parse_legacy_syntax` | 批次 8；仅作向后兼容 util，非一等语法 |
| streaming 的 train/val 切分 | `loader/base.py:split_streaming` | 批次 8；此前对 `IterableDataset` 会崩 |
| `_check_objects` 的 bbox 校验 | `preprocessor/base.py:check_objects` | 批次 8；key 顺序那半不需要（见批次 8） |
| `get_temporary_cache_files_directory` | `loader/base.py:use_swift_cache_for_temp_files` | 批次 8；改为显式调用而非 import 副作用 |
| `cache_file_name`（map 缓存） | `preprocessor/base.py:map_cache_path` | 批次 8；仅对无 `cache_files` 的内存数据集 |
| hub 抽象层（`get_hub` 分派） | `loader/base.py:load_from_hub` | 批次 8；复用 `swift.hub`，ModelScope 走 `MsDataset` |
| `ms_revision` / `hf_revision` | `loader/base.py:load_dataset` → `DatasetInfo.revision` | 批次 8；此前是死声明 |
| 本地目录加载 + `dataset_infos.json` 改名 | `build_dataset` / `hide_dataset_infos` | 批次 8 |
| csv `na_filter=False` | `build_dataset` | 批次 8 |
| `#N` 顺序取样（`shuffle=False`） | `loader/base.py:sample_dataset` | 批次 8；默认取前 N，与 legacy 默认一致 |

## 确认不需要迁的（dead code / 被设计替代 / 已有等价物）

| legacy 机制 | 理由 |
|---|---|
| `indexed_dataset.py`（132 行） | 零引用，confirmed dead code |
| `__@` streaming 前缀 | dev 的 `remove_columns=self._feature_columns` 用**输入列**，不与 standard_keys 碰撞。批次 8 实测：dev iterable 正常，legacy 反而 `IndexError` |
| `__#solution` rename hack | dev 不做 map 前全表 rename，solution 列作为透传列直接保留 |
| `origin_columns`（高优先级列重复保护） | dev 的 aliases 合并策略（caller wins）天然做到 |
| `remove_useless_columns`（丢掉非标准列） | dev 有意保留非标准列（对齐 legacy 实测行为：`label`/`junk` 等列不丢） |
| `safe_rename_columns`（case-insensitive, 去重） | dev 的 aliases 在 converter 层做，无需独立机制 |
| `disable_auto_column_mapping` | dev 的别名可按列 opt-out（`columns={'text':'text'}`）或整格式钉死（`format_name=`），不需要一刀切开关。legacy 需要它，是因为别名表和用户 `columns` 混在同一 dict，冲突时静默放弃双方 |
| `_cast_pil_image` | 批次 8 实测：两边输出 bytes 逐字节相同，`pin_features` 已达成同一效果（绑定 datasets 4.7.0） |
| `_check_rejected_response` | `template_inputs.py:166-186` 已做同样三项校验，legacy 那份是重复 |
| `_inject_dataset_routing_tag` | 全仓零消费者；channel loss 读的是 `channel` 列 |
| `get_dataset_list` | 唯一调用方 Web UI 已废弃重写 |
| `download_ms_dataset`（`dataset_meta.py:66`） | 零消费者，dead code |
| 多 subset 无 `default` 时报错 | legacy 要求用户显式指定，dev 默认加载全部非 weak subset。有意保留 dev 的宽松口径 |
| 本地文件 `cache_dir` 指向 swift cache | 用 datasets 默认位置即可 |

## 尚未迁的机制

按离训练的距离分组。批次 6/7/8 之后，这份清单只剩一项。

### A. 训练硬前提（数据 → DataLoader 的必经路径）—— **已迁（批次 6）**

五个组件（`PackingDataset` / `IterablePackingDataset` / `LazyLLMDataset` / `EncodePreprocessor` / `AddLengthPreprocessor`）已全部落地，与 legacy 逐项 parity 15/15；顺带对齐了 `MaxLengthError` 的静默丢弃语义（4/4）。详见上方「批次 6」。

### B. 多数据集混训编排 —— **已迁（批次 8）**

`shuffle` / `shuffle_buffer_size` / `stopping_strategy` / `interleave_prob` / `hub_token` 五个参数已在 `load_dataset` 上，`interleave_prob` 走 HF 官方 `interleave_datasets`。详见上方「批次 8」。

### C. 三个阻塞层 —— **已迁（批次 7）**

`GroundingMixin` / `ClsGenerationPreprocessor` / `TextGenerationPreprocessor` / self-cognition 注入全部落地，曾卡住的 7 个数据集已随批次 7 清零。回头看，这 7 项「阻塞」里只有 2 项真需要动基础设施（self-cognition 的 loader seam、MovieChat 的多文件下载模式）。

### D. 保护机制 —— **已迁 / 已判定不需要（批次 8）**

| 机制 | 结论 |
|---|---|
| `_check_objects` | bbox 校验已迁；key 顺序那半被 `Json()` 取代 |
| `_cast_pil_image` | 实测无差异，不迁 |
| `_check_rejected_response` | template 已做，不迁 |
| `cache_file_name` | 已迁（`map_cache_path`） |
| **`safe_ddp_context`** | 已迁，但换了实现：4 处接 `twinkle.utils.processing_lock`（不走 NCCL barrier，见批次 8 补节） |

### E. 环境与 CLI —— **已迁 / 已废弃（批次 8）**

| 机制 | 结论 |
|---|---|
| `get_temporary_cache_files_directory` | 已迁，改为显式调用 |
| `DatasetSyntax` 完整 DSL | 已迁为向后兼容 util（`parse_legacy_syntax`） |
| `get_dataset_list` | 不做，Web UI 废弃重写 |

## 结论

dev 完成的是：**「从 hub 拉数据 → 标准 messages 行 → input_ids → DataLoader」整条管线，加上混训编排与保护机制**。名称解析、格式探测、列别名、行变换、多模态下载、Arrow 类型钉住、template 编码、packing、延迟加载、shuffle/interleave、bbox 校验、缓存目录——全部到位。

机制层只剩**下载重试 `retry=3`**（抗网络抖动，不是正确性问题）。多卡串行化已全部接上，用的是 `twinkle.utils.processing_lock` 而非 `safe_ddp_context`。

**接线已完成**：`swift/dev/builders/dataset.py`、`recipe/cached_dataset.py`、`recipe/quantize.py` 都已走 `swift.dev.dataset.load_dataset`（曾卡在 template 的 `EncodePreprocessor` 已随 dev template 落地）。DataLoader 那一段在批次 9 换成了 twinkle `DataLoader`，`swift/dev/legacy_dataloader/` 已退役（无调用者，但按用户要求文件暂留）——dev 的数据路径已不再 import 任何 `swift.dataset`；`swift.dataloader` 仅剩那个待 review 的退役包在 import。

---

# 附：与模型迁移的口径差异

记录下来，避免照搬 `MODEL_MIGRATION.md` 的结论时误判。

| 维度 | 模型侧 | 数据集侧 |
|---|---|---|
| registry miss | `get_model_loader` **raise** | `get_dataset_loader` **返回基类**（任意 hub id / 本地文件都合法） |
| dev 专有名 | 69 个（按 template / task 拆分） | 0 个（数据集无拆分需要） |
| 未迁余量 | 真障碍（版本死 / 外部仓 / 重 patch）仍在 | **0**（批次 7 清零）。回头看，曾登记的 7 项「阻塞」里只有 2 项真要动基础设施 |
| 「迁移」的主体 | 每个模型一个 loader 类 | 四层基础设施；数据集本身多为填声明 |
| 声明形式 | 一律 Python 类 | Python 类（有逻辑，86）+ JSON（纯声明，111）|
