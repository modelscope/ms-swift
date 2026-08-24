# 数据集迁移结果表（legacy `swift/dataset/` → `swift/dev/dataset/`）

> 记录数据集从 legacy 迁到 dev 的进度、结论与依据。与 `MODEL_MIGRATION.md` 同体例，但有一处结构差异：模型侧只有「一个 loader 层」，数据集侧是**四层基础设施 + 数据集声明**，所以本文件先记录各层状态，再记录数据集批次——一个数据集能不能迁，往往取决于它依赖的层有没有到位。

> **本文只记「已经搬了什么」。接下来怎么改形状，见 [`DATASET_REDESIGN.md`](./DATASET_REDESIGN.md)**——那份含完整的猫腻清单（带代码位置）、目标类层次设计、`cache_encoded`（文本落盘 + 媒体运行时）、下游硬约束清单与分阶段计划。两者分工：**先有本文的逐字 parity，才有资格谈那份的重排。**

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

## 待建层

| 层 | 用途 | 影响的数据集 |
|---|---|---|
| syntax DSL 解析 | `hub::id:sub1/sub2#N` 全语法 | 无（当前支持 `#N` + `subsets` 参数，够用） |
| cls → 生成式改写 | `ClsGenerationPreprocessor`（把分类标签改写成生成任务的选项文本） | 2 个（jd、clue） |
| prompt 模板 | `TextGenerationPreprocessor` 的 `{{QUERY}}` 套模板 | 1 个（AdvertiseGen） |
| 多塔样本构造 | embedding / reranker 的 `positive_messages`/`negative_messages` | 3 个（stsb + MTEB ×2） |
| self-cognition 注入 | `set_name_author(model_name, model_author)` | 1 个 |

> 一个修正：上一版把「`ClsPreprocessor`（label→int）」也列为待建层。批次 4 迁 HC3 时实测发现：`label` 只需一个 `int` 穿透，`Preprocessor` 基类已经能做（`hc3_cls` 子集已与 legacy parity），**不需要单独一层**。真正缺的是 `ClsGenerationPreprocessor` 那种把标签重写成选项文本的改写。

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

legacy 的 `map` 外面还包了 `safe_ddp_context`（rank0 先跑、其余 rank 复用 cache）与显式 `cache_file_name`，dev 都没有。单机不影响正确性，多卡会变成每个 rank 各自编码一遍（浪费，非错误）。

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

> 批次 5 完成后做的系统对账，批次 6/7 后已按落地情况更新。问的是：不看数据集条目覆盖率（现为 100%），只看**「从 hub 拉数据 → 训练能跑」整条管线的功能模块**，dev 到底缺了什么。

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
| `EncodePreprocessor` / `AddLengthPreprocessor` | `encode_preprocessor.py` / `add_length_preprocessor.py` | 同 |
| `MaxLengthError` 静默丢弃 | `preprocessor/base.py:batched_preprocess` | 批次 6 对齐，parity 4/4 |

## 确认不需要迁的（dead code / 被设计替代）

| legacy 机制 | 理由 |
|---|---|
| `indexed_dataset.py`（132 行） | 零引用，confirmed dead code |
| `__@` streaming 前缀 | dev 的 `remove_columns=self._feature_columns` 用**输入列**，不与 standard_keys 碰撞 |
| `__#solution` rename hack | dev 不做 map 前全表 rename，solution 列作为透传列直接保留 |
| `origin_columns`（高优先级列重复保护） | dev 的 aliases 合并策略（caller wins）天然做到 |
| `remove_useless_columns`（丢掉非标准列） | dev 有意保留非标准列（对齐 legacy 实测行为：`label`/`junk` 等列不丢） |
| `safe_rename_columns`（case-insensitive, 去重） | dev 的 aliases 在 converter 层做，无需独立机制 |
| `disable_auto_column_mapping` | dev 的 auto-mapping 由格式探测替代，没有「一边 auto 一边想关」的局面 |

## 尚未迁的机制

按离训练的距离分组。

### A. 训练硬前提（数据 → DataLoader 的必经路径）—— **已迁（批次 6）**

五个组件（`PackingDataset` / `IterablePackingDataset` / `LazyLLMDataset` / `EncodePreprocessor` / `AddLengthPreprocessor`）已全部落地，与 legacy 逐项 parity 15/15；顺带对齐了 `MaxLengthError` 的静默丢弃语义（4/4）。详见上方「批次 6」。仅 `safe_ddp_context` + `cache_file_name`（多卡 cache 复用，属浪费而非错误）未迁。

### B. 多数据集混训编排

| 缺失机制 | legacy 位置 | 什么 |
|---|---|---|
| `shuffle` | `load_dataset` 参数 | concat 后打乱 |
| `shuffle_buffer_size` | 同 | streaming 模式下的 buffer 大小 |
| `stopping_strategy` | 同 | `first_exhausted` / `all_exhausted` |
| `interleave_prob` | 同 | 各数据集的采样权重 |
| `hub_token` | 同 | 私有 hub 鉴权 |

> dev 的 `load_dataset` 只有最简的 `concatenate_datasets`；混训编排没有独立实现。

### C. 三个阻塞层（7 个数据集卡在这里）

| 缺失机制 | legacy 位置 | 阻塞的数据集 |
|---|---|---|
| **GroundingMixin** | `extra.py:8` | refcoco、refcocog、Grit |
| **ClsGenerationPreprocessor** | `extra.py:72` | jd、clue |
| **TextGenerationPreprocessor** | `extra.py:55` | AdvertiseGen |
| **self-cognition 注入** | `load_dataset` + `loader.py` | swift/self-cognition |

### D. 保护机制

| 缺失机制 | legacy 位置 | 作用 |
|---|---|---|
| `_cast_pil_image` | `core.py` | `Image(decode=True)` 降为 `Image(decode=False)`，阻止 map 内 PIL decode——大图数据集不做会 OOM |
| `_check_objects` | `core.py:146` | 归一化 bbox（确保 x1<x2, y1<y2）+ 检查长度 2 or 4 |
| `_check_rejected_response` | `core.py` | 断言 rejected ≠ None 且 ≠ chosen（否则 DPO loss 全 0） |
| `safe_ddp_context` + `cache_file_name` | `core.py:352` | 多卡下 rank0 先跑 map、其余 rank 复用 cache；不做则每个 rank 各自编码一遍（浪费，非错误） |

### E. 环境与 CLI

| 缺失机制 | legacy 位置 | 作用 |
|---|---|---|
| `get_temporary_cache_files_directory` monkey patch | `__init__.py` + `utils.py:136` | 把 HF datasets 临时 cache 重定向到 swift cache，避免 /tmp 爆 |
| `DatasetSyntax` 完整 DSL | `dataset_syntax.py`（127 行） | 命令行 `hub::id:sub1/sub2#N` 语法 |
| `get_dataset_list` | `register.py` | `swift list-dataset` / UI |

## 结论

dev 完成的是：**「从 hub 拉数据 → 标准 messages 行 → input_ids → DataLoader」这条完整管线**。名称解析、格式探测、列别名、行变换、多模态下载、Arrow 类型钉住、template 编码、packing、延迟加载——全部到位。

未完成的剩下四类：**混训编排**（shuffle / interleave_prob / stopping_strategy）、**三个阻塞层**（grounding / cls→生成 / prompt 模板 / self-cognition）、**三个保护机制**（`_cast_pil_image` / `_check_objects` / `_check_rejected_response`，另加多卡 cache 复用）、**环境 CLI**。

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
