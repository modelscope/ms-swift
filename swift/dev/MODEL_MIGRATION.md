# 模型迁移结果表（legacy `swift/model/models/` → `swift/dev/model/loader/`）

> 记录每个 legacy `model_type` 迁移到 dev loader 的结论与依据。与 `PATCH_INVENTORY.md` 配套：patch 是否仍需要看那份，模型是否迁移看这份。

## 判定规则
- **迁移（migrated）**：在 dev 侧新建 `ModelLoader` 子类并 `@register_model`。
- **删除（dropped）**：不写 loader、不注册，仅在此表登记原因。判据：模型早于 2024 发布，或 2025 前但构造复杂/强依赖 legacy patch；构造简单（纯 `AutoModelForCausalLM`、无自定义 loader）的即便老也保留。
- **拆分**：legacy 一个 `model_type` 内若各 `ModelGroup` 的 template 或额外 pip 依赖不同，按 dev「一类一 requires / 模板变体独立子类」拆成多个 model_type。纯 transformers 版本差异则类级取最高版本、不拆。

## 列含义
`legacy model_type`：legacy 注册名 ｜ `dev loader`：dev 侧类名（dropped 留空）｜ `代表模型` ｜ `发布` ｜ `构造复杂度` ｜ `结论` ｜ `依据 / 备注`。

## mistral.py（首个 pilot）

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| mistral | `MistralLoader` | Mistral-7B-Instruct-v0.3 | 2023–2024 | 简单（AutoModelForCausalLM） | migrated | 常用基础模型，构造简单，保留 |
| mixtral | `MixtralLoader` | Mixtral-8x7B-Instruct-v0.1 | 2023-12 | 简单 | migrated | 保留主体；见下方 dropped 行 |
| —（mixtral 的 AQLM 组） | — | Mixtral-8x7b-AQLM-2Bit-1x16-hf | 2024 | 复杂（需 aqlm+torch>=2.2） | dropped | 小众 2-bit AQLM 量化 checkpoint，额外 pip 依赖，无法并入单一 requires |
| mistral_nemo | `MistralNemoLoader` | Mistral-Nemo-Instruct-2407 | 2024-07 | 简单 | migrated | 两组仅 transformers 版本差异，类级取 >=4.46 |
| mistral_2501 | `Mistral2501Loader` | Mistral-Small-24B-Instruct-2501 | 2025-01 | 简单 | migrated | |
| zephyr | `ZephyrLoader` | zephyr-7b-beta | 2023-10 | 简单 | migrated | 老但纯 AutoModelForCausalLM，保留 |
| wizardlm2_moe | `WizardLM2MoeLoader` | WizardLM-2-8x22B | 2024-04 | 简单 | migrated | |
| wizardlm2 | `WizardLM2Loader` | WizardLM-2-7B-AWQ | 2024-04 | 简单 | migrated | |
| devstral | `DevstralLoader` | Devstral-Small-2505 | 2025-05 | 中（借用 3.1 tokenizer） | migrated | `build_processor` override 借 Mistral-Small-3.1 的 tokenizer |
| mistral3 | `Mistral3Loader` | Mistral-Small-3.1-24B-Instruct-2503 | 2025-03 | 中（MLLM, llava_hf 分区） | migrated | 拆分见下：原 model_type 含 3 组 template/requires |
| mistral3（2512 组） | `Ministral3Loader` | Ministral-3-8B-Instruct-2512 | 2025-12 | 中 | migrated | 因 `mistral-common>=1.8.6`+transformers5 额外依赖拆为独立 model_type `ministral3` |
| mistral3（2512 reasoning 组） | `Ministral3ThinkingLoader` | Ministral-3-8B-Reasoning-2512 | 2025-12 | 中 | migrated | 模板变体，`architectures=[]`，新 model_type `ministral3_thinking` |
| mistral3_2506 | `Mistral3_2506Loader` | Mistral-Small-3.2-24B-Instruct-2506 | 2025-06 | 中（借用 3.1 processor） | migrated | `build_processor` override 借 3.1 的 processor |

> 修正（本轮）：给 base `ModelLoader` 补了 `ignore_patterns` 字段；`Mistral3Loader`（含 Ministral3/其 thinking/3.2 子类）回填 `ignore_patterns=[]`——Mistral 权重是 `consolidated*`，会被下载默认 skip，之前迁移漏了这个 correctness override。

关联 patch：mistral 系列在 `PATCH_INVENTORY.md` 第 3 节无条目，无 model-patch 依赖。

## yi.py

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| yi | `YiLoader` | Yi-1.5-34B-Chat | 2023-11 起 | 简单（LlamaForCausalLM） | migrated | 含 yi/yi1.5/quant，template=chatml |
| yi（coder 组） | `YiCoderLoader` | Yi-Coder-9B-Chat | 2024-09 | 简单 | migrated | 模板变体，新 model_type `yi_coder`，`architectures=[]` |
| yi（SUS 组） | `SusChatLoader` | SUS-Chat-34B | 2023-12 | 简单 | migrated | 模板变体，新 model_type `sus_chat`，`architectures=[]` |
| yi_vl | — | Yi-VL-6B | 2024-01 | 复杂（git clone + sys.path + 外部 llava 包） | dropped | 旧且构造复杂 |

## skywork.py

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| skywork | `SkyworkLoader` | Skywork-13B-chat | 2023-10 | 简单 | migrated | `process_tokenizer` 加 [USER]/[BOT]/[SEP] |
| llama3_2_reward | `Llama3_2RewardLoader` | Skywork-Reward-Llama-3.1-8B | 2024 | 简单 | migrated | reward = num_labels=1 的 seq_cls，pin `is_reward=True`；model_cls=AutoModelForSequenceClassification |
| gemma_reward | `GemmaRewardLoader` | Skywork-Reward-Gemma-2-27B | 2024 | 简单 | migrated | 同上 |

## codefuse.py

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| codefuse_codellama | `CodeFuseCodeLlamaLoader` | CodeFuse-CodeLlama-34B | 2023-09 | 简单（build_processor use_fast=False） | migrated | tags=coding |
| codefuse_qwen | — | CodeFuse-QWen-14B | 2023 | 复杂（依赖未迁移的 Qwen1 QwenLoader） | dropped | 旧且依赖未迁移基类 |
| codefuse_codegeex2 | — | CodeFuse-CodeGeeX2-6B | 2023 | 复杂（ChatGLMLoader + transformers<4.34） | dropped | 旧且依赖未迁移基类 |

## telechat.py

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| telechat | `TelechatLoader` | TeleChat-12B-v2 | 2024 | 中（build_processor 拷 generation_config token） | migrated | |
| telechat2 | `Telechat2Loader` | TeleChat2-35B-32K | 2024 | 简单 | migrated | |

## baai.py

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| bge_reranker | `BgeRerankerLoader` | bge-reranker-v2-m3 | 2024 | 简单 | migrated | pin `task_type=reranker`，model_cls=AutoModelForSequenceClassification |
| emu3_gen | — | Emu3-Gen | 2024 | 复杂（git clone Emu3 + VisionTokenizer 下载，t2i） | dropped | 旧且构造复杂 |
| emu3_chat | — | Emu3-Chat | 2024 | 复杂（同上） | dropped | 旧且构造复杂 |

## bert.py

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| modern_bert_gte | `GteModernBertLoader` | gte-modernbert-base | 2024-12 | 简单 | migrated | pin `task_type=embedding`，model_cls=AutoModel；池化/归一化交下游 loss 层 |
| modern_bert_gte_reranker | `GteModernBertRerankerLoader` | gte-reranker-modernbert-base | 2024-12 | 简单 | migrated | pin `task_type=reranker` |
| modern_bert | `ModernBertLoader` | ModernBERT-base | 2024-12 | 简单 | migrated | encoder 底座，任务由 `--task_type`(seq_cls/embedding) 决定；swift 无 mlm 任务。process_config 关 reference_compile |
| bert | `BertLoader` | nlp_structbert_backbone_base | — | 简单 | migrated | StructBERT encoder 底座，任务由 `--task_type` 决定 |

## mamba.py

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| mamba | `MambaLoader` | mamba-2.8b-hf | 2023-12 | 简单 | migrated | 纯 AutoModelForCausalLM；causal-conv1d/mamba-ssm 安装提示移到 docstring |

## seed.py

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| seed_oss | `SeedOssLoader` | Seed-OSS-36B-Instruct | 2025 | 简单 | migrated | 纯 AutoModelForCausalLM |

## openbuddy.py

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| openbuddy_llama | `OpenBuddyLlamaLoader` | openbuddy-llama2-70b-v10.1 | 2023–2024 | 简单 | migrated | 拆分：原 model_type 混用 openbuddy/openbuddy2 两种模板 |
| openbuddy_llama（openbuddy2 组） | `OpenBuddyLlama2Loader` | openbuddy-llama3.3-70b-v24.3 | 2024 | 简单 | migrated | 模板变体 `architectures=[]`，requires 取最高 >=4.45，新 model_type `openbuddy_llama2` |
| openbuddy_mistral | `OpenBuddyMistralLoader` | openbuddy-mistral-7b-v17.1 | 2023–2024 | 简单 | migrated | MistralForCausalLM |
| openbuddy_mixtral | `OpenBuddyMixtralLoader` | openbuddy-mixtral-7bx8-v18.1 | 2024 | 简单 | migrated | MixtralForCausalLM |

## baichuan.py

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| baichuan_m1 | `BaichuanM1Loader` | Baichuan-M1-14B-Instruct | 2025 | 中（rotary dtype patch） | migrated | `build_model` override：预加载 patch 远程 `RotaryEmbedding.forward` 做 q→k dtype 对齐 |
| baichuan | — | Baichuan-13B-Chat | 2023-07 | 中（get_input_embeddings fix） | dropped | 2023 且 pin `transformers<4.34` |
| baichuan2 | — | Baichuan2-7B-Chat | 2023-09 | 中（lm_head fp32 patch） | dropped | 2023 且依赖 `patch_baichuan2_lm_head_forward`（PATCH_INVENTORY 标「不迁移」） |

## baidu.py

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| ernie4_5 | `Ernie4_5Loader` | ERNIE-4.5-0.3B-PT | 2025 | 简单 | migrated | 纯 AutoModelForCausalLM |
| ernie4_5_moe | `Ernie4_5MoeLoader` | ERNIE-4.5-21B-A3B-PT | 2025 | 简单 | migrated | Base/PT 组，template=ernie |
| ernie4_5_moe（Thinking 组） | `Ernie4_5MoeThinkingLoader` | ERNIE-4.5-21B-A3B-Thinking | 2025 | 简单 | migrated | 模板变体 `architectures=[]`，新 model_type `ernie4_5_moe_thinking` |
| ernie_vl | — | ERNIE-4.5-VL-28B-A3B-PT | 2025 | 复杂（MLLM 后置钩子） | deferred | `ErnieVLLoader` 需 leaf_modules + `add_image_preprocess(processor)`；待 MLLM 后置 seam 决策 |
| paddle_ocr | — | PaddleOCR-VL | 2025 | 复杂（MLLM） | deferred | 待 MLLM seam 决策 |
| paddleocr_vl | — | PaddleOCR-VL-1.5 | 2025 | 复杂（MLLM, AutoModelForImageTextToText/trust_remote_code=False） | deferred | 待 MLLM seam 决策 |

## microsoft.py

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| phi2 | `Phi2Loader` | phi-2 | 2023-12 | 简单 | migrated | 纯 AutoModelForCausalLM |
| phi3 | `Phi3Loader` | Phi-3-mini-4k-instruct | 2024 | 简单 | migrated | 含 Phi-4-mini（同 Phi3ForCausalLM+phi3 模板） |
| phi4 | `Phi4Loader` | phi-4 | 2024-12 | 简单 | migrated | 模板变体（Phi3ForCausalLM+phi4 模板），`architectures=[]` |
| phi3_moe | `Phi3MoeLoader` | Phi-3.5-MoE-instruct | 2024 | 简单 | migrated | MoE，但 legacy z3 map 无此项→不设 moe_block |
| phi3_small | — | Phi-3-small-8k-instruct | 2024 | 中（逐层 rotary dtype patch） | deferred | 自定义 loader hardcode 32 层 patch rotary_emb.forward |
| phi3_vision / phi4_multimodal / florence | — | Phi-3-vision / Florence-2 | 2024 | 复杂（MLLM） | deferred | 多模态，待 MLLM seam |

## minimax.py

| legacy model_type | dev loader | 代表模型 | 发布 | 构造复杂度 | 结论 | 依据 / 备注 |
|---|---|---|---|---|---|---|
| minimax_m2 | `MinimaxM2Loader` | MiniMax-M2 | 2025 | 简单 | migrated | 纯 AutoModelForCausalLM，requires==4.57.1 |
| minimax_m2（M2.1/2.5/2.7 组） | `MinimaxM2_1/5/7Loader` | MiniMax-M2.5 | 2025 | 简单 | migrated | 模板变体，`architectures=[]`，各自 model_type |
| minimax / minimax_m1 | — | MiniMax-Text-01 / MiniMax-M1 | 2025 | 复杂（手工 device_map+Quanto，"不支持训练"） | deferred | `MinimaxTextLoader` 推理向多卡切分 |
| minimax_vl / minimax_m3_vl | — | MiniMax-VL-01 / MiniMax-M3 | 2025 | 复杂（MLLM） | deferred | 多模态，待 MLLM seam |

## moonshot.py / stepfun.py（无简单部分，全部 deferred）

| legacy model_type | 代表模型 | 结论 | 依据 / 备注 |
|---|---|---|---|
| kimi_vl / kimi_k25 / kimi_k3 | Kimi-VL / Kimi-K2.5 / Kimi-K3 | deferred | MLLM，需 model_arch + 动态模块 loader，待 MLLM seam |
| got_ocr2 / got_ocr2_hf / step_audio / step_audio2_mini / step3_vl | GOT-OCR2 / Step-Audio / Step3-VL | deferred | MLLM/audio，自定义 loader + model_arch，待 MLLM seam |

## llama.py

legacy 单个 `llama` model_type 内含 ~15 种模板家族，dev 按模板拆成独立 model_type（base `LlamaLoader` + `architectures=[]` 模板变体子类；共享 `process_config` 修 `pretraining_tp`）。

| legacy 组（template） | dev loader / model_type | 代表模型 | 结论 | 依据 / 备注 |
|---|---|---|---|---|
| llama2 / chinese-llama2（llama） | `LlamaLoader` / `llama` | Llama-2-7b | migrated | 反查 `LlamaForCausalLM` 落此；llama2 组 `.bin`-skip 未搬（见下注） |
| Atom（atom） | `AtomLoader` / `atom` | Atom-7B | migrated | 模板变体 |
| Mengzi3（mengzi） | `Mengzi3Loader` / `mengzi` | Mengzi3-13B-Base | migrated | 模板变体 |
| NuminaMath（numina） | `NuminaLoader` / `numina` | NuminaMath-7B-TIR | migrated | tags math |
| Ziya2（ziya） | `Ziya2Loader` / `ziya` | Ziya2-13B | migrated | |
| Megrez（megrez） | `MegrezLoader` / `megrez` | Megrez-3B-Instruct | migrated | |
| MiniMind2（minimind） | `MiniMindLoader` / `minimind` | MiniMind2 | migrated | requires>=4.57.1；MiniMind2-Small 无 ms 镜像，用裸 hf id |
| llama3 / chinese-llama3 / quant（llama3） | `Llama3Loader` / `llama3` | Meta-Llama-3-8B-Instruct | migrated | |
| llama3.1/3.2/3.3 + Nemotron + quant（llama3_2） | `Llama3_2Loader` / `llama3_2` | Llama-3.3-70B-Instruct | migrated | model_type 与原 template 名保持一致；requires>=4.43 |
| Skywork-o1（skywork_o1） | `SkyworkO1Loader` / `skywork_o1` | Skywork-o1-Open-Llama-3.1-8B | migrated | |
| LongWriter（longwriter_llama） | `LongWriterLlamaLoader` / `longwriter_llama` | LongWriter-llama3.1-8b | migrated | |
| Reflection（reflection） | `ReflectionLoader` / `reflection` | Reflection-Llama-3.1-70B | migrated | |
| deepseek-llm/math/coder（deepseek）、R1-Distill-Llama（deepseek_r1） | — | deepseek-llm-7b-chat | routed | 归 deepseek.py（本身是 DeepSeek 家族，非 Llama chat 变体） |
| MiniCPM5（minicpm5） | — | MiniCPM5-1B | routed | 归 minicpm.py（requires>=5.6） |
| llama3_2_vision / llama4 / llama3_1_omni | — | Llama-3.2-Vision / Llama-4 | deferred | MLLM/audio，待 MLLM seam |
| Llama-2-7b-AQLM-2Bit（llama） | — | Llama-2-7b-AQLM-2Bit-1x16-hf | dropped | 小众 2-bit AQLM + 额外依赖（同 Mixtral-AQLM 先例） |

> 注：legacy llama2 组的 `ignore_patterns=[r'.+\.bin$']` 未搬——它是 per-group 下载优化（regex 放进 glob 语境、实际近乎失效），且套到合并后的 base loader 会误伤无 safetensors 的 chinese-llama2。属 correctness-无关的优化，故省略。

