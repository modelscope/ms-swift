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

---

# MLLM A/B 批次

> 统一口径："全量功能还原" = 还原真实能力 + **不移植 dev 架构已废弃的 patch**（依据 `PATCH_INVENTORY.md`）。
> 已废弃且省略的 seam：`device_map`/`_no_split_modules`、`patch_output_clone`、`patch_get_input_embeddings`、
> `patch_qwen_vl_utils`/`keye_vl_utils`、`patch_output_to_input_device`、`enable_input_require_grads`（use_reentrant=False 下不需要）。
> MLLM 模式：固定 `model_cls` + `model_arch` property（`ModelArch(language_model/aligner/vision_tower/moe_block)`）+ `is_multimodal=True`；
> 分区串对齐 transformers 5.5 layout（`transformers_ge_4_52` → `model.*` 前缀分支）。
> **remote-code seam（g7）**：base `ModelLoader` 新增 opt-in `trust_remote_code: bool = False`；置 True 时 `build_config`/`build_processor`/`build_model` 经 setdefault 注入 `trust_remote_code=True`，取代 legacy 散落的 `get_class_from_dynamic_module`。in-tree 家族保持默认 False。

## gemma.py

| legacy model_type | dev loader | 结论 | 依据 / 备注 |
|---|---|---|---|
| gemma / gemma2 | `GemmaLoader` / `Gemma2Loader` | migrated | 纯文本 |
| gemma3_text | `Gemma3TextLoader` | migrated | `_EagerAttnDefault` mixin |
| paligemma / gemma3_vision | `PaligemmaLoader` / `Gemma3VisionLoader` | migrated | MLLM，llava_hf；gemma3 eager-attn |
| gemma3n | `Gemma3nLoader` | migrated | 双塔 vision+audio |
| gemma4 / gemma4_unified / diffusion_gemma / gemma_emb | — | deferred | forward 重写+MoE / diffusion / ST |

## llava.py

11 个 `*_hf`（Llava / LlavaNext / LlavaNextVideo），共享 llava_hf 分区。

| legacy model_type | dev loader | 结论 | 依据 / 备注 |
|---|---|---|---|
| llava_llama3_hf / llava1_5_hf / llava_onevision_hf | `LlavaLlama3HfLoader` / `Llava1_5HfLoader` / `LlavaOnevisionHfLoader` | migrated | onevision=AutoModelForImageTextToText |
| llava_next 6 变体（qwen/llama3/vicuna/mistral/llama3_1/yi） | `_LlavaNextHfLoader` ×6 | migrated | 6→1 反查符合预期；yi override 是死代码 |
| llava_next_video_hf / llava_next_video_yi_hf | `LlavaNextVideoHfLoader` / `LlavaNextVideoYiHfLoader` | migrated | yi 变体设 video/image_token_index |
| llava 老家族 / llava_onevision1_5 | — | deferred | git clone + 外部包 |

## mllm.py

| legacy model_type | dev loader | 结论 | 依据 / 备注 |
|---|---|---|---|
| idefics3 / pixtral | `Idefics3Loader` / `PixtralLoader` | migrated | Vision2Seq / llava_hf |
| keye_vl | `KeyeVLLoader` | migrated | 经 g7 remote-code；丢弃过时 keye_vl_utils patch |
| molmo / molmo2 | `MolmoLoader` / `Molmo2Loader` | migrated | 经 g7 remote-code（drop `_no_split_modules`/output_clone）；molmo2=AutoModelForImageTextToText |
| dots_ocr | `DotsOCRLoader` | migrated | 经 g7 remote-code；arch 仅 language_model='model' |
| molmoe / keye_vl_1_5 / megrez_omni / jina_reranker_m0 / sail_vl2 | — | deferred | molmoe float32 默认 / keye_vl_1_5 pinned `==4.52.4` / megrez_omni use_submodel_func + processor 取自已加载 model / jina forward 重写成 reranker 头 / sail_vl2 use_submodel_func |

> g6 审计结论：megrez_omni（use_submodel_func + processor 取自已加载 model）、jina_reranker_m0（forward 重写成 reranker 头）仍为 bucket C，g7/delegate 均无法覆盖。

## Ovis 家族（AIDC-AI，qwen.py）— 首个 C-seam 落地

新增 base helper `delegate_to_submodel`（legacy `use_submodel_func` 的忠实等价：把外层 wrapper 的 `forward`/`generate`/`get_input_embeddings` 代理到内部 `model.llm`；**剥离 legacy 的 device_map 挪移与 `fix device_map` 分支**，dev 由 twinkle 管理放置）。与 qwen.py 里的 `OvisOcr2Loader`（Qwen3.5 OCR 模板变体，仅撞名）无关。

| legacy model_type | dev loader | 结论 | 依据 / 备注 |
|---|---|---|---|
| ovis1_6 | `OvisLoader` + `Ovis1_6Llama3Loader`（ovis1_6_llama3） | migrated | 经 g7 remote-code + delegate_to_submodel(llm)；processor=AutoTokenizer；visual_tokenizer/vte 对齐 dtype + cache_implementation=None；丢弃过时 output_clone/get_input_embeddings |
| ovis2 | `Ovis2Loader` | migrated | 同 ovis1_6（architectures=['Ovis'] → 反查 ovis1_6/ovis2 many-to-many，id 消歧） |
| ovis2_5 | `Ovis2_5Loader` | migrated | arch=ovis2_5；delegate_to_submodel(llm) + dtype 对齐；无 cache_impl |

> 已知未 wired 差异：Ovis 的 `attn_impl_keys=['llm_attn_implementation']`（attn_implementation 经自定义 config 键路由到内层 LLM）——attn_impl 经 build_model 的传递本身尚未接主链路，故当前无影响。

## microsoft.py（补充 MLLM）

| legacy model_type | dev loader | 结论 | 依据 / 备注 |
|---|---|---|---|
| phi3_vision | `Phi3VisionLoader` | migrated | 经 g7 remote-code（Phi3VForCausalLM）；build_processor 保留 num_crops env 旋钮；丢弃过时 output_clone |
| phi4_multimodal / florence | — | deferred | phi4 pinned `<4.49` + processor 手术 + set_lora_adapter / florence use_submodel_func |

## deepseek.py（MLLM，全部 deferred）/ minicpm.py（部分 migrated）

| legacy model_type | dev loader | 结论 | 依据 / 备注 |
|---|---|---|---|
| deepseek_vl / deepseek_janus / deepseek_janus_pro / deepseek_vl2 | — | deferred | git_clone 外部包（deepseek_vl/Janus/DeepSeek-VL2）+ use_submodel_func |
| deepseek_ocr / deepseek_ocr2 | — | deferred | g7-able 但 pinned `==4.46.3`（dev 5.5 dead） |
| minicpmv / minicpmv2_5 / minicpmv2_6 / minicpmv4 / minicpmv4_5 | `MiniCPMVLoader` 及子类 | migrated | g7 remote-code + `delegate_to_submodel(llm)`；`build_model` override 内把 model 的 `get_slice_image_placeholder`/`transform` 绑到 processor（第二个 C-seam：processor 增强需 model+processor 同在，故落 build_model）；resampler 对齐 dtype；丢弃过时 `_patch_minicpmv_device_map`/output_clone |
| minicpmo | `MiniCPMOLoader` | migrated | 同上 + audio 塔（vision_tower=[vpm,apm]）；`process_config` 经 env 门控 init_tts/init_audio |
| minicpmv4_6 | — | deferred | pinned `transformers>=5.7.0`（未到）+ `_patch_qwen3_5_linear_attention_sequence_parallel` 活跃全局 patch |
| minicpmo（4_5 组） | — | deferred | pinned `transformers==4.51.3` + 额外 `minicpmo-utils` 依赖 |

## internlm.py

| legacy model_type | dev loader | 结论 | 依据 / 备注 |
|---|---|---|---|
| internvl | `InternVLLoader` | migrated | `-hf` 系，丢弃过时 enable_input_require_grads；requires 取最高 4.55.0 |
| internlm / internlm2 / internlm3 | `InternLMLoader` / `InternLM2Loader` / `InternLM3Loader` | migrated | remote-code（类不在 tf5.5），经 g7 `trust_remote_code=True` + AutoModelForCausalLM；纯文本用空 ModelArch |
| interns1 / internvl_chat / xcomposer2* / internlm2_reward | — | deferred | internvl_chat 含 use_submodel_func（C）/ xcomposer git_clone / interns1 pinned `<4.56` / reward 头 |

## glm.py

per-`ModelGroup` 多 template → base + `architectures=[]` 模板变体（命名用模板名）。MoE 经 `model_arch.moe_block`（类名）声明 z3 叶子。

| legacy model_type | dev loader / model_type | 结论 | 依据 / 备注 |
|---|---|---|---|
| glm4 | `Glm4Loader` + `Glm4Z1RuminationLoader`（glm4_z1_rumination） | migrated | 文本 |
| glm_edge | `GlmEdgeLoader` | migrated | 文本，template chatglm4 |
| glm4_moe | `Glm4MoeLoader` + `Glm4_7Loader`（glm4_7） | migrated | MoE=Glm4MoeMoE |
| glm4_moe_lite | `Glm4MoeLiteLoader` | migrated | MoE=Glm4MoeLiteMoE |
| glm_moe_dsa | `GlmMoeDsaLoader` + `Glm5_1/Glm5_2Loader`（glm5_1/glm5_2） | migrated | MoE=GlmMoeDsaMoE |
| glm4v | `Glm4vLoader` + `Glm4_5vLoader`（glm4_5v） | migrated | MLLM，arch=glm4v；丢弃过时 patch_get_input_embeddings |
| glm4v_moe | `Glm4vMoeLoader` | migrated | MLLM+MoE=Glm4vMoeTextMoE，template glm4_5v |
| glm_ocr | `GlmOcrLoader` | migrated | AutoModelForImageTextToText，arch=glm4v |
| glm_edge_v | `GlmEdgeVLoader` | migrated | processor_cls=AutoImageProcessor；含 legacy 重复 id glm-edge-4b-chat（忠实保留，解析归 glm_edge） |
| chatglm2/3/4 / codegeex4 / chatglm4v / cogvlm* / cogvlm2* | — | deferred | remote-code + `_patch_tokenizer`（不迁移）+ tf<4.42 |

## 单模型补充（并入既有文件）

| legacy model_type | dev 位置 | 结论 | 依据 / 备注 |
|---|---|---|---|
| qwen2_audio | qwen.py `Qwen2AudioLoader` | migrated | 忠实保留 `<4.49` pin |
| qwen_audio / qwen2_5_omni / qwen3_omni_moe / qwen3_asr / qwen3_tts | — | deferred | Qwen1 remote-code / omni keep-alive forward 重写 / 双通道 TTS |
| hunyuan_ocr | tencent.py `HunyuanOcrLoader` | migrated | eager-attn 默认 seam |
| paddleocr_vl | baidu.py `PaddleOCR1_5Loader` | migrated | AutoModelForImageTextToText，requires>=5.0 |
| minimax_m3_vl | minimax.py `MinimaxM3VLLoader` | migrated | processor trust_remote_code seam |
| got_ocr2_hf | stepfun.py `GotOCR2HfLoader` | migrated | llava_hf；删掉 legacy `_no_split_modules` |

---

# MLLM A/B 批次

> 统一口径："全量功能还原" = 还原真实能力 + **不移植 dev 架构已废弃的 patch**（依据 `PATCH_INVENTORY.md`）。
> 已废弃且在本批次中省略的 seam：`device_map`/`_no_split_modules`、`patch_output_clone`、`patch_get_input_embeddings`、
> `patch_qwen_vl_utils`/`keye_vl_utils`、`patch_output_to_input_device`、`enable_input_require_grads`（use_reentrant=False 下不需要）。
> MLLM 模式：固定 `model_cls` + `model_arch` property（`ModelArch(language_model/aligner/vision_tower/moe_block)`）+ `is_multimodal=True`；
> model_arch 分区串对 transformers 5.5 layout（`transformers_ge_4_52` → `model.*` 前缀分支）。

## gemma.py

| legacy model_type | dev loader | 代表模型 | 结论 | 依据 / 备注 |
|---|---|---|---|---|
| gemma / gemma2 | `GemmaLoader` / `Gemma2Loader` | gemma-7b / gemma-2-9b | migrated | 纯文本 |
| gemma3_text | `Gemma3TextLoader` | gemma-3-4b-text | migrated | `_EagerAttnDefault` mixin（eager-attn 默认） |
| paligemma | `PaligemmaLoader` | paligemma-3b | migrated | MLLM，model_arch=llava_hf |
| gemma3_vision | `Gemma3VisionLoader` | gemma-3-4b-it | migrated | MLLM + eager-attn；跳过 output_to_input_device |
| gemma3n | `Gemma3nLoader` | gemma-3n-E4B | migrated | 双塔（vision+audio），aligner/vision 各含 2 项 |
| gemma4 / gemma4_unified / diffusion_gemma / gemma_emb | — | Gemma-4 / T5Gemma | deferred | forward 重写+MoE / diffusion / SentenceTransformers |

## llava.py

11 个 `*_hf` model_type（`LlavaForConditionalGeneration` / `LlavaNextForConditionalGeneration` / `LlavaNextVideoForConditionalGeneration`），共享 llava_hf 分区。

| legacy model_type | dev loader | 结论 | 依据 / 备注 |
|---|---|---|---|
| llava_llama3_hf / llava1_5_hf / llava_onevision_hf | `LlavaLlama3HfLoader` / `Llava1_5HfLoader` / `LlavaOnevisionHfLoader` | migrated | onevision=AutoModelForImageTextToText |
| llava_next_qwen_hf / llama3_llava_next_hf / llava1_6_vicuna_hf / llava1_6_mistral_hf / llava_llama3_1_hf / llava1_6_yi_hf | `_LlavaNextHfLoader` 6 变体 | migrated | 6→1 反查（同 `LlavaNextForConditionalGeneration`）符合预期；llava1_6_yi_hf 的 legacy Yi override 是死代码 |
| llava_next_video_hf / llava_next_video_yi_hf | `LlavaNextVideoHfLoader` / `LlavaNextVideoYiHfLoader` | migrated | yi 变体 process_config 设 video/image_token_index |
| llava 老家族（git_clone）/ llava_onevision1_5 | — | deferred | git clone + 外部包 |

## mllm.py

| legacy model_type | dev loader | 结论 | 依据 / 备注 |
|---|---|---|---|
| idefics3 | `Idefics3Loader` | migrated | AutoModelForVision2Seq，arch=idefics3 |
| pixtral | `PixtralLoader` | migrated | LlavaForConditionalGeneration，llava_hf |
| keye_vl / molmo* / megrez / dots_ocr / jina_reranker_m0 / sail_vl2 | — | deferred | remote-code / forward-patch，待 remote-code seam |

## internlm.py

| legacy model_type | dev loader | 结论 | 依据 / 备注 |
|---|---|---|---|
| internvl | `InternVLLoader` | migrated | `-hf` 系（InternVLForConditionalGeneration），丢弃过时 enable_input_require_grads；requires 取最高 4.55.0 |
| internlm / internlm2 / internlm3 / interns1 / internvl_chat / xcomposer2* / internlm2_reward | — | deferred | remote-code（相应类不在 tf5.5）/ git_clone / interns1 pinned `<4.56` |

## glm.py

legacy per-`ModelGroup` 多 template → dev 拆成 base + `architectures=[]` 模板变体子类（命名用模板名，参照 llama）。MoE 经 `model_arch.moe_block`（类名）声明 z3 叶子。

| legacy model_type | dev loader / model_type | 代表模型 | 结论 | 依据 / 备注 |
|---|---|---|---|---|
| glm4 | `Glm4Loader` / `glm4` + `Glm4Z1RuminationLoader` / `glm4_z1_rumination` | GLM-4-9B-0414 / GLM-Z1-Rumination-32B | migrated | 文本；变体拆分 |
| glm_edge | `GlmEdgeLoader` / `glm_edge` | glm-edge-1.5b-chat | migrated | 文本，template chatglm4 |
| glm4_moe | `Glm4MoeLoader` / `glm4_moe` + `Glm4_7Loader` / `glm4_7` | GLM-4.5 / GLM-4.7 | migrated | MoE=Glm4MoeMoE；GLM-4.7 组 template glm4_7 |
| glm4_moe_lite | `Glm4MoeLiteLoader` / `glm4_moe_lite` | GLM-4.7-Flash | migrated | MoE=Glm4MoeLiteMoE |
| glm_moe_dsa | `GlmMoeDsaLoader` / `glm_moe_dsa` + `Glm5_1/Glm5_2Loader` / `glm5_1` `glm5_2` | GLM-5 / GLM-5.1 / GLM-5.2 | migrated | MoE=GlmMoeDsaMoE |
| glm4v | `Glm4vLoader` / `glm4v` + `Glm4_5vLoader` / `glm4_5v` | GLM-4.1V-9B / Glyph / GLM-4.6V-Flash | migrated | MLLM，arch=glm4v；丢弃过时 patch_get_input_embeddings |
| glm4v_moe | `Glm4vMoeLoader` / `glm4v_moe` | GLM-4.5V / GLM-4.6V | migrated | MLLM+MoE=Glm4vMoeTextMoE，template glm4_5v |
| glm_ocr | `GlmOcrLoader` / `glm_ocr` | GLM-OCR | migrated | AutoModelForImageTextToText，arch=glm4v |
| glm_edge_v | `GlmEdgeVLoader` / `glm_edge_v` | glm-edge-v-2b | migrated | processor_cls=AutoImageProcessor；arch=glm_edge_v（含 legacy 重复 id glm-edge-4b-chat，忠实保留） |
| chatglm2/3/4 / codegeex4 / chatglm4v / cogvlm / cogagent_* / cogvlm2 / cogvlm2_video | — | ChatGLM* / CogVLM* | deferred | remote-code + `_patch_tokenizer`（"不迁移"）+ tf<4.42 / 借用 vicuna tokenizer |

## 单模型补充（并入既有文件）

| legacy model_type | dev 位置 | 结论 | 依据 / 备注 |
|---|---|---|---|
| qwen2_audio | qwen.py `Qwen2AudioLoader` | migrated | 忠实保留 `<4.49` pin（dev tf5.5 会被 requires 标记，类仍在 in-tree） |
| qwen_audio / qwen2_5_omni / qwen3_omni_moe / qwen3_asr / qwen3_tts | — | deferred | Qwen1 remote-code / omni 混数据 keep-alive forward 重写 / 双通道 TTS |
| hunyuan_ocr | tencent.py `HunyuanOcrLoader` | migrated | eager-attn 默认 seam（build_model setdefault） |
| paddleocr_vl | baidu.py `PaddleOCR1_5Loader` | migrated | AutoModelForImageTextToText，无 architectures，requires>=5.0 |
| minimax_m3_vl | minimax.py `MinimaxM3VLLoader` | migrated | processor trust_remote_code seam |
| got_ocr2_hf | stepfun.py `GotOCR2HfLoader` | migrated | llava_hf；删掉 legacy `_no_split_modules`（dev 走 twinkle strategy，废弃 device_map） |

---

# Qwen 文本家族（qwen.py，本批次）

legacy 将多个 chat 格式塑到少数 model_type（`qwen2`/`qwen3`/...）下，每组一个 template。dev 按 llama 范式拆分：每架构一个 base（反查 owner，architectures 声明）+ `architectures=[]` 模板变体子类。文本模型用默认空 `ModelArch`；MoE 标 `is_moe=True`（暂不设 moe_block，同 phi3_moe 决策：等 z3 wiring 落地再补类名）。

> 命名决策（用户确认）：`qwen3_thinking`/`qwen3_nothinking`/`qwen3_coder` template 跨 dense/moe/next 三架构复用，model_type 必须唯一 → **dense 独占裸名**（`qwen3_thinking`/`qwen3_nothinking`），**moe/next 家族前缀限定**（`qwen3_moe_*`/`qwen3_next_*`）。

| legacy model_type | dev loader / model_type | 结论 | 依据 / 备注 |
|---|---|---|---|
| qwen2 | `Qwen2Loader`/`qwen2` + 变体 `code_qwen`/`qwen2_math`/`qwen2_5`/`qwen2_5_coder`/`qwen2_5_math`/`marco_o1`/`qwq_preview`/`qwq` | migrated | Qwen2ForCausalLM 反查 owner=qwen2（template qwen）；Qwen1.5/Qwen2/Qwen2.5/QwQ 等按 template 拆变体；coding/math 组用 tags 变体 |
| qwen2_moe | `Qwen2MoeLoader`/`qwen2_moe` | migrated | Qwen1.5-MoE/Qwen2-57B-A14B；template qwen；is_moe |
| qwen3 | `Qwen3Loader`/`qwen3` + 变体 `qwen3_guard`/`yufeng_xguard`/`qwen3_thinking`/`qwen3_nothinking` | migrated | Qwen3ForCausalLM 反查 owner；dense 独占裸变体名 |
| qwen3_moe | `Qwen3MoeLoader`/`qwen3_moe` + 变体 `qwen3_moe_nothinking`/`qwen3_moe_thinking`/`qwen3_moe_coder` | migrated | Qwen3MoeForCausalLM；is_moe；变体家族前缀限定 |
| qwen3_next | `Qwen3NextLoader`/`qwen3_next` + 变体 `qwen3_next_thinking`/`qwen3_next_coder` | migrated | Qwen3NextForCausalLM；base template=qwen3_nothinking（Instruct）；is_moe |
| （qwen2/qwen3 的 deepseek_r1 蒸馏组） | — | deferred | DeepSeek-R1-Distill-Qwen / DeepSeek-R1-0528-Qwen3：Qwen 架构但 DeepSeek 品牌，路由未来 deepseek.py（同 llama 先例）；反查落 qwen2/qwen3 base |

---

# DeepSeek 文本家族（deepseek.py，本批次）

legacy `DeepseekLoader.get_model` 仅对每个 MLP 做 `patch_output_to_input_device`（HF device 占位补丁，PATCH_INVENTORY 已废弃）→ 丢弃，降为普通 loader。同 llama/qwen 文本范式：每架构 base + `architectures=[]` 模板变体；文本用默认空 `ModelArch`；MoE 标 `is_moe`。

| legacy model_type | dev loader / model_type | 结论 | 依据 / 备注 |
|---|---|---|---|
| deepseek | `DeepseekLoader`/`deepseek` | migrated | v1 MoE；arch 不在 tf5.5（remote-code）→ g7 trust_remote_code + AutoModelForCausalLM |
| deepseek_v2 | `DeepseekV2Loader`/`deepseek_v2` + `deepseek_v2_5` | migrated | in-tree；base template deepseek + 变体 deepseek_v2_5；is_moe；requires>=4.39.3 |
| deepseek_v3 | `DeepseekV3Loader`/`deepseek_v3` + `deepseek_r1`/`deepseek_v3_1`/`kimi_k2` | migrated | in-tree；base template deepseek_v2_5；变体 R1(全)/V3.1/Kimi-K2；is_moe |
| deepseek_v32 / deepseek_v4 | — | deferred | arch 不在 tf5.5，mcore/Megatron-primary（deepseek_v4 仅见 swift/megatron）；v32 额外需 dummy-model 回退 seam（return_dummy_model） |
| deepseek_v3（moonlight 组） | — | deferred | pinned `transformers<4.49`（dev 5.5 dead） |
| （deepseek_r1 蒸馏组） | — | deferred | 在 qwen2/qwen3/llama 下（Qwen/Llama 架构），反查落对应 base |

---

# task 变体批次（qwen.py / internlm.py）

| legacy model_type | dev loader / model_type | 结论 | 依据 / 备注 |
|---|---|---|---|
| qwen3_emb | `Qwen3EmbLoader`/`qwen3_emb` | migrated | in-tree `Qwen3ForCausalLM`（legacy 无自定义 loader）；同 `Qwen3VLEmbLoader` 路子：task 变体声明 architectures（不能落回 qwen3 生成）；mcore_model_type='qwen3_emb'；task_type 不 pin（同 legacy，由 --task_type 定）；丢弃 additional_saved_files（export 关注点，dev loader 未建模） |
| qwen3_reranker | `Qwen3RerankerLoader`/`qwen3_reranker` | migrated | 同上；template qwen3_reranker；mcore_model_type='gpt'（legacy 值） |
| internlm2_reward | `InternLM2RewardLoader`/`internlm2_reward` | migrated | remote-code `InternLM2ForRewardModel`（不在 tf5.5）→ g7 trust_remote_code + `AutoModel`（legacy `RewardModelLoader` 在 auto_map 含 AutoModel 时走 AutoModel）；`is_reward=True`（声明，reward 路径接线时消费）；requires>=4.38 |
| qwen2_reward / qwen2_5_prm | — | deferred | remote-code `Qwen2ForRewardModel`/`Qwen2ForProcessRewardModel` 均不在 tf5.5；qwen2_reward 含两个不同 template 组（qwen / qwen2_5_math），qwen2_5_prm 需 `task_type='prm'` —— dev builders 仅认 seq_cls/reranker/embedding/generative_reranker，无 prm 头 |
| qwen2_gte / gemma_emb | — | deferred | 走 `SentenceTransformersLoader`（用 `SentenceTransformer(model_dir)` 构建而非 `from_pretrained`），dev 无此 seam |

---

# MLLM 补迁批次（baidu / stepfun / qwen / glm / moonshot）

> 前提修正：上一版对账把这几个笼统归进了“MLLM 待新 seam”，但逐个读完 legacy loader 后发现：它们的 seam 要么已废弃（device_map 系），要么现有 hook 就能表达。

| legacy model_type | dev loader / model_type | 结论 | 依据 / 备注 |
|---|---|---|---|
| ernie_vl | baidu.py `ErnieVLLoader`/`ernie_vl` + `ernie_vl_thinking` | migrated | in-tree `Ernie4_5_VLMoeForConditionalGeneration`；legacy `leaf_modules=MOEAllGatherLayerV2`（经 `get_class_from_dynamic_module`）→ dev 声明式 `ModelArch.moe_block=['MOEAllGatherLayerV2']`（按类名匹配，无需动态 import）；`add_image_preprocess(processor)` 需 model+processor 同在 → `build_model` override（同 MiniCPM-V 绑定）；thinking 组拆为 `architectures=[]` 模版变体 |
| step3_vl | stepfun.py `Step3VLLoader`/`step3_vl` | migrated | remote-code `StepVLForConditionalGeneration` → g7 + `AutoModelForImageTextToText`；`config.vocab_size = config.text_config.vocab_size` → `process_config`；权重扁平命名的 `key_mapping` → `build_model` 的 setdefault kwarg |
| midashenglm | qwen.py `MidashengLMLoader`/`midashenglm` | migrated | remote-code `MiDashengLMModel` → g7 + `AutoModel`；保留 `audio_encoder.float()`（Dasheng 编码器 bf16 不稳定，真实需求）；丢弃 `patch_output_clone`（PATCH_INVENTORY 已废弃） |
| cogvlm2_video | glm.py `CogVLM2VideoLoader`/`cogvlm2_video` | migrated | remote-code `CogVLMVideoForCausalLM` → g7；legacy `CogVLM2Loader` 整个身体均为 device_map 时代补丁（逐层 `patch_output_to_input_device` + 手动挑 boi/eoi device）→ 全丢，降为纯声明 loader；`>=4.42` 是下限（区别于兄弟 cogvlm/cogvlm2/cogagent_* 的 `<4.42` 上限） |
| kimi_k3 | moonshot.py `KimiK3Loader`/`kimi_k3`（新建文件） | migrated | remote-code `KimiK3ForConditionalGeneration` → g7 + `AutoModelForImageTextToText`；唔一 seam 是给嗡嗗的 remote tokenizer logger 降噪 → `build_processor` override |
| kimi_vl | — | deferred | pinned `transformers<4.49`（tf5.5 上死）；另有删 `_supports_sdpa` + `patch_get_input_embeddings`（后者已废弃） |
| kimi_k25 | — | deferred | pinned `>=4.57.1,<5.0.0` —— `<5.0.0` 上限排除 dev 的 5.5，**虽然它压根不需要任何 loader 逻辑**（本轮修正：上版误归为“版本窗口卡住”，实为版本死） |
| florence | — | deferred | `use_submodel_func`（`delegate_to_submodel` 可覆盖）+ `patch_ignore_check_imports`（dev 无对应）+ device_map；卡在 ignore-check-imports |

---

# 全量对账（ground truth，按注册表差集自动统计）

> 数据来源：`swift.model.model_meta.MODEL_MAPPING`（legacy）vs `swift.dev.model.loader.base.MODEL_MAPPING`（dev，导入全部 loader 模块后）。此段为自动对账结果，与上方逐文件表互为印证。

## 指标 1：支持（已迁移并注册）
- **dev 已注册 model_type：236**（其中 167 与 legacy 同名直接迁移，69 为迁移时拆分/新增的 dev 专有名）。
- legacy 总数 219；legacy 中已被 dev 覆盖（同名）= 167，剩 52 未进 dev（见指标 2）。236 = 167 同名 + 69 dev 专有名，已实测对账。

## 指标 1b：checkpoint id 层面完整性（二轮审计新增口径）

之前每批只验证了 model_type 注册数，**未验证 group-template 和 checkpoint 两层**，审计后补齐：

| 修补项 | 内容 |
|---|---|
| `minicpm5` 漏迁 | 文档曾写 “routed → minicpm.py” 但从未实现；已补（llama 架构 + `minicpm5` 模版变体） |
| `minicpm_moe` 真 bug | 子类未覆盖 `models`，**误继承父类 dense checkpoint**（声明成 MiniCPM-2B-sft-fp32），真实的 `MiniCPM-MoE-8x2B` 丢失；已修。全局扫描确认仅此一例 |
| hunyuan_v1_dense +12 | FP8 / AWQ-Int4 / GPTQ-Int4 各 4 个（同架构同模版，quant config 在 checkpoint 内） |
| qwen3 +11 | FP8×6、AWQ×4、swift/Qwen3-32B-AWQ |
| qwen3_moe +16 | 按模版归位：base（FP8×2+AWQ×2）/ nothinking（2507-FP8×2+AWQ+Marco×3）/ thinking（2507-FP8×2+AWQ）/ coder（FP8×2+AWQ） |
| DeepSeek 2025 新模型 +8 | `unsloth/DeepSeek-Prover-V2-671B-BF16` 进 `deepseek_v3`；新建 3 个 R1 蒸馏 loader（见下） |

### `deepseek_r1` 模版跮三架构的命名处理
`deepseek_r1` 模版同时出现在 DeepseekV3（原版 R1）/ Llama / Qwen2 / Qwen3 四种架构上。dev 的 model_type 必须唯一 → 沿用已确认的**家族前缀限定**约定：V3 保留裸名 `deepseek_r1`，蒸馏版分别为 `deepseek_r1_distill_llama`（2025-01, 8B/70B）、`deepseek_r1_distill_qwen2`（2025-01, 1.5B/7B/14B/32B + QwenLong-L1-32B）、`deepseek_r1_distill_qwen3`（2025-05, R1-0528-Qwen3-8B），均 `architectures=[]` 以保证反查仍落在 `llama`/`qwen2`/`qwen3`。

### 剩余 checkpoint 缺口：93（经用户确认不补）
- **旧量化 77**：Qwen1.5/Qwen2 GPTQ-Int4·AWQ（qwen2 下 77）、qwen2_moe 的 2 个 GPTQ、mixtral/llama 的 AQLM-2Bit。用户判定：2024 年旧模型量化变体已无人用，不补。
- **旧主 checkpoint 13**：`llama` 下的 deepseek-llm/math/coder（2023-24，Llama 架构 + `deepseek` 模版）。不在 25/26 年范围。
- **版本死 3**：`Moonlight-16B-A3B`×2（`<4.49`）、`MiniCPM-o-4_5`（`==4.51.3`）。

## 指标 1c：A/C/D 批次迁移（新 seam·keep-alive·remote-code 混合）

本批次按用户要求迁移 A 类全部 + C 类 gemma4/qwen-omni 系列 + diffusion_gemma + D 类，共 16 个 model_type（含 2 个模板变体），201→217：

| 迁移项 | 关键处理 |
|---|---|
| `deepseek_v32` | 继承 `DeepseekV3Loader` + g7 `trust_remote_code`；`build_config` 保留 V3 回退（arch 不在 tf5.5） |
| `deepseek_v4` | g7 `trust_remote_code` + `AutoModelForCausalLM`，`is_moe`；Megatron-primary，HF 路径走 remote-code |
| `qwen2_gte` | task 变体（`architectures=['Qwen2ForCausalLM']` + `task_type='embedding'`）；ST 构建在 model 层，loader 仅登记 |
| `gemma_emb` | 同上，backbone `Gemma3TextModel` + `task_type='embedding'` |
| `qwen2_reward`/`qwen2_5_math_reward` | remote-code `Qwen2ForRewardModel`（不在 tf5.5）→ `AutoModel`+`trust_remote_code`+`is_reward`；math 组为模板变体 |
| `qwen2_5_prm` | remote-code `Qwen2ForProcessRewardModel` → `AutoModel` 原生加载 + `is_reward`；**不 pin `task_type='prm'`**（dev builders 无 prm 头，PRM 逐步头由 checkpoint 自带；step-level loss 是独立 builder 任务） |
| `gemma4`/`gemma4_thinking`/`gemma4_unified` | in-tree（unified/diffusion 需 >=5.10/5.11，`model_cls` 惰性解析）；**丢弃 `_patch_gemma4_forward` 200+ 行 forward fork**，改用 dev 非侵入式 `apply_vision_keep_alive`；gemma4 MoE 叶子 `Gemma4TextExperts` |
| `diffusion_gemma` | block-diffusion：`process_model` 保留 `prepare_inputs_for_generation=None`+`use_cache=True` |
| `megrez_omni` | remote-code `MegrezO` + `delegate_to_submodel('llm')`；processor 由 model 创建 → `build_processor` 实例化 model 取 `_get_or_init_processor()`（processor-from-model 反转） |
| `jina_reranker_m0` | remote-code `AutoModel` + `task_type='reranker'`；`process_model` 绑定 forward 包装（`SequenceClassifierOutputWithPast`，logit_bias 2.65）+ `padding_free_fn` |
| `qwen2_5_omni`/`qwen3_omni_moe` | 继承 `Qwen2VLLoader` 复用 env/global_vars；`delegate_to_submodel('thinker')` + talker 配置；**丢弃 `patch_get_input_embeddings`/`_no_split_modules`**；qwen3 的 `_compat_qwen3_omni_mixed_data` forward fork → `apply_vision_keep_alive` |
| `minicpmv4_6` | transformers-native `AutoModelForImageTextToText`（需 >=5.7）；`build_model` 保留 `_patch_qwen3_5_linear_attention_sequence_parallel`（linear-attn SP，真实需求） |

反查未污染：`Qwen2ForCausalLM=[qwen2, qwen2_gte]`、`Gemma4ForConditionalGeneration=[gemma4]`（thinking 变体 `architectures=[]` 正确排除）、reward/prm/omni/megrez 各自唯一。

## 指标 1d：B 类重 patch 批次（chatglm4 / internvl_chat / minimax_vl / molmoe）

本批次按用户要求迁移 5 个 legacy model_type，产出 9 个 dev model_type（internvl_chat 家族按模板拆 5 个）；收尾审计又补上 `florence` 与 13 个旧 DeepSeek dense checkpoint（见下），共 13 个 dev model_type，217→230：

| 迁移项 | 关键处理 |
|---|---|
| `chatglm4` | remote-code `ChatGLMModel` + g7 `trust_remote_code`；`process_tokenizer` 保留 `_pad` 签名兼容（旧 tokenizer 不接受 `padding_side`）+ 未注册 special token 修复（`<|user|>` 编码成多 token 时 `add_tokens`）；**丢弃全局 `CrossEntropyLoss.forward` 覆写**（device_map 时代跨设备补丁，blast radius 过大）、`llm_int8_skip_modules`、动态类 tokenizer 手术（`_auto_class`/`remove_property`，已被 g7 覆盖） |
| `chatglm4v` | 同族 vision 半边，**同 `architectures` 有意共享**（模态差异，反查返回 `[chatglm4, chatglm4v]` 由 caller 按 id 消歧）；`build_processor` 保留 `init_kwargs['image_size']=1120`；丢弃 4-GPU `patch_output_to_input_device` + boi/eoi 挪设备 |
| `internvl_chat` 家族 5 个 | remote-code `InternVLChatModel`，`processor_cls` 钉 `AutoTokenizer`（checkpoint 带 `preprocessor_config.json`，否则文件探测会误走 `AutoProcessor`）；`process_model` 用 `delegate_to_submodel('language_model')` 还原 `use_submodel_func`；legacy 单 model_type 8 组 → base `internvl_chat`(internvl 模板) + `internvl_chat_v2`/`_v2_5`/`_v3_5`/`_v3_5_gpt`（`architectures=[]` 模板变体）；两个 `internvl2_5` 同模板组按最严 floor 合并；丢弃 `patch_output_clone` + bnb `force_no_igemmlt`。**id 覆盖 106/108**，未覆盖 2 个为 phi3 组（`<4.42` 版本死） |
| `minimax_vl` | remote-code；保留 `ignore_check_imports`（remote code 声明了缺失依赖，不放行则动态 import 直接失败）+ `build_processor` 回绑三个模板依赖的动态符号（`MiniMaxVL01ProcessorKwargs`/`get_hw_multiple_of`/`get_num_token`）；**丢弃手工 device_map 分片**（读 safetensors index 逐层分配）+ Quanto `modules_to_not_convert` |
| `molmoe` | 继承 `MolmoLoader`，remote-code 类名为 `OLMoForCausalLM`（非 `MolmoForCausalLM`，反查独占）；`build_model` 以 `setdefault('dtype', float32)` 还原 legacy `torch_dtype=torch.float32`（dev 无家族级默认 dtype 声明，用户显式 dtype 仍优先）；`process_model` 保留 `config.to_dict` 修补（`vision_backbone` 非嵌套 config，原生 `to_dict()` 会丢它）；丢弃 `patch_output_clone` |
| `florence` | **收尾审计发现的真遗漏**：原唯一障碍 `patch_ignore_check_imports` 已随 `minimax_vl` 落地。本次将 `ignore_check_imports` **提升为 `ModelLoader` 静态方法**（与 `delegate_to_submodel`/`apply_z3_leaf_modules` 一致）供两家共用；`process_config` 保留 `vision_config.model_type='davit'`（不设则 merge-lora 失败）、`process_model` 保留 `vision_tower.enable_checkpoint=True`（DaViT 自带开关，通用 `gradient_checkpointing_enable` 触不到）+ `delegate_to_submodel('language_model', ['generate','forward'])`；丢弃 `device_map='auto'` 覆写 |
| `deepseek_llm`/`deepseek_math`/`deepseek_coder` | 收尾审计的 id 层缺口，经用户确认补入（“反正没有额外代码”）。2023-11 初代 **dense** DeepSeek，legacy 挂在 `llama` model_type 下分 3 组（仅 `tags` 不同，模板同为 `deepseek`）。dev 的 `tags` 是 model_type 级属性，故拆 3 个而非合并 tag 并集；均 `architectures=[]`（`LlamaForCausalLM` 归 llama 家族，不参与反查）。与 dev 已有的 `deepseek`（MoE v1，`DeepseekForCausalLM`）架构不同，不能合并。共 13 个 checkpoint（4+3+6） |

反查未污染：`ChatGLMModel`/`ChatGLMForConditionalGeneration=[chatglm4, chatglm4v]`（有意共享）、`InternVLChatModel=[internvl_chat]`（与 -hf 的 `InternVLForConditionalGeneration=[internvl]` 完全分离）、`OLMoForCausalLM=[molmoe]` vs `MolmoForCausalLM=[molmo]`、`MiniMaxVL01ForConditionalGeneration=[minimax_vl]`。5 个 legacy 家族 id 覆盖 116/118（差 2 为版本死）。

## 指标 1e：用户指定的 6 个重点模型（版本 pin 根因排查批次）

用户指定：`deepseek_ocr2` / `llava_onevision1_5` / `qwen3_asr` / `qwen3_tts` / `phi4_multimodal` / `minicpmo`(MiniCPM-o-4_5)。230→236。本批次的核心是**逐个排查版本 pin 的真实根因**，而非直接接受 legacy 的声明；结果三类分明。

| 迁移项 | pin 根因与处理 | 可用性 |
|---|---|---|
| `llava_onevision1_5` | **之前的分类是错的**：我曾归入 git_clone，实际 requires 是 `transformers>=4.53.0`（floor）+ `qwen_vl_utils`（已装）——从未版本死。loader 三件事里两件废弃（`_no_split_modules` 是 device_map hint、`patch_get_input_embeddings` 是 reentrant 兼容），`get_class_from_dynamic_module` 仅为设置前者而存在 → g7 `trust_remote_code` 即可。保留 `vision_start_token_id=151652` | ✅ 可用 |
| `phi4_multimodal` | pin `<4.49` 守的是 remote-code `Phi4MMForCausalLM`。**tf5.5 自带原生端口** `transformers.models.phi4_multimodal` → 改用 `Phi4MultimodalForCausalLM`，pin 变 `>=5.0`，不需 trust_remote_code。三处差异：原生 `Phi4MultimodalProcessor` 字段已规范（legacy 六改三删的 hack 全丢）；**原生版零 LoRA 代码**（实测 `set_lora_adapter` 不存在、模块源码 0 处 `lora`）故 `set_lora_adapter(['vision','speech'])` 无处可接 → **微调语义与 legacy 不同**，模态适配从内置 adapter 激活变为普通 tuner；model_arch 按原生布局（`img_projection` 拆成 up/down 对） | ⚠️ 可用，语义有变 |
| `qwen3_asr` | pin `==4.57.6` 的根因**已定位到单点**：`qwen_asr` 用 `@check_model_inputs()`（工厂式），tf5.x 改成裸装饰器 → import 即 `TypeError`。新增 `Qwen3ASRLoader.compat_check_model_inputs`（静态方法，只 patch `transformers.utils.generic` 一处——实测那是 tf5.5 唯一持有该符号的模块），使两种写法都可用。**已实测**：shim 后 `qwen_asr` import 成功、`Qwen3ASRForConditionalGeneration`/`Config`/`Processor` 均可达、幂等。requires 去掉 transformers pin。残留风险已写入 docstring：shim 只恢复调用约定，`check_model_inputs` 在 5.x 负责 output_attentions/hidden_states 收集，forward 辅助输出未对真 checkpoint 验证 | ⚠️ 可加载，forward 待实跑 |
| `qwen3_tts` | pin `<5`。`qwen_tts` **本环境未装**，用户选择不装先写 loader。保留全部真实逻辑：Auto* 三注册（`exist_ok=True`）、`_patch_qwen3_tts_forward`（**非废弃 patch，而是双通道训练本体**：text/codec 双通道 embedding 求和 + speaker embedding 注入 codec position 6 + 15 层 sub-talker embedding + sub-talker CE loss；从 legacy import 而非拷贝）、freeze `speaker_encoder`、外部 `Qwen3TTSTokenizer` 下载、delegate `get_input_embeddings`/`gradient_checkpointing_enable` 到 talker。ceiling 按与 ASR 同理由丢弃（pin 守的是包而非权重），但**无 import 级证据** | ❓ 未验证 |
| `deepseek_ocr2` | pin `==4.46.3`（距 tf5.5 跳 9 个小版本）+ `easydict` 未装。loader 代码几乎全是废弃 patch（`patch_output_clone` + `patch_output_to_input_device`×3），声明很轻；但**pin 保留不动**——与 ASR 不同，这里没有定位到具体不兼容点，乐观放宽没依据。注册使 id 可解析、版本检查报真实冲突 | ❌ 版本死（已声明）|
| `minicpmo4_5` | 家族已支持，只缺这个 id。加模板变体（`architectures=[]`，2_6 保留 `MiniCPMO` 反查所有权）。pin `==4.51.3` + `minicpmo-utils==1.0.6` 同样保留不动（无证据支持放宽） | ❌ 版本死（已声明）|

方法论沉淀：**版本 pin 分三类**——(a) 错归类（llava_onevision1_5，实际是 floor）、(b) 守第三方包且不兼容点可定位可 shim（qwen3_asr）、(c) 守 remote-code 旧 API 且未定位（deepseek_ocr2 / minicpmo4_5）。只有 (b) 允许放宽，且必须附实测证据 + 残留风险说明。另一条：**当 transformers 已官方端口某 remote-code 模型时，改基到原生类比移植旧 patch 更划算**（phi4_multimodal），但必须核对能力差异并写明。

反查未污染：6 个 arch 各自唯一，`MiniCPMO -> [minicpmo]`（4_5 变体正确排除）。

## 指标 2：未进 dev 的 legacy model_type = 52（按障碍类型，已逐个归属，无悬空项）

> 关键区分：**"待迁"≠"无法迁"**。下面前两类是**待迁（可迁，只是还没轮到）**，后四类是**真障碍（需新 seam / 版本死 / 外部仓）**。

### A. 待迁类（可迁，无真障碍）
已全部完成：纯文本 backbone（qwen2/3/3_moe/3_next + deepseek v1/v2/v3 + gpt_oss/dbrx/grok/olmoe/dots1/ling/bailing_*/hunyuan/hy_v3/mimo/aya/c4ai/bluelm/orion/xverse/yuan2/polylm/longchat/youtu_llm/minicpm/minicpm3/minicpm_chatml/minicpm_moe）及 task 变体（qwen3_emb/qwen3_reranker/internlm2_reward）均已迁移。

### B. 文本 deferred（20，真障碍或非核心已终止支持）
| legacy model_type | 障碍 | 分类 |
|---|---|---|
| baichuan | transformers<4.34 版本死 | 版本死 |
| baichuan2 | lm_head fp32 patch + remote-code | 重 patch |
| chatglm2 | transformers<4.42 版本死 + _patch_tokenizer | 版本死 |
| chatglm3 | transformers<4.42 版本死 + _patch_tokenizer | 版本死 |
| chatglm4 | remote-code + _patch_tokenizer（add_tokens class-level hack） | 重 patch |
| codefuse_codegeex2 | transformers<4.34 版本死 + ChatGLM2 动态类 | 版本死 |
| codefuse_qwen | Qwen1 QWenLMHeadModel remote-code 2023 遗留 | Qwen1 遗留 |
| codegeex4 | transformers<4.42 版本死 + ChatGLM4 动态类 | 版本死 |
| deepseek_v32 | arch 不在 tf5.5, mcore-primary, 需 dummy-fallback seam | mcore-primary |
| deepseek_v4 | arch 不在 tf5.5, mcore-primary | mcore-primary |
| gemma_emb | SentenceTransformersLoader（dev 无此 seam） | 新 seam |
| iquestcoder | transformers==4.52.4 版本死 | 版本死 |
| minimax | 手工 device_map + Quanto modules_to_not_convert + "不支持训练" | 重 patch |
| minimax_m1 | 同 minimax（手工 device_map + "不支持训练"） | 重 patch |
| modelscope_agent | Qwen1 QWenLMHeadModel remote-code 2023 遗留 | Qwen1 遗留 |
| phi3_small | 逐层 rotary_emb dtype patch + arch 不在 tf5.5 | 重 patch |
| qwen | Qwen1 QWenLMHeadModel remote-code + fix registered_causal_mask + eos_token_id patch | Qwen1 遗留 |
| qwen2_5_prm | remote-code Qwen2ForProcessRewardModel (不在 tf5.5) + task_type='prm'（dev 尚无 prm 头） | 新 seam |
| qwen2_gte | SentenceTransformersLoader（dev 无此 seam） | 新 seam |
| qwen2_reward | remote-code Qwen2ForRewardModel (不在 tf5.5) + is_reward + 多 template 组 | 新 seam |

### C. MLLM deferred（57，真障碍）
- **版本死锁 pin（16）**：`cogvlm`/`cogvlm2`/`cogagent_chat`/`cogagent_vqa`(<4.42)、`deepseek_ocr`/`deepseek_ocr2`/`unlimited_ocr`(==4.46.3)、`keye_vl_1_5`(==4.52.4)、`kimi_vl`(<4.49)、`kimi_k25`(<5.0.0)、`sail_vl2`(<=4.51.3)、`step_audio2_mini`(==4.53.3)、`qwen3_asr`(==4.57.6)、`qwen3_tts`(<5)、`minicpmo4_5`(==4.51.3)、`paddle_ocr`(<5.0)。
- **git_clone / 外部包（25）**：`deepseek_vl`/`deepseek_vl2`/`deepseek_janus`/`deepseek_janus_pro`、`yi_vl`、`mplug_owl2`/`mplug_owl2_1`/`mplug_owl3`/`mplug_owl3_241101`/`doc_owl2`、`valley`、`llava_next_qwen`/`llama3_llava_next`/`llava1_6_mistral`/`llava1_6_yi`/`llava_onevision1_5`、`xcomposer2`/`xcomposer2_4khd`/`xcomposer2_5`/`xcomposer2_5_ol_audio`、`emu3_gen`/`emu3_chat`、`got_ocr2`、`step_audio`、`llama3_1_omni`。
- **Qwen1 VL/Audio remote-code（2）**：`qwen_vl`/`qwen_audio`（QWenLMHeadModel 2023 动态类 + 各自专属 patch）。
- **MLLM 待新 seam / 真耦合（~16）**：`megrez_omni`（processor 由 model 创建、顺序反转）、`jina_reranker_m0`（forward 整体重写 + padding_free_fn）、`molmoe`（float32 默认 + config.to_dict patch）、`qwen2_5_omni`/`qwen3_omni_moe`（omni keep-alive forward 重写）、`gemma4`/`gemma4_unified`（`_patch_gemma4_forward` 200+ 行 forward 重写）/`diffusion_gemma`（block-diffusion：`prepare_inputs_for_generation=None`）、`florence`（patch_ignore_check_imports）、`internvl_chat`/`interns1`、`chatglm4v`、`minimax_vl`、`minicpmv4_6`。

> 小结：52 未进 dev = **48 真障碍**（19 版本死 + 24 git_clone 外部仓 + 5 Qwen1-2023 遗留）+ **2 用户明确否决**（`baichuan2`/`phi3_small`）+ **2 无行为可迁**（`minimax`/`minimax_m1`）。
>
> **双层审计（归零）**：model_type 层 52 个全部有归属，悬空项 = 0。checkpoint id 层：已迁家族内缺口 81 = **77 旧量化**（用户确认不补）+ **4 版本死组内 id**（Moonlight-16B-A3B×2 `<4.49`、Mini-InternVL-Chat-4B-V1-5 / InternVL2-4B `<4.42`），待定夺项 = **0**。

