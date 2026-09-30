# 组件测试补全计划（本地文档，勿提交）

> 本文档是「正式进入训练修复之前，把尾巴处理好」这一批工作的唯一跟踪清单。
> 按用户要求：一次只精细化推进一个任务，做完一个勾一个，不因任务多而降低单个任务的
> attention。测试要求统一遵循既有纪律（见下）。

## 全局纪律（每个任务都适用）

1. **端到端优先**：驱动真实链路（真实组件 × 真实引擎/真实数据集跑完整流程），不用桩孤立
   手调单个方法。判据：一次运行能暴露同一接缝上「还不知道会错」的缺口，而不是只验证已
   写好的那个函数。
2. **模型太大用小模型代替**：加载/启动冒烟即可，这批**不跑训练**（model 任务）；需要真实
   forward 的 twinkle 组件测试用最小可跑配置 + 单卡 `@pytest.mark.accel(1)`。
3. **分层标记**：无 GPU 的走默认（fast）；需要加速器的加 `@pytest.mark.accel(N)`；重的加
   `@pytest.mark.slow`。CI 用 `pytest -m "not slow"`。
4. **解释器** `/usr/local/bin/python`；仓库根为 cwd；twinkle 已 editable 安装，无需 PYTHONPATH。
5. **template 本轮放过**（用户明确）。
6. **先核实现有覆盖，只补缺口，不重写、不复制**（剃刀 + 不打补丁）。每个任务开工前先读对应
   现有测试文件，列出「已覆盖 / 缺口」，只针对缺口写。
7. 需要真实下载的多模态数据集：优先用体积极小的公开集；下载类用例标 `slow`，并对离线环境
   给出 skip 条件（拿不到网络就 skip 而非 fail）。

## 现有覆盖盘点（开工前基线，执行时再逐文件确认）

- `swift/dev/tests/component/dataset/`：已有 test_api.py(1081)、test_parity.py、test_store.py、
  test_swift_dataset.py、test_cached_dataset_store.py。覆盖 store(arrow/jsonl/bin)、packing
  (static/streaming)、lazy、group_by_length、data_sharding、多模态 preprocessor(coco/clevr/
  science_qa/grit/voc/captcha/geometry3k)、部分格式转换。**偏组件级，非「一条命令跑通整链」的 e2e。**
- `swift/dev/tests/component/model/`：test_model.py(409) 是对 swift 模型包装类的**契约测试**
  （方法存在性/forward 形状），**不做真实家族加载冒烟** → 这是主要缺口。
- `swift/dev/tests/component/processor/`：test_packing.py、test_vl_mm.py 等已覆盖 collate/多模态 batch。
- `twinkle/tests/dataset/`：已覆盖 csv/json/jsonl/**lance**/parquet 加载、多模态、packing、lazy、
  mixing、ray、save_as。**twinkle 侧数据集已相当完整**（含 lance）。
- `twinkle/tests/loss|metric|model|sampler/`：均已有大量用例（见 grep 清单），需逐文件比对缺口。

## 任务分解（一个一个做）

### 任务 A —— swift/dev/dataset 端到端测试补全
目标：补「数据集能跑通即功能完整可用」的 e2e，覆盖：纯文本、多模态、**需下载文件的多模态**、
各 format_converter、packing、lazy、store 往返。
- A1 现状 gap 分析：通读 test_api.py / test_swift_dataset.py / test_store.py，列出已有 vs 缺口。
- A2 format_converter e2e：alpaca / openai / anthropic / response 四种转换器，raw→messages→encode
      整链跑通（真实 template，load_model=False）。
- A3 纯文本 e2e：load_dataset → 预处理 → encode → SwiftDataset/Packing/Lazy 取 batch，断言
      input_ids/labels/lengths 存在且正确。
- A4 多模态 e2e（本地小样本，不下载）：图像/音频占位符 → encode → pixel/视觉张量随包产出。
- A5 需下载多模态 e2e（标 slow + 离线 skip）：走 mm_download 拉取媒体归档再整链跑通。
- A6 packing / lazy / store 的 e2e 补口（只补 A1 认定的缺口）。
- 验收：新增/补全用例 `pytest -m "not slow"` 全绿；slow 用例在联网单卡跑通。

### 任务 B —— 新特性：数据湖格式支持（swift/dev/dataset）
目标：让 `swift/dev/dataset` 的加载器支持常用数据湖格式，与 twinkle 侧能力对齐。
- B1 调研定型：确认 `datasets` 原生 builder 覆盖哪些（parquet/arrow/csv/json/text/orc/sql/
      imagefolder/audiofolder），lance 在本环境经 `load_dataset('lance', ...)` 是否可用（twinkle
      测试已证明可用）。**决定支持集：parquet、lance、arrow(IPC/feather)、orc**（单文件 + 分片目录）。
- B2 定位缺口：`swift/dev/dataset/loader/base.py::build_dataset` 现在单文件按扩展名透传
      `hf_load_dataset(ext,...)`（parquet/lance 单文件可能已通），但**目录**走 `load_from_hub`
      的 HF folder 自动探测，对 lance 目录不识别 → 需镜像 twinkle 的「目录取首文件扩展名 +
      data_dir」处理。
- B3 实现：通用化扩展名/目录分派；datasets 有 builder 的直接用，没有的才自写；fail-loudly
      对不支持格式给清晰报错。复用优先，不在 dev 重复 twinkle 已有逻辑（评估是否下沉/共享）。
- B4 测试：单文件 + 分片目录 × 各格式，端到端 load→encode 跑通；对齐 twinkle/tests/dataset 的
      lance/parquet 用例风格。
- 验收：新格式 `load_dataset` 能跑通并进 encode；不支持格式早失败。

### 任务 C —— swift/dev/model 重点家族加载冒烟
目标：重点模型「正常加载启动」冒烟，太大用小模型代替，不跑训练。
- C1 选家族：从 MODEL_MAPPING 选代表性家族（qwen2_5、gemma、llama、以及一个多模态如 qwen-vl、
      一个 MoE），每个用其最小 checkpoint 或缩层配置。
- C2 经 `_resolve_model_loader` + `TransformersModel(model_loader=...)` 真实构造 config/processor/model，
      断言加载成功、六 hook 生效（如 gemma 的 eager attn / vision keep-alive）。
- C3 megatron 家族：仅 config 路径冒烟（loader 只用于 build/process_config）。
- 验收：`@pytest.mark.accel(1)` 下加载冒烟全绿；无网络时对需下载的标 slow/skip。

### 任务 D —— twinkle/loss 完整测试补全
- D1 逐 loss 文件比对现有 test（bnpo/ce_mse/channel/dpo/grpo_gkd/liger/opsd/ppo/sampling_replay）
      vs `twinkle/src/twinkle/loss/` 全部实现（含 gkd/infonce/mse/reranker/reward/seq_cls/value/
      chunked_cross_entropy），列缺口。
- D2 补缺口：每个 loss 端到端（真实张量 forward+backward，断言 loss 值/梯度/metric 聚合）。
- 验收：loss 全家族有用例且绿。

### 任务 E —— twinkle/metric 完整测试补全
- E1 比对 test_metrics.py(553) vs `twinkle/src/twinkle/metric/` 全部（accuracy/completion_and_reward/
      dpo/embedding/generation/grpo/loss/ppo/train_metric/buffer/reporting），列缺口。
- E2 补缺口：每 metric accumulate/calculate/reset 端到端 + 边界（空/长度不匹配/mask）。
- 验收：metric 全家族有用例且绿。

### 任务 F —— twinkle/model 完整测试补全
- F1 比对现有 test（megatron_offload/micro_batch/multi_lora*/value_model）vs `twinkle/src/twinkle/model/`
      （transformers/megatron/hybrid/moe/strategy/multi_lora），列缺口。
- F2 补缺口：TransformersModel/MultiLora/Hybrid/MoE-EP/strategy(accelerate/native_fsdp/deepspeed)
      的真实构造 + forward_backward 冒烟（小模型、单卡）。
- 验收：model 各能力端到端用例绿。

### 任务 G —— twinkle/sampler 完整测试补全
- G1 比对现有 test（sampler_e2e/sglang/vllm_*/weight_sync/megatron/torch）vs `twinkle/src/twinkle/sampler/`
      （vllm/sglang/transformers 三引擎 + base_engine），列缺口。
- G2 补缺口：三引擎 sample/sample_to_data_plane/encode、LoRA 热加载、weight sync 的端到端
      （小模型 + 脚本化输入，单卡）。
- 验收：sampler 各引擎能力端到端用例绿。

## 执行顺序
A → B → C → D → E → F → G（先 dev 数据集与新特性，再 model 冒烟，最后 twinkle 组件）。
每完成一个任务，在对应小节勾掉并记录：改了哪些文件、跑通命令、结果、遗留。

## 进度日志
- [x] A swift/dev/dataset e2e —— 全部子项完成
  - [x] A2 format_converter e2e —— 新增 `swift/dev/tests/component/dataset/test_format_converter.py`
        (30 用例全绿)。覆盖：get_converter 解析序(priority/fallback/显式 pin/caller aliases)、
        resolve_aliases 两条竞争规则、四转换器经真实 Preprocessor+HF map 的 detect→convert→
        check_messages→cast_mm_data(alpaca 合并/openai tool_calls/anthropic blocks+images 提升/
        response 别名+history+多答案+rejected 保持扁平)，并把标准 messages 经真实 template
        (load_model=False) encode 到 input_ids/labels。
  - [x] A3 纯文本 e2e —— 新增 `test_text_e2e.py`(7 用例全绿)。真实磁盘文件(jsonl/csv)→
        load_dataset(源探测/按扩展名读/自动格式检测/预处理/split 切分/#N 预算/caller columns)→
        标准 messages → SwiftDataset encode → input_ids/labels/lengths → 真实 DataLoader +
        template.data_collator 得到 padding 对齐的整型 batch(attention_mask 求和==真实 token 数)。
        补的是 test_api(全 mock)/test_swift_dataset(内存行起步) 都没打通的「真文件→整链」接缝。
  - [x] A4 多模态本地 e2e —— 新增 `test_multimodal_e2e.py`(3 用例全绿, 本地生成图, 纯 CPU)。
        真实图片路径 jsonl → load_dataset(cast_mm_data 规范化 images 为 {bytes,path}) →
        qwen3_5 模板 encode → pixel_values/image_grid_thw(patch 数==grid 各维乘积之和)/
        mm_token_type_ids → ragged batch(1图 vs 2图) data_collator 视觉张量 concat(grid 3 行,
        pixel dim0 求和)。collator 的 mrope 路径需 base model, 用 meta-device dummy 挂到 template.model。
        补的是 test_vl_mm(下载 Qwen2.5-VL + 内存样本 + slow/CUDA) 未跑 loader 的本地图片路径接缝。
  - [x] A5 需下载多模态 e2e —— 新增 `test_download_multimodal_e2e.py`(4 offline + 1 slow 全绿)。
        file:// 归档驱动真实 MediaDownloader.fetch/extract + 原子提升(.tmp→rename) + cache fast-path
        + crash 遗留 .tmp 清理重试；LocalArchiveImagePreprocessor 继承生产 ArchiveImagePreprocessor，
        整链跑 fetch→resolve(join media_dir)→缺失行丢弃(len==1)→encode(pixel_values/image_grid_thw)。
        一个 @slow 用真 https 证明传输跳可用，离线 skip。
  - [x] A6 packing/lazy/store e2e 缺口补全 —— 新增 `test_packing_e2e.py`(4 用例全绿)。
        gap 分析：store 层(test_store 28 + test_cached_dataset_store 13，含真 store→dataset e2e)已全覆盖；
        lazy(LazyLLMDataset) 已在 test_swift_dataset 覆盖 parity；processor/test_packing.py 只用合成行
        ({'input_ids':[1,2,3]})驱动 collate 侧、test_api.py 只 mock PackingDataset 验装配——**真实文件→
        PackingDataset 规划→__getitem__ 出组→template.data_collator 拼接** 与 **流式→IterablePackingDataset
        迭代出组** 两条集成链无人驱动，是唯一缺口。本文件补：map-style 规划(索引精确划分/每组
        packed_length==成员长度和/至少一组>1成员)、出组 collate 成单条拼接序列(input_ids (1,ΣT)、
        position_ids 每成员从 0 重启的 multiple-0 形式)、sequential 策略保序(展开==0..N-1)、流式窗口打包
        (worker 内编码，迭代出组后同样 collate)。PACKING_LENGTH=100(行 33~47 token)保证成组。
        → 任务 A 全部完成。
- [x] B 数据湖格式特性 —— 完成实现 + 测试
  - 调研定型：`datasets` 4.8.4 的 PACKAGED builders 含 arrow/csv/json/parquet/**lance**/text 等，
        **但无 orc builder**（'orc' 被当 hub id → ConnectionError）；`.lance` 已在 `_EXTENSION_TO_MODULE`
        映射到 lance builder，但 folder 扫描不会为 `.lance` 目录选中它，且 builder 需 `pylance`。
        实证 gap = **orc（datasets 完全无 builder）** + **lance 目录（folder 不探测）**；parquet/arrow
        单文件 + 目录经现有 dispatch 已工作（不动，剃刀）。
  - 实现 `swift/dev/dataset/loader/base.py`：build_dataset 收敛为统一 `if info.source == 'path':`
        分支，先经 `resolve_lake_format` 拦截 orc/lance。新增三个 staticmethod：
        `resolve_lake_format`（按扩展名/目录内文件/`.lance` 目录名判定，只认 orc/lance）、
        `load_orc`（自写 pyarrow.orc 读取，单文件或目录分片排序拼接，streaming 走
        `to_iterable_dataset()`）、`load_lance`（复用 datasets builder：文件→data_files，目录→
        data_dir；缺 pylance 时 fail-loudly 给 `pip install pylance` 提示）。
  - 测试 `test_datalake_formats_e2e.py`（**9 passed, 1 skipped**）：resolve 检测边界、orc 单文件/分片
        目录/streaming/空目录 fail-loudly、parquet+arrow 回归、orc→SwiftDataset encode 到 input_ids、
        lance 缺 pylance 的 ImportError 路径。`test_lance_dataset_roundtrip`（真 lance.write_dataset→
        load→messages）用 `importorskip('lance')` 守卫。
  - **遗留（需向用户明确）**：本环境**离线且未装 pylance，无法 pip install**，故 lance 真实往返用例
        在本环境 skip、未实跑；lance 路径按 twinkle 已验证的 builder 契约与 canonical
        `lance.write_dataset` API 编写，fail-loudly 分支已实测。
  - 回归门禁：test_text_e2e + test_api → 80 passed, 1 failed。该 failed
        (`test_single_backend_table_defaults_match_the_config_declarations`, KeyError 'model_config')
        经核实为 **pre-existing config-validation drift**（validate.py 的 holders 表含 model_config，
        test_api 内 holders dict 未列），与本次 loader 改动无关（git status 确认只 M 了 loader/base.py）。
- [x] C swift/dev/model 加载冒烟 —— 完成
  - gap：现有 test_model.py(409) 是契约/继承测试（方法存在性/data format identity），自己声明
        “Constructing a real twinkle Model needs a downloadable model + process group, so those paths
        are covered by skip-guarded integration tests”——**真实家族加载冒烟无人做**，是唯一缺口。
  - 关键发现（probe 实证）：`TransformersModel(model_id=..., model_loader=..., mixed_precision='no')`
        可**无需分布式/GPU/网络**构造（`_try_init_process_group` 在 world size 1 是 no-op；
        `HubOperation.download_model` 对本地路径原样返回），六 hook 精确对应：build_config/
        process_config(343)、build_processor/process_tokenizer(359)、build_model/process_model(363-364)。
        故用**缩层配置在 tmp 现场构造 tiny checkpoint**（2 层/hidden 16，几千参）代替大模型，
        比计划预期的 `@accel(1)` 更轻：**全部进 fast 道（CPU、离线、~12s）**，无需 GPU。
  - 新增 `swift/dev/tests/component/model/test_model_loading_e2e.py`（**8 用例全绿**），每家族钉一个
        hook 的可观测效果（非 mock）：
        · 六 hook 顺序契约（Recording 子类包真 loader，断言调用序 == build_config→process_config→
          build_processor→process_tokenizer→build_model→process_model）；
        · qwen2_5 加载 + 真实 forward（logits (1,4,64) 全 finite）；
        · llama `process_config` 把落盘的 pretraining_tp=2 强制为 1；
        · gemma3_text `build_model` 默认 attn=eager（_EagerAttnDefault）；
        · qwen3_moe 构出 expert 层（num_experts=4，每解码层一个 experts 块）；
        · qwen2_5_vl `build_processor` 走 AutoProcessor→Qwen2_5_VLProcessor，`process_model` 装
          vision keep-alive（_vision_keep_alive），model_arch vision_tower/aligner 正确（importorskip
          qwen_vl_utils，本环境已装故实跑）；
        · resolve 逻辑（显式 model_type 权威 / 按 basename 推断 / 未知→None→通用 AutoModel 路径）；
        · **C3 megatron config 路径**：`_build_megatron_model` 复用同一 loader 的 `_build_hf_config`
          (=build_config+process_config+rope/max_len)，用 llama tiny ckpt 断言 process_config 在
          megatron 路径也生效(pretraining_tp→1)、max_model_len=4096、rope factor=64。不构造
          MegatronModel（需 mcore+Ray+device mesh，超出加载冒烟范围）。
  - 回归门禁：`pytest swift/dev/tests/component/model/`（GPU 可见）→ **47 passed**（39 既有 + 8 新）。
        注：若用 `CUDA_VISIBLE_DEVICES=""` 跑，test_model.py 的 3 个 TestInputProcessor 用例会因
        `No CUDA GPUs are available` fail——那是既有 GPU-依赖用例 + gate 命令配置问题，非本次引入。
- [x] D twinkle/loss —— 完成（缺口补全 + 顺带修一个 loss 家族产品 bug）
  - D1 gap 分析：现有 test（bnpo_token_mean/ce_mse/channel/dpo/grpo_gkd/liger_fused_linear_ce/
        opsd/ppo/sampling_replay）已覆盖 CE/ChunkedCE/MSE/Channel/DPO 家族(CPO/ORPO/SimPO)/GRPO 家族
        (GRPO/PPO/GSPO/SAPO/CISPO/BNPO/DRGRPO)/GKD/OPSD/Liger-CE/PPOValue。**真缺口 = 无任何测试
        引用的 5 个模块共 10 个类**：infonce.py(EmbeddingLoss/InfonceLoss/CosineSimilarityLoss/
        ContrastiveLoss/OnlineContrastiveLoss)、reranker.py(Pointwise/Listwise)、seq_cls.py(SeqClsLoss)、
        reward.py(RewardLoss)、liger_fused_linear_grpo.py(LigerFusedLinearGRPOLoss)。
  - D2 新增 4 个测试文件（全 CPU、真实张量 forward+backward、对已知值/独立参考做数值断言，无桩）：
        · `test_infonce.py`（**35 passed**）：MRL 加权前缀和(禁用时单次不变/全 dim 超 hidden→ValueError/
          supports_mrl=False 拒绝 mrl_dims)、InfonceLoss(intra vs in-batch 目标不同、ragged 走 unbatched、
          hard_negatives 截断+上采样、include_qq/dd、mask_fake_negative、fake_neg_margin<=0→ValueError、
          MRL、无组→零损失保图、logits 回退+3D CLS 池化)、CosineSimilarityLoss(相同对 label1→0/正交对→1/
          逐对标签)、ContrastiveLoss(相似相同→0/不相似超 margin→0/margin 内被罚/三种命名度量+callable/
          未知→ValueError)、OnlineContrastiveLoss(重叠距离触发 hard 对→非零梯度、easy 对全丢弃→零、
          单侧空→零保图)。
        · `test_reranker.py`（**18 passed**）：Pointwise(零分单位标签→ln2、对齐手写 BCEWithLogits、[B,1]
          squeeze、置信→0、梯度)、Listwise(完美排序→0、对齐手写 CE、双等组均值、temperature 锐化、
          min_group_size 跳过 positive-only 组、全组过小/无 positive→零保图、temperature<=0→ValueError)。
        · `test_seq_cls_reward.py`（**18 passed**）：SeqCls(problem_type 必填校验、regression num_labels=1
          squeeze 已知值=1.0、regression 多输出对齐 MSE、single_label 对齐 CE+置信近零、multi_label 对齐
          BCE、三 problem_type 各自梯度)、Reward(tied→ln2、对齐手写 -logsigmoid、chosen 压制→0、奇数 batch
          →AssertionError、center_rewards_coefficient 正则项可分离、梯度)。
        · `test_liger_fused_linear_grpo.py`（**12 passed**）：**关键事实修正——Liger fused GRPO kernel 在
          CPU 上可运行且与 materialised GRPO 数值精确一致(diff=0.0)**，故 fused 路径正确性无需 GPU 即可
          常驻验证。覆盖：fused 运行(_fused_broken=False)+对齐独立 F.linear 参考、fused 反传到 hidden+head、
          beta>0+ref_logps 的 fused KL 路径；无 lm_head→直接基类 GRPO；advantages=None→零；
          **defensive fallback 用 monkeypatch 令 `_get_liger_module` 抛 ImportError 诚实触发**
          (_fused_broken/_warned=True、materialise 后对齐基类、二次调用跳过 fused)。
  - 顺带修复：`twinkle/src/twinkle/loss/liger_fused_linear_cross_entropy.py:184` 的 `except Exception:`
        未绑定 `e`，但 192 行 `logger.warning(..., e)` 引用 `e` → **defensive fallback 路径 NameError 崩溃**
        （一旦 liger CE fused kernel 在任何后端抛异常，本应优雅回退却崩）。按根因改为 `except Exception as e:`
        （与 GRPO 版一致），既有 GPU 门控用例 `test_fused_kernel_failure_falls_back_to_materialised_ce`
        即为其证明，修复后 GPU 下 11 passed。
  - 回归门禁：`pytest twinkle/tests/loss/`（CUDA_VISIBLE_DEVICES=1）→ **191 passed, 1 failed**。唯一
        failed = `test_channel_loss.py::test_loss_metric_aggregates_channels_and_resets`，断言
        `result['loss']=='2.0000'`(4 位) 但 `metric/loss.py:89` 源码用 `.5f`→'2.00000'（通道 loss 用 `.4f`
        且匹配）。这是**既有 metric 模块的源码/测试精度漂移，非本次 loss 补全引入**（我只新增 4 个 loss
        测试文件 + 改 1 行 liger_ce）。主 loss 用 5 位、通道用 4 位看似有意区分 → 疑为该既有测试断言过时；
        `LossMetric` 属**任务 E（twinkle/metric）范畴**，留待任务 E 完整过 metric 时判定 .5f 是否有意并处置。
- [x] E twinkle/metric —— 完成（缺口补全 + 处置任务 D 遗留的精度漂移）
  - E1 缺口盘点：`test_metrics.py`(554) 已覆盖 Accuracy/LossMetric(基础)/TrainMetric/
        CompletionRewardMetric/DPOMetric/GRPO/GSPO/CISPO/EmbeddingMetric/ExactMatch/RougeBleu；
        `reporting.py`/`types.py`(MetricRecord)/`buffer.py`(MetricBuffer)/_QueuedBackend 已由
        `twinkle_agentic/test_async_rl_metrics.py` 充分覆盖，`TextMetric` 基类经 ExactMatch/RougeBleu
        子类覆盖 —— 均不重复。**真缺口 = `PPOValueMetric`（ppo.py，全仓无测试引用，逻辑实质）
        + `PPOMetric`（grpo.py，GRPOMetric 纯改名子类，全仓无测试引用）**。
  - E2 补缺口：新增 `twinkle/tests/metric/test_ppo_metric.py`（20 用例，CPU 常驻，真实张量
        accumulate→calculate→数值断言，对齐 `_no_dist_metric` 风格）：
        * PPOValueMetric：values==returns→explained_variance=1.0；手算 ev=0.0 的已知值；
          records 只保留 mask 位（prompt 位的 9.0/7.0 被丢弃，钉住 ignore_index 对齐）；
          value_clip_ratio 全超/全不超 epsilon 两端；clipped_values=old+clamp(diff) 手算；
          advantage_mean/std（[2,4]→3.0/1.0）；多 micro-batch cursor 递进（values 传 list、
          old/returns 传单张量按 cursor 切片）；无有效 token 跳过；5 类缺操作数(outputs/values/
          old/returns 为 None)早返回→records 空→calculate()={}；空 metric→{}；calculate/reset 清空。
        * PPOMetric：is subclass of GRPOMetric；基础 policy_confidence；**同输入下 calculate() 与
          GRPOMetric 完全一致**（钉住「纯改名子类」契约）；old_logps→approx_kl/logp_diff_mean；reset。
  - E2 遗留处置（channel_loss 精度漂移，任务 D 转来）：git blame 时间线证实 —— 测试断言
        `'2.0000'` 由 `99638ec`(16:02) 针对当时源码 `.4f` 写入；`d1c1f28`("Async rl #273", 16:38,
        晚 36 分钟)**有意**把主 loss 从 `.4f` 提到 `.5f`（头部 loss 更高分辨率），但漏更新这条既有断言。
        故 `.5f` 是刻意设计而非回归 → 属**测试断言过时**，修 `test_channel_loss.py:110` 断言为 `'2.00000'`
        并加注释说明「主 loss 5 位 / 通道 loss 4 位为有意区分」，不动源码。
  - 回归门禁：`pytest twinkle/tests/loss/test_channel_loss.py twinkle/tests/metric/`（CPU）→ **86 passed**。
        （channel_loss 8 + test_metrics 58 + test_ppo_metric 20）
- [x] F twinkle/model —— 完成（CPU 可测纯逻辑缺口补全；重型模型类归任务 C 冒烟范畴）
  - F1 缺口盘点：twinkle/model 共 11280 行。已覆盖：micro_batch/multi_lora_target_parameters/
        value_model（tests/model）、native_fsdp/sequence_parallel/hybrid（tests/transformers）、
        moe ep（tests/moe）、megatron offload（tests/model）。重型类（transformers.py 2121 /
        megatron.py 2120 / strategies / multi_lora 重方法）需真模型+GPU+分布式，非 CPU 端到端可行，
        其家族加载冒烟已由任务 C（swift/dev/model）覆盖。**真缺口 = CPU 可测且零覆盖的纯逻辑**：
        `optimizer_group.py::BaseOptimizerGroup`（do_grad_sync 累积门控 / accumulate_metrics /
        calculate_metrics / __setattr__ 的 safe_loss 包装接缝 / TrainStatus）与
        `base.py::rotate_checkpoints`+`copy_checkpoint_args`+`_should_bind_device_id_for_process_group`。
  - F2 补缺口：新增 `twinkle/tests/model/test_optimizer_group.py`（19 用例，CPU，驱动真实
        LossMetric/CrossEntropyLoss）：TrainStatus 默认值与 per-instance 可变默认；do_grad_sync
        全边界（gas=1 恒同步 / 首步不同步 / 窗口中段不同步 / 边界步同步 / 显式入参回写状态 /
        None 用状态不改）；safe_loss 包装（包成 SafeLossWrapper 且 isinstance Loss / 幂等不重复包 /
        None 不包 / fail-fast 下透明委托数值不变）；accumulate_metrics（train/eval 双状态 / 缺 io 跳过 /
        空 metrics noop）、calculate_metrics（返回结果并清空 io / grad_norm 透传，'0.750000' 字符串）。
  - F3 补缺口：新增 `twinkle/tests/model/test_model_base.py`（15 用例，CPU + tmp 目录，
        单进程下 Platform.is_master()=True）：rotate_checkpoints（limit=None 全保留 / <1 → ValueError /
        output_dir 缺失 noop / 按 mtime 保留最新 N / **current 即使最旧也被保护**（is_current 排序键）/
        checkpoint-final 命中且非 checkpoint 目录与同名文件不误删）；copy_checkpoint_args（正常拷贝 /
        源缺失 noop / 源=目标 realpath 去重跳过自拷）；_should_bind_device_id（nccl/hccl→True，gloo/cpu/''→False）。
  - F4 回归门禁：`PYTHONPATH=peft/src CUDA_VISIBLE_DEVICES="" pytest twinkle/tests/model/` → **62 passed,
        15 skipped**（含本次新增 34）。**环境陷阱**：裸跑（不挂 fork）会有 1 个既有失败
        `test_multi_lora_target_parameters.py::test_peft_target_parameter_key_shapes_for_3d_experts` —— 因默认
        `import peft` 命中 site-packages 0.19.0，遮蔽了项目认定的工作区 fork `peft/`(0.20.1.dev0)，二者对
        3D experts 的 target_parameters LoRA 的 lora_A/lora_B 布局互为转置。用 fork 跑即通过 → 属**环境版本遮蔽，
        非产品 bug 亦非过时断言，未改测试**（同文件走 twinkle TargetParameterLoraManager 的其余用例两版 peft 均绿，
        佐证产品功能正常）。已记入 common_pitfalls 记忆。
- [x] G twinkle/sampler —— 完成（CPU 可测纯逻辑接缝补全 + 处置回归红灯的产品缺口）
  - G1 缺口盘点：twinkle/sampler 共 7424 行，vllm/sglang/transformers 三引擎均重 GPU（真实生成、
        权重同步、多卡），既有 `tests/sampler/`(2353 行) 已以 e2e + GPU 门控覆盖（accel/skipif CUDA）。
        引擎生成非 CPU 端到端可行，属既有 e2e 范畴。**真缺口 = CPU 可测且零覆盖的 `sampler/base.py::Sampler`
        纯逻辑接缝**：`_not_encoded`/`_is_trajectory`/`_normalize_inputs`（输入分类）、`encode_trajectory`
        （Template→InputFeature，过滤 labels、input_ids→tolist）、`decode_response`、默认 `encode`（拒绝 pooling）。
  - G2 补缺口：新增 `twinkle/tests/sampler/test_sampler_base.py`（19 用例，CPU，`_ConcreteSampler` 实现两抽象方法
        + `_FakeTemplate` 镜像 Template.encode/decode 契约，延续本目录「pin the seam, not generation」哲学）：
        输入分类全分支（Trajectory/encoded/列表取首/空列表 False/非 dict AssertionError）；encode_trajectory
        （无 template→ValueError 'Template not set'、构建 InputFeature 且 input_ids=[1,2,3] list、labels 被过滤、
        attention_mask/images 保留、add_generation_prompt 转发、encode 无 input_ids→ValueError）；decode_response
        （无 template→ValueError、委托 template.decode）；默认 encode→NotImplementedError 'does not support pooling'。
  - G3 处置回归红灯（既有失败，非本次引入）：`test_sampler_over_existing_model.py::test_megatron_refuses_to_generate_in_place`
        期望 `MegatronModel.generate(['prompt'])` 抛 `NotImplementedError('sharded across TP/PP')`，实际抛
        `AttributeError`（MegatronModel 继承 `TrainableModel, nn.Module, CheckpointEngineMixin`，**不含 PreTrainedModel**，
        故根本无 generate；TransformersModel 的 generate 来自 HF PreTrainedModel）。git 考证：`sharded across TP/PP` 全史
        仅见于 `dd719d5 wip`（即添加该测试的同一提交），`def generate` 在 megatron.py 全史从未出现 —— 该提交同时落地了
        facade（transformers_sampler 转发 model.generate/generate_stream）与测试，却漏实现 MegatronModel 的拒绝。判定为
        **spec 先于实现的产品缺口**（非过时断言）：契约合理且与整个测试文件哲学一致（会撒谎的方法应显式拒绝而非静默近似）。
        按纪律在根因接缝补实现，**不削弱断言**：给 MegatronModel 加 `generate`/`generate_stream` 两方法直接 raise
        NotImplementedError（消息含 'sharded across TP/PP' + 指向改用 vLLM/SGLang/Transformers 专用 sampler）。
        实现要点：`generate_stream` 是普通方法而非 generator（否则调用不抛、迭代才抛，`pytest.raises` 直接包裹调用会漏）；
        不加 `@remote_function`（测试用 `object.__new__` 绕过 __init__ 直调，远程派发会走样）。纯追加、CPU 安全。
  - G4 回归门禁：`CUDA_VISIBLE_DEVICES="" pytest twinkle/tests/sampler/` → **42 passed, 23 skipped**（红灯转绿）；
        因改动 megatron.py（model 层），复跑 `PYTHONPATH=peft/src CUDA_VISIBLE_DEVICES="" pytest twinkle/tests/model/`
        → **62 passed, 15 skipped**，确认无回归。
