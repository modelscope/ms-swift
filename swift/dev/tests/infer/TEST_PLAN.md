# swift infer 测试计划

本目录覆盖 dev 侧统一推理入口 `swift infer`（`swift/dev/cli/infer.py` → `swift/dev/recipe/run_infer.py`
/ `infer_tui.py`）。测试分两层：

- **fast 层**（默认套件运行，无 GPU、不载权重）：纯函数、行整形（emit）、写盘/续跑（writer）、
  候选缓存（cache）、TUI 状态机、CLI 解析、以及两类端到端接线——
  (a) 用 `infer_backend='no'` + `cache_files` 回放候选驱动 **采样→打分→整形→落盘** 全链（不建 sampler）；
  (b) 用**真实模板**（`load_model=False`，只加载已缓存 tokenizer）+ **脚本化 sampler**（返回预置 token，
  不碰模型权重）驱动**真实 twinkle 引擎**跑完整多轮工具循环 / generative-judge 打分（`test_tools_multiturn_e2e.py`、
  `test_reward_channels.py`、`test_template_contract.py`）。(b) 类证明「完整引擎级端到端不需要 GPU」：
  真组件 × 真引擎，只是把权重换成脚本化替身。
- **slow 层**（`@pytest.mark.slow`，需 CUDA，用 `TinyModel` 本地建 4 层随机权重小模型或
  `Qwen/Qwen2.5-0.5B-Instruct`）：真实后端生成、各任务类型 forward 路径、多模态 VL 生成、
  TUI 端到端。CI 用 `-m slow` 选择。（megatron 不是 infer 后端，见维度 4。）

设计原则：fast 层断言**行为契约**（形状、排序、阈值、续跑语义、参数映射），slow 层断言**链路接通**
（真实引擎能跑通并写出可加载结果），数值正确性交给既有 `feature/sft/test_alignment.py` 一类测试。

---

## 维度对照（用户 12 项 → 测试文件）

### 1. 若干小模型测试
- `test_backends_e2e.py`（slow）：`TinyModel.build()` 本地建 4 层随机 Qwen2，`transformers` 后端真实
  生成，断言写出 jsonl、每行有 `response`/`responses`/`messages`，行数 == prompt 数。
- fast 层用合成 `_Candidate` 与 `cache_files`，不依赖任何模型。

### 2. 多模态
- `test_multimodal_e2e.py`（slow）：tiny Qwen2.5-VL 检查点（`tiny_loader.build_tiny_multimodal`，只缩层数、
  同时保存 model 与 processor）经 `run_infer` 在 **transformers 后端**真实生成；断言单图与双图同批都产出
  `response`、图片引用随行保留、写盘可回读。生成成功本身即最强断言：VL 模型拿到 `<image>` 占位 token 却
  没有对应 pixel_values 会因 token/patch 数不匹配报错，故干净补全证明图像张量确实进了 `generate`。
  - 支撑改动（twinkle 属本次重构范围，多模态 generate 要支持）：`build_sampler` 对 transformers 后端把
    family loader 声明的 `model_cls`（`transformers:Qwen2_5_VLForConditionalGeneration`）传进引擎；
    `TransformersEngine` 用它 `from_pretrained`（默认仍 `AutoModelForCausalLM`，文本模型天然 no-op）；
    `TransformersSampler._run_chunk` 把 template 编码出的 `pixel_values`/`image_grid_thw` 经引擎既有的
    `extra_model_inputs` 通道按 dim 0 拼接后传给 `batch_sample`。未新增架构，全部复用现有入口。
- `test_tui_state.py`（fast）：`_CliState.prompt_media` 用 monkeypatch `input` 收集媒体路径，
  `to_trajectory` 把非空媒体挂到轨迹上。

### 3. 各任务类型（causal_lm / seq_cls / embedding / reranker / generative_reranker）
- `test_pure.py`（fast）：`is_pooling_task` / `pooling_task_for` 映射（embedding→embed、
  seq_cls/reranker→classify、causal_lm/generative_reranker→None）；`_sigmoid` 数值稳定性。
- `test_task_types_e2e.py`（slow）：pooling 三类 + generative_reranker 走 forward 路径
  （`TrainAssembly.initialize_twinkle` 'model' 组），断言每行一个值；causal_lm 走生成路径。

### 4. megatron 与 transformers
- `test_backends_e2e.py`（slow）：`transformers` 后端生成（单卡即可，all/dpo/callable-reward 三形状）
  + `vllm` 后端（`importorskip` 门控，小显存引擎参数）。
- **megatron 不是 infer 后端**（经用户决策：仅文档说明，不加测试）：`InferConfig.infer_backend` 的
  Literal 只含 `vllm/transformers/sglang/pt/client/no`，不含 megatron；`run_infer`/`build_sampler`
  也无 megatron 分支。megatron 在 dev 侧只作训练后端（`DistributedConfig.backend='megatron'` →
  `build_model` 建 `MegatronModel`）。推理没有 megatron 生成路径，故无对应 e2e 测试。

### 5. TUI 与写文件
- `test_tui_state.py`（fast）：`_CliState` 全状态机——`clear`/`add_query`/`add_response`/`_prune`
  （在 user 边界切，不孤立 assistant 答复）/`to_trajectory`（带 system+media）/`read_query`
  （quit/clear/reset-system/multi-line/single-line 命令，monkeypatch `input`）。
- `test_tui_e2e.py`（slow）：`infer_cli` 用 scripted sampler + monkeypatch `input` 跑一轮对话。
- `test_writers.py`（fast）：`_IncrementalWriter`（追加语义、batch_size 为假时 finish 一次性写、
  非写 rank 返回 None）、`_CheckpointWriter` + `_CheckpointPaths`（tmp/resume/state/final 四文件、
  prepare/checkpoint/finalize 顺序、续跑跳过已完成 batch）。

### 6. 使用 rm（reward model）
- `test_reward_channels.py`（fast，覆盖 example `math_prm_orm_ray.sh` 的 model-reward 通道）：
  - **四 kind 解析矩阵**（monkeypatch `swift.model.get_model_info_meta` 控制 task_type，不下载模型）：
    URL→`api`、seq_cls/reranker→`scalar`、causal_lm/generative_reranker 且本地且等于 sampler 模型→
    `generative_reuse`、否则→`generative_independent`、仅 adapter→复用 sampler、client 后端从不复用、
    无模型无 adapter→无通道、未知 task_type→报错。
  - **judge 纯函数**：`_render_conversation`（Role: content 行）、`_judge_query`（剥离候选轮 + 填模板）、
    `_parse_judge_score`（`Reward: <score>` 优先、回退裸数、无则 None→记 nan 不猜 0）。
  - **ray 接线**：`plan_sampling_device_groups`（每角色独立不相交 GPU 块）、`_exclusive_reward_groups`
    （只点名 GPU 常驻通道）、scalar+非 ray 经 `run_infer` 驱动到守卫报错（`mode='ray'`）。
  - **generative-judge 全链路 fast e2e**：`_ScriptedJudgeSampler` 双模式（按 `=== Candidate response ===`
    标记区分「生成候选」与「被复用为 judge 打分」），`orm_model==sampler 模型`→`generative_reuse`，
    经 `run_infer` 断言 `responses` 与 `scores==pytest.approx([...])`（float32 舍入，用 approx）。
- `test_pipeline_fast.py`（fast）：`rlhf_config.reward_funcs=[callable ORM]` 经 `_build_channels`
  → `_RewardChannels.score` 给每条候选打分；断言 'all' 行带 `scores`、'dpo' 行按分排序选正负。
- `test_pure.py`（fast）：`plan_sampling_device_groups` / `_exclusive_reward_groups` 的纯函数性质。
- `test_backends_e2e.py`（slow）：真实 seq_cls RM 通道（如环境允许）。

### 7. 产出 dpo / grpo / sft 数据形状
- `test_emit.py`（fast）：
  - **sft/all 形状**：`_emit_all` 一行一 prompt，`response`=首条、`responses`=全部、`labels`=参考、
    多轮带 `all_messages`、有 logprobs 时带 `logprobs`。
  - **dpo 形状**：`_emit_dpo` best-of-n——无分全为正例、有分按 `reward_threshold` 过滤、
    `easy_query_threshold` 整组跳过、`n_best_to_keep` 条正例配最差一条负例、nan 候选整体排除、
    多轮保留完整 chosen/rejected 轨迹、`id` 组内一致。
  - **grpo 形状**： scored group（每候选一分）是 GRPO 组相对优势的输入；断言 'all'+score 的
    `responses`/`scores` 一一对应、组内可还原。
- `test_tools_multiturn_e2e.py`（fast，覆盖 example `dataset_tools_grpo.sh` 的核心）：**真 grpo 落盘 e2e**——
  `output_format='grpo'` + `save_rollout_tokens=True` 经 `run_infer` 多轮工具 rollout，断言 jsonl 行键集
  （id/messages/completions/all_messages/num_generations/scores/rollout_tokens）、`scores is None`（无 reward
  通道时）、`len(rollout_tokens)==num_generations`、jsonl round-trip；以及 **NPZ sidecar 6 字段契约**
  （input_ids/labels int64 全长、response_token_ids int64 可训练、rollout_logprobs float32 可训练、
  response_loss_mask int8 可训练、completion_mask int8 全长），文件名 `{prompt_id}_c{index}.npz`。
- `test_pipeline_fast.py`（fast）：端到端 'dpo' 落盘走 `_CheckpointWriter`，'all' 走 `_IncrementalWriter`。

### 8. local 沙箱与 tools mock
- `test_tools_multiturn_e2e.py`（fast，覆盖 example `tui_tools_multiturn.sh` + `dataset_tools_grpo.sh`）：
  **真多轮工具端到端**，两条接线各驱动一次真实 twinkle 引擎——
  - `run_infer(dataset)` 路径：`external_plugins` 加载 `examples/v5/infer/custom_tools.py` 的 calculator
    （断言注册进 `TOOL.entries`），多轮轨迹 roles==[user,assistant(tool_call),tool,assistant]、
    tool_calls 结构、**工具观察消息含 name 字段**（`{'role':'tool','content':'14','name':'calculator'}`，
    calculator 真执行返回 int `'14'`）、completions/num_generations。
  - `infer_cli(交互)` 路径：`tools=['sandbox']` + monkeypatch `input`，脚本化 sampler 发 `run_command`
    工具调用，断言打印 `called tools: run_command` 且 **proof 文件被真 LocalEnv 的 shell subprocess 写出**
    （`echo executed > proof` → 回读 == 'executed'），证明不是桩。
  - **`_trainable_positions` off-by-one 回归**（`test_response_token_ids_align_with_sampled_tokens`）：
    从 NPZ 取 response_token_ids/rollout_logprobs，断言逐位对齐且解码含答案文本。此 bug 曾因 dev 翻译层
    把 labels 反 unroll 到 input order 却用 output-order 的 completion_mask 相交（差一个循环移位），
    令多轮 tools+save_rollout_tokens 崩在 `sampled token ... missing from its returned logprobs`；
    `test_template_contract.py` 直驱 twinkle 引擎不过 dev 翻译层，故漏掉它——本用例钉住。
- `test_sandbox.py`（fast）：`build_tool_sandbox(None)` / 空 `tools` → `(None, [])`（tools 默认关）；
  `resolve_tool_plugins(['sandbox'], config)` 构造 `SandboxTools`；`SandboxTools.build(env)` 经
  mock env 返回 EnvTool 列表（monkeypatch `twinkle_agentic.envs.EnvTool`）。
- `test_pipeline_fast.py`（fast）：多轮 rollout——`RolloutEngine` 注入 scripted sampler +
  `configure_multi_turn`（mock tool_manager），断言多轮候选带 `messages`/`rollout_infos`/`truncated`，
  `_emit_all` 存 `all_messages`、`_emit_dpo` 存 `rejected_messages`。
- `test_template_contract.py`（fast）：真实 `Shifted*Template`（`DevMixin`）× 真实 twinkle `MultiTurnRollout`
  × 真 `ToolManager` × 脚本 sampler 的**引擎级端到端**（真模板 `load_model=False`、无 GPU），钉住
  `concat_input_feature`/`parse_tool_call`/`observe` 路径的模板契约（曾藏四个生产 bug：漏追加 assistant
  消息、agent 工具面方法缺失、observe 路径缺 `_invoke_post_pipeline`、单轮捷径致多轮 completion_mask/labels
  错位）；断言工具 schema 烤进 opening prompt、单轮全字段等长、三轮双 observe 对齐、畸形标记经
  `tool_call_errors` 反馈重试、`appended_as='context'` 掩码。
- `test_rollout_logprobs.py`（fast）：单轮 old_logps 对齐守卫——`force_logprobs=False`（plain infer）时
  `require_logprobs=False`，logprobs=None 产出空 old_logps 不报错；GRPO（默认 `require_logprobs=True`）
  对缺失/部分 logprobs 仍致命报错（`logprobs misaligned`）。

### 9. 命令行使用
- `test_cli_parse.py`（fast）：`parse_infer_configs(argv)` 无模型解析——`test_parse_returns_every_config_key`
  断言全部 config 键齐全；`infer_main` 用 monkeypatch 把 `run_infer`/`infer_cli` 换成桩，验证 argv→recipe
  入参映射（interactive vs dataset 分派、resolve_model 等价式、result_path 派生、backend 统一）。
- `test_cli_parse.py`（fast）：**四个 example 脚本的 flag 逐条落到正确 Config 字段**——
  `test_tui_multigpu_flags` / `test_tui_tools_multiturn_flags` / `test_dataset_tools_grpo_flags` /
  `test_math_prm_orm_ray_flags` 分别拿真 example 的 argv 断言：`vllm_` 前缀剥离（`_build_engine_args`）、
  字段归属（`temperature` 在 GenerationConfig 不在 RLHFConfig）、`tools`+`max_turns`、`mode`/`nproc_per_node`。
- `test_cli_parse.py`（fast）：**validate 校验**——`tools` 需 `max_turns`、`sandbox_num_envs>0`、
  `save_rollout_tokens` 拒 client 后端、多轮边界、grpo 需 ≥2 候选且需 save_rollout_tokens、
  dpo `n_best_to_keep<num_return`、save_rollout_tokens 需 output_path。
- `test_cli_parse.py`（fast）：`_guard_interactive`（pooling/reranker 交互拒绝、dp>1 拒绝、
  ray dp=1 放行）、`_interactive_dp_width`、`_derive_result_path`。

### 10. 回归 legacy 能力对比 + 命令参数测试
- `test_cli_parse.py`（fast）：legacy 拼写折叠——`num_samples`↔`num_return_sequences` 双向同步、
  `pt`→`transformers`、`prm_threshold`→`reward_threshold`（仅当后者未显式传）、
  `num_sampling_batch_size`→`batch_size`、`num_sampling_batches`→`max_batches`、
  `padding_side` 默认 left、`stream` 由有无 dataset 派生、`output_format='all'` 下设
  best-of-n 旋钮告警。
- `test_pure.py`（fast）：`compute_metric` 'acc' 用精确字符串相等（== legacy `--metric acc`，
  非 token 级 Accuracy）、'rouge' 走 RougeBleu、只评第一条、无参考行跳过。
- `test_writers.py`（fast）：`_CandidateCache` == legacy `cache_files`（按 prompt messages keyed、
  候选数不足不命中、多轮按 id keyed、截断行跳过）。

### 11. 其他重要功能覆盖
- `test_pure.py`（fast）：`_normalize`（min-max 到 [0,1]、退化组塌缩为常数、nan 透传不污染）、
  `_is_too_easy`、`_plan_batches`（固定区间、保留尾部不满批、batch_size<1 报错、max_batches 截断）、
  `_prompt_key`（md5 稳定、与顺序无关）、`_sigmoid`（正负分支数值稳定）、`_tool_names`。
- `test_pipeline_fast.py`（fast）：`infer_backend='no'` 缺 cache 覆盖时报错；已完成 checkpoint +
  `override_exist_file=False` 短路返回 []；空数据集报错；dpo 校验（num_return<2、
  n_best_to_keep>=num_return 报错）；save_rollout_tokens 无 output_path 报错；metric+dpo 跳过告警。

### 12. 目录与运行
- 全部测试落在 `swift/dev/tests/infer/`，`__init__.py` + `conftest.py` 提供共享 fixture
  （合成候选、cache 文件构造器、scripted sampler、monkeypatch 后的 `load_prompt_rows`）。
- 建好后运行：fast 层 `pytest swift/dev/tests/infer -m "not slow"` 必须全绿；
  slow 层 `pytest swift/dev/tests/infer -m slow`（需 CUDA）。

---

## 文件清单

| 文件 | 层 | 用例数 | 覆盖维度 |
|---|---|---|---|
| `conftest.py` | - | - | 共享 fixture |
| `test_pure.py` | fast | 11 | 3,6,10,11 |
| `test_emit.py` | fast | 12 | 7,8 |
| `test_writers.py` | fast | 14 | 5,10,11 |
| `test_tui_state.py` | fast | 14 | 2,5 |
| `test_cli_parse.py` | fast | 33 | 9,10 |
| `test_sandbox.py` | fast | 11 | 8 |
| `test_pipeline_fast.py` | fast | 9 | 1,6,7,8,11 |
| `test_template_contract.py` | fast | 5 | 8 |
| `test_rollout_logprobs.py` | fast | 3 | 7,8 |
| `test_tools_multiturn_e2e.py` | fast | 6 | 7,8 |
| `test_reward_channels.py` | fast | 15（参数化后 22） | 6 |
| `test_backends_e2e.py` | slow | 4 | 1,4,6 |
| `test_task_types_e2e.py` | slow | 3 | 3 |
| `test_multimodal_e2e.py` | slow | 2 | 2 |
| `test_tui_e2e.py` | slow | 4 | 5 |

---

## examples 覆盖矩阵（验收标准：测试能过 ⟹ examples 能过）

`examples/v5/infer/` 四个脚本的定义性能力，各由哪些测试端到端驱动：

| example | 定义性能力 | 驱动测试 | 说明 |
|---|---|---|---|
| `tui_multigpu.sh` | 交互 vllm TP=2 | `test_cli_parse.py`（CLI parse）+ `test_tui_e2e.py`（slow，交互 REPL） | 多 GPU TP 属 slow；CLI flag 映射 fast 覆盖 |
| `tui_tools_multiturn.sh` | 交互 + `--tools sandbox --max_turns 8` | `test_tools_multiturn_e2e.py::test_infer_cli_multiturn_sandbox_executes_a_real_command` | 真 LocalEnv shell subprocess 执行 `run_command` |
| `dataset_tools_grpo.sh` | `--external_plugins custom_tools.py --tools calculator --max_turns 8 --num_samples 8 --output_format grpo --save_rollout_tokens true` | `test_tools_multiturn_e2e.py`（grpo_run fixture 一次驱动多轮 tools+grpo+sidecar+external_plugins） | 正是 `_trainable_positions` off-by-one bug 路径 |
| `math_prm_orm_ray.sh` | `--reward_funcs accuracy --prm_model <PRM> --mode ray --nproc_per_node 2` | `test_reward_channels.py`（scalar 解析 + ray 守卫 + DeviceGroup 规划 + generative-judge 全链路） | tiny/脚本替身，不下载 7B、不占多卡 |

> ray/多 GPU 的真实分布式启动（`--mode ray`）不在 fast 层实跑（需集群），但 ray 独有的
> **接线逻辑**（DeviceGroup 规划、scalar 通道必须 ray 的守卫、generative_reuse 复用 sampler）已在 fast 层覆盖。
