# Sampling / Infer 统一与 Rollout Token 存储 —— 进度与剩余工作

> 本地工作文档（按约定保留在工作区，不加入暂存区、不提交）。用于跨会话记录进度，防止遗忘。
> 定位一律以符号名为准，行号会漂移。

## 0. 任务由来

原始三段任务（上一轮 plan `reward_devicegroup_and_tool_sandbox`）：

- **Part A**：reward function 的 remote_group + heterogeneous 禁 local。
- **Part B**：twinkle 原生工具/沙箱接入 dev 侧 sample 链路。
- **Part C**：rollout 产出的 encoded(input_ids/labels) + logprobs 落盘存储。

本轮用户追加（范围扩大，需重新出 plan 批准后执行）：

1. **单轮/多轮彻底统一到 `RolloutEngine`**（单轮 = 第一轮后即停的多轮特例）。
2. **`sample` 命令并入 `infer`，做成大而全**：infer 按 sample 现有模式补齐能力（best-of-n、reward 打分/筛选、DPO 输出、断点续采、多轮、工具、token 存储），之后下线 `sample` 命令。
3. **infer 的无数据集实时对话改为 TUI + 工具调用**，参考 `twinkle/src/twinkle_client/auto`。

## 1. 已完成（AST 通过，未跑测试/lint —— 遵循"实现阶段仅 AST 校验、未授权不改/跑测试"）

### Part A —— reward remote_group + heterogeneous 禁 local
- `swift/dev/config/rlhf_config.py`：reward function 的 device-group / heterogeneous 相关字段。
- `swift/dev/recipe/grpo.py`：`_build_reward_function` + `_reward_device_groups`。
- `swift/dev/config/process.py`：`_derive_reward`。
- `swift/dev/config/validate.py`：`_check_reward`。
- `swift/dev/builders/reward.py`：reward 构建对 device-group 的支持。

### Part B —— 工具/沙箱接入（b1–b5）
- b1 `swift/dev/rollout/sandbox.py`（新建，**不是 env_pool.py**）。`__all__` =
  `TOOL, SandboxTools, build_env_pool, build_tool_sandbox, resolve_tool_plugins, tool_manager_for`：
  - `TOOL = PluginRegistry.register_kind('tool', ToolPlugin, config_field='tools')`；
    `@PluginRegistry.register('tool','sandbox') class SandboxTools(ToolPlugin)`，`build(env)` 返回 `EnvTool.from_env(env)`。
  - `build_env_pool(rollout_config) -> EnvLeases`（twinkle 的池，带 `.lease()` 上下文管理器/`__len__`/`.close()`）：
    无 `sandbox_template` → `sandbox_num_envs` 个 `LocalEnv`（workspace 在 `sandbox_workspace_root` 或临时目录下 `slot_i`）；
    有 template → `AgentEnv` microVM。
  - `resolve_tool_plugins(names, rollout_config)`：按名解析 'tool' 插件（`PluginRegistry.resolve(TOOL, name, config=rollout_config)`）。
  - `tool_manager_for(env, plugins) -> Optional[ToolManager]`：每 env 单独建 manager（绑该 episode 租到的 workspace），无工具返回 None。
  - `build_tool_sandbox(rollout_config) -> (env_pool, tool_plugins)`：**两 recipe 的统一入口**；tools 空/无 config → `(None, [])`。
- b2 sandbox.py **不经** `rollout/__init__.py` 再导出；消费方直接 `from swift.dev.rollout.sandbox import build_tool_sandbox`，
  并 `import swift.dev.rollout.sandbox`（触发 'tool' 扩展点与 'sandbox' 插件注册）。
- b3 `swift/dev/rollout/multi_turn.py`：`MultiTurnRollout.__init__` 接收 `tool_manager/harness/followup_fn/env_pool/tool_plugins`；
  `env_pool 非空且有 plugins` → `_per_episode_tools=True`，`_generate_with_envs` 每 episode `with env_pool.lease() as env`
  + `tool_manager_for(env, plugins)` 单轨迹 rollout（线程池并发 = `len(env_pool)`）。`RolloutEngine.configure_multi_turn` 同签名透传。
- b4 `swift/dev/recipe/run_sampling.py`：`multi_turn_enabled and sampler is not None` 时
  `env_pool, tool_plugins = build_tool_sandbox(rollout_config)`，**直接构造** `MultiTurnRollout(sampler, template,
  max_turns, max_trajectory_tokens, env_pool=env_pool, tool_plugins=tool_plugins)`（未经 RolloutEngine）；finally `rollout.close()`。
- b5 `swift/dev/config/rollout_config.py`（新建）`RolloutConfig` 的工具/沙箱字段（真实名）：
  `tools: List[str]`、`sandbox_template`、`sandbox_workspace_root`、`sandbox_num_envs:int=1`、`sandbox_api_url`、
  `sandbox_timeout`、`sandbox_command_timeout:int=60`、`sandbox_memory_limit_gb:Optional[float]=2.0`
  （**无 sandbox_host/sandbox_port**）。`swift/dev/config/__init__.py` 导出 `RolloutConfig`；`cli/sample.py` 纳入解析。
  注：`SamplingConfig` 目前**没有** `rollout_config` 字段，rollout_config 由 CLI 单独解析后传给 run_sampling。

### b6 —— tools 需多轮启用的校验
- `swift/dev/config/validate.py`：新增 `validate_rollout_config` + `_check_rollout_tools`
  （`sandbox_num_envs >= 1`；tools 非空但 max_turns 为空 → raise）。
- `swift/dev/config/process.py`：`process_and_validate_configs` 内接线，
  `multi_turn_carrier = multi_turn_config or rlhf_config`（grpo 的 max_turns 在 rlhf_config 上）。

### c1（第一步，已落地）
- `swift/dev/rollout/__init__.py`：把 `RolloutEngine._samples_from_responses` 的逻辑提升为
  **模块级公开函数 `samples_from_responses(responses, prompt_extras)`**；原 staticmethod 改为委托它
  （保留 staticmethod 是因为 `run_grpo` 的 `SamplerRollout` 继承 `generate` 会调
  `self._samples_from_responses`，且 `test_rollout.py` 直接驱动该 staticmethod）。
  目的：让 run_sampling 单轮路径无需自建 engine 即可把采样响应转成带 encoded+logprobs 的 RolloutSample。

## 2. 原 plan 剩余（Part C，未开始）

- **c1 余下**：run_sampling 单轮走统一 encode，`_Candidate` 携带 encoded/logprobs。
- **c2**：`SamplingConfig` 加 `save_rollout_tokens: bool = False`（已确认无跨 Config flag 冲突）；
  新建 dev-native recorder（建议 `swift/dev/rollout/recorder.py`）。
  设计：**NPZ sidecar + 把相对路径嵌进已有 jsonl 行**（jsonl 本身即索引，避免 index.jsonl 与续采的一致性负担）。
  NPZ 每候选存 `input_ids/labels/completion_mask/response_token_ids/rollout_logprobs/response_loss_mask`
  （+ 有 prompt 时 `prompt_text`）。命名用**位置确定性 key** `{prompt_id}_c{candidate_index}`，
  使 resume 重放覆盖旧文件而非重复。参考 challenger `RolloutRecorder`（NPZ+index.jsonl）但做 dev-native 适配。
- **c3**：把分值 + npz 路径嵌进 `_emit_rows`/`_dpo_line` 输出（仅 `save_rollout_tokens` 开启时改变 schema）；
  与断点续采（tmp/resume/state/final）的落盘一致性对齐。

## 3. 本轮新增范围（待 plan 批准）

### 3.1 单轮/多轮统一到 RolloutEngine
- **现状**（已核实）：run_sampling 多轮是直接构造 `MultiTurnRollout`（非 RolloutEngine），单轮是
  `sampled_texts(sampler.sample([trajectories[i] for i in to_sample], params, **kwargs))`（`kwargs={'strict':...}` 仅 transformers）。
  `RolloutEngine.__init__(model_id, template, *, engine_args)` 只会从 model_id 建 vllm sampler，**无注入口**；
  `generate` 单轮分支硬编 `Trajectory(messages=)` + `self.sampler.sample(trajectories, params)`，**无 `**kwargs`**（strict 透不进）。
- 让 `RolloutEngine.__init__` 支持**注入已建 sampler**（`RolloutEngine(sampler=..., template=..., config=...)`
  则不重建 sampler），参照 `run_grpo.SamplerRollout` 的注入范式。
- run_sampling 单轮/多轮都只调 `rollout.generate(trajectories, params, ...)`：
  多轮（max_turns）走 twinkle 循环；单轮走 `else` 分支（`sampler.sample` + `samples_from_responses`），
  天然产出同样的 RolloutSample。
- `strict` 等 sample kwargs 需能透传（generate 的 `**kwargs` → sampler.sample）。
- **client 后端**：响应无 token → 不能走 `samples_from_responses`（会 raise）。
  方案：单轮 client 走 message-only（RolloutSample 只带 decoded/messages，无 encoded），
  由 `save_rollout_tokens` 与 client 互斥（或 generate 内识别无 token 响应退回 message-only）。

### 3.2 sample 并入 infer（大而全），下线 sample 命令
- infer 数据集路径（`run_infer`）吸收 run_sampling 能力：best-of-n、reward 打分/筛选、
  DPO 输出、断点续采、多轮、工具、token 存储；同时保留 infer 原有 pooling/reranker/generative + metric。
- infer CLI（`cli/infer.py`）需解析 `SamplingConfig`（+ reward/multi-turn 载体），
  按模式分派：交互（无数据集/eval_human）→ TUI；数据集 → 统一 run_infer。
- 下线：`swift/cli/main.py` 的 `DEV_ROUTE_MAPPING` 删 `'sample'`；删 `swift/dev/cli/sample.py`；
  `swift/dev/cli/__init__.py` 删 sample 相关导出/惰性映射。
- 注意：`run_infer` 与 `run_sampling` 输出契约不同（infer 每输入一行 response/responses/labels/messages；
  sample 是 DPO chosen/rejected + reward + resume）。统一 recipe 需按 flag 选输出形态。

### 3.3 infer 无数据集实时对话 → TUI + 工具调用（参考 auto）
- 参考 `auto/app.py`（async input 循环 + 流式输出，无 Textual/curses）与
  `auto/agent/core.py`（`AgentLoop`：OpenAI 兼容流式工具循环、`MAX_TOOL_ROUNDS`、历史裁剪、
  "↳ calling tools" 展示）。
- **但工具执行走 dev 原生**：复用 Part B 的 `SandboxEnvPool` + twinkle `ToolManager` + `MultiTurnRollout`
  （对本地 sampler，token 模式；client 后端 message-only）。auto 的 OpenAI `ToolExecutor` 仅借 UX，不借实现。
- 交互模型：每个人类输入 → 驱动一次 rollout（模型生成、按需多轮工具调用、直到不再调工具）→
  打印 assistant 终态 → 等待下一输入；跨轮累积 messages。
- 替换现有 `infer_cli` 的纯 `input()` REPL（`run_infer.py`）为 TUI 版。

## 4. 关键已核实事实（避免重复推导）

- `RolloutEngine.generate`：`self._multi_turn is not None` → 多轮；else → 批量 `sampler.sample` + `samples_from_responses`（单轮）。
- `enable_continous_work=True` 是 twinkle `MultiTurnRollout`（token 模式）构造期硬校验：
  vLLM(`vllm_sampler.py`)、SGLang(`sglang_sampler.py`) 已声明；**transformers(`transformers_sampler.py`) 未声明**（走多轮会 raise）。
  message-only 模式（template=None）免此校验。
- transformers 引擎**返回 token + logprobs**：`transformers_engine.py` 产 `prompt_token_ids`、
  `logprobs=step_logprobs[...]`（当 `params.logprobs is not None`）。故单轮 transformers 可走 `samples_from_responses`。
- `to_sampling_params`（builders/sampler.py）：仅当 `generation_config.logprobs` 为真才设 `params['logprobs']`；
  要拿 old_logps 必须强制 logprobs 非 None（镜像 `generate` 里 `sp['logprobs']=max(int(sp.get('logprobs') or 0),1)`）。
- `RolloutSample.encoded` 契约：`input_ids/labels/completion_mask`(+`SHIFTED_KEY`)，1-D、等长、无 padding；
  单轮 `samples_from_responses` 自建并自行 shift；多轮 `trajectory_to_rollout_sample` 吃账本已 shift 的 labels。
- run_sampling 断点续采：`*.tmp`(活写)/`.resume`(每批快照)/`sampling_state.json`(最后完成 batch)/`*`(收尾原子替换)；
  先 copyfile(tmp→resume) 再写 state，崩溃一致性优先保 state。
- infer CLI 现状：`InferCliConfig{eval_human,merge_lora,num_samples,multi_round}`；
  `infer_main`：eval_human 或无数据集 → `infer_cli`，否则 → `run_infer`。infer 目前**不解析** SamplingConfig/RLHFConfig。
- CLI 路由：`swift/cli/main.py` `DEV_ROUTE_MAPPING`（USE_SWIFT_V5 时生效）；`resolve_route` 对缺失 dev 路由 raise。
- `recipe/__init__.py` 导出**公开** `run_sampling`（infer 可合法委托，不违反“禁互导私有符号”）；也导出 `infer_cli`/`run_infer`。
- CLI 撞 flag 解决机制：`parse_configs_strict(classes, argv, command=, field_owners=, load_args_default=True)`，
  `field_owners` 声明同名字段归哪个 Config。sample CLI 用 `{'strict': SamplingConfig, 'temperature': GenerationConfig,
  'reward_funcs': SamplingConfig, 'reward_weights': SamplingConfig}`。infer 合并 SamplingConfig 时必须复用同一机制（`--strict`
  当前在 infer 是 `dataset_config.strict`，改由 SamplingConfig 拥有后 run_infer 需改读 sampling_config.strict）。
- sample CLI 还做：`sampler_engine=='pt'→'transformers'`；`num_sampling_batch_size→batch_size`、`num_sampling_batches→max_batches`；
  `prm_threshold→reward_threshold`（未显式传 reward_threshold 时）；`output_file` 默认时间戳.jsonl；`padding_side='left'`（未显式传时）；
  `sampling.reward_config = result['reward_config']`；`result['multi_turn_config'] = result['reward_config']`（RLHFConfig 是采样的多轮载体，持 max_turns）。
- 约束：recipe 间禁止互导私有符号（run_*.py 之间）；共享 helper 归 builders 层；
  CLI 新字段自动成 flag、字段名须跨 Config 唯一；计划/迁移文档放 `swift/dev/*.md` 且不提交。

## 5. 执行结果（四阶段全部完成，仅 AST 校验，未跑测试/lint）

- **阶段一（3.1）**：`RolloutEngine.__init__` 支持注入 sampler（不 own 生命周期）；`generate(**kwargs, force_logprobs)`
  单轮透传 strict、可关 logprobs；`samples_from_responses(allow_message_only)` 支持 client message-only；
  `close()` 只关 env_pool，`shutdown()` = close()+sampler.shutdown()。run_sampling 单/多轮统一走 RolloutEngine。
- **阶段二（Part C / c2,c3）**：`SamplingConfig.save_rollout_tokens` 开关；新建 `rollout/recorder.py` 的
  `RolloutRecorder`（NPZ sidecar，位置确定性命名 `{prompt_id}_c{idx}.npz`，prompt_id 复用 `_prompt_key`==`out['id']`）；
  接入 `_emit_rows`/`_dpo_line`；`validate_sampling_config` 校验 save_rollout_tokens 与 client 互斥（接入 process.py 校验链）。
- **阶段三（3.2）**：`cli/infer.py` 解析 SamplingConfig+RLHFConfig，`field_owners={strict:Sampling, temperature:Generation,
  reward_funcs:Sampling, reward_weights:Sampling}`，复刻 sample 的归一（pt→transformers、sampler_engine 权威、
  num_sampling_*→batch_size/max_batches、prm_threshold→reward_threshold、padding_side=left、reward_config/multi_turn_config 接线、
  num_samples↔num_return_sequences 双向同步）；`_is_sampling_mode`+`infer_main` 三路分派（交互/采样/纯推理）；
  run_infer 纯推理 strict 改读 `sampling_config.strict`；下线 sample（main.py DEV_ROUTE_MAPPING + 删 cli/sample.py + cli/__init__.py 导出）。
- **阶段四（3.3）**：新建 `recipe/infer_tui.py`（升级版 `infer_cli`）——ANSI 颜色/`You:`/`Agent:`/dim 工具 trace/历史裁剪（借 auto UX）；
  三种 turn：`_run_tool_turn`（tools+max_turns 时走 `RolloutEngine.generate`+`build_tool_sandbox`+`configure_multi_turn`，事后打印工具名与 truncated）、
  `_run_stream_turn`（sample_stream）、`_run_plain_turn`（sample）；保留 clear/reset-system/multi-line/quit；**不借** auto 的 OpenAI ToolExecutor。
  旧 REPL 从 run_infer.py 移除；`recipe/__init__.py` 改从 `infer_tui` 导出 `infer_cli`（`from swift.dev.recipe import infer_cli` 仍有效）；
  cli/infer.py 交互分支补传 `rollout_config`/`multi_turn_config`。

### 遗留 / 计划外（未处理）
- `legacy_coverage.py` 的 `CLI_LEGACY_ONLY['sample']` 可能成为死数据条目（无消费者）——计划外，后续清理。
- 受影响测试（未授权不动）：test_all_cli_coverage.py、test_cli_mapping.py、test_cli_entrypoints.py（曾 import parse_sample_configs/sample_main）、test_rollout.py。
- 当前环境缺 `twinkle.data_format`（run_infer/builders 同样依赖），属环境未装全，非本次改动引入。
- 工具路径的多轮 token 模式要求 sampler 声明 `enable_continous_work=True`：vllm/sglang 有、transformers 无（见 §4），故 TUI 工具路径实际限 vllm/sglang。
