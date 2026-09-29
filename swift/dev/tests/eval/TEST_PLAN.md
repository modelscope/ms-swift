# `swift eval` 测试计划

`swift eval`（v5）不再起 HTTP 服务、不再评测远程模型，而是在**本进程**里用 `build_sampler` 造一个 twinkle
sampler，交给 `twinkle_agentic.evaluator.Evaluator`（`EvalType.CUSTOM` + `EvalBackend.NATIVE`），由 EvalScope
的 Native runner 驱动生成与打分。sampler 就是被测模型本身：`--infer_backend` 选本地引擎，任何引擎参数原样透传，
LoRA adapter **在线加载**（引擎建好 LoRA 槽位、每个请求选中 `adapters[0]`，不做 merge）。continuous-batching
后端（vllm/sglang）下每条 trajectory 单独提交，引擎保持饱和、短答案不等最长的一条。

本套件覆盖从「命令行参数 → 配置装配 → 真实 Evaluator × EvalScope Native runner × SamplerModelAPI → report /
`result_jsonl`」的完整链路，分两层，镜像 `swift/dev/tests/deploy`：

- **fast 层**（无 GPU、无下载）：`run_eval` 的纯 helper 与每个 CLI 参数落地（`test_config_build` /
  `test_cli_parse`）；以及**真实 `run_eval` 链路**在无 GPU 下端到端跑通（`test_evaluator_e2e`）——只替换三处
  模型构建调用（`build_sampler` / `load_model_processor` / `build_template`）为脚本化 sampler，**不桩
  Evaluator、不桩 EvalScope、不桩 `_summarize` / `_validate_eval_datasets`**。数据集是 hermetic 本地
  `general_qa`（`dataset_args` 的 `local_path` 指向本地 jsonl，EvalScope 离线 load_from_disk），行内容为中文，
  使 BLEU/Rouge 走 jieba 路径、不触发 NLTK `punkt_tab` 下载；脚本化 sampler 精确返回参考答案，得分确定性 1.0。
- **slow 层**（`@pytest.mark.slow` + `@pytest.mark.accel(1)`，真实 Qwen2.5-0.5B-Instruct + vLLM + EvalScope）：
  子进程真跑 `swift eval`，参数与 `examples/v5/eval/{eval,eval_lora}.sh` 一致，证明 example 按原样可运行。

## 用例矩阵

| 文件 | 层 | 覆盖 | 对应 example / 需求 |
| --- | --- | --- | --- |
| `test_config_build.py` | fast | `_model_name`（basename/去尾斜杠/None→'model'）；`_guard_backend`（client/no 报错含替代方案，vllm/sglang/transformers 放行）；`_validate_eval_datasets`（大小写就地归一、未知 benchmark 报错列 supported、空 dataset 报错）；`_build_task_config`（work_dir/limit/eval_batch_size=eval_num_proc/dataset_args/generation_config、`extra_eval_args` 最后 merge、绝不带 owned key）；`Evaluator` 构造守卫（sampler XOR api、owned key 冒泡 `EvaluatorConfigError`、model_id 不可推断报错）；sampler adapter（continuous→`batcher is None`、非 continuous→走 `SamplerBatcher`、不支持的 generation 字段 / `stream=True`→`UnsupportedCapabilityError`） | 全部参数 / 功能 |
| `test_cli_parse.py` | fast | `parse_eval_configs` 每个 eval flag 落到正确 Config 字段（eval_dataset/eval_limit/eval_num_proc/eval_generation_config/eval_dataset_args/extra_eval_args/eval_output_dir 绝对化/result_jsonl 默认 None）；`infer_backend` vllm/sglang/transformers + `pt→transformers` 改写；`build_engine_args` 引擎参数剥前缀原样透传 + 排除 server-only 键、transformers→`max_batch_size`；`--adapters` 裸 path / 多 adapter 顺序；`quantize_config` 在面上；`eval_main` 透传 backend/engine_args/adapters/quantize；退役 flag 分类拒绝（eval_url/eval_backend/local_dataset/merge_lora/temperature/top_p/top_k/max_new_tokens/host） | 全部参数 |
| `test_evaluator_e2e.py` | fast | 真实 `run_eval` × 真实 Evaluator × EvalScope Native × SamplerModelAPI：continuous 路径逐 trajectory（每次 sample 只收 1 条）跑通、report 行 `general_qa`/num=3/score=1.0、report 键集、`eval_output_dir` 产物、`result_jsonl` 单行；非 continuous 路径走 batcher 仍正确；`--adapters` 选首个、per-request `adapter_path`、多余 adapter 告警、`shutdown()` 被调用 | 端到端 / 功能在线 |
| `test_eval_example_e2e.py` | slow | 子进程真跑 `swift eval`（eval.sh 参数）：exit 0、EvalScope 在 `eval_output_dir` 落报告、`result_jsonl` 单行含 gsm8k 真实评分、model/adapters/eval_limit 正确 | `eval.sh` |
| `test_eval_lora_example_e2e.py` | slow | peft 造随机 LoRA → 子进程真跑 `swift eval --adapters`（eval_lora.sh 参数）：exit 0（证明引擎接受在线 adapter）、report `adapters==[ckpt]`、gsm8k 评分行存在、**磁盘无合并权重**（`{adapter}-merged` 不存在、adapter 目录未增权重，证明在线加载而非 merge） | `eval_lora.sh` |

## 取代的旧桩测

删除 `swift/dev/tests/component/config/test_cli_entrypoints.py::test_eval_builds_local_sampler_and_drives_evaluator`：
它把 `Evaluator` / `_summarize` / `_validate_eval_datasets` 全桩掉，只验 recipe 的接线顺序，是伪端到端。其
接线与 report 断言已被 `test_evaluator_e2e.py` 用**真实组件**覆盖。该文件其余用例不动（删后 33 passed）。

## 运行

```bash
# fast（无 GPU、无下载）
CUDA_VISIBLE_DEVICES="" PYTHONPATH=twinkle/src:. /usr/local/bin/python -m pytest swift/dev/tests/eval -m "not slow" -q
# slow（单卡，真起 vLLM + 真下载 gsm8k；每个模块拉起一次引擎）
CUDA_VISIBLE_DEVICES=<free> PYTHONPATH=twinkle/src:. /usr/local/bin/python -m pytest swift/dev/tests/eval -m slow -q
```

### 依赖前提（examples / slow 测试能直接跑的关键）

- `swift eval` 在 `USE_SWIFT_V5=1` 下路由到 `python -m swift.dev.cli.eval`；`run_eval_cli` 优先用已安装的
  `swift` 控制台脚本（example 的真实入口），不在 `PATH` 时回退模块形式。
- 环境 `/usr/local/bin/python`（3.12），已装 `ray` + `vllm` + `evalscope 1.9.0`（Native benchmark 注册表含
  `gsm8k`、`general_qa`），`swift` / `twinkle` 均 editable 安装。
- slow 层允许联网下载真实权重（`Qwen/Qwen2.5-0.5B-Instruct`）与 gsm8k 数据集（markers 定义：slow=下载真实
  权重 / 跑真实循环，daily CI）。
- hermetic `general_qa` 的评分陷阱：BLEU 对**英文**预测走 nltk `word_tokenize`，需下载 `punkt_tab`（非离线）；
  对**中文**走 jieba（离线）。故 fast 层数据集必须中文，且脚本化 sampler 返回值须精确等于参考答案。

## 结果

fast 层（无 GPU、离线）全绿：

| 文件 | 层 | 结果 |
| --- | --- | --- |
| `test_config_build.py` | fast | 21 passed |
| `test_cli_parse.py` | fast | 23 passed |
| `test_evaluator_e2e.py` | fast | 3 passed |
| 合计（`-m "not slow"`） | fast | **47 passed, 2 deselected** |

slow 层（真实 Qwen2.5-0.5B-Instruct + vLLM + gsm8k，单卡）全绿：

| 文件 | 层 | 结果 |
| --- | --- | --- |
| `test_eval_example_e2e.py` | slow | 1 passed (~62s) |
| `test_eval_lora_example_e2e.py` | slow | 1 passed (~76s) |

两个 slow 用例都在子进程真跑了 `swift eval`（真实 vLLM 引擎 + 真实 gsm8k），例证 `eval.sh` / `eval_lora.sh`
按原样可运行；LoRA 用例证明 adapter 在线加载（exit 0 + `adapters` 记录 + 磁盘无 `-merged` 合并权重）。

编写端到端测试期间暴露并修复的**测试自身**问题（非产品缺陷）：

1. **`report['model']` 断言用了原始 hub id**：config 校验会把 `--model` 解析成本地缓存 snapshot 路径
   （`.../Qwen--Qwen2.5-0.5B-Instruct/snapshots/master`），故 `row['model'] == MODEL` 失败。改为断言模型
   leaf 名是该路径的子串（仍钉住「被测模型就是这个 checkpoint」）。
2. **conftest 硬编码了错误的 `MODEL_TYPE='qwen2_5'`**：legacy 模型注册表里没有 `qwen2_5`（纯文本 Qwen2.5
   归到 `qwen2`，只有 `qwen2_5_vl`/`qwen2_5_omni` 等变体），LoRA fixture 造 adapter 时 `load_model_processor`
   直接 `ValueError`。改为不硬编码 `model_type`/`template`，交给 swift 从 `--model` 自动解析——正是 example
   CLI 的行为，既修掉错误常量也消除潜在陷阱。
