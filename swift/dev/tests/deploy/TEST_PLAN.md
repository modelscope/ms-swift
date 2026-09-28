# `swift deploy` 测试计划

`swift deploy` 不再手搓 FastAPI，而是把 dev 的 Config 拼成 twinkle-server 的 `ServerConfig`
（gateway + sampler [+ data-plane]）交给 `launch_server` 跑在 Ray Serve 上。本套件覆盖从「命令行参数
→ 配置装配 → 真起服务 → OpenAI 端点 / token-in-token-out」的完整链路，分两层：

- **fast 层**（无 GPU、无模型下载）：配置装配与 CLI 解析，直接调 `build_server_config` /
  `parse_deploy_configs`，钉住 sampler_type 派生、`_validate` 守卫、route_prefix/data-plane URL、
  merge-lora 守卫、每个 CLI 参数落地到正确的 Config 字段。
- **slow 层**（`@pytest.mark.slow` + `@pytest.mark.accel(1)`，真实 Qwen3.5-4B + vLLM + Ray Serve）：
  经 `run_deploy_process` 在子进程真起服务，用 HTTP / twinkle_client 打真实请求，端到端验证。

模型：`Qwen/Qwen3.5-4B`（swift 注册为多模态 `qwen3_5`，twinkle 模板 `Qwen3_5Template`），
一个模型同时覆盖纯文本与图文两类服务。

## 用例矩阵

| 文件 | 层 | 覆盖 | 对应用户需求 |
| --- | --- | --- | --- |
| `test_config_build.py` | fast | sampler_type 派生（vllm/vllm_async/sglang/sglang_async/torch）、`_validate` 四类守卫、`_url_scheme`、route_prefix 与 sampler 路由一致、data_plane_url 随 TLS 变 scheme、merge-lora 单/多 adapter 守卫、deployment 选项（autoscaling/num_replicas）、persistence、application 组装（gateway/[data-plane]/sampler） | 8 |
| `test_cli_parse.py` | fast | `parse_deploy_configs` 逐个 CLI 参数落到正确 Config：served_model_name/route_prefix/api_key/owned_by/host/port/ssl_*/max_logprobs/max_concurrency/enable_data_plane/sampler_type/ray_address/ray_namespace/num_replicas/autoscaling/persistence_*/merge_lora/adapters(name=path 与裸 path)/infer_backend(vllm/sglang/pt→transformers)/mode/nproc_per_node；`deploy_main` 透传 | 8 |
| `test_openai_surface_e2e.py` | slow | /health、/ping(GET+POST)、/v1/models(id+owned_by)、/v1/chat/completions(非流式+流式)、/v1/completions、/v1/infer、api_key 鉴权(401/200)、model_not_found(404)、bad_request(400) | OpenAI 面 |
| `test_multimodal_e2e.py` | slow | /v1/chat/completions 带 image_url 的图文请求，Qwen3.5-4B 走 vLLM 返回 completion（证明图片张量到达引擎） | 1 |
| `test_merge_lora_e2e.py` | slow | peft 造随机 LoRA → `merge_lora=True` 部署 → /v1/models 只剩合并后的 base 名、能正常 chat；`_merge_single_adapter` 对 >1 adapter 报错 | 3 |
| `test_legacy_parity_e2e.py` | slow | 新 deploy 的端点面与响应形状对齐 legacy `SwiftDeploy`（health/ping/models/chat/completions/infer + OpenAI 兼容响应对象 + api_key 语义） | 4 |
| `test_restart_e2e.py` | slow | 起服务→就绪→退出上下文（SIGTERM→serve.shutdown，进程退出、端口释放）→同端口重启→再次就绪 | 5 |
| `test_token_in_token_out_e2e.py` | slow | 轻量路径 `sample(InputFeature(input_ids))` 的 `new_input_feature` 字段/偏移正确性（prompt 前缀 completion_mask=0、labels=-100；生成段 mask=1、labels==input_ids；三字段等长）；多轮 trajectory 的偏移正确性；data-plane 路径 `asample_to_data_plane`→DataRef→aget/aappend/arelease | 6 |

## 运行

```bash
# fast（无 GPU）
PYTHONPATH=twinkle/src:. python -m pytest swift/dev/tests/deploy -m "not slow" -q
# slow（单卡，真起服务；每个模块会拉起一次 Ray Serve + vLLM）
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=twinkle/src:. python -m pytest swift/dev/tests/deploy -m slow -q
```

### 依赖前提（examples / slow 测试能直接跑的关键）

slow 层与 `examples/v5/deploy` 走真实 twinkle-server，除 `swift`/`ray`/`vllm` 外还需两个 extra：

- `twinkle[server]` + `twinkle[client]`（拉入 `tinker`）：gateway 的 `tinker_handlers` 顶层硬导入 `tinker`，
  起 gateway 必须装。
- `twinkle[async-rl]`（拉入 `TransferQueue`，import 名 `transfer_queue`）：`--enable_data_plane` 时
  `data_plane/store.py` 硬依赖它，token-in-token-out 的 data-plane 路径必须装（`pip install --pre
  "TransferQueue>=0.1.9.dev0"`）。
- token-in-token-out 的 `twinkle_client` 有个硬约束：`get_base_url()` 会给 base_url 强制追加 `/api/v1`，
  所以服务端 `--route_prefix` 必须是 `/api/v1`，客户端 base_url 必须是网关根或以 `/api/v1` 结尾，否则
  会拼成 `/v1/api/v1/...` 而 404。OpenAI SDK 客户端不受此约束，指向同一前缀即可。

## 结果

真实 Qwen3.5-4B + vLLM 0.23 + Ray Serve，单卡顺序跑，全绿：

| 文件 | 层 | 结果 |
| --- | --- | --- |
| `test_config_build.py` + `test_cli_parse.py` | fast | 57 passed |
| `test_openai_surface_e2e.py` | slow | 10 passed |
| `test_multimodal_e2e.py` | slow | 5 passed |
| `test_merge_lora_e2e.py` | slow | 2 passed |
| `test_legacy_parity_e2e.py` | slow | 6 passed |
| `test_restart_e2e.py` | slow | 1 passed |
| `test_token_in_token_out_e2e.py` | slow | 3 passed |

全量一次跑通（fast + slow 同进程顺序）：**84 passed in ~15min**。`examples/v5/deploy` 已用真实
`swift deploy --route_prefix /api/v1 --enable_data_plane true` + `token_in_token_out_client.py` 冒烟验证：
轻量路径 `new_tokens=64 / trainable_len=70`（prompt 6 + completion 64，偏移正确），data-plane 路径生成 4
条并成功 `arelease`，`/models` 返回带 `created` 且 `owned_by=swift`。

编写端到端测试期间暴露并修复的真实缺陷（非测试自身问题）：

1. **多模态响应序列化崩溃**：sampler 的 `new_input_feature` 是整个 trajectory，多模态编码后其
   `messages[].content[]` 媒体块里带 `PIL.Image`；`_serialize_input_feature` 只做顶层 numpy/torch→list，
   不递归、不处理 PIL，导致 FastAPI 序列化抛 `PydanticSerializationError`（HTTP 500）。改为递归 sanitizer。
2. **OpenAI 标准 `image_url` content-part 被丢弃**：`_process_mm_list_format` 只认 twinkle 原生
   `{'type':'image',...}`，OpenAI 的 `{'type':'image_url','image_url':{'url':...}}` 原样透传不进
   `preprocess_images`、不产 pixel。改为在循环开头把 `image_url`/`video_url`/`audio_url` 就地解包成原生格式。
3. **token-in-token-out 轻量路径崩溃**：裸 `InputFeature(input_ids=...)`（无 labels）经
   `concat_input_feature`→`_prefix_completion_mask` 抛 `ValueError`（prefix completion_mask 0 entries）。
   改为 labels 为空时默认 `[-100]*len(prompt_ids)`，使 prefix mask 全 0、只有追加的 completion 段可训练。
4. **gateway `/models` 缺 `created` 字段**：OpenAI 规范与 legacy `Model` dataclass 都含 `created`
   （openai SDK 视其为必需），gateway 的 `list_models` 漏了。补上 `'created': int(time.time())`。

另记录两处「客户端需知道」的服务端行为（非缺陷，测试已适配、examples 已注明）：

- **sampler 就绪竞态**：`run_deploy_process` 只探 gateway 的 `/models`（秒级就绪）即返回，但 sampler
  replica（加载权重 + vLLM warmup，1~2min）健康后才注册 `{route_prefix}/sampler/{name}` 路由；直连
  sampler 的 `twinkle_client` 需重试等就绪（gateway 代理路径会等待，故 OpenAI 面不受影响）。
- **与 legacy 的有意差异**：鉴权失败 legacy 返回 400 `{'message','object':'error'}`，新 gateway 返回
  401 + OpenAI 风格 `{'error':{...}}`；unknown model legacy 400 拒绝，新 gateway 单模型时回退到唯一模型；
  `owned_by` legacy 默认 `'ms-swift'`、新 gateway 报 `'swift'`。这些由 `test_legacy_parity_e2e.py` 钉住。

