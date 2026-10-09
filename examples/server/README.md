# v5 server examples — twinkle server-client multi-LoRA cookbook

v5 has **no `swift server` command**. Serving a *trainable* model over HTTP is done by driving
`twinkle-server` directly: a `ServerConfig` deploys four Ray Serve applications — a gateway, a model app, a
sampler app and a processor app — and a thin Python client attaches LoRA adapters, trains them and samples
them, all over HTTP. This directory is a complete, **e2e-verified** walkthrough of that path: two named
adapters (`assistant`, `coder`) were trained against one resident base and each sampled back correctly.

## Files

| file | role |
| --- | --- |
| `server_config.yaml` | the deployment: gateway + model + sampler + processor apps, `Qwen2.5-0.5B-Instruct`, `max_loras: 4` |
| `serve.sh` | `ray start --head` (idempotent) → `twinkle-server check-config` → `twinkle-server launch`; blocks in the foreground |
| `train.sh` | waits for the gateway, then runs `client.py` from a neutral cwd |
| `client.py` | the client half: attach several named LoRA adapters → train each → save → sample base + each adapter |

## How to run

Two shells. The server blocks, so it needs its own:

```shell
# shell 1 — start the server (2 GPUs: one for the model app, one for the vLLM sampler)
GPUS=1,2 RAY_PORT=6379 bash examples/server/serve.sh

# shell 2 — once serve.sh prints that the gateway is up, drive the client
bash examples/server/train.sh
```

Env overrides (all optional, read by `train.sh`/`client.py`): `TWINKLE_SERVER_URL` (default
`http://localhost:8000`), `TWINKLE_SERVER_TOKEN` (`EMPTY_TOKEN`), `TWINKLE_MODEL_ID`
(`Qwen/Qwen2.5-0.5B-Instruct`), `TWINKLE_ADAPTER_NAMES` (`assistant,coder`), `TWINKLE_MAX_STEPS` (`6`),
`TWINKLE_DATA_SLICE` (`200`), `TWINKLE_TEMPLATE` (`Template`).

## Load-bearing details (why the scripts look the way they do)

- **`ray start --head` is mandatory before launch.** `twinkle-server`'s launcher calls
  `ray.init(address='auto')`, which only *attaches* to an existing cluster and never starts one. `serve.sh`
  starts a head (reusing one already on `RAY_PORT` if present) and pins `CUDA_VISIBLE_DEVICES` for it.
- **The model app is loaded through swift.** `model_loader: swift.dev.model.loader:build_model_loader` is a
  `module:callable` factory resolved inside the Ray actor at startup, so **ms-swift must be importable in
  this env** (from the repo root, `pip install -e .`). `serve.sh` fails fast if it is not.
- **`max_loras: N` makes it a multi-LoRA host** — several named adapters co-resident on one base. Adding a
  second name via `add_adapter_to_model(name, ...)` is what exercises the multi-LoRA path.
- **Do not pass a client-side `save_dir`.** The server owns the checkpoint store
  (`TWINKLE_DEFAULT_SAVE_DIR`, default `./outputs` under the server cwd); `save()` returns a `twinkle://`
  URI. A client-side `save_dir` only resolves when client and server share a filesystem *and* the directory
  already exists on the server, so the managed store is the portable choice.
- **Resolve the tokenizer through modelscope, not HF hub.** The client decodes sampler token ids with the
  base tokenizer; `swift.dev.utils.hub.safe_snapshot_download(base_model)` points at the local modelscope
  snapshot so it works on an offline, modelscope-cached box (`AutoTokenizer.from_pretrained('Qwen/...')`
  would otherwise hit HF hub and fail).
- **Template namespace is only three classes.** twinkle ships `Template`, `Qwen3_5Template` and
  `DeepseekV4Template`. Any other family — including `Qwen2.5` here — uses the base `Template`, which
  renders through the checkpoint's own jinja `chat_template`.
- **The sampler is vLLM with LoRA enabled** (`enable_lora: true`); `sample(..., adapter_uri=twinkle_path)`
  selects a trained adapter, `adapter_uri=None` samples the base.

See the twinkle cookbooks under `twinkle/cookbook/client/{swift_model,server/transformer}/` for the upstream
shapes this directory adapts.
