# Server-client mode: resident models driven over HTTP

Two serving shapes — pick by what you need:

| You need | Use |
| --- | --- |
| Just an OpenAI-compatible endpoint for inference | `swift deploy` (one command, no YAML) |
| A resident **trainable** model: attach adapters, run optimizer steps, sample — all over HTTP, multi-tenant multi-LoRA | twinkle-server via `server_config.yaml` |
| The same swift commands you already know, but executing against a remote server | `--mode remote` |

## 1. `swift deploy`: OpenAI endpoint in one command

```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 USE_SWIFT_V5=1 swift deploy \
    --model Qwen/Qwen3.5-4B \
    --sampler vllm \
    --served_model_name policy \
    --host 0.0.0.0 --port 8000 \
    --route_prefix /api/v1 \
    --api_key EMPTY \
    --enable_data_plane true \
    --vllm_gpu_memory_utilization 0.85
```

What this builds: a twinkle-server `ServerConfig` (a gateway app + a sampler app) launched on
Ray Serve — dev starts an in-process local Ray head when no cluster exists; pass `--mode ray`
(or `--ray_address auto`) to attach to an existing cluster instead.

- Endpoints: `{route_prefix}/chat/completions`, `/completions`, `/embeddings`, `/models`,
  `/infer`, `/health`. A plain OpenAI SDK client works — point it at the same prefix.
- `--served_model_name` is the `model` clients pass and the name the sampler is mounted under.
- **`--route_prefix /api/v1` matters for the twinkle_client path**: the client's
  `get_base_url()` appends `/api/v1` itself, so the sampler only resolves when the server is
  actually mounted at `/api/v1`. Plain OpenAI clients are unaffected.
- `--enable_data_plane` turns on token-in-token-out for RL rollouts: it adds a data-plane app
  and switches the sampler to its async variant (vllm -> vllm_async, sglang -> sglang_async).
  Drop it for the lightweight path.
- Generation knobs (temperature/top_p/max_new_tokens) are per-request, NOT deploy flags (a
  server-side default would be silently ignored — dev warns if you set one).
- A worked client: `examples/deploy/vllm/client.py` and `token_in_token_out_client.py`.

## 2. twinkle-server: resident trainable model (multi-LoRA)

The full serving surface is defined by one YAML (`examples/server/server_config.yaml` is the
annotated reference). Four applications deploy onto one Ray Serve cluster:

```yaml
proxy_location: EveryNode          # gateway replica on every node
http_options: { host: 0.0.0.0, port: 8000 }

applications:
  - name: server                   # 1. the gateway clients talk to (route_prefix /api/v1)
    route_prefix: /api/v1
    import_path: server
    args:
      supported_models: [Qwen/Qwen2.5-0.5B-Instruct]
    deployments:
      - name: TinkerCompatServer
        max_ongoing_requests: 50
        autoscaling_config: { min_replicas: 1, max_replicas: 1, target_ongoing_requests: 128 }

  - name: models-Qwen2.5-0.5B-Instruct   # 2. the trainable base model
    route_prefix: /api/v1/model/Qwen/Qwen2.5-0.5B-Instruct
    import_path: model
    args:
      backend: transformers
      model_id: "ms://Qwen/Qwen2.5-0.5B-Instruct"
      model_loader: swift.dev.model.loader:build_model_loader   # swift's per-family loader
      max_loras: 4                 # several named adapters co-resident on one base
      device_group: { name: model, ranks: 1, device_type: cuda }
      device_mesh: { device_type: cuda, dp_size: 1 }

  # 3. sampler app (a vLLM engine, enable_lora) and 4. processor app (CPU tokenization)
  #    — see examples/server/server_config.yaml for their blocks
```

Launch (the counter-intuitive prerequisite: **`ray start --head` first** — the launcher's
`ray.init(address='auto')` only attaches to an existing cluster, it never starts one):

```shell
ray start --head --port=6379 --num-gpus=2 --include-dashboard=false --disable-usage-stats
twinkle-server check-config -c server_config.yaml
twinkle-server launch -c server_config.yaml
```

### The client half: train and sample over HTTP

```python
from peft import LoraConfig
from twinkle import init_twinkle_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.sampler import vLLMSampler

init_twinkle_client(base_url='http://localhost:8000', api_key='EMPTY_TOKEN')

# Data is loaded and encoded CLIENT-side (twinkle Dataset works locally); only the
# model/sampler are remote. Delegating encoding to the processor app is also possible.
dataset = Dataset(DatasetMeta('ms://swift/self-cognition', data_slice=range(200)))
dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=512)
dataset.map('SelfCognitionProcessor', init_args={'model_name': '...', 'model_author': '...'})
dataset.encode(batched=True)
dataloader = DataLoader(dataset=dataset, batch_size=4)

model = MultiLoraTransformersModel(model_id='Qwen/Qwen2.5-0.5B-Instruct')

# Attach a NAMED adapter; max_loras on the server bounds how many coexist
model.add_adapter_to_model('assistant', LoraConfig(target_modules='all-linear'),
                           gradient_accumulation_steps=2)
model.set_template('Template')
model.set_processor('InputProcessor', padding_side='right')
model.set_loss('CrossEntropyLoss')
model.set_optimizer('Adam', lr=1e-4)

for epoch in range(3):
    for cur_step, batch in enumerate(dataloader, start=1):
        model.forward_backward(inputs=batch)      # same method names as the local TrainableModel
        model.clip_grad_and_step()
        metric = model.calculate_metric(is_training=True)

uri = model.save('assistant')                     # -> 'twinkle://...' managed-store URI
# sampling from the saved adapter: twinkle_client.sampler.vLLMSampler
```

The complete runnable pair: `examples/server/{serve.sh, server_config.yaml, client.py, train.sh}`.
More remote flows in `twinkle/cookbook/client/twinkle/` (self_cognition, dpo, sample,
embedding, multi_turn_rollout, upload_to_hub).

**Client-handle contract**: same method names as the local `TrainableModel`, but everything
crossing the wire must be serializable — class **names** plus scalar kwargs for
`set_loss`/`set_optimizer`/`set_processor`/`set_template`; a live `Template` instance cannot
cross. Checkpoints come back as `twinkle://` URIs in the server's managed store (the server
owns checkpoint storage; a client-side `save_dir` only works when both sides share a
filesystem).

## 3. `--mode remote`: swift commands against a server

The supervised commands run as thin clients against a twinkle-server:

```shell
USE_SWIFT_V5=1 swift sft \
    --mode remote \
    --server_base_url http://127.0.0.1:8300 \
    --server_api_key <token> \
    --model Qwen/Qwen3.5-4B \        # the SERVER-side served model name here
    --dataset 'swift/self-cognition#600' \
    --tuner lora ...
```

- `--server_base_url` is the gateway root WITHOUT `/api/v1` (the client appends it);
  `--server_api_key` is the Bearer token (omit if unauthenticated).
- Data still loads locally (tokenizer needs to be resolvable); model and sampler are remote.
- Constraints enforced up front by validation (with guidance in the error): RL is refused
  (awaiting the streaming-driver client port); tuner must be `lora` or `full`; base-modifying
  LoRA inits (pissa/olora...) and non-default freeze knobs are refused; DP engines need the
  forward batch rows >= data world size (pass `--gradient_accumulation_steps` >= DP width).
- `--backend megatron` under remote names the SERVER-side model app's engine — both backends
  are legal.

## Known gotchas

- **`ray start --head` first** for `twinkle-server launch` (attach-only `ray.init`);
  `swift deploy` handles the local head for you.
- Gateway responses use a task envelope (`status` in pending/queued/running/completed/failed/
  cancelled, payload under `result`) — consume through the client handles rather than raw HTTP
  when you can.
- Under remote/DP, single `forward_backward` row count must be >= data world size, or the
  pre-flight check rejects the step.
- vLLM memory floor applies per server app: budget `gpu_memory_utilization` against what the
  model app's GPUs actually have free.
