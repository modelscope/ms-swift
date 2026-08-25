"""run_deploy: an OpenAI-compatible server on twinkle's Sampler.

dev counterpart of legacy ``swift deploy`` (``swift/pipelines/infer/deploy.py::SwiftDeploy``), with the
same endpoint set: ``/v1/chat/completions``, ``/v1/completions``, ``/v1/embeddings``, ``/v1/models``,
``/health``, ``/ping`` and ``/infer/``. Auth, HTTPS, throughput stats, request logging, multi-LoRA
routing and the ``run_deploy_process`` context manager are all here too.

**Concurrency is the one thing that is not a simplification, and getting it wrong is silent.**
twinkle's Sampler is a synchronous API: every method funnels through ``_run_in_loop``, which is
``asyncio.run_coroutine_threadsafe(...).result()`` -- it BLOCKS the calling thread until generation
finishes. Calling it directly from an async handler would block uvicorn's event loop for the whole
generation, so requests would serialize and the engine's continuous batching would never see more than
one at a time. Throughput would collapse without a single error appearing. Every sampler call therefore
goes through :func:`_off_loop` onto a thread pool.

Multi-LoRA is routed per request: ``request.model`` is looked up in ``adapter_mapping`` and the adapter
path travels with the individual request (``adapter_paths=``), so vLLM and sglang keep several adapters
resident and mix them in one running batch. The transformers backend groups by adapter instead,
because peft activates one at a time.

Backend caveat: ``/v1/embeddings`` needs a pooling forward, which no generation engine provides, so it
is served by a separately-built model rather than by the sampler -- and is available only when the
server was started for an embedding model.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from swift.dev.config import GenerationConfig, ModelConfig, TemplateConfig

logger = logging.getLogger(__name__)

#: twinkle StopReason -> OpenAI finish_reason. twinkle's is a plain str Literal
#: ('length' | 'stop' | 'abort' | 'error'), and OpenAI has no vocabulary for a cancelled or failed
#: generation, so those report 'stop' -- the response body already carries whatever was produced.
_FINISH_REASONS = {'length': 'length', 'stop': 'stop', 'abort': 'stop', 'error': 'stop'}


def run_deploy(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    backend: Literal['vllm', 'sglang', 'transformers'] = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    adapter_mapping: Optional[Dict[str, str]] = None,
    merge_lora: bool = False,
    host: str = '0.0.0.0',
    port: int = 8000,
    served_model_name: Optional[str] = None,
    owned_by: str = 'swift',
    api_key: Optional[str] = None,
    max_logprobs: Optional[int] = None,
    max_concurrency: int = 64,
    log_interval: float = 0.0,
    request_log_path: Optional[str] = None,
    verbose: bool = False,
    ssl_keyfile: Optional[str] = None,
    ssl_certfile: Optional[str] = None,
    log_level: str = 'info',
) -> None:
    """Serve the model until interrupted.

    Args:
        model_config / template_config / generation_config: as in ``run_infer``. The generation config
            supplies the server-side defaults a request may override.
        backend: generation engine.
        engine_args: forwarded verbatim to the engine.
        adapter_mapping: ``{served name: adapter path}``. Each name shows up in ``/v1/models`` and a
            request naming it is served by that adapter. The engine is built with LoRA enabled and
            sized for these, because vLLM cannot turn it on afterwards.
        merge_lora: merge a single adapter into the base weights at startup instead of applying it per
            request. Only meaningful with exactly one adapter; per-request routing is the point of
            ``adapter_mapping``, and a merged model can no longer switch.
        host / port: bind address.
        served_model_name: the id reported in ``/v1/models`` and echoed in responses. Defaults to the
            model path's basename.
        owned_by: the ``owned_by`` field of ``/v1/models`` entries.
        api_key: when set, requests must carry ``Authorization: Bearer <key>``. When unset the port is
            UNAUTHENTICATED -- do not expose it to an untrusted network.
        max_logprobs: server-side ceiling on ``top_logprobs``. Requests above it are rejected with 400
            rather than silently truncated, so a client cannot quietly get fewer than it asked for.
        max_concurrency: size of the thread pool that runs sampler calls, i.e. the ceiling on requests
            in flight. Too low starves the engine's batching; too high just queues threads.
        log_interval: seconds between throughput log lines. 0 disables.
        request_log_path: jsonl file recording each request and its response.
        verbose: also log each request/response through the logger.
        ssl_keyfile / ssl_certfile: serve HTTPS. Both are required together.
        log_level: uvicorn log level.
    """
    import uvicorn

    if bool(ssl_keyfile) != bool(ssl_certfile):
        raise ValueError('HTTPS needs both ssl_keyfile and ssl_certfile; one alone would start a plain '
                         'HTTP server while the operator believed it was encrypted.')
    if api_key is None:
        logger.warning('run_deploy is starting WITHOUT api_key: anyone who can reach the port can use '
                       'the model. Set api_key= before exposing it beyond localhost.')

    app, sampler = build_app(
        model_config,
        template_config,
        generation_config,
        backend=backend,
        engine_args=engine_args,
        adapter_mapping=adapter_mapping,
        merge_lora=merge_lora,
        served_model_name=served_model_name,
        owned_by=owned_by,
        api_key=api_key,
        max_logprobs=max_logprobs,
        max_concurrency=max_concurrency,
        log_interval=log_interval,
        request_log_path=request_log_path,
        verbose=verbose,
    )
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
        )
    finally:
        sampler.shutdown()


def build_app(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    backend: Literal['vllm', 'sglang', 'transformers'] = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    adapter_mapping: Optional[Dict[str, str]] = None,
    merge_lora: bool = False,
    served_model_name: Optional[str] = None,
    owned_by: str = 'swift',
    api_key: Optional[str] = None,
    max_logprobs: Optional[int] = None,
    max_concurrency: int = 64,
    log_interval: float = 0.0,
    request_log_path: Optional[str] = None,
    verbose: bool = False,
):
    """Build the FastAPI app and its sampler, without serving.

    Separate from :func:`run_deploy` because ``uvicorn.run`` blocks: tests need the app object, and so
    does anyone mounting these routes inside a larger service.

    Returns:
        ``(app, sampler)``. The caller owns the sampler's lifetime.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    from swift.dev.builders import build_sampler, build_template, to_sampling_params
    from swift.model import get_model_processor

    adapter_mapping = dict(adapter_mapping or {})
    if merge_lora and adapter_mapping:
        model_config, adapter_mapping = _merge_single_adapter(model_config, template_config, adapter_mapping)

    _, processor = get_model_processor(model_config.model, load_model=False)
    template = build_template(template_config, processor)
    model_name = served_model_name or os.path.basename(str(model_config.model).rstrip('/'))
    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args=engine_args,
        template=template,
        adapters=list(adapter_mapping.values()) or None)

    # A thread pool, not the event loop: see the module docstring. max_workers is the real concurrency
    # ceiling, since each in-flight request holds one thread for the length of its generation.
    executor = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix='deploy-sample')

    async def _off_loop(fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, lambda: fn(*args, **kwargs))

    stats = _Stats()
    request_log = _RequestLog(request_log_path, verbose)
    context = _ServerContext(
        off_loop=_off_loop,
        sampler=sampler,
        backend=backend,
        model_name=model_name,
        owned_by=owned_by,
        adapter_mapping=adapter_mapping,
        api_key=api_key,
        max_logprobs=max_logprobs,
        generation_config=generation_config,
        template_config=template_config,
        to_sampling_params=to_sampling_params,
        model_config=model_config,
        stats=stats,
        request_log=request_log,
    )

    app = FastAPI(lifespan=_lifespan_factory(stats, log_interval, executor))

    @app.get('/health')
    async def health():
        return Response(status_code=200)

    @app.get('/ping')
    async def ping():
        """SageMaker's health path. Same check as /health, different URL by convention."""
        return Response(status_code=200)

    @app.get('/v1/models')
    async def models():
        created = int(time.time())
        names = [model_name] + list(adapter_mapping.keys())
        return {
            'object': 'list',
            'data': [{
                'id': name,
                'object': 'model',
                'created': created,
                'owned_by': owned_by
            } for name in names],
        }

    @app.post('/v1/chat/completions')
    async def chat_completions(request: Request):
        return await _handle_chat(await request.json(), request, context)

    @app.post('/v1/completions')
    async def completions(request: Request):
        """Text completion, expressed as a single-user-turn chat and converted back.

        The prompt is NOT run through the chat template: a completion request means "continue this
        text", and wrapping it in chat markup would change what the model sees.
        """
        return await _handle_completion(await request.json(), request, context)

    @app.post('/v1/embeddings')
    async def embeddings(request: Request):
        return await _handle_embeddings(await request.json(), request, context)

    @app.post('/infer/')
    async def infer(request: Request):
        return await _handle_rollout_infer(await request.json(), request, context)

    return app, sampler


class _ServerContext:
    """Everything the handlers need, gathered so each handler takes one object instead of twelve."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _lifespan_factory(stats: '_Stats', log_interval: float, executor: ThreadPoolExecutor):
    """Start the throughput logger on startup and stop it (plus the pool) on shutdown."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        task = None
        if log_interval > 0:
            task = asyncio.create_task(_log_stats_forever(stats, log_interval))
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
            # Not wait=True: a request still generating would hold shutdown open for minutes.
            executor.shutdown(wait=False)

    return lifespan


async def _log_stats_forever(stats: '_Stats', interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        snapshot = stats.compute()
        if snapshot:
            logger.info(f'infer_stats: {snapshot}')


class _Stats:
    """Request/token counters behind the periodic throughput line, i.e. legacy's ``InferStats``.

    Deliberately plain arithmetic under no lock: the counters are only ever incremented from the event
    loop thread (handlers await their generation, then update), so there is no race to protect, and a
    lock on the hot path for a log line would be the wrong trade.
    """

    def __init__(self):
        self.start = time.perf_counter()
        self.requests = 0
        self.prompt_tokens = 0
        self.generated_tokens = 0

    def update(self, prompt_tokens: int, generated_tokens: int) -> None:
        self.requests += 1
        self.prompt_tokens += prompt_tokens
        self.generated_tokens += generated_tokens

    def compute(self) -> Dict[str, float]:
        runtime = time.perf_counter() - self.start
        if not self.requests or runtime <= 0:
            return {}
        return {
            'num_requests': self.requests,
            'num_prompt_tokens': self.prompt_tokens,
            'num_generated_tokens': self.generated_tokens,
            'runtime': round(runtime, 2),
            'samples/s': round(self.requests / runtime, 3),
            'tokens/s': round(self.generated_tokens / runtime, 3),
        }


class _RequestLog:
    """Append ``{request, response}`` pairs to jsonl, and optionally mirror them to the logger."""

    def __init__(self, path: Optional[str], verbose: bool):
        self.path = path
        self.verbose = verbose
        if path:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

    def record(self, request_body: Dict[str, Any], response: Any) -> None:
        if not self.path and not self.verbose:
            return
        entry = {'request': _inline_media(request_body), 'response': response}
        if self.verbose:
            logger.info(f'request_info: {entry}')
        if self.path:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + '\n')


def _inline_media(body: Any) -> Any:
    """Rewrite local media paths in a logged body as base64 data URIs.

    A log line saying ``/tmp/x.jpg`` is worthless once that file is gone, and the whole point of the
    request log is to be replayable later. URLs and existing base64 are passed through by
    ``to_base64``, so only genuine local files are inlined.
    """
    if not isinstance(body, dict) or not isinstance(body.get('messages'), list):
        return body

    from swift.infer_engine.protocol import MultiModalRequestMixin

    messages = []
    for message in body['messages']:
        content = message.get('content')
        if not isinstance(content, list):
            messages.append(message)
            continue
        blocks = []
        for block in content:
            kind = block.get('type') if isinstance(block, dict) else None
            if kind in ('image', 'image_url') and isinstance(block, dict):
                block = dict(block)
                for key in ('image', 'url', 'path', 'image_url'):
                    if isinstance(block.get(key), str):
                        encoded = MultiModalRequestMixin.to_base64(block[key])
                        block[key] = encoded if encoded.startswith(('http', 'data:')) else \
                            f'data:image/jpg;base64,{encoded}'
                        break
            blocks.append(block)
        messages.append({**message, 'content': blocks})
    return {**body, 'messages': messages}


async def _handle_chat(body: Dict[str, Any], request: Any, ctx: _ServerContext):
    """The /v1/chat/completions body, lifted out of the route so it is testable without an app."""
    from fastapi.responses import JSONResponse, StreamingResponse

    denied = _check_api_key(request, ctx.api_key)
    if denied is not None:
        return denied
    try:
        trajectory, params, stream = _parse_request(body, ctx)
        adapter_path = _resolve_adapter(body.get('model'), ctx)
    except ValueError as e:
        return JSONResponse(status_code=400, content=_error(str(e)))
    except LookupError as e:
        return JSONResponse(status_code=404, content=_error(str(e)))

    request_id = f'chatcmpl-{uuid.uuid4().hex[:24]}'
    if stream:
        return StreamingResponse(
            _stream_sse(ctx, trajectory, params, adapter_path, request_id),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            },
        )
    try:
        responses = await _sample_one(ctx, trajectory, params, adapter_path)
    except Exception as e:  # noqa: BLE001 -- a bad request must not take the server down
        logger.exception('sample() failed')
        return JSONResponse(status_code=500, content=_error(str(e), 'server_error'))

    payload = _to_chat_response(responses[0], ctx.model_name, request_id)
    ctx.stats.update(len(responses[0].prompt_token_ids or []), payload['usage']['completion_tokens'])
    ctx.request_log.record(body, payload)
    return payload


async def _handle_completion(body: Dict[str, Any], request: Any, ctx: _ServerContext):
    """/v1/completions: generate from a raw prompt and answer in completion shape."""
    from fastapi.responses import JSONResponse

    denied = _check_api_key(request, ctx.api_key)
    if denied is not None:
        return denied
    prompt = body.get('prompt')
    if not isinstance(prompt, str) or not prompt:
        return JSONResponse(
            status_code=400,
            content=_error('"prompt" must be a non-empty string. A list of prompts or of token ids is '
                           'not supported; send one request per prompt.'))
    try:
        adapter_path = _resolve_adapter(body.get('model'), ctx)
        params = _sampling_params_from_body(body, ctx)
    except ValueError as e:
        return JSONResponse(status_code=400, content=_error(str(e)))
    except LookupError as e:
        return JSONResponse(status_code=404, content=_error(str(e)))

    # Pre-encoded input, not a Trajectory: the chat template must not touch a completion prompt.
    tokenizer = _tokenizer(ctx)
    feature = {'input_ids': tokenizer.encode(prompt, add_special_tokens=True)}
    request_id = f'cmpl-{uuid.uuid4().hex[:24]}'
    try:
        responses = await _sample_one(ctx, feature, params, adapter_path)
    except Exception as e:  # noqa: BLE001
        logger.exception('sample() failed')
        return JSONResponse(status_code=500, content=_error(str(e), 'server_error'))

    response = responses[0]
    choices = [{
        'index': index,
        'text': sequence.decoded or '',
        'finish_reason': _finish_reason(sequence),
        'logprobs': None,
    } for index, sequence in enumerate(response.sequences)]
    completion_tokens = sum(len(sequence.tokens or []) for sequence in response.sequences)
    prompt_tokens = len(response.prompt_token_ids or feature['input_ids'])
    payload = {
        'id': request_id,
        'object': 'text_completion',
        'created': int(time.time()),
        'model': ctx.model_name,
        'choices': choices,
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        },
    }
    ctx.stats.update(prompt_tokens, completion_tokens)
    ctx.request_log.record(body, payload)
    return payload


async def _handle_embeddings(body: Dict[str, Any], request: Any, ctx: _ServerContext):
    """/v1/embeddings: a pooling forward, which is why it does not go through the sampler.

    Only available when the server was started for an embedding model. Refusing loudly beats returning
    hidden states from a causal LM, which would look like embeddings and rank nothing correctly.
    """
    from fastapi.responses import JSONResponse

    denied = _check_api_key(request, ctx.api_key)
    if denied is not None:
        return denied
    if (ctx.model_config.task_type or 'causal_lm') != 'embedding':
        return JSONResponse(
            status_code=400,
            content=_error('This server was started for task_type='
                           f'{ctx.model_config.task_type or "causal_lm"!r}, so it cannot produce embeddings. '
                           'Start a second deployment with ModelConfig(task_type="embedding").'))

    inputs = body.get('input')
    if isinstance(inputs, str):
        inputs = [inputs]
    if not inputs or not all(isinstance(text, str) for text in inputs):
        return JSONResponse(status_code=400, content=_error('"input" must be a string or a list of strings.'))

    try:
        vectors = await ctx.off_loop(_embed, ctx, inputs)
    except Exception as e:  # noqa: BLE001
        logger.exception('embedding forward failed')
        return JSONResponse(status_code=500, content=_error(str(e), 'server_error'))

    payload = {
        'object': 'list',
        'model': ctx.model_name,
        'data': [{
            'object': 'embedding',
            'index': index,
            'embedding': vector
        } for index, vector in enumerate(vectors)],
        'usage': {
            'prompt_tokens': 0,
            'total_tokens': 0
        },
    }
    ctx.request_log.record(body, {'num_embeddings': len(vectors)})
    return payload


def _embed(ctx: _ServerContext, texts: List[str]) -> List[List[float]]:
    """Lazily build the pooling model on first use and run it. Blocking; called off the event loop."""
    from swift.dev.builders import build_model
    from swift.dev.config import DistributedConfig

    if getattr(ctx, 'embedding_model', None) is None:
        ctx.embedding_model = build_model(ctx.model_config, DistributedConfig())
    features = [{'messages': [{'role': 'user', 'content': text}]} for text in texts]
    outputs = ctx.embedding_model.forward_only(inputs=features, task='embedding')
    tensor = outputs['embedding'] if isinstance(outputs, dict) and 'embedding' in outputs else outputs
    return tensor.tolist() if hasattr(tensor, 'tolist') else list(tensor)


async def _handle_rollout_infer(body: Any, request: Any, ctx: _ServerContext):
    """``/infer/``: batch inference for RL rollout, returning token ids and logprobs.

    Differs from /v1/chat/completions in what it returns, not in how it generates: rollout needs the
    token ids and per-token logprobs that produced the text, because the trainer recomputes ratios
    against them. ``max_tokens=0`` is honoured as "score the prompt, do not generate", which is
    twinkle's ``logprobs_only`` path.

    This is NOT the full legacy rollout contract. Legacy's ``SwiftRolloutDeploy`` also owns weight
    synchronisation and the placeholder-token protocol used by GRPO; those live in
    ``swift.dev.rollout`` and are not reproduced here. Use this endpoint for rollout generation, not as
    a drop-in for that server.
    """
    from fastapi.responses import JSONResponse

    denied = _check_api_key(request, ctx.api_key)
    if denied is not None:
        return denied
    requests = body if isinstance(body, list) else [body]
    try:
        trajectories = [_trajectory_from_messages(item['messages'], ctx) for item in requests]
        params = _sampling_params_from_body(requests[0], ctx, want_logprobs=True)
        adapter_paths = [_resolve_adapter(item.get('model'), ctx) for item in requests]
    except KeyError:
        return JSONResponse(status_code=400, content=_error('every entry needs "messages".'))
    except ValueError as e:
        return JSONResponse(status_code=400, content=_error(str(e)))
    except LookupError as e:
        return JSONResponse(status_code=404, content=_error(str(e)))

    try:
        responses = await ctx.off_loop(_sample_many, ctx, trajectories, params, adapter_paths)
    except Exception as e:  # noqa: BLE001
        logger.exception('rollout sample() failed')
        return JSONResponse(status_code=500, content=_error(str(e), 'server_error'))

    return [{
        'prompt_token_ids': response.prompt_token_ids,
        'prompt_logprobs': response.prompt_logprobs,
        'sequences': [{
            'token_ids': sequence.tokens,
            'text': sequence.decoded,
            'logprobs': sequence.logprobs,
            'finish_reason': _finish_reason(sequence),
        } for sequence in response.sequences],
    } for response in responses]


async def _sample_one(ctx: _ServerContext, feature: Dict[str, Any], params: Any, adapter_path: Optional[str]):
    """One request, off the event loop, with the adapter carried per request."""
    return await ctx.off_loop(_sample_many, ctx, [feature], params, [adapter_path])


def _sample_many(ctx: _ServerContext, features: List[Dict[str, Any]], params: Any,
                 adapter_paths: List[Optional[str]]):
    """Blocking ``sample()``. Uses ``adapter_paths=`` only when an adapter is actually involved, so a
    plain deployment never pays for the per-request machinery."""
    if any(path is not None for path in adapter_paths):
        return ctx.sampler.sample(features, params, adapter_paths=adapter_paths)
    return ctx.sampler.sample(features, params)


def _resolve_adapter(model: Optional[str], ctx: _ServerContext) -> Optional[str]:
    """``request.model`` -> adapter path, or None for the base model.

    Raises:
        LookupError: the name is neither the base model nor a configured adapter. Returning the base
            model for an unknown name would silently serve the wrong weights.
    """
    if model is None or model == ctx.model_name:
        return None
    if model in ctx.adapter_mapping:
        return ctx.adapter_mapping[model]
    known = [ctx.model_name] + list(ctx.adapter_mapping)
    raise LookupError(f'Unknown model {model!r}; this server serves {known}.')


def _check_api_key(request: Any, api_key: Optional[str]):
    """None when the request may proceed, a 401 response otherwise."""
    from fastapi.responses import JSONResponse

    if api_key is None:
        return None
    header = ''
    if request is not None and getattr(request, 'headers', None) is not None:
        header = request.headers.get('authorization') or ''
    if not header.startswith('Bearer ') or header[len('Bearer '):] != api_key:
        return JSONResponse(
            status_code=401, content=_error('Missing or invalid API key; send "Authorization: Bearer <key>".'))
    return None


def _parse_request(body: Dict[str, Any], ctx: _ServerContext):
    """OpenAI chat body -> ``(trajectory, SamplingParams, stream)``."""
    messages = body.get('messages')
    if not messages:
        raise ValueError('"messages" is required and must be non-empty.')
    return _trajectory_from_messages(messages, ctx), _sampling_params_from_body(body, ctx), bool(body.get('stream'))


def _trajectory_from_messages(messages: List[Dict[str, Any]], ctx: _ServerContext) -> Dict[str, Any]:
    """Apply the server's default system prompt, replacing rather than stacking (as ``run_infer`` does)."""
    system = getattr(ctx.template_config, 'system', None)
    if system and not any(message.get('role') == 'system' for message in messages):
        messages = [{'role': 'system', 'content': system}] + list(messages)
    return {'messages': list(messages)}


def _sampling_params_from_body(body: Dict[str, Any], ctx: _ServerContext, want_logprobs: bool = False) -> Any:
    """Request overrides on top of the server's GenerationConfig defaults."""
    overrides: Dict[str, Any] = {}
    max_tokens = body.get('max_completion_tokens', body.get('max_tokens'))
    if max_tokens is not None:
        overrides['max_tokens'] = max_tokens
    # OpenAI name -> twinkle SamplingParams name. Fields absent here (presence_penalty,
    # frequency_penalty, tools, response_format) have no SamplingParams counterpart and are ignored.
    for openai_name, twinkle_name in (('temperature', 'temperature'), ('top_p', 'top_p'), ('seed', 'seed'),
                                      ('n', 'num_samples')):
        if body.get(openai_name) is not None:
            overrides[twinkle_name] = body[openai_name]
    if body.get('stop'):
        stop = body['stop']
        overrides['stop'] = [stop] if isinstance(stop, str) else list(stop)
    if body.get('logprobs') or want_logprobs:
        top_logprobs = body.get('top_logprobs') or 0
        if ctx.max_logprobs is not None and top_logprobs > ctx.max_logprobs:
            raise ValueError(f'top_logprobs={top_logprobs} exceeds this server\'s max_logprobs='
                             f'{ctx.max_logprobs}.')
        overrides['logprobs'] = top_logprobs
    return ctx.to_sampling_params(ctx.generation_config, **overrides)


def _tokenizer(ctx: _ServerContext):
    """The sampler's tokenizer, for the endpoints that must bypass the chat template."""
    if getattr(ctx, '_tokenizer_cache', None) is None:
        from swift.model import get_model_processor

        _, processor = get_model_processor(ctx.model_config.model, load_model=False)
        ctx._tokenizer_cache = getattr(processor, 'tokenizer', processor)
    return ctx._tokenizer_cache


def _finish_reason(seq: Any) -> str:
    return _FINISH_REASONS.get(getattr(seq, 'stop_reason', None), 'stop')


async def _stream_sse(ctx: _ServerContext, trajectory: Dict[str, Any], params: Any, adapter_path: Optional[str],
                      request_id: str):
    """Server-sent events for a streaming chat completion.

    ``next()`` on the generator is what actually blocks (it drives the engine), so it is pumped through
    the thread pool one token at a time rather than iterated on the event loop.
    """
    created = int(time.time())
    yield _chunk(request_id, created, ctx.model_name, {'role': 'assistant', 'content': ''}, None)
    generator = ctx.sampler.sample_stream(trajectory, params, adapter_path=adapter_path)
    generated = 0
    try:
        while True:
            item = await ctx.off_loop(next, generator, None)
            if item is None:
                break
            delta_text, finish_reason = item
            if delta_text:
                generated += 1
                yield _chunk(request_id, created, ctx.model_name, {'content': delta_text}, None)
            if finish_reason:
                yield _chunk(request_id, created, ctx.model_name, {}, _FINISH_REASONS.get(finish_reason, 'stop'))
                break
    except Exception as e:  # noqa: BLE001 -- the stream has already started, so 500 is not available
        logger.exception('sample_stream failed')
        yield f'data: {json.dumps(_error(str(e), "server_error"))}\n\n'
    ctx.stats.update(0, generated)
    yield 'data: [DONE]\n\n'


def _chunk(request_id: str, created: int, model: str, delta: Dict[str, Any], finish_reason: Optional[str]) -> str:
    payload = {
        'id': request_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': model,
        'choices': [{
            'index': 0,
            'delta': delta,
            'finish_reason': finish_reason
        }],
    }
    return f'data: {json.dumps(payload, ensure_ascii=False)}\n\n'


def _to_chat_response(response: Any, model: str, request_id: str) -> Dict[str, Any]:
    """A twinkle SampleResponse in OpenAI chat-completion shape."""
    choices = []
    completion_tokens = 0
    for index, sequence in enumerate(response.sequences):
        completion_tokens += len(sequence.tokens or [])
        choices.append({
            'index': index,
            'message': {
                'role': 'assistant',
                'content': sequence.decoded or ''
            },
            'finish_reason': _finish_reason(sequence),
        })
    prompt_tokens = len(response.prompt_token_ids or [])
    return {
        'id': request_id,
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model,
        'choices': choices,
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        },
    }


def _error(message: str, kind: str = 'invalid_request_error') -> Dict[str, Any]:
    return {'error': {'message': message, 'type': kind}}


def _merge_single_adapter(model_config: ModelConfig, template_config: TemplateConfig, adapter_mapping: Dict[str, str]):
    """Merge the one adapter into the base weights; returns the new config and an empty mapping."""
    import dataclasses

    from swift.dev.config import TunerConfig
    from swift.dev.recipe.merge_lora import run_merge_lora

    if len(adapter_mapping) > 1:
        raise ValueError(f'merge_lora=True cannot serve {len(adapter_mapping)} adapters: merging bakes one '
                         'adapter into the weights, after which per-request routing is impossible. Drop '
                         'merge_lora to route between them, or start one deployment per adapter.')
    adapter = next(iter(adapter_mapping.values()))
    merged = run_merge_lora(
        model_config, TunerConfig(adapters=[adapter]), template_config=template_config, device_map='cpu')
    logger.info(f'run_deploy: merged {adapter} into {merged}')
    return dataclasses.replace(model_config, model=merged), {}


def run_deploy_process(*args, port: int = 8000, timeout: float = 300.0, **kwargs):
    """Context manager that serves in a subprocess and yields the base URL once it answers.

    Legacy's ``run_deploy``. Kept because evaluation harnesses want a live endpoint inside a Python
    block without giving up the current process, and because a server in-process would have to share
    the caller's event loop and GPU.

    Spawn, not fork: the parent may already hold CUDA context, and forking that produces a child whose
    CUDA state is unusable in ways that surface much later.
    """
    import multiprocessing
    from contextlib import contextmanager

    @contextmanager
    def _manager():
        process = multiprocessing.get_context('spawn').Process(
            target=run_deploy, args=args, kwargs={
                **kwargs, 'port': port
            }, daemon=True)
        process.start()
        try:
            _wait_until_accessible(port, timeout, process)
            yield f'http://127.0.0.1:{port}/v1'
        finally:
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()

    return _manager()


def _wait_until_accessible(port: int, timeout: float, process) -> None:
    """Poll the port until /health answers, failing fast if the child dies first."""
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process.is_alive():
            raise RuntimeError(f'deploy subprocess exited with code {process.exitcode} before the port opened; '
                               'its traceback is on this process\'s stderr.')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                return
        time.sleep(1.0)
    raise TimeoutError(f'deploy did not become accessible on port {port} within {timeout}s.')
