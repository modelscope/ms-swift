"""run_deploy: an OpenAI-compatible server, as a thin wrapper over twinkle-server.

dev counterpart of legacy ``swift deploy``. It no longer hand-rolls a FastAPI app: it turns dev's
Configs into a twinkle-server ``ServerConfig`` (a gateway app + a sampler app, plus a data-plane app
for token-in-token-out) and hands it to ``launch_server``, which runs it on Ray Serve. The OpenAI
surface -- ``/v1/chat/completions``, ``/v1/completions``, ``/v1/embeddings``, ``/v1/models``,
``/v1/infer``, ``/health`` -- is served by twinkle's gateway, so this module's only job is the config
translation and the Ray bring-up. That mirrors how ``run_infer`` delegates the engine to twinkle's
Sampler instead of reimplementing it.

Layering (mirrors the infer module): ``cli/deploy.py`` parses and validates, ``config/deploy_config.py``
carries the knobs, this recipe assembles the ``ServerConfig``, and ``builders/`` supplies the reusable
config -> object glue (``build_engine_args`` / ``resolve_twinkle_template`` / ``build_device_mesh_if_dp``).

Two things are deliberately the client's job, not the server's:

- **Multi-turn** stays client-side. The server answers one request; a conversation is a loop the caller
  drives, exactly as an OpenAI client does.
- **Generation defaults** are per-request. twinkle's gateway translates each request's own sampling
  knobs, so a server-side ``GenerationConfig`` default is not injected here (see the warning in
  :func:`run_deploy`); send ``temperature`` / ``top_p`` / ... in the request body.

**token-in-token-out** (rollout training) needs no extra endpoint: the sampler already serves
``/twinkle/sample`` (returns each sequence's ``new_input_feature``) and, with ``enable_data_plane``,
``/twinkle/sample_to_data_plane`` (returns a ``DataRef``). ``swift deploy`` just deploys those apps; the
training loop that consumes them lives in the client (see ``examples`` for a rollout client sketch).
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (DeployConfig, DistributedConfig, GenerationConfig, ModelConfig, QuantizeConfig,
                                  TemplateConfig)

logger = get_logger()


def run_deploy(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    backend: Literal['vllm', 'sglang', 'transformers'] = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    adapter_mapping: Optional[Dict[str, str]] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    distributed_config: Optional[DistributedConfig] = None,
    deploy_config: Optional[DeployConfig] = None,
) -> None:
    """Serve the model until interrupted, by launching twinkle-server on Ray Serve.

    Args:
        model_config / template_config: the model to serve and how to render its chat. ``template_config``
            is resolved to a *serializable* twinkle template spec (a class name + kwargs) rather than the
            in-process ``ShiftedTemplate`` instance, because the sampler runs inside a Ray actor.
        generation_config: accepted for parity with ``run_infer``; its sampling defaults are NOT applied
            server-side (see the module docstring). A warning names any that were set so nothing is
            silently dropped.
        backend: the sampler engine. ``vllm``, ``sglang`` and ``transformers`` (``pt``) are all served.
            With ``enable_data_plane``, vLLM and sglang use their non-blocking ``_async`` sampler
            (``vllm_async`` / ``sglang_async``); transformers has no async variant, so
            ``transformers`` + ``enable_data_plane`` raises.
        engine_args: forwarded to the engine (from ``build_engine_args``); ``max_logprobs`` and the LoRA
            sizing for ``adapter_mapping`` are merged in here.
        adapter_mapping: ``{served name: adapter path}``. Each name is listed in ``/v1/models`` and the
            engine is built with LoRA enabled and sized for them. With ``deploy_config.merge_lora`` a
            single adapter is baked into the base weights instead (and the mapping cleared).
        quantize_config: a load-time ``quant_method`` is a transformers-only setting the Ray Serve sampler
            cannot apply, so it is rejected with a pointer to the engine's own quantization args.
        distributed_config: ``mode``/``nproc_per_node`` drive the Ray bring-up and the sampler's device
            group / data-parallel mesh.
        deploy_config: the serving surface -- address, TLS, auth, identity, route prefix, data plane,
            persistence and scaling.
    """
    from twinkle.server import launch_server

    from swift.dev.config import DeployConfig as _DeployConfig
    from swift.dev.config import DistributedConfig as _DistributedConfig

    deploy_config = deploy_config or _DeployConfig()
    distributed_config = distributed_config or _DistributedConfig()

    _validate(deploy_config, quantize_config)
    _warn_unapplied_generation_defaults(generation_config)
    if deploy_config.api_key is None:
        logger.warning('run_deploy is starting WITHOUT api_key: anyone who can reach the port can use the '
                       'model. Set DeployConfig.api_key before exposing it beyond localhost.')

    _ensure_ray(distributed_config, deploy_config)
    config = build_server_config(
        model_config,
        template_config,
        generation_config,
        backend=backend,
        engine_args=engine_args,
        adapter_mapping=adapter_mapping,
        quantize_config=quantize_config,
        distributed_config=distributed_config,
        deploy_config=deploy_config,
    )
    # Blocks until SIGINT/SIGTERM; twinkle's launcher installs the handlers and calls serve.shutdown().
    launch_server(config=config, ray_namespace=deploy_config.ray_namespace)


def build_server_config(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    backend: str = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    adapter_mapping: Optional[Dict[str, str]] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    distributed_config: Optional[DistributedConfig] = None,
    deploy_config: Optional[DeployConfig] = None,
) -> Any:
    """dev Configs -> a twinkle-server ``ServerConfig`` (gateway + sampler [+ data plane]).

    Separate from :func:`run_deploy` so the assembled config can be inspected, dumped to YAML, or handed
    to ``launch_server`` by a caller that manages Ray itself.

    The gateway is mounted at ``deploy_config.route_prefix`` (``/v1`` by default) and the sampler at
    ``{route_prefix}/sampler/{public_name}``; twinkle's proxy rebuilds that exact path from the request's
    model, so the public name and the sampler's route must agree. ``ServerArgs.route_prefix`` is set to
    the same ``route_prefix`` for that reason.
    """
    from twinkle.server.config import (ApplicationSpec, DataPlaneArgs, HttpOptions, PersistenceConfig, SamplerArgs,
                                       ServerArgs, ServerConfig)

    from swift.dev.builders import build_device_mesh_if_dp, resolve_twinkle_template
    from swift.dev.builders.sampler import _derive_sampler_type, _enable_lora
    from swift.dev.config import DeployConfig as _DeployConfig
    from swift.dev.config import DistributedConfig as _DistributedConfig

    deploy_config = deploy_config or _DeployConfig()
    distributed_config = distributed_config or _DistributedConfig()
    adapter_mapping = dict(adapter_mapping or {})

    if deploy_config.merge_lora and adapter_mapping:
        model_config, adapter_mapping = _merge_single_adapter(model_config, template_config, adapter_mapping)
    if model_config.model is None:
        raise ValueError('ModelConfig.model is required to deploy (it is the model id/path the sampler loads).')

    public_name = deploy_config.served_model_name or os.path.basename(str(model_config.model).rstrip('/'))
    sampler_type = deploy_config.sampler_type or _derive_sampler_type(backend, deploy_config.enable_data_plane)

    sampler_engine_args = _sampler_engine_args(backend, engine_args, adapter_mapping, deploy_config, _enable_lora)
    device_group, device_mesh, nproc = _device_spec(distributed_config, build_device_mesh_if_dp)
    template_spec = resolve_twinkle_template(template_config, model_config)

    route_prefix = deploy_config.route_prefix
    # 127.0.0.1 is correct here, not a placeholder: ``data_plane_url`` is what the *sampler actor* dials to
    # reach the data-plane app, and Ray Serve runs one HTTP proxy per node listening on that node's
    # loopback. The sampler and the proxy it talks to are co-located, so loopback routes to the data-plane
    # application without exposing it on the external interface. The scheme follows the TLS config so an
    # HTTPS deployment does not have its sampler silently dial plaintext at its own proxy.
    scheme = _url_scheme(deploy_config)
    data_plane_url = (f'{scheme}://127.0.0.1:{deploy_config.port}{route_prefix}/data-plane'
                      if deploy_config.enable_data_plane else None)

    supported_models: List[str] = [public_name] + list(adapter_mapping.keys())
    applications: List[Any] = [
        ApplicationSpec(
            name='server',
            route_prefix=route_prefix,
            import_path='server',
            args=ServerArgs(
                supported_models=supported_models,
                route_prefix=route_prefix,
                api_key=deploy_config.api_key,
                owned_by=deploy_config.owned_by,
            ),
            deployments=[_deployment('GatewayServer', deploy_config)],
        ),
    ]
    # Three Ray Serve applications make up the deployment, each mounted at its own route_prefix:
    #   - 'server' (the gateway): the OpenAI surface (/v1/chat/completions, /v1/models, ...) and the proxy
    #     that forwards token-in-token-out calls to the sampler app below.
    #   - 'data-plane' (only when enable_data_plane): stores rollout groups server-side and hands back a
    #     DataRef, so a trainer can consume generations without shipping tokens over the wire.
    #   - 'sampler-<name>': the engine replica that actually runs the model, mounted where the gateway's
    #     proxy rebuilds its target URL ({route_prefix}/sampler/{public_name}).
    if deploy_config.enable_data_plane:
        applications.append(
            ApplicationSpec(
                name='data-plane',
                route_prefix=f'{route_prefix}/data-plane',
                import_path='data_plane',
                args=DataPlaneArgs(config={'backend': {'SimpleStorage': {'num_data_storage_units': 2}}}),
                deployments=[_deployment('DataPlaneManagement', deploy_config)],
            ))
    applications.append(
        ApplicationSpec(
            name=f'sampler-{public_name.replace("/", "_")}',
            route_prefix=f'{route_prefix}/sampler/{public_name}',
            import_path='sampler',
            args=SamplerArgs(
                model_id=str(model_config.model),
                nproc_per_node=nproc,
                device_group=device_group,
                device_mesh=device_mesh,
                sampler_type=sampler_type,
                engine_args=sampler_engine_args or None,
                data_plane_url=data_plane_url,
                template=template_spec,
            ),
            deployments=[_deployment('SamplerManagement', deploy_config)],
        ))

    return ServerConfig(
        ray_namespace=deploy_config.ray_namespace,
        http_options=HttpOptions(
            host=deploy_config.host,
            port=deploy_config.port,
            ssl_keyfile=deploy_config.ssl_keyfile,
            ssl_certfile=deploy_config.ssl_certfile,
        ),
        persistence=_persistence(deploy_config, PersistenceConfig),
        applications=applications,
    )


def _url_scheme(deploy_config: DeployConfig) -> str:
    """``https`` when TLS is configured, else ``http`` -- the one place the scheme is decided.

    Every URL this module builds for the *running* server (the data-plane dial target, the readiness
    probe, and the base URL yielded to callers) must agree with how Ray Serve's proxy was started, which
    ``HttpOptions`` drives off ``ssl_certfile``. Deriving the scheme from that same field keeps an HTTPS
    deployment from being handed plaintext URLs.
    """
    return 'https' if deploy_config.ssl_certfile else 'http'


def _validate(deploy_config: DeployConfig, quantize_config: Optional[QuantizeConfig]) -> None:
    """Fail loudly on knob combinations the twinkle-server path cannot honour, before anything starts."""
    if bool(deploy_config.ssl_keyfile) != bool(deploy_config.ssl_certfile):
        raise ValueError('HTTPS needs both ssl_keyfile and ssl_certfile; one alone would start a plain HTTP '
                         'server while the operator believed it was encrypted.')
    # An explicitly pinned sampler_type bypasses _derive_sampler_type, so guard data-plane compatibility
    # here: sample_to_data_plane needs the sampler's non-blocking submit_generation, which only the
    # *_async variants provide. A plain vllm/sglang/torch/mock with the data plane would start a server
    # whose token-in-token-out endpoint 503s on every call.
    if deploy_config.enable_data_plane and deploy_config.sampler_type not in (None, 'vllm_async', 'sglang_async'):
        raise ValueError(f'sampler_type={deploy_config.sampler_type!r} cannot serve the data plane: '
                         'sample_to_data_plane needs the sampler\'s non-blocking submit_generation, which '
                         'only "vllm_async" and "sglang_async" provide. Use one of those, or leave '
                         'sampler_type=None to derive it from the backend, or drop enable_data_plane.')
    quant_method = getattr(quantize_config, 'quant_method', None)
    if quant_method is not None:
        raise ValueError(f'quant_method={quant_method!r} is a transformers load-time setting and cannot be '
                         'applied by the Ray Serve sampler. Deploy a pre-quantized checkpoint, or pass the '
                         "engine's own quantization arg via engine_args (e.g. vllm_quantization).")


def _warn_unapplied_generation_defaults(generation_config: Optional[GenerationConfig]) -> None:
    """Name any server-side generation defaults that the twinkle-server path does not inject.

    The gateway translates each request's own sampling knobs, so a ``GenerationConfig`` default set here
    would be silently ignored. Warning (only when something was actually set) keeps that visible instead
    of letting a client get a different distribution than the operator configured.
    """
    if generation_config is None:
        return
    watched = ('temperature', 'top_p', 'top_k', 'max_new_tokens', 'repetition_penalty')
    set_fields = [name for name in watched if getattr(generation_config, name, None) is not None]
    if generation_config.stop_words:
        set_fields.append('stop_words')
    if set_fields:
        logger.warning(f'swift deploy serves through twinkle-server, which applies sampling per request: the '
                       f'GenerationConfig default(s) {set_fields} are NOT injected server-side. Send them in '
                       'each request body (temperature/top_p/max_tokens/...) instead.')


def _ensure_ray(distributed_config: DistributedConfig, deploy_config: DeployConfig) -> None:
    """Bring up Ray so ``launch_server`` can deploy onto it.

    ``launch_server._init_ray`` only calls ``ray.init(address='auto')`` when Ray is not already up, and
    ``address='auto'`` needs a cluster that already exists. So:

    - ``mode='ray'`` or ``ray_address='auto'``: leave Ray alone and let the launcher attach to the
      existing cluster.
    - otherwise (the single-command default): start a local Ray head here, seeding workers with the same
      runtime_env the launcher would (``build_ray_runtime_env``), so one ``swift deploy`` serves without a
      pre-existing cluster.
    """
    import ray

    if ray.is_initialized():
        return
    if distributed_config.mode == 'ray' or deploy_config.ray_address == 'auto':
        return

    from twinkle.server.launcher import build_ray_runtime_env

    init_kwargs: Dict[str, Any] = {
        'namespace': deploy_config.ray_namespace,
        'runtime_env': build_ray_runtime_env(),
        'ignore_reinit_error': True,
    }
    address = deploy_config.ray_address
    if address and address != 'local':
        init_kwargs['address'] = address
    ray.init(**init_kwargs)
    logger.info(f'run_deploy started a local Ray head (namespace={deploy_config.ray_namespace}).')


def _sampler_engine_args(
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    adapter_mapping: Dict[str, str],
    deploy_config: DeployConfig,
    enable_lora,
) -> Dict[str, Any]:
    """The engine args for the server sampler: the caller's, plus max_logprobs and LoRA sizing.

    ``max_logprobs`` is a vLLM engine flag (the server-side ceiling on a request's ``top_logprobs``); the
    LoRA machinery is sized for ``adapter_mapping`` via the same helper the in-process sampler uses, so
    both paths reserve adapter slots identically.
    """
    resolved = dict(engine_args or {})
    # max_logprobs is a vLLM *engine-startup* flag (the server-side ceiling on a request's top_logprobs),
    # so it is set once here. sglang has no startup equivalent -- its logprobs are configured per request
    # (return_logprob / top_logprobs_num / logprob_start_len), so there is nothing to reserve at build time
    # and injecting max_logprobs would be an unknown kwarg. Hence vllm-only.
    if backend == 'vllm' and deploy_config.max_logprobs:
        resolved.setdefault('max_logprobs', deploy_config.max_logprobs)
    if adapter_mapping:
        enable_lora(resolved, backend, list(adapter_mapping.values()))
    return resolved


def _device_spec(distributed_config: DistributedConfig, build_device_mesh_if_dp) -> tuple:
    """``(device_group, device_mesh, nproc_per_node)`` dicts for the sampler deployment.

    ``ranks`` is the GPU count handed to the sampler's device group; Ray sets ``CUDA_VISIBLE_DEVICES`` for
    the actor from it, which is what a vLLM engine without an explicit ``tensor_parallel_size`` falls back
    to, so ``ranks=nproc`` gives tensor parallelism across all allocated GPUs by default. ``dp_size`` comes
    from the shared DP-mesh builder and is 1 unless data parallelism was actually requested.
    """
    nproc = distributed_config.nproc_per_node or 1
    mesh = build_device_mesh_if_dp(distributed_config)
    dp_size = int(getattr(mesh, 'data_world_size', 1)) if mesh is not None else 1
    device_group = {'name': 'sampler', 'ranks': nproc, 'device_type': 'cuda'}
    device_mesh = {'device_type': 'cuda', 'dp_size': dp_size}
    return device_group, device_mesh, nproc


def _deployment(name: str, deploy_config: DeployConfig) -> Dict[str, Any]:
    """One Ray Serve deployment options block, from the scaling/concurrency knobs.

    The launcher strips ``name`` and applies the rest via ``.options(...)``, so ``num_replicas`` /
    ``max_ongoing_requests`` / ``autoscaling_config`` are Ray Serve's own deployment options.
    """
    options: Dict[str, Any] = {'name': name}
    if deploy_config.autoscaling:
        options['autoscaling_config'] = {
            'min_replicas': 1,
            'max_replicas': max(1, deploy_config.num_replicas or 1),
            'target_ongoing_requests': deploy_config.max_concurrency,
        }
    else:
        if deploy_config.num_replicas is not None:
            options['num_replicas'] = deploy_config.num_replicas
        options['max_ongoing_requests'] = deploy_config.max_concurrency
    return options


def _persistence(deploy_config: DeployConfig, PersistenceConfig) -> Any:
    """``ServerConfig.persistence`` from the deploy knobs, or the in-memory default."""
    if not deploy_config.persistence_mode:
        return PersistenceConfig()
    return PersistenceConfig(
        mode=deploy_config.persistence_mode,
        file_path=deploy_config.persistence_file_path,
    )


def _merge_single_adapter(model_config: ModelConfig, template_config: TemplateConfig,
                          adapter_mapping: Dict[str, str]):
    """Merge the one adapter into the base weights; returns the new config and an empty mapping."""
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
    return replace(model_config, model=merged), {}


def run_deploy_process(*args, port: int = 8000, timeout: float = 300.0, **kwargs):
    """Context manager that serves in a subprocess and yields the base URL once it answers.

    Legacy's ``run_deploy``. Kept because evaluation harnesses want a live endpoint inside a Python block
    without giving up the current process, and because a server in-process would have to share the
    caller's event loop, GPU and Ray instance.

    Spawn, not fork: the parent may already hold a CUDA context (and its own Ray), and forking that
    produces a child whose state is unusable in ways that surface much later.
    """
    import multiprocessing
    from contextlib import contextmanager

    from swift.dev.config import DeployConfig

    deploy_config = kwargs.get('deploy_config') or DeployConfig()
    deploy_config = replace(deploy_config, port=port)
    kwargs['deploy_config'] = deploy_config
    scheme = _url_scheme(deploy_config)

    @contextmanager
    def _manager():
        process = multiprocessing.get_context('spawn').Process(
            target=run_deploy, args=args, kwargs=kwargs, daemon=True)
        process.start()
        try:
            _wait_until_accessible(port, timeout, process, deploy_config.route_prefix, scheme)
            yield f'{scheme}://127.0.0.1:{port}{deploy_config.route_prefix}'
        finally:
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()

    return _manager()


def _wait_until_accessible(port: int, timeout: float, process, route_prefix: str, scheme: str = 'http') -> None:
    """Poll ``{route_prefix}/models`` until it answers 200, failing fast if the child dies first.

    An HTTP probe rather than a bare socket connect: the port opens when Ray Serve's proxy is up, but the
    gateway app (and its model list) is only ready a little later, and yielding before that would hand the
    caller a URL that 404s on the first request.

    ``scheme`` comes from the same TLS config that started the proxy. For an HTTPS probe the cert is
    typically self-signed, so verification is disabled for this readiness check only -- otherwise the
    handshake error would be swallowed by the retry loop and surface as a bogus timeout.
    """
    import ssl
    import time
    import urllib.error
    import urllib.request

    url = f'{scheme}://127.0.0.1:{port}{route_prefix}/models'
    context = ssl._create_unverified_context() if scheme == 'https' else None
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process.is_alive():
            raise RuntimeError(f'deploy subprocess exited with code {process.exitcode} before the endpoint '
                               "opened; its traceback is on this process's stderr.")
        try:
            with urllib.request.urlopen(url, timeout=2.0, context=context) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f'deploy did not become accessible at {url} within {timeout}s.')
