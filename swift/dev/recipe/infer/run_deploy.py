# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (DeployConfig, DistributedConfig, GenerationConfig, ModelConfig,
                                    QuantizeConfig, TemplateConfig)

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

    ``template_config`` resolves to a serializable twinkle template spec (the sampler runs inside a
    Ray actor); ``generation_config`` is accepted for parity with ``run_infer`` but its sampling
    defaults are NOT applied server-side -- a warning names any that were set. Blocks until
    SIGINT/SIGTERM; twinkle's launcher installs the handlers and calls serve.shutdown().
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

    Separate from :func:`run_deploy` so the assembled config can be inspected, dumped to YAML, or
    handed to ``launch_server`` by a caller that manages Ray itself. The gateway is mounted at
    ``route_prefix`` and the sampler at ``{route_prefix}/sampler/{public_name}`` -- twinkle's proxy
    rebuilds that exact path from the request's model, so the two must agree.
    """
    from twinkle.server.config import (ApplicationSpec, DataPlaneArgs, HttpOptions, PersistenceConfig, SamplerArgs,
                                       ServerArgs, ServerConfig)

    from swift.dev.builders import resolve_twinkle_template
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
    device_group, device_mesh, nproc = _device_spec(distributed_config)
    template_spec = resolve_twinkle_template(template_config, model_config)

    route_prefix = deploy_config.route_prefix
    # 127.0.0.1 is deliberate: the sampler actor dials its own node's co-located Ray Serve proxy, so
    # loopback reaches the data plane without exposing it; the scheme follows TLS (no plaintext dial).
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
    # Three Ray Serve applications: 'server' (the OpenAI gateway + proxy), optional 'data-plane'
    # (server-side rollout groups via DataRef), and 'sampler-<name>' (the engine replica).
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
        persistence=PersistenceConfig(),
        applications=applications,
    )


def _url_scheme(deploy_config: DeployConfig) -> str:
    """``https`` when TLS is configured, else ``http`` -- the one place the scheme is decided.

    Every URL this module builds for the running server (the data-plane dial target, the readiness
    probe, the base URL yielded to callers) must agree with how Ray Serve's proxy was started, which
    ``HttpOptions`` drives off ``ssl_certfile``.
    """
    return 'https' if deploy_config.ssl_certfile else 'http'


def _validate(deploy_config: DeployConfig, quantize_config: Optional[QuantizeConfig]) -> None:
    """Fail loudly on knob combinations the twinkle-server path cannot honour, before anything starts."""
    if bool(deploy_config.ssl_keyfile) != bool(deploy_config.ssl_certfile):
        raise ValueError('HTTPS needs both ssl_keyfile and ssl_certfile; one alone would start a plain HTTP '
                         'server while the operator believed it was encrypted.')
    # A pinned sampler_type bypasses _derive_sampler_type: the data plane rides the non-blocking
    # submit on the async engine backends -- guard here or every token-in-token-out call 503s.
    if deploy_config.enable_data_plane and deploy_config.sampler_type not in (None, 'vllm_async', 'sglang_async'):
        raise ValueError(f'sampler_type={deploy_config.sampler_type!r} cannot serve the data plane: '
                         'sample_to_data_plane needs the sampler\'s non-blocking submit, which '
                         'only "vllm_async" and "sglang_async" provide. Use one of those, or leave '
                         'sampler_type=None to derive it from the backend, or drop enable_data_plane.')
    quant_method = getattr(quantize_config, 'quant_method', None)
    if quant_method is not None:
        raise ValueError(f'quant_method={quant_method!r} is a transformers load-time setting and cannot be '
                         'applied by the Ray Serve sampler. Deploy a pre-quantized checkpoint, or pass the '
                         "engine's own quantization arg via engine_args (e.g. vllm_quantization).")


def _warn_unapplied_generation_defaults(generation_config: Optional[GenerationConfig]) -> None:
    """Name any server-side generation defaults that the twinkle-server path does not inject.

    The gateway translates each request's own sampling knobs, so a ``GenerationConfig`` default set
    here would be silently ignored; warning (only when something was set) keeps that visible.
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

    ``launch_server._init_ray`` only attaches (``address='auto'``) to an existing cluster, so
    ``mode='ray'`` / ``ray_address='auto'`` are left alone; the single-command default starts a
    local Ray head via the launcher's own ``ensure_ray_cluster``, so worker seeding (runtime_env,
    namespace) stays single-sourced with the launcher.
    """
    if distributed_config.mode == 'ray' or deploy_config.ray_address == 'auto':
        return
    from twinkle.server.launcher import ensure_ray_cluster

    address = deploy_config.ray_address
    if not address or address == 'local':
        address = None
    ensure_ray_cluster(namespace=deploy_config.ray_namespace, address=address)
    logger.info(f'run_deploy started a local Ray head (namespace={deploy_config.ray_namespace}).')


def _sampler_engine_args(
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    adapter_mapping: Dict[str, str],
    deploy_config: DeployConfig,
    enable_lora,
) -> Dict[str, Any]:
    """The engine args for the server sampler: the caller's, plus max_logprobs and LoRA sizing."""
    resolved = dict(engine_args or {})
    # max_logprobs is a vLLM engine-startup flag (sglang has no equivalent -- its logprobs are
    # per-request), so vllm-only; LoRA sizing uses the same helper the in-process sampler uses.
    if backend == 'vllm' and deploy_config.max_logprobs:
        resolved.setdefault('max_logprobs', deploy_config.max_logprobs)
    if adapter_mapping:
        enable_lora(resolved, backend, list(adapter_mapping.values()))
    return resolved


def _device_spec(distributed_config: DistributedConfig) -> tuple:
    """``(device_group, device_mesh, nproc_per_node)`` dicts for the sampler deployment.

    ``ranks=nproc`` gives tensor parallelism across all allocated GPUs by default (Ray sets
    CUDA_VISIBLE_DEVICES from it, which a vLLM engine without explicit TP falls back to); ``dp_size``
    comes from the shared DP-mesh builder, 1 unless data parallelism was requested.
    """
    nproc = distributed_config.nproc_per_node or 1
    from swift.dev.builders import build_parallel_mesh

    mesh = build_parallel_mesh(distributed_config, none_if_single_rank=True)
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


def _merge_single_adapter(model_config: ModelConfig, template_config: TemplateConfig,
                          adapter_mapping: Dict[str, str]):
    """Merge the one adapter into the base weights; returns the new config and an empty mapping."""
    from swift.dev.config import TunerConfig
    from swift.dev.recipe.export import run_merge_lora

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

    Legacy's ``run_deploy``, kept because evaluation harnesses want a live endpoint without giving
    up the current process (an in-process server would share its event loop, GPU and Ray). Spawn,
    not fork: forking a parent that already holds a CUDA context breaks the child much later.
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

    An HTTP probe, not a bare socket connect: the port opens when Ray Serve's proxy is up, but the
    gateway is ready a little later -- yielding before that hands the caller a URL that 404s. For
    HTTPS the cert is typically self-signed, so verification is off for this probe only.
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
