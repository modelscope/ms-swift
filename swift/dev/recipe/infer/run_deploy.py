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
    SIGINT/SIGTERM; twinkle's launcher installs the handlers and calls serve.shutdown(). With
    ``deploy_config.detach`` the driver returns once the applications are RUNNING and the
    deployment keeps living on the Ray cluster; deployments of different model names coexist
    behind one shared gateway (the registry widens per deployment).
    """
    from twinkle.server import launch_server

    from swift.dev.config import DeployConfig as _DeployConfig
    from swift.dev.config import DistributedConfig as _DistributedConfig

    deploy_config = deploy_config or _DeployConfig()
    distributed_config = distributed_config or _DistributedConfig()

    _validate(deploy_config, quantize_config, backend)
    _warn_unapplied_generation_defaults(generation_config)
    if deploy_config.api_key is None:
        logger.warning('run_deploy is starting WITHOUT api_key: anyone who can reach the port can use the '
                       'model. Set DeployConfig.api_key before exposing it beyond localhost.')

    _ensure_ray(distributed_config, deploy_config)

    # Coexistence gate, straight off ServeController's live list: one sampler app per model
    # name, so a second deployment of the same name is refused instead of silently rolling it.
    public_name = _public_name(deploy_config, model_config)
    existing = _serve_status()
    sampler_app = _sampler_app_name(public_name)
    if sampler_app in existing:
        raise ValueError(f'model {public_name!r} is already deployed: Ray Serve app '
                         f'{sampler_app!r} is {existing[sampler_app].status}. Run with '
                         '--stop true to remove it first, or choose another '
                         '--served_model_name.')

    # A second deployment shares the running gateway/data-plane instead of re-creating them:
    # the gateway's registry is widened (rolling update) with every live sampler's name, while
    # the data-plane is left untouched -- re-running it would drop in-flight DataRefs.
    gateway_models, skip_apps = [public_name], set()
    if existing:
        gateway_models = sorted(set(_deployed_model_names(existing)) | {public_name})
        if 'data-plane' in existing:
            skip_apps.add('data-plane')

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
        gateway_supported_models=gateway_models,
        skip_apps=skip_apps,
    )
    launch_server(config=config, ray_namespace=deploy_config.ray_namespace, blocking=not deploy_config.detach)
    if deploy_config.detach:
        # Detached drivers should not hand back a half-started endpoint: park until Serve
        # reports the sampler RUNNING (or fail loudly on DEPLOY_FAILED), then exit -- the
        # deployment now lives on the Ray cluster and outlives this process.
        _wait_for_application(sampler_app, timeout=600.0)
        scheme = _url_scheme(deploy_config)
        logger.info(f'detached deployment ready: {scheme}://{deploy_config.host}:{deploy_config.port}'
                    f'{deploy_config.route_prefix} (model={public_name!r}); it stays up on the '
                    'Ray cluster until stopped with --stop true.')


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
    gateway_supported_models: Optional[List[str]] = None,
    skip_apps: Optional[set] = None,
) -> Any:
    """dev Configs -> a twinkle-server ``ServerConfig`` (gateway + sampler [+ data plane]).

    Separate from :func:`run_deploy` so the assembled config can be inspected, dumped to YAML, or
    handed to ``launch_server`` by a caller that manages Ray itself. The gateway is mounted at
    ``route_prefix`` and the sampler at ``{route_prefix}/sampler/{public_name}`` -- twinkle's proxy
    rebuilds that exact path from the request's model, so the two must agree.

    ``gateway_supported_models`` overrides the gateway registry (defaults to this deployment's
    own name); coexisting deployments pass the union with every live sampler's name.
    ``skip_apps`` drops application names from the assembled config -- callers use it to leave
    an already-running shared app (e.g. the data-plane) untouched.
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
    if adapter_mapping:
        # Multi-adapter serving is not wired through yet: the gateway registers the adapter names,
        # but requests carry only ``adapter_name`` while the sampler resolves LoRA from ``adapter_uri``
        # -- the mapping's paths never reach the engine, so every request would silently answer from
        # the BASE model. Fail loudly instead (single adapter: use merge_lora; several: one deploy each).
        raise ValueError(
            f'Deploying {len(adapter_mapping)} adapters with per-request routing is not supported yet: the '
            'adapter paths cannot reach the sampler engine. Use --merge_lora true to fold a single '
            'adapter into the weights, or start one deployment per adapter.')
    if model_config.model is None:
        raise ValueError('ModelConfig.model is required to deploy (it is the model id/path the sampler loads).')

    public_name = _public_name(deploy_config, model_config)
    sampler_type = deploy_config.sampler_type or _derive_sampler_type(backend, deploy_config.enable_data_plane)
    if sampler_type == 'torch':
        # The terminal-value guard (not a backend flag): a pinned sampler_type='torch' or a direct
        # build_server_config call with the transformers backend would otherwise skip run_deploy's
        # _validate and assemble a server the transformers sampler cannot serve.
        raise ValueError('deploy with the transformers (pt) sampler backend is not supported yet: the '
                         'server needs the vLLM/SGLang engines. Use --backend vllm or sglang, or leave '
                         'sampler_type unset to derive it from the backend.')

    sampler_engine_args = _sampler_engine_args(backend, engine_args, adapter_mapping, deploy_config, _enable_lora)
    device_group, device_mesh, nproc = _device_spec(distributed_config)
    template_spec = resolve_twinkle_template(template_config, model_config)

    route_prefix = deploy_config.route_prefix
    # 127.0.0.1 is deliberate: the sampler actor dials its own node's co-located Ray Serve proxy, so
    # loopback reaches the data plane without exposing it; the scheme follows TLS (no plaintext dial).
    scheme = _url_scheme(deploy_config)
    data_plane_url = (f'{scheme}://127.0.0.1:{deploy_config.port}{route_prefix}/data-plane'
                      if deploy_config.enable_data_plane else None)

    supported_models: List[str] = list(gateway_supported_models or [public_name])
    if adapter_mapping:
        supported_models += [name for name in adapter_mapping if name not in supported_models]
    skip = set(skip_apps or ())
    applications: List[Any] = []
    if 'server' not in skip:
        applications.append(_gateway_app_spec(deploy_config, supported_models))
    # Three Ray Serve applications: 'server' (the OpenAI gateway + proxy), optional 'data-plane'
    # (server-side rollout groups via DataRef), and 'sampler-<name>' (the engine replica).
    if deploy_config.enable_data_plane and 'data-plane' not in skip:
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


def _public_name(deploy_config: DeployConfig, model_config: ModelConfig) -> str:
    """The name clients pass as ``model``: an explicit served_model_name, else the loaded
    checkpoint's basename (serving one's own merged/finetuned artifacts under their dir name)."""
    return deploy_config.served_model_name or os.path.basename(str(model_config.model).rstrip('/'))


def _sampler_app_name(public_name: str) -> str:
    """Ray Serve app name of a model's sampler; ``/`` is not legal in an app name."""
    return f'sampler-{public_name.replace("/", "_")}'


def _serve_status() -> Dict[str, Any]:
    """ServeController's live application list; empty when no Serve instance is running.

    This is the single source of truth for what is deployed -- no local registry exists.
    A failed poll is logged rather than swallowed: a silently broken status reader would
    present as an endless wait.
    """
    from ray import serve

    try:
        return dict(serve.status().applications)
    except Exception:
        logger.warning(f'serve.status() unavailable; treating as no applications '
                       f'({__import__("traceback").format_exc().splitlines()[-1]})')
        return {}


def _serve_details() -> Dict[str, Any]:
    """Per-application detail dicts (route_prefix included), keyed by application name.

    ``serve.status()`` entries carry no route prefix; the controller client's details do, and
    the route is the only lossless carrier of a model name (app names replace ``/`` with
    ``_``). A failed read is logged -- callers widen or narrow the gateway registry with it.
    """
    from ray.serve.context import _get_global_client

    try:
        return dict(_get_global_client().get_serve_details()['applications'])
    except Exception:
        logger.warning(f'serve details unavailable; treating as no applications '
                       f'({__import__("traceback").format_exc().splitlines()[-1]})')
        return {}


def _deployed_model_names(apps: Dict[str, Any]) -> List[str]:
    """Model names behind the running sampler apps.

    App names alone are lossy (``/`` became ``_``); the route prefix keeps the original name
    verbatim in its ``.../sampler/{name}`` tail, so the names are read off the routes.
    """
    names = []
    for name, details in _serve_details().items():
        if not name.startswith('sampler-'):
            continue
        route = (details.get('route_prefix') if isinstance(details, dict) else getattr(details, 'route_prefix', None)) or ''
        if '/sampler/' in route:
            names.append(route.split('/sampler/', 1)[1])
    return names


def _gateway_app_spec(deploy_config: DeployConfig, supported_models: List[str]) -> Any:
    """The gateway application spec -- shared by build_server_config and stop_deployment (the
    latter re-runs it to re-register the surviving models after removing one)."""
    from twinkle.server.config import ApplicationSpec, ServerArgs

    route_prefix = deploy_config.route_prefix
    return ApplicationSpec(
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
    )


def _wait_for_application(app_name: str, timeout: float, poll: float = 2.0) -> None:
    """Park until Serve reports the app RUNNING; DEPLOY_FAILED/UNHEALTHY fail loudly."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        details = _serve_status().get(app_name)
        if details is not None:
            # ApplicationStatus is an enum: str() yields 'ApplicationStatus.RUNNING', the
            # value is the plain word -- compare against the value.
            status = getattr(details.status, 'value', str(details.status))
            if status == 'RUNNING':
                return
            if status in ('DEPLOY_FAILED', 'UNHEALTHY'):
                raise RuntimeError(f'application {app_name!r} reached status {status}; '
                                   'check the driver logs above for the underlying error.')
        time.sleep(poll)
    raise TimeoutError(f'application {app_name!r} did not reach RUNNING within {timeout}s.')


def stop_deployment(deploy_config: DeployConfig, model_config: Optional[ModelConfig] = None) -> None:
    """Remove the deployment serving ``deploy_config``'s model name (``--stop``).

    Pure Ray Serve primitives: the sampler app is ``serve.delete``d, then the shared gateway's
    registry is rebuilt without the removed name (a rolling update, the same route an adding
    deployment takes). When the last model goes, the gateway and data-plane are deleted too.
    The gateway's ``api_key``/``route_prefix`` after the update are whatever THIS command
    passes -- they are gateway-global knobs, not per-model state.
    """
    import ray

    public_name = deploy_config.served_model_name
    if public_name is None and model_config is not None and getattr(model_config, 'model', None):
        public_name = os.path.basename(str(model_config.model).rstrip('/'))
    if not public_name:
        raise ValueError('--stop needs the model name: pass --served_model_name (or --model '
                         'whose basename serves as the name).')

    try:
        ray.init(address='auto', namespace=deploy_config.ray_namespace, ignore_reinit_error=True)
    except ConnectionError as e:
        raise ValueError(f'no Ray cluster is running, so there is no deployment for model '
                         f'{public_name!r}.') from e

    from ray import serve

    from twinkle.server import launch_server

    apps = _serve_status()
    sampler_app = _sampler_app_name(public_name)
    if sampler_app not in apps:
        running = sorted(n for n in apps if n.startswith('sampler-'))
        raise ValueError(f'no deployment found for model {public_name!r} '
                         f'(running sampler apps: {running or "none"}).')

    # serve.delete blocks until the app (and its replicas) are gone.
    serve.delete(sampler_app)
    remaining = [name for name in _deployed_model_names(apps) if name != public_name]
    if 'server' in apps:
        if remaining:
            logger.info(f're-registering gateway without {public_name!r}: {remaining}')
            launch_server(
                config=_server_config_with(_gateway_app_spec(deploy_config, remaining), deploy_config),
                ray_namespace=deploy_config.ray_namespace,
                blocking=False,
            )
        else:
            serve.delete('server')
            if 'data-plane' in apps:
                serve.delete('data-plane')
    logger.info(f'stopped deployment {sampler_app!r} (model {public_name!r}).')


def _server_config_with(gateway_spec: Any, deploy_config: DeployConfig) -> Any:
    """A minimal ServerConfig carrying just the gateway application (stop's re-registration)."""
    from twinkle.server.config import HttpOptions, PersistenceConfig, ServerConfig

    return ServerConfig(
        ray_namespace=deploy_config.ray_namespace,
        http_options=HttpOptions(
            host=deploy_config.host,
            port=deploy_config.port,
            ssl_keyfile=deploy_config.ssl_keyfile,
            ssl_certfile=deploy_config.ssl_certfile,
        ),
        persistence=PersistenceConfig(),
        applications=[gateway_spec],
    )


def _url_scheme(deploy_config: DeployConfig) -> str:
    """``https`` when TLS is configured, else ``http`` -- the one place the scheme is decided.

    Every URL this module builds for the running server (the data-plane dial target, the readiness
    probe, the base URL yielded to callers) must agree with how Ray Serve's proxy was started, which
    ``HttpOptions`` drives off ``ssl_certfile``.
    """
    return 'https' if deploy_config.ssl_certfile else 'http'


def _validate(deploy_config: DeployConfig, quantize_config: Optional[QuantizeConfig], backend: str) -> None:
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
    if backend in ('pt', 'transformers'):
        # The server rides the vLLM/SGLang engines (async submit, LoRA slots, sleep/wake); the
        # transformers sampler has none of that plumbing. Fail loudly instead of mis-serving.
        raise ValueError('deploy with the transformers (pt) sampler backend is not supported yet: the '
                         'server needs the vLLM/SGLang engines. Use --backend vllm or sglang.')


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


def _local_head_running() -> bool:
    """Whether a standalone local Ray head is already up (its GCS listens on 6379)."""
    import socket

    try:
        with socket.create_connection(('127.0.0.1', 6379), timeout=1.0):
            return True
    except OSError:
        return False


def _start_local_head() -> None:
    """Start a standalone ``ray start --head`` that outlives this driver process."""
    import subprocess
    import sys
    from pathlib import Path

    # Same interpreter's ray CLI first: PATH may front a different (incompatible) install,
    # and a head from another interpreter is unreachable from this driver's ray.init.
    ray_bin = str(Path(sys.executable).parent / 'ray')
    if not Path(ray_bin).exists():
        import shutil
        ray_bin = shutil.which('ray')
    result = subprocess.run([ray_bin, 'start', '--head'], capture_output=True, text=True)
    if result.returncode != 0 and 'already' not in (result.stdout + result.stderr).lower():
        raise RuntimeError(f'ray start --head failed:\n{result.stdout}\n{result.stderr}')
    logger.info('started a standalone local Ray head (detached; outlives this command).')


def _ensure_ray(distributed_config: DistributedConfig, deploy_config: DeployConfig) -> None:
    """Bring up Ray so ``launch_server`` can deploy onto it.

    ``launch_server._init_ray`` only attaches (``address='auto'``) to an existing cluster, so
    ``mode='ray'`` / ``ray_address='auto'`` are left alone; the single-command default starts a
    local Ray head via the launcher's own ``ensure_ray_cluster``, so worker seeding (runtime_env,
    namespace) stays single-sourced with the launcher.

    A DETACHED deployment additionally needs the local head to be its own process: the
    in-process mini-cluster ``ray.init()`` would start dies with this driver, taking every
    Serve application with it. ``ray start --head`` owns the cluster lifetime instead.
    """
    if distributed_config.mode == 'ray' or deploy_config.ray_address == 'auto':
        return
    from twinkle.server.launcher import ensure_ray_cluster

    address = deploy_config.ray_address
    if not address or address == 'local':
        address = None
        if deploy_config.detach and not _local_head_running():
            _start_local_head()
            address = 'auto'
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
    comes from the shared DP-mesh builder, 1 unless data parallelism was requested. The device types
    come from the platform probe (NPU clusters address ``npu`` devices); the mesh dict needs none --
    DeviceMesh fills its own device_type the same way on the worker side.
    """
    from twinkle import Platform

    nproc = distributed_config.nproc_per_node or 1
    from swift.dev.builders import build_parallel_mesh

    mesh = build_parallel_mesh(distributed_config, none_if_single_rank=True)
    dp_size = int(getattr(mesh, 'data_world_size', 1)) if mesh is not None else 1
    device_group = {'name': 'sampler', 'ranks': nproc, 'device_type': Platform.get_platform().__name__}
    device_mesh = {'dp_size': dp_size}
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
            _wait_until_accessible(
                port, timeout, process, deploy_config.route_prefix, scheme, api_key=deploy_config.api_key)
            yield f'{scheme}://127.0.0.1:{port}{deploy_config.route_prefix}'
        finally:
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()

    return _manager()


def _wait_until_accessible(port: int, timeout: float, process, route_prefix: str, scheme: str = 'http',
                           api_key: Optional[str] = None) -> None:
    """Poll ``{route_prefix}/models`` until it answers 200, failing fast if the child dies first.

    An HTTP probe, not a bare socket connect: the port opens when Ray Serve's proxy is up, but the
    gateway is ready a little later -- yielding before that hands the caller a URL that 404s. For
    HTTPS the cert is typically self-signed, so verification is off for this probe only.

    ``api_key`` rides along as a Bearer header: when auth is on the gateway answers an unauthenticated
    probe with 401 FOREVER (the server is healthy but the probe can never see it), so the readiness
    handshake must present the same credential the caller will use.
    """
    import ssl
    import time
    import urllib.error
    import urllib.request

    url = f'{scheme}://127.0.0.1:{port}{route_prefix}/models'
    headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
    context = ssl._create_unverified_context() if scheme == 'https' else None
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process.is_alive():
            raise RuntimeError(f'deploy subprocess exited with code {process.exitcode} before the endpoint '
                               "opened; its traceback is on this process's stderr.")
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=2.0, context=context) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f'deploy did not become accessible at {url} within {timeout}s.')
