import os

os.environ.setdefault('USE_SWIFT_V5', '1')

import ray  # noqa: E402
from twinkle.server import launch_server  # noqa: E402

from swift.dev.config import DeployConfig, DistributedConfig, ModelConfig, TemplateConfig  # noqa: E402
from swift.dev.recipe.infer.run_deploy import build_server_config  # noqa: E402


def build_configs():
    """Assemble the deployment as Config objects (the same knobs as ../vllm/deploy.sh).

    ``served_model_name`` is the ``model`` clients pass and the name the sampler is mounted under
    (``{route_prefix}/sampler/{served_model_name}``). ``route_prefix`` is ``/api/v1`` (not ``/v1``) so the
    token-in-token-out ``twinkle_client`` -- which appends ``/api/v1`` to its base_url -- resolves too.
    ``api_key`` gates every request with a Bearer token; leaving it None serves unauthenticated, which is
    only safe on a port unreachable from outside the host.
    """
    model_config = ModelConfig(model='Qwen/Qwen2.5-0.5B-Instruct', torch_dtype='bfloat16')
    # An empty TemplateConfig lets the template be resolved from the model; set ``template=`` to pin one.
    template_config = TemplateConfig()
    deploy_config = DeployConfig(
        host='0.0.0.0',
        port=8000,
        route_prefix='/api/v1',
        served_model_name='policy',
        api_key='EMPTY',
    )
    return model_config, template_config, deploy_config


def main() -> None:
    """Deploy in three steps -- the orchestration ``run_deploy`` wraps, spelled out.

    ``run_deploy`` is validate + warn + ``_ensure_ray`` + ``build_server_config`` + ``launch_server``. This
    scaffold keeps only the load-bearing three so the assembly is visible: bring up Ray, translate dev's
    Configs into a twinkle-server ``ServerConfig``, then hand that config to the launcher. Use
    ``swift.dev.recipe.infer.run_deploy`` directly when you want the validation and generation-default warnings
    too; drop down to these building blocks when you need to inspect or customise the ``ServerConfig``.
    """
    model_config, template_config, deploy_config = build_configs()
    distributed_config = DistributedConfig()

    # 1. Ray bring-up. With no cluster running, start a local head so this one process serves; set
    #    ``distributed_config.mode='ray'`` (or ``deploy_config.ray_address='auto'``) to attach to one that
    #    already exists, and skip this step.
    if not ray.is_initialized():
        ray.init(namespace=deploy_config.ray_namespace, ignore_reinit_error=True)

    # 2. dev Configs -> a twinkle-server ServerConfig (a gateway app + a sampler app). engine_args are the
    #    raw vLLM engine kwargs (unprefixed, unlike the CLI's --vllm_* flags). Generation knobs
    #    (temperature / top_p / max_new_tokens) are NOT here: the gateway applies each request's own
    #    sampling params, so a server-side default would be silently ignored.
    config = build_server_config(
        model_config,
        template_config,
        backend='vllm',
        engine_args={
            'gpu_memory_utilization': 0.9,
            'max_model_len': 4096,
            'enforce_eager': True,
        },
        distributed_config=distributed_config,
        deploy_config=deploy_config,
    )

    # 3. Launch on Ray Serve. Blocks until interrupted; the launcher installs the signal handlers and calls
    #    serve.shutdown(). backend='vllm' picks the sampler engine ('sglang' / 'transformers' also work).
    launch_server(config=config, ray_namespace=deploy_config.ray_namespace)


if __name__ == '__main__':
    main()
