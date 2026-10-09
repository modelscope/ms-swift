# Programmatic Python entry for v5 deploy -- the code equivalent of ../vllm/deploy.sh.
#
# A .sh script spells the deployment as CLI flags; this file builds the SAME deployment as typed Config
# OBJECTS and drives it through swift's public serving recipe (``run_deploy``), so you can compute the knobs
# in Python (a model chosen from a registry, a port from a service discovery call, engine args derived from
# the box's free memory) instead of hand-editing a shell string. ``swift deploy`` on the CLI is exactly
# ``parse_deploy_configs`` -> ``process_and_validate_configs`` -> ``run_deploy``; here we skip the argv
# parser and hand ``run_deploy`` the Configs directly.
#
# ``run_deploy`` blocks serving until SIGINT/SIGTERM (it launches twinkle-server on Ray Serve), just like the
# shell command. If you need a live endpoint INSIDE a Python block -- e.g. an eval harness that serves, then
# drives it over HTTP, then tears it down -- use ``run_deploy_process(...)`` instead: same arguments, but it
# spawns the server in a subprocess and is used as a context manager that yields the base URL once ready.
#
# Run: python examples/deploy/code/deploy.py   (set CUDA_VISIBLE_DEVICES to your free cards first).
# Then, from another shell:  python examples/deploy/vllm/client.py
import os

from swift.dev.config import DeployConfig, ModelConfig, TemplateConfig
from swift.dev.recipe import run_deploy


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
    os.environ.setdefault('USE_SWIFT_V5', '1')
    model_config, template_config, deploy_config = build_configs()
    # engine_args are the raw vLLM engine kwargs (unprefixed, unlike the CLI's --vllm_* flags). Generation
    # knobs (temperature / top_p / max_new_tokens) are NOT here: the gateway applies each request's own
    # sampling params, so a server-side default would be silently ignored.
    engine_args = {
        'gpu_memory_utilization': 0.9,
        'max_model_len': 4096,
        'enforce_eager': True,
    }
    # Blocks until interrupted; backend='vllm' picks the sampler engine ('sglang' / 'transformers' also work).
    run_deploy(
        model_config,
        template_config,
        backend='vllm',
        engine_args=engine_args,
        deploy_config=deploy_config,
    )


if __name__ == '__main__':
    main()
