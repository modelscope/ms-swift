"""Standalone vLLM rollout service for external online-RL trainers."""
from __future__ import annotations
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from swift.dev.config import DeployConfig, GenerationConfig, ModelConfig, RLHFConfig, RolloutConfig, TemplateConfig


def build_rollout_app(  # noqa: C901
    model_config: ModelConfig,
    template_config: TemplateConfig,
    rollout_config: RolloutConfig,
    rlhf_config: RLHFConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    engine_args: Optional[Dict[str, Any]] = None,
):
    """Build the external rollout API without starting uvicorn."""
    from fastapi import Body, FastAPI, HTTPException

    from swift.dev.builders import build_template, to_sampling_params
    from swift.dev.rollout import RolloutEngine
    from swift.model import get_model_processor

    _, processor = get_model_processor(model_config.model, model_type=model_config.model_type, load_model=False)
    template = build_template(template_config, processor)
    engine = RolloutEngine(model_config.model, template, engine_args=engine_args)
    app = FastAPI()
    app.state.rollout_engine = engine

    def call_sampler(method: str, *args, **kwargs):
        sampler = engine.sampler
        target = getattr(sampler, method, None)
        if target is not None:
            return target(*args, **kwargs)
        collective_rpc = getattr(sampler, 'collective_rpc', None)
        if collective_rpc is not None:
            return collective_rpc(method=method, args=args, kwargs=kwargs)
        raise HTTPException(status_code=501, detail=f'The active sampler does not expose {method}.')

    @app.get('/health/')
    async def health():
        return {'status': 'ok'}

    @app.get('/get_world_size/')
    async def get_world_size():
        return {'world_size': rollout_config.vllm_tensor_parallel_size * rollout_config.vllm_data_parallel_size}

    @app.get('/get_model_state_keys/')
    async def get_model_state_keys():
        result = call_sampler('get_state_keys')
        return {'keys': result or []}

    @app.get('/get_engine_type/')
    async def get_engine_type():
        return {
            'engine_type': 'AsyncLLMEngine' if rollout_config.vllm_use_async_engine else 'LLMEngine',
            'enable_multi_turn': bool(rlhf_config.multi_turn_scheduler),
            'use_gym_env': bool(rlhf_config.use_gym_env),
            'enable_lora': rollout_config.vllm_enable_lora,
        }

    @app.post('/infer/', response_model=None)
    async def infer(payload: Dict[str, Any] = Body(...)):  # noqa: B008
        prompts = payload.get('prompts') or payload.get('infer_requests') or []
        normalized = [item.get('messages', item) if isinstance(item, dict) else item for item in prompts]
        overrides = dict(payload.get('request_config') or {})
        num_samples = int(overrides.pop('n', overrides.pop('num_samples', 1)))
        params = asdict(to_sampling_params(generation_config, **overrides))
        samples = engine.generate(normalized, num_samples=num_samples, sampling_params=params)
        return [asdict(sample) for sample in samples]

    def register_passthrough(path: str, method: str):
        async def handler(payload: Dict[str, Any] = Body(default_factory=dict)):  # noqa: B008
            return call_sampler(method, **payload)
        app.post(path)(handler)

    for path, method in {
        '/init_communicator/': 'init_communicator',
        '/update_named_param/': 'update_named_param',
        '/update_adapter_flattened_param/': 'update_adapter_flattened_param',
        '/update_adapter_param/': 'update_adapter_param',
        '/update_flattened_params/': 'update_flattened_params',
        '/process_weights_after_loading/': 'process_weights_after_loading',
        '/reset_prefix_cache/': 'reset_prefix_cache',
        '/reset_encoder_cache/': 'reset_encoder_cache',
        '/reset_mm_cache/': 'reset_mm_cache',
        '/close_communicator/': 'close_communicator',
    }.items():
        register_passthrough(path, method)

    @app.on_event('shutdown')
    async def shutdown():
        engine.shutdown()

    return app


def run_rollout(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    rollout_config: RolloutConfig,
    rlhf_config: RLHFConfig,
    deploy_config: DeployConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    engine_args: Optional[Dict[str, Any]] = None,
) -> None:
    """Run the standalone rollout API until interrupted."""
    import uvicorn

    app = build_rollout_app(
        model_config, template_config, rollout_config, rlhf_config, generation_config, engine_args=engine_args)
    uvicorn.run(
        app,
        host=deploy_config.host,
        port=deploy_config.port,
        log_level=deploy_config.log_level,
        ssl_keyfile=deploy_config.ssl_keyfile,
        ssl_certfile=deploy_config.ssl_certfile,
    )
