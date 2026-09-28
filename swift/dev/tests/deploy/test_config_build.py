# Copyright (c) ModelScope Contributors. All rights reserved.
"""Config-assembly tests for ``swift deploy`` (no GPU, no model download).

``build_server_config`` turns dev's Configs into a twinkle-server ``ServerConfig``; ``_validate`` rejects
knob combinations the server cannot honour; ``_derive_sampler_type`` maps a backend + the data-plane flag
onto a twinkle ``sampler_type``. These pin that translation -- the surface a wrong field name or a missed
guard would silently break -- without launching anything. ``build_server_config`` only *names* the model
(a template lookup by string), so no weights are touched here.
"""
import pytest

from swift.dev.builders.sampler import _derive_sampler_type
from swift.dev.config import DeployConfig, ModelConfig, QuantizeConfig, TemplateConfig
from swift.dev.recipe.run_deploy import (_deployment, _sampler_engine_args, _url_scheme, _validate,
                                         build_server_config)

MODEL = 'Qwen/Qwen3.5-4B'


def _model_config():
    return ModelConfig(model=MODEL, model_type='qwen3_5', torch_dtype='bfloat16')


def _apps_by_name(server_config):
    return {app.name: app for app in server_config.applications}


# --- sampler_type derivation ----------------------------------------------------------


@pytest.mark.parametrize('backend,data_plane,expected', [
    ('vllm', False, 'vllm'),
    ('vllm', True, 'vllm_async'),
    ('sglang', False, 'sglang'),
    ('sglang', True, 'sglang_async'),
    ('transformers', False, 'torch'),
    ('pt', False, 'torch'),
])
def test_derive_sampler_type(backend, data_plane, expected):
    assert _derive_sampler_type(backend, data_plane) == expected


def test_derive_sampler_type_transformers_cannot_serve_data_plane():
    with pytest.raises(ValueError, match='cannot serve the data plane'):
        _derive_sampler_type('transformers', True)


def test_derive_sampler_type_unknown_backend():
    with pytest.raises(ValueError, match='has no twinkle-server sampler'):
        _derive_sampler_type('nope', False)


# --- _validate guards -----------------------------------------------------------------


def test_validate_rejects_one_sided_tls():
    with pytest.raises(ValueError, match='both ssl_keyfile and ssl_certfile'):
        _validate(DeployConfig(ssl_certfile='/tmp/c.pem'), None)


def test_validate_rejects_non_async_sampler_with_data_plane():
    for sampler_type in ('vllm', 'sglang', 'torch', 'mock'):
        with pytest.raises(ValueError, match='cannot serve the data plane'):
            _validate(DeployConfig(enable_data_plane=True, sampler_type=sampler_type), None)


def test_validate_allows_async_samplers_with_data_plane():
    for sampler_type in (None, 'vllm_async', 'sglang_async'):
        _validate(DeployConfig(enable_data_plane=True, sampler_type=sampler_type), None)  # must not raise


def test_validate_rejects_load_time_quant_method():
    with pytest.raises(ValueError, match='quant_method'):
        _validate(DeployConfig(), QuantizeConfig(quant_method='gptq'))


# --- url scheme -----------------------------------------------------------------------


def test_url_scheme_follows_tls():
    assert _url_scheme(DeployConfig()) == 'http'
    assert _url_scheme(DeployConfig(ssl_certfile='/tmp/c.pem', ssl_keyfile='/tmp/k.pem')) == 'https'


# --- server config assembly -----------------------------------------------------------


def test_build_server_config_gateway_and_sampler():
    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template='qwen3_5'),
        backend='vllm',
        deploy_config=DeployConfig(served_model_name='policy', api_key='EMPTY', route_prefix='/v1'),
    )
    apps = _apps_by_name(cfg)
    assert 'server' in apps and 'sampler-policy' in apps
    assert 'data-plane' not in apps  # off unless enable_data_plane
    gateway = apps['server']
    assert gateway.route_prefix == '/v1'
    assert gateway.args.api_key == 'EMPTY'
    assert 'policy' in gateway.args.supported_models
    sampler = apps['sampler-policy']
    assert sampler.route_prefix == '/v1/sampler/policy'
    assert sampler.args.sampler_type == 'vllm'
    assert sampler.args.model_id == MODEL
    assert sampler.args.data_plane_url is None
    # the template is pinned at construction, resolved by name from the model id
    assert sampler.args.template['template_cls'] == 'Qwen3_5Template'


def test_build_server_config_data_plane_adds_app_and_async_sampler():
    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template='qwen3_5'),
        backend='vllm',
        deploy_config=DeployConfig(served_model_name='policy', route_prefix='/v1', enable_data_plane=True),
    )
    apps = _apps_by_name(cfg)
    assert 'data-plane' in apps
    assert apps['data-plane'].route_prefix == '/v1/data-plane'
    sampler = apps['sampler-policy']
    assert sampler.args.sampler_type == 'vllm_async'
    assert sampler.args.data_plane_url == 'http://127.0.0.1:8000/v1/data-plane'


def test_build_server_config_data_plane_url_follows_tls():
    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template='qwen3_5'),
        backend='vllm',
        deploy_config=DeployConfig(
            served_model_name='policy',
            port=8443,
            route_prefix='/v1',
            enable_data_plane=True,
            ssl_certfile='/tmp/c.pem',
            ssl_keyfile='/tmp/k.pem'),
    )
    sampler = _apps_by_name(cfg)['sampler-policy']
    assert sampler.args.data_plane_url == 'https://127.0.0.1:8443/v1/data-plane'
    assert cfg.http_options.ssl_certfile == '/tmp/c.pem'


def test_build_server_config_sglang_data_plane_derives_async():
    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template='qwen3_5'),
        backend='sglang',
        deploy_config=DeployConfig(served_model_name='policy', enable_data_plane=True),
    )
    assert _apps_by_name(cfg)['sampler-policy'].args.sampler_type == 'sglang_async'


def test_build_server_config_lists_adapters_as_supported_models():
    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template='qwen3_5'),
        backend='vllm',
        adapter_mapping={'adapter-1': '/tmp/lora1'},
        deploy_config=DeployConfig(served_model_name='policy'),
    )
    gateway = _apps_by_name(cfg)['server']
    assert 'policy' in gateway.args.supported_models
    assert 'adapter-1' in gateway.args.supported_models


def test_build_server_config_requires_model():
    with pytest.raises(ValueError, match='ModelConfig.model is required'):
        build_server_config(
            ModelConfig(model=None),
            TemplateConfig(template='qwen3_5'),
            backend='vllm',
            deploy_config=DeployConfig(),
        )


def test_build_server_config_merge_lora_rejects_multiple_adapters():
    with pytest.raises(ValueError, match='cannot serve 2 adapters'):
        build_server_config(
            _model_config(),
            TemplateConfig(template='qwen3_5'),
            backend='vllm',
            adapter_mapping={'a': '/tmp/lora1', 'b': '/tmp/lora2'},
            deploy_config=DeployConfig(merge_lora=True),
        )


# --- engine args ----------------------------------------------------------------------


def test_sampler_engine_args_max_logprobs_is_vllm_only():
    vllm = _sampler_engine_args('vllm', {}, {}, DeployConfig(max_logprobs=20), lambda *a, **k: None)
    assert vllm['max_logprobs'] == 20
    sglang = _sampler_engine_args('sglang', {}, {}, DeployConfig(max_logprobs=20), lambda *a, **k: None)
    assert 'max_logprobs' not in sglang


def test_sampler_engine_args_does_not_override_caller_max_logprobs():
    resolved = _sampler_engine_args('vllm', {'max_logprobs': 5}, {}, DeployConfig(max_logprobs=20),
                                    lambda *a, **k: None)
    assert resolved['max_logprobs'] == 5


def test_sampler_engine_args_sizes_lora_when_adapters_present():
    calls = {}

    def fake_enable_lora(resolved, backend, paths):
        calls['args'] = (backend, list(paths))
        resolved['enable_lora'] = True

    resolved = _sampler_engine_args('vllm', {}, {'a': '/tmp/lora1'}, DeployConfig(), fake_enable_lora)
    assert resolved['enable_lora'] is True
    assert calls['args'] == ('vllm', ['/tmp/lora1'])


# --- deployment options ---------------------------------------------------------------


def test_deployment_fixed_replicas():
    opts = _deployment('SamplerManagement', DeployConfig(num_replicas=2, max_concurrency=32))
    assert opts['name'] == 'SamplerManagement'
    assert opts['num_replicas'] == 2
    assert opts['max_ongoing_requests'] == 32
    assert 'autoscaling_config' not in opts


def test_deployment_autoscaling():
    opts = _deployment('SamplerManagement', DeployConfig(autoscaling=True, num_replicas=4, max_concurrency=16))
    assert opts['autoscaling_config'] == {
        'min_replicas': 1,
        'max_replicas': 4,
        'target_ongoing_requests': 16,
    }
    assert 'num_replicas' not in opts


def test_persistence_default_and_explicit():
    from twinkle.server.config import PersistenceConfig
    from swift.dev.recipe.run_deploy import _persistence
    # no deploy knob set -> twinkle's own in-memory default
    assert _persistence(DeployConfig(), PersistenceConfig).mode == 'memory'
    p = _persistence(DeployConfig(persistence_mode='file', persistence_file_path='/tmp/s.json'), PersistenceConfig)
    assert p.mode == 'file' and p.file_path == '/tmp/s.json'
