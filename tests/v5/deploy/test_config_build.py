# Copyright (c) ModelScope Contributors. All rights reserved.
"""Config-assembly tests for ``swift deploy`` (no GPU, no model download).

``build_server_config`` turns dev's Configs into a twinkle-server ``ServerConfig``; ``_validate`` rejects
knob combinations the server cannot honour; ``_derive_sampler_type`` maps a backend + the data-plane flag
onto a twinkle ``sampler_type``. These pin that translation -- the surface a wrong field name or a missed
guard would silently break -- without launching anything. ``build_server_config`` only *names* the model
(a template lookup by string), so no weights are touched here.
"""

import importlib
from unittest.mock import Mock

import pytest

from swift.dev.builders.sampler import _derive_sampler_type
from swift.dev.config import DeployConfig, ModelConfig, QuantizeConfig, TemplateConfig
from swift.dev.recipe.infer.run_deploy import (
    _deployment,
    _sampler_engine_args,
    _sampler_pool,
    _url_scheme,
    _validate,
    build_server_config,
    run_deploy,
)

MODEL = "Qwen/Qwen3.5-4B"


def _model_config():
    return ModelConfig(model=MODEL, model_type="qwen3_5", torch_dtype="bfloat16")


def _apps_by_name(server_config):
    return {app.name: app for app in server_config.applications}


# --- sampler_type derivation ----------------------------------------------------------


@pytest.mark.parametrize(
    "backend,data_plane,expected",
    [
        ("vllm", False, "vllm"),
        ("vllm", True, "vllm_async"),
        ("sglang", False, "sglang"),
        ("sglang", True, "sglang_async"),
        ("transformers", False, "torch"),
        ("pt", False, "torch"),
    ],
)
def test_derive_sampler_type(backend, data_plane, expected):
    assert _derive_sampler_type(backend, data_plane) == expected


def test_derive_sampler_type_transformers_cannot_serve_data_plane():
    with pytest.raises(ValueError, match="cannot serve the data plane"):
        _derive_sampler_type("transformers", True)


def test_derive_sampler_type_unknown_backend():
    with pytest.raises(ValueError, match="has no twinkle-server sampler"):
        _derive_sampler_type("nope", False)


# --- _validate guards -----------------------------------------------------------------


def test_validate_rejects_one_sided_tls():
    with pytest.raises(ValueError, match="both ssl_keyfile and ssl_certfile"):
        _validate(DeployConfig(ssl_certfile="/tmp/c.pem"), None, "vllm")


def test_validate_rejects_non_async_sampler_with_data_plane():
    for sampler_type in ("vllm", "sglang", "torch", "mock"):
        with pytest.raises(ValueError, match="cannot serve the data plane"):
            _validate(DeployConfig(enable_data_plane=True, sampler_type=sampler_type), None, "vllm")


def test_validate_allows_async_samplers_with_data_plane():
    for sampler_type in (None, "vllm_async", "sglang_async"):
        _validate(DeployConfig(enable_data_plane=True, sampler_type=sampler_type), None, "vllm")  # must not raise


def test_validate_rejects_load_time_quant_method():
    with pytest.raises(ValueError, match="quant_method"):
        _validate(DeployConfig(), QuantizeConfig(quant_method="gptq"), "vllm")


# --- url scheme -----------------------------------------------------------------------


def test_url_scheme_follows_tls():
    assert _url_scheme(DeployConfig()) == "http"
    assert _url_scheme(DeployConfig(ssl_certfile="/tmp/c.pem", ssl_keyfile="/tmp/k.pem")) == "https"


# --- server config assembly -----------------------------------------------------------


def test_build_server_config_gateway_and_sampler():
    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template="qwen3_5"),
        backend="vllm",
        deploy_config=DeployConfig(served_model_name="policy", api_key="EMPTY", route_prefix="/v1"),
    )
    apps = _apps_by_name(cfg)
    assert "server" in apps and "sampler-policy" in apps
    assert "data-plane" not in apps  # off unless enable_data_plane
    gateway = apps["server"]
    assert gateway.route_prefix == "/v1"
    assert gateway.args.api_key == "EMPTY"
    assert "policy" in gateway.args.supported_models
    sampler = apps["sampler-policy"]
    assert sampler.route_prefix == "/v1/sampler/policy"
    assert sampler.args.sampler_type == "vllm"
    assert sampler.args.model_id == MODEL
    assert sampler.args.api_key == "EMPTY"
    assert sampler.args.data_plane_url is None
    # the template is pinned at construction. f2871af62 changed the pin from twinkle's family-map
    # class name to SwiftTemplate + a serialized dev TemplateSpec (the same shape the remote client
    # paths post); the spec must still carry the configured template family.
    assert sampler.args.template["template_cls"] == "SwiftTemplate"
    assert sampler.args.template["template_spec"]["template_type"] == "qwen3_5"


def test_sampler_app_names_distinguish_slash_and_underscore():
    samplers = []
    for model in ("team/model", "team_model"):
        cfg = build_server_config(
            _model_config(), TemplateConfig(template="qwen3_5"), deploy_config=DeployConfig(served_model_name=model)
        )
        sampler = next(app for app in cfg.applications if app.import_path == "sampler")
        assert sampler.route_prefix == f"/v1/sampler/{model}"
        samplers.append(sampler.name)
    assert samplers[0] != samplers[1]


def test_run_deploy_rejects_duplicate_public_model_on_legacy_route(monkeypatch):
    import twinkle_server
    from twinkle_server import launcher

    deployment = importlib.import_module("swift.dev.recipe.infer.run_deploy")
    monkeypatch.setattr(launcher, "ensure_ray_cluster", Mock())
    monkeypatch.setattr(twinkle_server, "launch_server", Mock())
    monkeypatch.setattr(deployment, "_serve_routes", lambda: {"sampler-team_model": "/v1/sampler/team/model"})
    with pytest.raises(ValueError, match="Model 'team/model' is already deployed"):
        run_deploy(
            _model_config(),
            TemplateConfig(template="qwen3_5"),
            deploy_config=DeployConfig(served_model_name="team/model"),
        )
    twinkle_server.launch_server.assert_not_called()


def test_build_server_config_data_plane_adds_app_and_async_sampler():
    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template="qwen3_5"),
        backend="vllm",
        deploy_config=DeployConfig(served_model_name="policy", route_prefix="/v1", enable_data_plane=True),
    )
    apps = _apps_by_name(cfg)
    assert "data-plane" in apps
    assert apps["data-plane"].route_prefix == "/v1/data-plane"
    sampler = apps["sampler-policy"]
    assert sampler.args.sampler_type == "vllm_async"
    assert sampler.args.data_plane_url == "http://127.0.0.1:8000/v1/data-plane"


def test_build_server_config_data_plane_url_follows_tls():
    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template="qwen3_5"),
        backend="vllm",
        deploy_config=DeployConfig(
            served_model_name="policy",
            port=8443,
            route_prefix="/v1",
            enable_data_plane=True,
            ssl_certfile="/tmp/c.pem",
            ssl_keyfile="/tmp/k.pem",
        ),
    )
    sampler = _apps_by_name(cfg)["sampler-policy"]
    assert sampler.args.data_plane_url == "https://127.0.0.1:8443/v1/data-plane"
    assert cfg.http_options.ssl_certfile == "/tmp/c.pem"


def test_build_server_config_sglang_data_plane_derives_async():
    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template="qwen3_5"),
        backend="sglang",
        deploy_config=DeployConfig(served_model_name="policy", enable_data_plane=True),
    )
    assert _apps_by_name(cfg)["sampler-policy"].args.sampler_type == "sglang_async"


def test_build_server_config_wires_per_request_adapter_routing():
    """A name->path adapter mapping must reach BOTH ends of the route (581a10e1c replaced the old
    refusal): the gateway registers each adapter name with its base_model so _resolve_base_model
    routes the request here, and the sampler receives the paths so the engine actually loads them.
    The failure this kills is the original silent one -- names registered but paths dropped, so
    every request answers from the BASE model while looking healthy.
    """
    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template="qwen3_5"),
        backend="vllm",
        adapter_mapping={"adapter-1": "/tmp/lora1"},
        deploy_config=DeployConfig(served_model_name="policy"),
    )
    apps = _apps_by_name(cfg)
    assert {"model_name": "adapter-1", "base_model": "policy"} in apps["server"].args.supported_models
    sampler = apps["sampler-policy"]
    assert sampler.args.adapters == {"adapter-1": "/tmp/lora1"}
    # the engine's LoRA machinery is on and sized for the mapping (rank falls back to 16 when the
    # adapter_config.json is absent, as for this nonexistent path)
    assert sampler.args.engine_args["enable_lora"] is True
    assert sampler.args.engine_args["max_loras"] == 1


def test_build_server_config_requires_model():
    with pytest.raises(ValueError, match="ModelConfig.model is required"):
        build_server_config(
            ModelConfig(model=None),
            TemplateConfig(template="qwen3_5"),
            backend="vllm",
            deploy_config=DeployConfig(),
        )


def test_build_server_config_merge_lora_rejects_multiple_adapters():
    with pytest.raises(ValueError, match="cannot serve 2 adapters"):
        build_server_config(
            _model_config(),
            TemplateConfig(template="qwen3_5"),
            backend="vllm",
            adapter_mapping={"a": "/tmp/lora1", "b": "/tmp/lora2"},
            deploy_config=DeployConfig(merge_lora=True),
        )


# --- engine args ----------------------------------------------------------------------


def test_sampler_engine_args_max_logprobs_is_vllm_only():
    vllm = _sampler_engine_args("vllm", {}, {}, DeployConfig(max_logprobs=20), lambda *a, **k: None)
    assert vllm["max_logprobs"] == 20
    sglang = _sampler_engine_args("sglang", {}, {}, DeployConfig(max_logprobs=20), lambda *a, **k: None)
    assert "max_logprobs" not in sglang


def test_sampler_engine_args_does_not_override_caller_max_logprobs():
    resolved = _sampler_engine_args(
        "vllm", {"max_logprobs": 5}, {}, DeployConfig(max_logprobs=20), lambda *a, **k: None
    )
    assert resolved["max_logprobs"] == 5


def test_sampler_engine_args_sizes_lora_when_adapters_present():
    calls = {}

    def fake_enable_lora(resolved, backend, paths):
        calls["args"] = (backend, list(paths))
        resolved["enable_lora"] = True

    resolved = _sampler_engine_args("vllm", {}, {"a": "/tmp/lora1"}, DeployConfig(), fake_enable_lora)
    assert resolved["enable_lora"] is True
    assert calls["args"] == ("vllm", ["/tmp/lora1"])


# --- deployment options ---------------------------------------------------------------


def test_deployment_fixed_replicas():
    opts = _deployment("SamplerManagement", DeployConfig(num_replicas=2, max_concurrency=32))
    assert opts["name"] == "SamplerManagement"
    assert opts["num_replicas"] == 2
    assert opts["max_ongoing_requests"] == 32
    assert "autoscaling_config" not in opts


def test_deployment_autoscaling():
    opts = _deployment("SamplerManagement", DeployConfig(autoscaling=True, num_replicas=4, max_concurrency=16))
    assert opts["autoscaling_config"] == {
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 16,
    }
    assert "num_replicas" not in opts


# --- sampler pool replicas (G28: --num_replicas must reach the engine pool) -------------


def test_sampler_pool_unset_keeps_twinkle_defaults():
    assert _sampler_pool(DeployConfig()) == {}


def test_sampler_pool_fixed_replicas():
    assert _sampler_pool(DeployConfig(num_replicas=2)) == {"min_replicas": 2, "max_replicas": 2}


def test_sampler_pool_autoscaling_maps_target_concurrency():
    assert _sampler_pool(DeployConfig(autoscaling=True, num_replicas=3, max_concurrency=8)) == {
        "min_replicas": 1,
        "max_replicas": 3,
        "target_ongoing_requests": 8,
    }


def test_sampler_pool_rejects_zero_replicas():
    with pytest.raises(ValueError, match="num_replicas"):
        _sampler_pool(DeployConfig(num_replicas=0))


def test_build_server_config_num_replicas_sizes_sampler_pool():
    """twinkle forces num_replicas=1 on every per-instance Serve deployment and pops
    autoscaling_config, so the pool size must ride on SamplerArgs.min/max_replicas --
    otherwise --num_replicas only scales the gateway and the engine pool stays at 1."""
    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template="qwen3_5"),
        backend="vllm",
        deploy_config=DeployConfig(served_model_name="policy", num_replicas=2),
    )
    sampler = _apps_by_name(cfg)["sampler-policy"]
    assert sampler.args.min_replicas == 2
    assert sampler.args.max_replicas == 2


def test_build_server_config_autoscaling_sizes_sampler_pool():
    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template="qwen3_5"),
        backend="vllm",
        deploy_config=DeployConfig(served_model_name="policy", autoscaling=True, num_replicas=3, max_concurrency=4),
    )
    sampler = _apps_by_name(cfg)["sampler-policy"]
    assert sampler.args.min_replicas == 1
    assert sampler.args.max_replicas == 3
    assert sampler.args.target_ongoing_requests == 4


def test_build_server_config_default_pool_is_one_replica():
    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template="qwen3_5"),
        backend="vllm",
        deploy_config=DeployConfig(served_model_name="policy"),
    )
    sampler = _apps_by_name(cfg)["sampler-policy"]
    assert sampler.args.min_replicas == 1
    assert sampler.args.max_replicas == 1


def test_persistence_is_the_server_default():
    """The persistence knobs moved to the server config file; the assembled config carries twinkle's
    own in-memory default (memory mode), not a deploy-side persistence mapping."""
    from twinkle_server.config.persistence import PersistenceConfig

    cfg = build_server_config(
        _model_config(),
        TemplateConfig(template="qwen3_5"),
        backend="vllm",
        deploy_config=DeployConfig(served_model_name="policy"),
    )
    assert isinstance(cfg.persistence, PersistenceConfig)
    assert cfg.persistence.mode == "memory"
