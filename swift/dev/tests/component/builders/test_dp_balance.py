"""The same TrainConfig option reaches each model backend, not bridge/model-loader kwargs."""

from types import SimpleNamespace

import pytest

from swift.dev.builders import model as builders
from swift.dev.config import DistributedConfig, ModelConfig, TrainConfig


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("backend", ["transformers", "unsloth", "sentence_transformers", "megatron"])
def test_builder_passes_common_dp_balance_switch(monkeypatch, enabled, backend):
    import twinkle.model.megatron as megatron

    import swift.dev.model as models
    import swift.dev.utils.naming as naming

    monkeypatch.setattr(builders, "_resolve_model_loader", lambda *args: None)
    monkeypatch.setattr(builders, "_build_hf_config", lambda *args: SimpleNamespace(model_type="qwen2"))
    monkeypatch.setattr(builders, "_apply_unsloth_kwargs", lambda *args: None)
    monkeypatch.setattr(
        builders,
        "_resolve_model_framework",
        lambda *args: ("sentence_transformers" if backend == "sentence_transformers" else None),
    )
    monkeypatch.setattr(naming, "resolve_megatron_attn_backend", lambda *args, **kwargs: "flash")
    monkeypatch.setattr(models, "TransformersModel", lambda **kwargs: kwargs)
    monkeypatch.setattr(models, "UnslothModel", lambda **kwargs: kwargs)
    monkeypatch.setattr(models, "SentenceTransformerModel", lambda **kwargs: kwargs)
    monkeypatch.setattr(megatron, "MegatronModel", lambda **kwargs: kwargs)
    model = ModelConfig(model="m", task_type="embedding" if backend == "sentence_transformers" else "causal_lm")
    distributed = DistributedConfig(
        backend="megatron" if backend == "megatron" else ("unsloth" if backend == "unsloth" else "transformers"),
        mode="local",
        nproc_per_node=2,
    )
    train = TrainConfig(dp_balance=enabled)
    build = builders._build_megatron_model if backend == "megatron" else builders._build_transformers_model
    result = build(model, distributed, train)
    assert result["dp_balance"] is enabled
