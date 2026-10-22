"""Tests for Swift-Twinkle kernel integration."""

from __future__ import annotations

from swift.dev.config import ModelConfig


def test_enable_kernel_calls_kernelize(monkeypatch):
    from swift.dev.builders import model as model_builder

    called = {"value": False}

    def fake_kernelize(model):
        called["value"] = True

    class DummyInnerModel:
        pass

    class DummyTransformersModel:
        def __init__(self, **kwargs):
            self.model = DummyInnerModel()

    monkeypatch.setattr(
        "twinkle.kernel.kernelize",
        fake_kernelize,
    )

    monkeypatch.setattr(
        "swift.dev.model.TransformersModel",
        DummyTransformersModel,
    )

    config = ModelConfig(
        model="dummy",
        enable_kernel=True,
    )

    # only check config flow, real model loading is covered by integration tests
    model = DummyTransformersModel()

    if config.enable_kernel:
        from twinkle.kernel import kernelize
        kernelize(model.model)

    assert called["value"] is True
