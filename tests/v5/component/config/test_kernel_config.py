"""Tests for Twinkle kernel integration config."""

from __future__ import annotations


def test_kernel_disabled_by_default():
    from swift.dev.config import ModelConfig

    config = ModelConfig(model="dummy")

    assert config.enable_kernel is False


def test_kernel_can_be_enabled():
    from swift.dev.config import ModelConfig

    config = ModelConfig(
        model="dummy",
        enable_kernel=True,
    )

    assert config.enable_kernel is True
