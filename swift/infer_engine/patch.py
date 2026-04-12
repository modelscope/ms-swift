# Copyright (c) ModelScope Contributors. All rights reserved.
from contextlib import contextmanager
from functools import wraps
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase


@contextmanager
def patch_auto_tokenizer(tokenizer: PreTrainedTokenizerBase):
    _old_from_pretrained = AutoTokenizer.from_pretrained

    @wraps(_old_from_pretrained)
    def _from_pretrained(*args, **kwargs):
        return tokenizer

    AutoTokenizer.from_pretrained = _from_pretrained
    try:
        yield
    finally:
        AutoTokenizer.from_pretrained = _old_from_pretrained


@contextmanager
def patch_auto_config(config: PretrainedConfig):
    _old_from_pretrained = AutoConfig.from_pretrained

    patched_config = _maybe_convert_config_for_vllm(config)

    @wraps(_old_from_pretrained)
    def _from_pretrained(*args, **kwargs):
        return (patched_config, {}) if 'return_unused_kwargs' in kwargs else patched_config

    AutoConfig.from_pretrained = _from_pretrained
    try:
        yield
    finally:
        AutoConfig.from_pretrained = _old_from_pretrained


def _maybe_convert_config_for_vllm(config: PretrainedConfig) -> PretrainedConfig:
    """Convert config to vLLM's registered config class if needed.

    When both transformers and vLLM define a config for the same model_type
    (e.g. qwen3_5), vLLM's isinstance checks require its own config class.
    """
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
    except ImportError:
        return config
    model_type = getattr(config, 'model_type', None)
    if model_type is None or model_type not in _CONFIG_REGISTRY:
        return config
    vllm_config_cls = _CONFIG_REGISTRY[model_type]
    if isinstance(config, vllm_config_cls):
        return config
    return vllm_config_cls.from_dict(config.to_dict())
