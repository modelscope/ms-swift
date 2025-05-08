# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys

from swift.llm import git_clone_github
from swift.utils import is_megatron_available, safe_ddp_context, subprocess_run


def _patch_megatron():
    try:
        from transformer_engine.pytorch.attention import FusedRoPEFunc
    except ImportError:
        try:
            import transformer_engine
            transformer_engine.pytorch.attention.FusedRoPEFunc = (
                transformer_engine.pytorch.dot_product_attention.rope.FusedRoPEFunc)
        except (ImportError, AttributeError):
            pass


def init_megatron_env() -> None:
    if 'MEGATRON_LM_PATH' not in os.environ:
        os.environ['MEGATRON_LM_PATH'] = git_clone_github(
            'https://github.com/NVIDIA/Megatron-LM', branch='core_r0.11.0')
    with safe_ddp_context(hash_id='megatron-lm'):
        if not is_megatron_available():
            subprocess_run([sys.executable, '-m', 'pip', 'install', '-e', os.environ['MEGATRON_LM_PATH']])
    sys.path.insert(0, os.environ['MEGATRON_LM_PATH'])
    _patch_megatron()
