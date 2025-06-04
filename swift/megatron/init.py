# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys

from swift.llm import git_clone_github
from swift.utils import get_logger, is_megatron_available, safe_ddp_context, subprocess_run

logger = get_logger()


def _patch_transformer_engine():
    import transformer_engine
    try:
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb
    except ImportError:
        try:
            transformer_engine.pytorch.attention.apply_rotary_pos_emb = (
                transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb)
            logger.info('Patch apply_rotary_pos_emb successfully applied.')
        except (ImportError, AttributeError):
            pass
    try:
        from transformer_engine.pytorch.attention import _SplitAlongDim
    except ImportError:
        try:
            transformer_engine.pytorch.attention._SplitAlongDim = (transformer_engine.pytorch.utils.SplitAlongDim)
            logger.info('Patch _SplitAlongDim successfully applied.')
        except (ImportError, AttributeError):
            pass


def _patch__batched_p2p_ops():
    from megatron.core.pipeline_parallel import p2p_communication

    _batched_p2p_ops_origin = p2p_communication._batched_p2p_ops

    def _batched_p2p_ops(**kwargs):
        kwargs['group'] = None
        return _batched_p2p_ops_origin(**kwargs)

    p2p_communication._batched_p2p_ops = _batched_p2p_ops


def _patch_megatron():
    _patch_transformer_engine()
    _patch__batched_p2p_ops()


def init_megatron_env() -> None:
    if 'MEGATRON_LM_PATH' not in os.environ:
        os.environ['MEGATRON_LM_PATH'] = git_clone_github(
            'https://github.com/NVIDIA/Megatron-LM', branch='core_r0.12.0')
    with safe_ddp_context(hash_id='megatron-lm'):
        if not is_megatron_available():
            subprocess_run([sys.executable, '-m', 'pip', 'install', '-e', os.environ['MEGATRON_LM_PATH']])
    sys.path.insert(0, os.environ['MEGATRON_LM_PATH'])
    _patch_megatron()
