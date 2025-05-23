# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from contextlib import contextmanager

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


def new_cyclic_iter(iter):
    from megatron.training import get_args
    args = get_args()
    max_epochs = args.max_epochs
    i = 0
    while True:
        if getattr(args, 'is_training', False):
            if max_epochs and i >= max_epochs:
                logger.info(f'Training of {i} epochs has been completed, the training has finished.')
                break
            logger.info(f'The training of Epoch {i} starts...')
        for x in iter:
            yield x
        i += 1


@contextmanager
def _training_context():
    from megatron.training import get_args
    args = get_args()
    args.is_training = True
    try:
        yield
    finally:
        args.is_training = False


def _patch_max_epochs():
    # support max_epochs
    from megatron.training import training
    train_step_origin = training.train_step

    def train_step(*args, **kwargs):
        with _training_context():
            try:
                return train_step_origin(*args, **kwargs)
            except StopIteration:
                return {}, True, True, True, 0, None, None

    training.train_step = train_step

    training.cyclic_iter = new_cyclic_iter


def _patch_megatron():
    _patch_transformer_engine()
    _patch_max_epochs()


def init_megatron_env() -> None:
    if 'MEGATRON_LM_PATH' not in os.environ:
        os.environ['MEGATRON_LM_PATH'] = git_clone_github(
            'https://github.com/NVIDIA/Megatron-LM', branch='core_r0.12.0')
    with safe_ddp_context(hash_id='megatron-lm'):
        if not is_megatron_available():
            subprocess_run([sys.executable, '-m', 'pip', 'install', '-e', os.environ['MEGATRON_LM_PATH']])
    sys.path.insert(0, os.environ['MEGATRON_LM_PATH'])
    _patch_megatron()
