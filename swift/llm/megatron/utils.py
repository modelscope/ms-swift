import os
import sys
from functools import wraps

import torch
import torch.distributed as dist

from swift.llm import git_clone_github, is_megatron_available
from swift.utils import get_dist_setting, subprocess_run


def init_megatron_env() -> None:

    if 'MEGATRON_LM_PATH' not in os.environ:
        megatron_path = git_clone_github('https://github.com/NVIDIA/Megatron-LM')
        os.environ['MEGATRON_LM_PATH'] = megatron_path
    else:
        megatron_path = os.environ['MEGATRON_LM_PATH']
    if not is_megatron_available():
        subprocess_run(['pip', 'install', '-e', megatron_path])
    sys.path.append(megatron_path)

    if 'PAI_MEGATRON_PATCH_PATH' not in os.environ:
        megatron_patch_path = git_clone_github('https://github.com/alibaba/Pai-Megatron-Patch')
        os.environ['PAI_MEGATRON_PATCH_PATH'] = megatron_patch_path
    sys.path.append(os.environ['PAI_MEGATRON_PATCH_PATH'])


def get_model_seires(model_type: str) -> str:
    if model_type.startswith('qwen2'):
        return 'qwen2'
    else:
        raise ValueError(f'model_type: {model_type} not supported')


def patch_megatron(tokenizer):

    def build_tokenizer(args):
        args.extra_vocab_size = args.padded_vocab_size - tokenizer.vocab_size
        return tokenizer

    from megatron.training import global_vars
    global_vars.build_tokenizer = build_tokenizer

    from megatron.training import get_args
    from megatron.training import initialize
    _old_initialize_distributed = initialize._initialize_distributed

    @wraps(_old_initialize_distributed)
    def _initialize_distributed():
        args = get_args()
        if dist.is_initialized():
            args.rank, args.local_rank, args.world_size, args.local_world_size = get_dist_setting()
            torch.cuda.set_device(args.local_rank)
        return _old_initialize_distributed()

    initialize._initialize_distributed = _initialize_distributed

    from megatron.training import training
    _old_load_checkpoint = training.load_checkpoint

    @wraps(_old_load_checkpoint)
    def load_checkpoint(model, optimizer, opt_param_scheduler, load_arg='load', strict=False):
        # default: strict=False
        return _old_load_checkpoint(model, optimizer, opt_param_scheduler, load_arg=load_arg, strict=strict)

    training.load_checkpoint = load_checkpoint
