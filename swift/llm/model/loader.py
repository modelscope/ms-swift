import os
from typing import Any, Dict, Optional, List
from modelscope.hub.utils.utils import get_cache_dir

from datasets.utils.filelock import FileLock
from swift import get_logger
from swift.hub import MSHub, HFHub
from swift.utils import is_unsloth_available, safe_ddp_context, is_dist, is_dist_ta
import torch.distributed as dist

import hashlib

logger = get_logger()

# Model Home: 'https://modelscope.cn/models/{model_id_or_path}'

def safe_snapshot_download(model_id_or_path: str,
                           revision: Optional[str] = None,
                           download_model: bool = True,
                           use_hf: bool = False,
                           ignore_file_pattern: Optional[List[str]] = None,
                           **kwargs) -> str:
    """Download model protected by DDP context

    Args:
        model_id_or_path: The model id or model path
        revision: The model revision
        download_model: Download model bin/safetensors files or not
        use_hf: use huggingface or modelscope

    Returns:
        model_dir
    """
    if (is_dist() or is_dist_ta()) and not dist.is_initialized():
        # Distributed but uninitialized
        lock_dir = os.path.join(get_cache_dir(), 'lockers')
        file_path = hashlib.md5(model_id_or_path.encode('utf-8')).hexdigest() + '.lock'
        file_path = os.path.join(lock_dir, file_path)
        context = FileLock(file_path)
    else:
        context = safe_ddp_context()
    hub = HFHub if use_hf else MSHub
    with context:
        if os.path.exists(model_id_or_path):
            model_dir = model_id_or_path
        else:
            if model_id_or_path[:1] in {'~', '/'}:  # startswith
                raise ValueError(f"path: '{model_id_or_path}' not found")
            model_dir = hub.download_model(model_id_or_path, revision, download_model, ignore_file_pattern, **kwargs)

        logger.info(f'Loading the model using model_dir: {model_dir}')

    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    assert os.path.isdir(model_dir), f'model_dir: {model_dir}'
    return model_dir


# [TODO:not impl]
# def load_by_unsloth(model_dir, torch_dtype, **kwargs):
#     """Load model by unsloth"""
#     assert is_unsloth_available(), 'please install unsloth if using `use_unsloth=True`'
#     from unsloth import FastLanguageModel
#     return FastLanguageModel.from_pretrained(
#         model_name=model_dir,
#         max_seq_length=kwargs.get('max_length', None),
#         dtype=torch_dtype,
#         load_in_4bit=kwargs.get('load_in_4bit', True),
#         trust_remote_code=True,
#     )
#

# def load_by_transformers(automodel_class, model_dir, model_config, torch_dtype, is_aqlm, is_training, model_kwargs,
#                          **kwargs):
#     """Load model by transformers"""
#     context = kwargs.get('context', None)
#     if is_aqlm and is_training:
#         require_version('transformers>=4.39')
#         import aqlm
#         context = aqlm.optimize_for_training()
#     if context is None:
#         context = nullcontext()
#     with context:
#         model = automodel_class.from_pretrained(
#             model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
#     return model
