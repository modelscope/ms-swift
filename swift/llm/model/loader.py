import os
from contextlib import nullcontext
from typing import Optional, Dict, Any

from transformers.utils.versions import require_version

from swift import get_logger
from swift.hub import hub
from swift.utils.env import use_hf_hub
from swift.utils.import_utils import is_unsloth_available
from swift.utils.torch_utils import safe_ddp_context

logger = get_logger()

# Model Home: 'https://modelscope.cn/models/{model_id_or_path}'
MODEL_MAPPING: Dict[str, Dict[str, Any]] = {}


def safe_snapshot_download(model_type: str,
                           model_id_or_path: Optional[str] = None,
                           revision: Optional[str] = None,
                           download_model: bool = True,
                           **kwargs) -> str:
    """Download model protected by DDP context

    Args:
        model_type: The model type, can be None
        model_id_or_path: The model id or model path
        revision: The model revision
        download_model: Download model bin files or not
        **kwargs:

    Returns:
        The model dir
    """
    # Perform snapshot_download (ms or hf) based on model_type and model_id_or_path.
    model_info = MODEL_MAPPING.get(model_type, {})

    model_dir = None
    if model_id_or_path is None:
        model_dir = kwargs.pop('model_dir', None)  # compat with swift<1.7
        if model_dir is not None:
            model_id_or_path = model_dir
        else:
            model_id_or_path = model_info['hf_model_id' if use_hf_hub() else 'model_id_or_path']

    with safe_ddp_context():
        if model_id_or_path is not None and not os.path.exists(model_id_or_path):
            if model_id_or_path.startswith('/'):
                raise ValueError(f"path: '{model_id_or_path}' not found")
            ignore_file_pattern = model_info.get('ignore_file_pattern')
            model_dir = hub.download_model(model_id_or_path, revision, download_model, ignore_file_pattern, **kwargs)
        else:
            model_dir = model_id_or_path
        logger.info(f'Loading the model using model_dir: {model_dir}')

    model_dir = os.path.expanduser(model_dir)
    assert os.path.isdir(model_dir), f'model_dir: {model_dir}'
    return model_dir


def load_by_unsloth(model_dir, torch_dtype, **kwargs):
    assert is_unsloth_available(), 'please install unsloth if using `use_unsloth=True`'
    from unsloth import FastLanguageModel
    return FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=kwargs.get('max_length', None),
        dtype=torch_dtype,
        load_in_4bit=kwargs.get('load_in_4bit', True),
        trust_remote_code=True,
    )


def load_by_transformers(automodel_class, model_dir, model_config, torch_dtype,
                      is_aqlm, is_training, model_kwargs, **kwargs):
    context = kwargs.get('context', None)
    if is_aqlm and is_training:
        require_version('transformers>=4.39')
        import aqlm
        context = aqlm.optimize_for_training()
    if context is None:
        context = nullcontext()
    with context:
        model = automodel_class.from_pretrained(
            model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
    return model