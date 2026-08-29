# Copyright (c) ModelScope Contributors. All rights reserved.
"""Hub access for swift.dev: model snapshot download + dataset loading across ModelScope / HF.

Internalized from ``swift.hub`` and ``swift.utils.hub_utils`` so the dev stack no longer couples to
legacy. Only the subset dev actually uses is kept: ``get_hub`` selects a backend, ``HFHub`` / ``MSHub``
expose ``download_model`` + ``load_dataset`` (with ``try_login`` for ModelScope), and
``safe_snapshot_download`` wraps ``download_model`` with local-path handling and cross-rank
serialization. The push-to-hub / repo-creation surface of legacy ``swift.hub`` is intentionally not
copied -- dev never uploads.

Cross-rank serialization uses ``twinkle.utils.processing_lock`` (dev's coordination primitive) rather
than legacy ``safe_ddp_context``: a snapshot download is idempotent, content-addressed work, so
``sticky=True`` lets a late rank read the finished download instead of blocking on ``dist.barrier``.
"""
import logging
import os
from contextlib import contextmanager
from typing import List, Literal, Optional

from .env import use_hf_hub
from .logger import get_logger

logger = get_logger()


@contextmanager
def _ms_logger_context(level):
    """Temporarily raise the ModelScope logger level (silence its dataset-load chatter)."""
    from modelscope.utils.logger import get_logger as get_ms_logger
    ms_logger = get_ms_logger()
    origin_level = ms_logger.level
    ms_logger.setLevel(level)
    try:
        yield
    finally:
        ms_logger.setLevel(origin_level)


class MSHub:

    ms_token = None

    @classmethod
    def try_login(cls, token: Optional[str] = None) -> bool:
        from modelscope import HubApi
        if token is None:
            token = os.environ.get('MODELSCOPE_API_TOKEN')
        if token:
            api = HubApi()
            api.login(token)
            return True
        return False

    @classmethod
    def load_dataset(cls,
                     dataset_id: str,
                     subset_name: str,
                     split: str,
                     streaming: bool = False,
                     revision: Optional[str] = None,
                     download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
                     token: Optional[str] = None,
                     **kwargs):
        import modelscope
        from modelscope import MsDataset
        from packaging import version
        cls.try_login(token)
        if revision is None or revision == 'main':
            revision = 'master'
        load_kwargs = {}
        if version.parse(modelscope.__version__) >= version.parse('1.29.1'):
            load_kwargs['trust_remote_code'] = True
        with _ms_logger_context(logging.ERROR):
            return MsDataset.load(
                dataset_id,
                subset_name=subset_name,
                split=split,
                version=revision,
                download_mode=download_mode,
                use_streaming=streaming,
                **load_kwargs,
            )

    @classmethod
    def download_model(cls,
                       model_id_or_path: Optional[str] = None,
                       revision: Optional[str] = None,
                       ignore_patterns: Optional[List[str]] = None,
                       token: Optional[str] = None,
                       **kwargs):
        cls.try_login(token)
        if revision is None or revision == 'main':
            revision = 'master'
        logger.info(f'Downloading the model from ModelScope Hub, model_id: {model_id_or_path}')
        from modelscope import snapshot_download
        return snapshot_download(model_id_or_path, revision, ignore_patterns=ignore_patterns, **kwargs)


class HFHub:

    @classmethod
    def try_login(cls, token: Optional[str] = None) -> bool:
        pass

    @classmethod
    def load_dataset(cls,
                     dataset_id: str,
                     subset_name: str,
                     split: str,
                     streaming: bool = False,
                     revision: Optional[str] = None,
                     download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
                     num_proc: Optional[int] = None,
                     **kwargs):
        from datasets import load_dataset
        if revision is None or revision == 'master':
            revision = 'main'
        return load_dataset(
            dataset_id,
            name=subset_name,
            split=split,
            streaming=streaming,
            revision=revision,
            download_mode=download_mode,
            num_proc=num_proc,
            trust_remote_code=True)

    @classmethod
    def download_model(cls,
                       model_id_or_path: Optional[str] = None,
                       revision: Optional[str] = None,
                       ignore_patterns: Optional[List[str]] = None,
                       **kwargs):
        from transformers.utils import strtobool
        if revision is None or revision == 'master':
            revision = 'main'
        logger.info(f'Downloading the model from HuggingFace Hub, model_id: {model_id_or_path}')
        use_hf_transfer = strtobool(os.environ.get('USE_HF_TRANSFER', 'False'))
        if use_hf_transfer:
            from huggingface_hub import _snapshot_download
            _snapshot_download.HF_HUB_ENABLE_HF_TRANSFER = True
        from huggingface_hub import snapshot_download
        return snapshot_download(
            model_id_or_path, repo_type='model', revision=revision, ignore_patterns=ignore_patterns, **kwargs)


def get_hub(use_hf: Optional[bool] = None):
    if use_hf is None:
        use_hf = True if use_hf_hub() else False
    return {True: HFHub, False: MSHub}[use_hf]


def safe_snapshot_download(model_id_or_path: str,
                           revision: Optional[str] = None,
                           download_model: bool = True,
                           use_hf: Optional[bool] = None,
                           hub_token: Optional[str] = None,
                           ignore_patterns: Optional[List[str]] = None,
                           check_local: bool = False,
                           **kwargs) -> str:
    """Download a model snapshot, resolving local paths and serializing across ranks.

    Handles local directories, ``model_id:subfolder`` syntax, HF vs ModelScope selection, and
    filtering of non-weight files. Returns the absolute model directory.

    Args:
        model_id_or_path: Hub id (e.g. ``'Qwen/Qwen2.5-7B-Instruct'``) or a local path; supports a
            ``model_id:subfolder`` suffix.
        revision: Branch / tag / commit to download (``None`` = latest).
        download_model: If ``False``, skip weight files (``*.bin`` / ``*.safetensors``).
        use_hf: Force HF (``True``) or ModelScope (``False``); ``None`` follows the ``USE_HF`` env.
        hub_token: Auth token for private / gated models.
        ignore_patterns: Glob patterns to exclude; ``None`` uses the default auxiliary-file excludes.
        check_local: If ``True``, prefer a local dir matching the last path component before downloading.

    Returns:
        Absolute path to the model directory.

    Raises:
        ValueError: an absolute path (``/...``) that does not exist.
    """
    from twinkle.utils import processing_lock
    if check_local:
        model_suffix = model_id_or_path.rsplit('/', 1)[-1]
        if os.path.exists(model_suffix):
            model_dir = os.path.abspath(os.path.expanduser(model_suffix))
            logger.info(f'Loading the model using local model_dir: {model_dir}')
            return model_dir
    if ignore_patterns is None:
        ignore_patterns = [
            '*.zip', '*.gguf', '*.pth', '*.pt', 'consolidated*', 'onnx/*', '*.safetensors.md', '*.msgpack', '*.onnx',
            '*.ot', '*.h5'
        ]
    if not download_model:
        ignore_patterns += ['*.bin', '*.safetensors']
    hub = get_hub(use_hf)
    if model_id_or_path.startswith('~'):
        model_id_or_path = os.path.abspath(os.path.expanduser(model_id_or_path))
    model_path_to_check = '/'.join(model_id_or_path.split(':', 1))
    if os.path.exists(model_id_or_path):
        model_dir = model_id_or_path
        sub_folder = None
    elif os.path.exists(model_path_to_check):
        model_dir = model_path_to_check
        sub_folder = None
    else:
        if model_id_or_path.startswith('/'):  # startswith
            raise ValueError(f"path: '{model_id_or_path}' not found")
        model_id_or_path = model_id_or_path.split(':', 1)  # get sub_folder
        if len(model_id_or_path) == 1:
            model_id_or_path = [model_id_or_path[0], None]
        model_id_or_path, sub_folder = model_id_or_path
        if sub_folder is not None:
            kwargs['allow_patterns'] = [f"{sub_folder.rstrip('/')}/*"]
        with processing_lock(model_id_or_path, sticky=True):
            model_dir = hub.download_model(model_id_or_path, revision, ignore_patterns, token=hub_token, **kwargs)

        logger.info(f'Loading the model using model_dir: {model_dir}')

    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    if sub_folder:
        model_dir = os.path.join(model_dir, sub_folder)
    assert os.path.isdir(model_dir), f'model_dir: {model_dir}'
    return model_dir
