# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from typing import List, Optional

import requests
from modelscope.hub.api import ModelScopeConfig
from modelscope.hub.utils.utils import get_cache_dir
from tqdm import tqdm

from .logger import get_logger
from .torch_utils import is_local_master, safe_ddp_context
from .utils import subprocess_run

logger = get_logger()


def safe_snapshot_download(model_id_or_path: str,
                           revision: Optional[str] = None,
                           download_model: bool = True,
                           use_hf: Optional[bool] = None,
                           hub_token: Optional[str] = None,
                           ignore_patterns: Optional[List[str]] = None,
                           check_local: bool = False,
                           **kwargs) -> str:
    """Download model snapshot safely with DDP context protection.

    This function attempts to download a model from HuggingFace or ModelScope hub,
    with support for local paths, subfolder specification, and distributed training
    context protection. It handles various path formats and provides flexible
    file filtering options.

    Args:
        model_id_or_path (str): The model identifier on the hub (e.g., 'Qwen/Qwen2.5-7B-Instruct')
            or a local path to the model directory. Supports subfolder specification
            using colon syntax (e.g., 'model_id:subfolder').
        revision (Optional[str], optional): Specific model version/revision to download
            (branch name, tag, or commit hash). Defaults to None (latest version).
        download_model (bool, optional): Whether to download model weight files
            (.bin, .safetensors). If False, only config and tokenizer files are
            downloaded. Defaults to True.
        use_hf (Optional[bool], optional): Force using HuggingFace Hub (True) or ModelScope (False).
            If None, it is controlled by the environment variable `USE_HF`, which defaults to '0'.
            Default: None.
        hub_token (Optional[str], optional): Authentication token for accessing private
            or gated models. Defaults to None.
        ignore_patterns (Optional[List[str]], optional): List of glob patterns for files
            to exclude from download. If None, uses default patterns to exclude zip,
            gguf, pth, pt, and other auxiliary files. Defaults to None.
        check_local (bool, optional): Whether to check for a local directory matching
            the last component of model_id_or_path before attempting download.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the underlying hub download function.

    Returns:
        str: Absolute path to the model directory where files are stored.

    Raises:
        ValueError: If model_id_or_path starts with '/' (absolute path) and the path
            does not exist.
    Examples:
        >>> # Download from hub
        >>> model_dir = safe_snapshot_download('Qwen/Qwen2.5-7B-Instruct')

        >>> # Download config only (no weights)
        >>> model_dir = safe_snapshot_download('Qwen/Qwen2.5-7B-Instruct', download_model=False)
    """
    from swift.hub import get_hub
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
        with safe_ddp_context(hash_id=model_id_or_path):
            model_dir = hub.download_model(model_id_or_path, revision, ignore_patterns, token=hub_token, **kwargs)

        logger.info(f'Loading the model using model_dir: {model_dir}')

    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    if sub_folder:
        model_dir = os.path.join(model_dir, sub_folder)
    assert os.path.isdir(model_dir), f'model_dir: {model_dir}'
    return model_dir


def git_clone_github(github_url: str,
                     *,
                     local_repo_name: Optional[str] = None,
                     branch: Optional[str] = None,
                     commit_hash: Optional[str] = None) -> str:
    if github_url.endswith('.git'):
        github_url = github_url[:-4]
    git_cache_dir = os.path.join(get_cache_dir(), '_github')
    os.makedirs(git_cache_dir, exist_ok=True)
    if local_repo_name is None:
        github_url = github_url.rstrip('/')
        local_repo_name = github_url.rsplit('/', 1)[1]
    github_url = f'{github_url}.git'
    local_repo_path = os.path.join(git_cache_dir, local_repo_name)
    with safe_ddp_context('git_clone', use_barrier=True):
        repo_existed = os.path.exists(local_repo_path)
        if not is_local_master() and repo_existed:
            return local_repo_path
        if repo_existed:
            command = ['git', '-C', local_repo_path, 'fetch']
            subprocess_run(command)
            if branch is not None:
                command = ['git', '-C', local_repo_path, 'checkout', branch]
                subprocess_run(command)
        else:
            command = ['git', '-C', git_cache_dir, 'clone', github_url, local_repo_name]
            if branch is not None:
                command += ['--branch', branch]
            subprocess_run(command)

        if commit_hash is not None:
            command = ['git', '-C', local_repo_path, 'reset', '--hard', commit_hash]
            subprocess_run(command)
        elif repo_existed:
            command = ['git', '-C', local_repo_path, 'pull']
            subprocess_run(command)
    logger.info(f'local_repo_path: {local_repo_path}')
    return local_repo_path


def download_ms_file(url: str, local_path: str, cookies=None) -> None:
    if cookies is None:
        cookies = ModelScopeConfig.get_cookies()
    resp = requests.get(url, cookies=cookies, stream=True)
    with open(local_path, 'wb') as f:
        for data in tqdm(resp.iter_lines()):
            f.write(data)
