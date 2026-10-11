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

from .env import dataset_offline, use_hf_hub
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


@contextmanager
def _forced_offline():
    """Force every hub access layer into offline mode for the duration of the block.

    Two kinds of switch are set together, because the readers differ:

    * env vars ``HF_DATASETS_OFFLINE`` / ``HF_HUB_OFFLINE`` -- picked up by late imports, subprocesses
      and modelscope's env-based checks;
    * the in-process module flags that ``datasets`` and ``huggingface_hub`` read *dynamically at call
      time*: ``datasets.config.HF_HUB_OFFLINE`` (consulted by ``_raise_if_offline_mode_is_enabled`` and
      the ``CachedDatasetModuleFactory`` fallback) and ``huggingface_hub.constants.HF_HUB_OFFLINE``
      (consulted by ``is_offline_mode()``). ``datasets.config.HF_DATASETS_OFFLINE`` is deliberately not
      touched: it is an import-time alias no runtime path reads.

    All imports are lazy so importing this module never pulls in the heavy hub stacks. Previous values
    are restored in ``finally`` -- including deleting env keys that did not exist before -- so a failed
    offline load leaves no residue that could poison a later online load.
    """
    import datasets.config as ds_config
    import huggingface_hub.constants as hf_constants

    env_keys = ('HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE')
    saved_env = {key: os.environ.get(key) for key in env_keys}
    saved_ds_offline = ds_config.HF_HUB_OFFLINE
    saved_hf_offline = hf_constants.HF_HUB_OFFLINE
    try:
        for key in env_keys:
            os.environ[key] = '1'
        ds_config.HF_HUB_OFFLINE = True
        hf_constants.HF_HUB_OFFLINE = True
        yield
    finally:
        for key in env_keys:
            saved = saved_env[key]
            if saved is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved
        ds_config.HF_HUB_OFFLINE = saved_ds_offline
        hf_constants.HF_HUB_OFFLINE = saved_hf_offline


def _is_network_error(exc: BaseException) -> bool:
    """Whether ``exc`` from a hub load is a *connectivity* failure (eligible for offline cache fallback)
    rather than a *real* error (missing repo / auth / gated) that must propagate untouched.

    Precedence is deliberate and each step exists to kill a specific misclassification:

    1. connectivity types first -- ``LocalEntryNotFoundError`` subclasses ``EntryNotFoundError`` (a real
       error), so the specific offline/connectivity type must be matched before its parent, otherwise a
       cold-cache offline failure would be read as "genuinely absent" and never retried;
    2. real-error types next -- a true 404 / auth / gated failure is never retried as offline;
    3. message keywords, only for hub-specific errors whose type we do not know (e.g. modelscope-native
       resolution failures), scanning real-error keywords before connectivity keywords so a 404 message
       is never swept into the network bucket;
    4. anything unrecognized counts as a real error -- fail loudly, never silently mask it as offline.
    """
    import socket

    import requests.exceptions as req_exc
    from datasets import exceptions as ds_exceptions
    from huggingface_hub import errors as hf_errors

    network_types = (
        ConnectionError,  # builtin; also catches hf OfflineModeIsEnabled and modelscope's wrapped error
        socket.gaierror,
        socket.timeout,
        req_exc.ConnectionError,
        req_exc.ConnectTimeout,
        req_exc.ReadTimeout,
        req_exc.Timeout,
        hf_errors.LocalEntryNotFoundError,
        hf_errors.OfflineModeIsEnabled,
    )
    if isinstance(exc, network_types):
        return True

    real_types = (
        ds_exceptions.DatasetNotFoundError,
        hf_errors.RepositoryNotFoundError,
        hf_errors.EntryNotFoundError,
        hf_errors.GatedRepoError,
    )
    if isinstance(exc, real_types):
        return False

    msg = str(exc).lower()
    real_keywords = ('401', '403', '404', 'gated', 'private', 'unauthorized', 'forbidden', 'auth')
    if any(keyword in msg for keyword in real_keywords):
        return False
    network_keywords = ('connection', 'timed out', 'timeout', 'name or service not known',
                        'name resolution', 'failed to resolve', 'unreachable', 'getaddrinfo',
                        'max retries', "couldn't reach", 'offline', 'temporary failure', 'network')
    if any(keyword in msg for keyword in network_keywords):
        return True
    return False


def _offline_cache_error(describe: str, exc: BaseException) -> Exception:
    """The fail-loudly error for "offline and this dataset is not in the local cache".

    Names the dataset and the two ways out (download once while online, or point ``--dataset`` at the
    local copy), and chains the underlying exception so the low-level cause is still inspectable.
    """
    return ConnectionError(
        f'Cannot load dataset {describe} offline: it is not present in the local cache '
        f'({type(exc).__name__}: {exc}). Either connect once and load it so it gets cached, '
        f'or point --dataset at the local directory/file that already holds it.')


def _load_with_cache_fallback(online_fn, offline_fn, *, describe: str):
    """Run a hub dataset load, transparently falling back to the local cache when the network is down.

    Two callables rather than one, because on ModelScope the online and the offline load are different
    operations:

    * ``online_fn`` -- the normal hub load (``MsDataset.load`` on ModelScope, ``datasets.load_dataset``
      on HuggingFace);
    * ``offline_fn`` -- loads straight from the local cache through ``datasets.load_dataset(...,
      cache_dir=<hub cache>)`` under the forced-offline mode this wrapper sets, which routes to
      datasets' ``CachedDatasetModuleFactory``. On ModelScope it deliberately does *not* re-invoke
      ``MsDataset.load``: that loader resolves the repo over the network first (``get_repo_info``) and
      so never reaches modelscope's own cache fallback when offline. Going through ``datasets`` against
      ``MS_DATASETS_CACHE`` reproduces exactly the ``CachedDatasetModuleFactory(path,
      cache_dir=MS_DATASETS_CACHE)`` modelscope would have fallen back to, minus the unreachable
      network prelude.

    Policy:

    * explicit offline env -> take the fast path: force offline and load straight from cache, never
      opening a socket;
    * otherwise try the normal (online) load; if -- and only if -- it fails with a *connectivity* error,
      retry once against the local cache with offline mode forced;
    * a real error (missing repo / auth / gated) propagates untouched and is never retried as offline;
    * if the offline attempt also fails (cache miss), raise a clear, actionable error rather than
      letting a cryptic low-level exception escape.

    Args:
        online_fn: zero-arg callable performing the normal (online) hub load.
        offline_fn: zero-arg callable loading from the local cache under forced-offline mode.
        describe: human-readable dataset identity used in log/error messages.
    """
    if dataset_offline():
        with _forced_offline():
            try:
                return offline_fn()
            except Exception as exc:
                raise _offline_cache_error(describe, exc) from exc

    try:
        return online_fn()
    except Exception as exc:
        if not _is_network_error(exc):
            raise
        logger.warning(f'Failed to reach the hub for {describe} ({type(exc).__name__}: {exc}); '
                       f'falling back to the local dataset cache.')
        with _forced_offline():
            try:
                return offline_fn()
            except Exception as retry_exc:
                raise _offline_cache_error(describe, retry_exc) from retry_exc


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

        def online_fn():
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

        def offline_fn():
            # MsDataset.load cannot be reused offline: it resolves the repo over the network first and
            # never reaches modelscope's cache fallback. Load through datasets against the ModelScope
            # dataset cache instead -- the same CachedDatasetModuleFactory(path, MS_DATASETS_CACHE) that
            # modelscope's loader falls back to, just without the unreachable network prelude. revision
            # and download_mode are dropped: the cache scan ignores them, and only an already-cached
            # build can be served offline.
            from datasets import load_dataset
            from modelscope.utils.config_ds import MS_DATASETS_CACHE
            return load_dataset(
                dataset_id,
                name=subset_name,
                split=split,
                streaming=streaming,
                cache_dir=MS_DATASETS_CACHE,
                **load_kwargs,
            )

        return _load_with_cache_fallback(
            online_fn, offline_fn, describe=f'ms://{dataset_id}:{subset_name}#{split}')

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

        def online_fn():
            return load_dataset(
                dataset_id,
                name=subset_name,
                split=split,
                streaming=streaming,
                revision=revision,
                download_mode=download_mode,
                num_proc=num_proc,
                trust_remote_code=True)

        def offline_fn():
            # Same loader as online, but the forced-offline mode set by the fallback wrapper makes
            # datasets resolve the id through CachedDatasetModuleFactory against the default
            # HF_DATASETS_CACHE instead of reaching the Hub. revision/download_mode are dropped: the
            # cache scan ignores them, and only the already-cached build can be served offline.
            return load_dataset(
                dataset_id,
                name=subset_name,
                split=split,
                streaming=streaming,
                num_proc=num_proc,
                trust_remote_code=True)

        return _load_with_cache_fallback(
            online_fn, offline_fn, describe=f'hf://{dataset_id}:{subset_name}#{split}')

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
