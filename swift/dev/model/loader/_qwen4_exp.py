# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import os
from contextvars import ContextVar
from functools import wraps
from types import MethodType

import torch
from torch import nn
from transformers import PreTrainedModel
from twinkle.model.host_embedding import HostEmbedding as _HostEmbeddingProxy
from swift.dev.utils import get_logger

logger = get_logger()
_QWEN4_EXP_PLE_INPUT_IDS = ContextVar('qwen4_exp_ple_input_ids', default=None)


def _qwen4_exp_ple_ngram_cache_path(model_dir: str) -> str:
    import hashlib
    from modelscope.hub.utils.utils import get_cache_dir

    model_key = hashlib.sha256(os.path.realpath(model_dir).encode()).hexdigest()
    return os.path.join(get_cache_dir(), 'ms_swift', 'ple', model_key, 'ple_ngram_embedding.safetensors')


def _iter_qwen4_exp_ple_ngram_modules(model: PreTrainedModel):
    language_model = getattr(getattr(model, 'model', None), 'language_model', None)
    if language_model is None or not hasattr(language_model, 'layers'):
        return
    for layer in language_model.layers:
        ple = getattr(layer, 'ple', None)
        if ple is None:
            continue
        ngram_mod = getattr(getattr(ple, 'ple_embedding', None), 'ngram_embedding', None)
        if isinstance(ngram_mod, nn.Embedding):
            yield ple, ngram_mod


def _qwen4_exp_ple_ngram_source_fingerprint(model_dir: str) -> str:
    import hashlib
    import json

    # The cache is derived from this checkpoint; track source metadata without
    # rereading the full (~96 GiB) PLE shards on every launch.
    manifest = {}
    index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    for name in ('config.json', 'model.safetensors.index.json'):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                manifest[name] = hashlib.sha256(f.read()).hexdigest()
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)['weight_map']
        shards = {filename for name, filename in weight_map.items() if '.ngram_embedding.' in name}
    else:
        shards = {'model.safetensors'}
    for name in sorted(shards):
        stat = os.stat(os.path.join(model_dir, name))
        manifest[name] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _qwen4_exp_single_ngram_module(model: PreTrainedModel):
    pairs = list(_iter_qwen4_exp_ple_ngram_modules(model))
    if len(pairs) > 1:
        raise ValueError('Qwen4Exp FSDP2 currently supports at most one PLE n-gram embedding table.')
    return pairs[0][1] if pairs else None


def _read_qwen4_exp_ple_ngram_cache(cache_path: str, fingerprint: str, expected_weight: torch.Tensor) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(cache_path, framework='pt', device='cpu') as f:
        metadata = f.metadata() or {}
        if metadata.get('source_fingerprint') != fingerprint:
            raise ValueError('Qwen4Exp PLE cache does not match the source checkpoint.')
        weight = f.get_tensor('weight')
    if weight.shape != expected_weight.shape or weight.dtype != expected_weight.dtype:
        raise ValueError('Qwen4Exp PLE cache shape or dtype does not match the model.')
    return weight


def _export_qwen4_exp_ple_ngram_cache(model: PreTrainedModel, cache_path: str, model_dir: str) -> None:
    import tempfile
    from safetensors import SafetensorError
    from safetensors.torch import save_file

    module = _qwen4_exp_single_ngram_module(model)
    if module is None:
        return
    weight = module.weight
    if weight.is_meta:
        raise ValueError('Cannot export a Qwen4Exp PLE cache from a meta tensor.')
    fingerprint = _qwen4_exp_ple_ngram_source_fingerprint(model_dir)
    if os.path.isfile(cache_path):
        try:
            _read_qwen4_exp_ple_ngram_cache(cache_path, fingerprint, weight)
            return
        except (OSError, ValueError, SafetensorError) as exc:
            logger.warning(f'Rebuilding invalid Qwen4Exp PLE cache: {exc}')
    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='.ple_ngram_', suffix='.safetensors', dir=cache_dir)
    os.close(fd)
    try:
        save_file({'weight': weight.detach().cpu().contiguous()},
                  tmp_path,
                  metadata={'source_fingerprint': fingerprint})
        _read_qwen4_exp_ple_ngram_cache(tmp_path, fingerprint, weight)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    logger.info(f'Exported Qwen4Exp PLE n-gram cache to {cache_path}')


def _load_qwen4_exp_ple_ngram_from_cache(model: PreTrainedModel, cache_path: str, model_dir: str) -> None:
    module = _qwen4_exp_single_ngram_module(model)
    if module is None:
        return
    fingerprint = _qwen4_exp_ple_ngram_source_fingerprint(model_dir)
    weight = _read_qwen4_exp_ple_ngram_cache(cache_path, fingerprint, module.weight)
    module.weight = nn.Parameter(weight, requires_grad=module.weight.requires_grad)
    logger.info(f'Loaded Qwen4Exp PLE n-gram embedding from cache: {cache_path}')


def _pin_qwen4_exp_ple_ngram_on_host(model: PreTrainedModel) -> None:
    pinned = 0
    for ple, ngram_mod in _iter_qwen4_exp_ple_ngram_modules(model):
        if ngram_mod.weight.requires_grad:
            continue
        ple.ple_embedding.ngram_embedding = _HostEmbeddingProxy(ngram_mod)
        pinned += 1

    if pinned:
        logger.info('Pinned Qwen4Exp PLE n-gram embedding(s) on host CPU and excluded them from FSDP/LoRA.')


def _patch_qwen4_exp_ple_input_ids(model) -> None:
    """Supply PLE ids for precomputed multimodal embeddings without reverse lookup."""
    if not getattr(model.config.text_config, 'ple_layer_ids', None):
        return
    core = model.model
    if getattr(core, '_swift_ple_input_ids_patched', False):
        return
    forward = core.forward
    if 'ple_input_ids' in inspect.signature(forward).parameters:
        return
    language_model = core.language_model
    # Scope the ids to one forward, including nested calls and exception cleanup.
    # Do not store batch tensors on the model or change Transformers globally.

    @wraps(getattr(language_model.reverse_embedding, '__func__', language_model.reverse_embedding))
    def _reverse_embedding(self, inputs_embeds):
        context = _QWEN4_EXP_PLE_INPUT_IDS.get()
        if context is not None and context[0] is self and context[1] is not None:
            return context[1]
        return self._swift_reverse_embedding(inputs_embeds)

    @wraps(getattr(forward, '__func__', forward))
    def _forward(self, *args, ple_input_ids=None, **kwargs):
        inputs_embeds = kwargs.get('inputs_embeds')
        if ple_input_ids is not None and inputs_embeds is not None:
            if ple_input_ids.shape != inputs_embeds.shape[:2]:
                raise ValueError('ple_input_ids must match the batch and sequence dimensions of inputs_embeds.')
        token = _QWEN4_EXP_PLE_INPUT_IDS.set((self.language_model, ple_input_ids))
        try:
            return self._swift_ple_forward(*args, **kwargs)
        finally:
            _QWEN4_EXP_PLE_INPUT_IDS.reset(token)

    language_model._swift_reverse_embedding = language_model.reverse_embedding
    language_model.reverse_embedding = MethodType(_reverse_embedding, language_model)
    core._swift_ple_forward = forward
    core.forward = MethodType(_forward, core)
    core._swift_ple_input_ids_patched = True


