# Copyright (c) ModelScope Contributors. All rights reserved.
"""Scope TF5 construction changes to the official tiny default-RoPE remote code.

Keep AutoModel loading and checkpoint parameter names intact. Temporary class
patches exist only during controlled construction; attention math lives separately.
"""
import torch
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from importlib import import_module
from threading import Lock, get_ident
from types import MethodType

TF4_VERSION = '4.57.1'
TF5_VERSION = '5.14.1'
SUPPORTED_TRANSFORMERS = (TF4_VERSION, TF5_VERSION)

_LING3_ROPE_CONSTRUCTION_LOCK = Lock()


@contextmanager
def ling3_rope_construction(model_dir, config, model_kwargs):
    """Yield a post-load buffer finalizer; TF4 construction is a no-op.

    TF5 validates the local tiny config before patching shared remote constructors.
    The lock rejects nested/concurrent adapter contexts; owner checks also reject
    foreign-thread calls to patched constructors. Call the finalizer after weights
    materialize and before leaving this context. Always restore class patches,
    including when loading fails. Unrelated direct remote loads must not overlap.
    """
    import os
    import transformers
    from packaging.version import Version

    version = Version(transformers.__version__)
    if version == Version(TF4_VERSION):
        yield None
        return
    if version != Version(TF5_VERSION):
        raise NotImplementedError('Ling3 RoPE compatibility has only been verified with Transformers 4.57.1 and 5.14.1')
    scaling = getattr(config, 'rope_scaling', None)
    if scaling is not None and not isinstance(scaling, dict):
        raise ValueError('Ling3 default RoPE requires a flat parameter dictionary or None')
    scaling = scaling or {}
    if (scaling.get('rope_type', scaling.get('type', 'default')) != 'default'
            or scaling.get('type', 'default') != 'default'
            or any(isinstance(value, dict) for value in scaling.values())):
        raise NotImplementedError('Ling3 compatibility supports default RoPE only; scaling is not converted')
    if scaling.get('factor', 1.0) != 1.0:
        raise ValueError('Ling3 default RoPE does not accept a scaling factor other than one')
    if (getattr(config, 'hidden_size', None), getattr(config, 'num_hidden_layers', None),
            getattr(config, 'qk_rope_head_dim', None), getattr(config, 'rope_theta', None)) != (1536, 24, 64, 6000000):
        raise NotImplementedError('Ling3 Transformers 5 compatibility is limited to the tiny default configuration')
    params = getattr(config, 'rope_parameters', None)
    if not isinstance(params, dict) or params.get('rope_theta') != config.rope_theta:
        raise ValueError('Ling3 rope_parameters must contain the original rope_theta')
    class_ref = getattr(config, 'auto_map', {}).get('AutoModelForCausalLM', '')
    if (not os.path.isdir(model_dir) or '--' in class_ref
            or class_ref != 'modeling_bailing_moe_v3.BailingMoeV3ForCausalLM'):
        raise NotImplementedError('Ling3 RoPE adaptation requires a local tiny model directory and local remote code')
    if model_kwargs.get('force_download'):
        raise NotImplementedError('Ling3 RoPE adaptation does not support forced dynamic-module reload')
    if not _LING3_ROPE_CONSTRUCTION_LOCK.acquire(blocking=False):
        raise RuntimeError('Ling3 nested or concurrent model construction is not supported')
    model_cls = None
    original_tied = None
    tied_captured = False
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # Match AutoModel's dynamic resolver options, without replacing its loader.
        options = {
            key: model_kwargs[key]
            for key in ('cache_dir', 'revision', 'code_revision', 'token', 'local_files_only', 'proxies', 'repo_type')
            if key in model_kwargs
        }
        model_cls = get_class_from_dynamic_module(class_ref, model_dir, **options)
        original_tied = model_cls._tied_weights_keys
        tied_captured = True
        if config.tie_word_embeddings:
            raise NotImplementedError('Ling3 TF5 metadata adapter currently supports untied embeddings only')
        if model_cls._tied_weights_keys == ['lm_head.weight']:
            model_cls._tied_weights_keys = {}
        elif model_cls._tied_weights_keys not in ({}, None):
            raise ValueError('Ling3 has unexpected tied-weight metadata')
        modeling = import_module(model_cls.__module__)
        cls = modeling.BailingMoeV3RotaryEmbedding
        original_init = cls.__init__
        mla_cls = getattr(modeling, 'BailingMoeV3MultiLatentAttention', None)
        original_mla_init = mla_cls.__init__ if mla_cls is not None else None
        owner = get_ident()
        attn_impl = getattr(config, '_attn_implementation', None)
        if attn_impl in ('flash_attention_3', 'flash_attention_4'):
            raise NotImplementedError('Ling3 compatibility supports Flash Attention 2 only')
        base_cls = None
        if attn_impl == 'flash_attention_2':
            base_cls = modeling.BailingMoeV3PreTrainedModel
            if getattr(base_cls, '_supports_flash_attn_2', False) is not True:
                raise ValueError('Ling3 remote model must declare Flash Attention 2 support')
            original_base_init = base_cls.__init__
            own_base_init = base_cls.__dict__.get('__init__')

            @wraps(original_base_init)
            def initialize_base(self, *args, **kwargs):
                if get_ident() != owner:
                    raise RuntimeError('Ling3 base construction requires the controlled thread')
                # HF5 checks the new name before constructing both model levels.
                # Bridge declared FA2 support only; do not advertise FA3/4 or
                # change existing instances through a class capability flag.
                self._supports_flash_attn = True
                return original_base_init(self, *args, **kwargs)

        hits = 0

        def restore_rope(module):
            """Rebuild nonpersistent frequencies after loading or a device/dtype move."""
            if module.inv_freq.device.type == 'meta':
                raise RuntimeError('Ling3 RoPE buffers must be materialized before finalization')
            inv_freq, scaling = module.rope_init_fn(module.config, module.inv_freq.device)
            # Explicit FP64 storage still uses canonical FP32 frequencies.
            # BF16/FP16 moves must not quantize these operands.
            if module.inv_freq.dtype == torch.float64:
                inv_freq = inv_freq.to(torch.float64)
            # Re-register through Module.__setattr__; keep nonpersistent membership,
            # but never copy exact FP32 frequencies into a quantized destination.
            module.inv_freq = inv_freq
            module.original_inv_freq = inv_freq.clone()
            module.attention_scaling = scaling

        def apply_rope(self, fn, recurse=True):
            # to_empty() allocates meta buffers without initializing values.
            # Rebuild only after the ordinary move has made storage real.
            torch.nn.Module._apply(self, fn, recurse=recurse)
            if self.inv_freq.device.type != 'meta':
                with torch.no_grad():
                    restore_rope(self)
            return self

        def initialize(self, config, device=None):
            """Replace only the removed default-RoPE construction interface."""
            nonlocal hits
            if get_ident() != owner:
                raise RuntimeError('Ling3 RoPE adaptation requires controlled single-threaded construction')
            torch.nn.Module.__init__(self)
            self.config = deepcopy(config)
            self.config.head_dim = config.qk_rope_head_dim
            self.config.partial_rotary_factor = 1.0
            self.config.rope_parameters = dict(params, partial_rotary_factor=1.0)
            self.rope_type = 'default'
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

            def default_rope(cfg, device=None, seq_len=None):
                dim = cfg.qk_rope_head_dim
                base = cfg.rope_parameters['rope_theta']
                # Match TF4's CPU FP32 initialization before moving the buffer.
                # A target-device formula can change backward operands even
                # when the resulting BF16 forward outputs are equal.
                inv_freq = 1.0 / (base**(torch.arange(0, dim, 2, device='cpu', dtype=torch.int64).float() / dim))
                return inv_freq.to(device=device), 1.0

            self.rope_init_fn = default_rope
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
            self.register_buffer('inv_freq', inv_freq, persistent=False)
            self.register_buffer('original_inv_freq', inv_freq.clone(), persistent=False)
            self._apply = MethodType(apply_rope, self)
            hits += 1

        @wraps(original_mla_init)
        def initialize_mla(self, config, *args, **kwargs):
            """Supply the legacy factor to this leaf without mutating shared config."""
            if get_ident() != owner:
                raise RuntimeError('Ling3 MLA adaptation requires controlled single-threaded construction')
            rope = getattr(config, 'rope_scaling', None)
            if not isinstance(rope, dict) or 'factor' in rope:
                return original_mla_init(self, config, *args, **kwargs)
            if (rope.get('rope_type', rope.get('type', 'default')) != 'default' or rope.get('mscale_all_dim', 0) != 0):
                raise NotImplementedError('Ling3 missing-factor adaptation requires default RoPE and mscale_all_dim=0')
            # Only this leaf constructor sees the temporary dictionary.
            local_config = deepcopy(config)
            local_config.rope_scaling = dict(rope, factor=1.0)
            try:
                original_mla_init(self, local_config, *args, **kwargs)
            finally:
                if getattr(self, 'config', None) is local_config:
                    self.config = config
            if any(getattr(module, 'config', None) is local_config for module in self.modules()):
                raise RuntimeError('Ling3 MLA retained a temporary configuration reference')

        def finalize(model):
            """Finalize the resolved instance after AutoModel materializes its buffers."""
            if not isinstance(model, model_cls):
                raise RuntimeError('Ling3 AutoModel resolved a different remote model class')
            model._tied_weights_keys = {}
            # HF 5 allocates nonpersistent meta buffers; Ling3's _init_weights
            # initializes only parameters, so rebuild RoPE after loading.
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, cls):
                        restore_rope(module)
            return model

        cls.__init__ = initialize
        if mla_cls is not None:
            mla_cls.__init__ = initialize_mla
        try:
            if base_cls is not None:
                base_cls.__init__ = initialize_base
            yield finalize
            if hits == 0 or modeling.BailingMoeV3RotaryEmbedding is not cls or cls.__init__ is not initialize:
                raise RuntimeError('Ling3 dynamic model construction did not use the adapted RotaryEmbedding class')
            if mla_cls is not None and (modeling.BailingMoeV3MultiLatentAttention is not mla_cls
                                        or mla_cls.__init__ is not initialize_mla):
                raise RuntimeError('Ling3 dynamic model resolved a different MLA constructor')
        finally:
            if base_cls is not None:
                if own_base_init is None:
                    del base_cls.__init__
                else:
                    base_cls.__init__ = own_base_init
            cls.__init__ = original_init
            if mla_cls is not None:
                mla_cls.__init__ = original_mla_init
    finally:
        if model_cls is not None and tied_captured:
            model_cls._tied_weights_keys = original_tied
        _LING3_ROPE_CONSTRUCTION_LOCK.release()
