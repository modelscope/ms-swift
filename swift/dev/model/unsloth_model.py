# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import dataclasses
import inspect
import os
from typing import Any, Dict, Optional

from twinkle.model.transformers.transformers import TransformersModel as TwinkleTransformersModel
from twinkle.model.transformers.transformers import _default_adapter_name
from twinkle.patch import Patch, apply_context
from twinkle.utils import get_logger

from swift.dev.patch import UnslothDistributedFunctionPatch

logger = get_logger()

# unsloth's get_peft_model takes no adapter_name, so the adapter always carries peft's default
# name; twinkle keys its bookkeeping off the requested name, so any other name would disagree.
_PEFT_DEFAULT_ADAPTER_NAME = 'default'

# Set before the first unsloth import. UNSLOTH_RETURN_LOGITS is load-bearing: unsloth fuses the CE
# loss and returns ``logits=None`` by default, while twinkle's Loss objects need ``outputs['logits']``.
_UNSLOTH_ENV = {
    'UNSLOTH_RETURN_LOGITS': '1',
    'UNSLOTH_DISABLE_STATISTICS': '1',
    'UNSLOTH_IS_PRESENT': '1',
}

# Load kwargs that mean "the parameters are sharded", which unsloth cannot consume.
_SHARDED_LOAD_KEYS = ('tp_plan', 'tp_size', 'distributed_config', 'device_mesh')

# LoraConfig fields not forwarded to unsloth's get_peft_model: save-time bookkeeping, a live object
# peft rebuilds, a training disabler, and a variant-specific default unsloth owns.
_LORA_BOOKKEEPING_FIELDS = frozenset({
    'peft_type',
    'peft_version',
    'base_model_name_or_path',
    'revision',
    'auto_mapping',
    'inference_mode',
    'runtime_config',
    'task_type',
})


def _accepted_params(fn) -> frozenset:
    """The keyword names ``fn`` declares, excluding ``*args`` / ``**kwargs``.

    A wrapper such as ``FastModel.get_peft_model(model, *args, **kwargs)`` reports almost nothing,
    so this must only be used on entry points with a real signature (the ``from_pretrained``
    variants); filtering LoRA kwargs against it would drop everything.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return frozenset()
    return frozenset(name for name, p in params.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))


def _unsloth_lora_kwargs(config,
                         max_seq_length: int,
                         use_gradient_checkpointing: Any = 'unsloth',
                         extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """peft ``LoraConfig`` -> ``get_peft_model`` kwargs for unsloth.

    Every LoRA field the caller set is forwarded, including the ones unsloth does not declare
    (unsloth filters its ``**kwargs`` through ``inspect.signature(LoraConfig)``, so ``use_dora`` /
    ``trainable_token_indices`` land where they should). ``target_modules`` passes through untouched
    (``'all-linear'`` is unsloth's own shorthand -- expanding it here would narrow the run). Only the
    tuner *type* is rejected: trainable_tokens produces a PeftConfig unsloth cannot represent.
    """
    peft_type = getattr(config, 'peft_type', None)
    peft_type = str(getattr(peft_type, 'value', peft_type) or '').upper()
    if peft_type != 'LORA':
        raise NotImplementedError(f'unsloth implements plain LoRA only, got peft_type={peft_type or None!r} '
                                  f'(trainable_tokens / ...). Use the default backend for those.')

    kwargs: Dict[str, Any] = {}
    for field in dataclasses.fields(config):
        if field.name in _LORA_BOOKKEEPING_FIELDS:
            continue
        value = getattr(config, field.name, None)
        if value is not None:
            kwargs[field.name] = value

    if isinstance(kwargs.get('target_modules'), (set, frozenset)):
        # peft's __post_init__ turns a list into a set; sort for a deterministic layer order.
        kwargs['target_modules'] = sorted(kwargs['target_modules'])

    # unsloth's offloaded checkpointing: replaces the HF enable() twinkle already ran; False keeps
    # checkpointing off, so a caller asking for none is not overridden here.
    kwargs['use_gradient_checkpointing'] = use_gradient_checkpointing
    kwargs['max_seq_length'] = max_seq_length
    # unsloth-only LoRA knobs (finetune_vision_layers, finetune_last_n_layers, qat_scheme, ...) have
    # no LoraConfig equivalent, so they arrive separately and win over anything derived above.
    kwargs.update(extra or {})
    return kwargs


class _UnslothLoader:
    """A ``model_cls`` stand-in whose ``from_pretrained`` routes construction through unsloth.

    twinkle builds the model with ``model_cls.from_pretrained(model_id, config=..., **load_kwargs)``
    and only ever touches that one attribute, so passing this object as ``model_cls`` reuses the
    entire parent ``__init__`` and swaps just the construction call -- cheaper and far less
    drift-prone than re-implementing ``__init__``.
    """

    def __init__(self, owner: 'UnslothModel') -> None:
        self._owner = owner

    def from_pretrained(self, model_id: str, config=None, **load_kwargs):
        # ``config`` is dropped on purpose: unsloth reads the checkpoint's own config and rebuilds
        # the module graph around it, so an externally built PretrainedConfig has nowhere to go.
        return self._owner._load_unsloth_model(model_id, **load_kwargs)


class _UnslothPeftPatch(Patch):
    """Swap twinkle's module-global ``get_peft_model`` for unsloth's while ``_patch_adapter`` runs.

    The first-class patch channel (not mock.patch): idempotent, reversible, and the same pattern
    UnslothDistributedFunctionPatch uses.
    """

    _MARKER = '_swift_origin_get_peft_model'

    def __init__(self, owner: 'UnslothModel') -> None:
        self._owner = owner

    def __call__(self, module=None, *args, **kwargs):
        import twinkle.model.transformers.transformers as tmt
        if hasattr(tmt, self._MARKER):
            return module
        setattr(tmt, self._MARKER, tmt.get_peft_model)
        tmt.get_peft_model = self._owner._unsloth_get_peft_model
        return module

    def unpatch(self, module=None, *args, **kwargs):
        import twinkle.model.transformers.transformers as tmt
        origin = getattr(tmt, self._MARKER, None)
        if origin is not None:
            tmt.get_peft_model = origin
            delattr(tmt, self._MARKER)
        return module


class UnslothModel(TwinkleTransformersModel):
    """A ``TransformersModel`` whose weights and LoRA come from unsloth.

    ``model_id`` (required) plus unsloth's own knobs -- max_seq_length / full_finetuning /
    load_in_{4,8,16}bit / variant / is_multimodal / is_moe / use_gradient_checkpointing /
    offload_embedding / unsloth_tiled_mlp / float32_mixed_precision / qat_scheme, signature-filtered
    against the chosen entry class; unsloth_kwargs / unsloth_lora_kwargs escape to
    from_pretrained / get_peft_model.
    """

    def __init__(self,
                 model_id: Optional[str] = None,
                 *,
                 max_seq_length: int = 2048,
                 full_finetuning: bool = False,
                 load_in_4bit: bool = False,
                 load_in_8bit: bool = False,
                 load_in_16bit: bool = False,
                 variant: Optional[str] = None,
                 is_multimodal: bool = False,
                 is_moe: bool = False,
                 use_gradient_checkpointing: Any = 'unsloth',
                 offload_embedding: bool = False,
                 unsloth_tiled_mlp: bool = False,
                 float32_mixed_precision: Optional[bool] = None,
                 qat_scheme: Optional[str] = None,
                 unsloth_kwargs: Optional[Dict[str, Any]] = None,
                 unsloth_lora_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        if model_id is None:
            raise ValueError('UnslothModel requires `model_id`: unsloth loads a checkpoint by name or path '
                             'and cannot build a blank model from a config.')
        if kwargs.get('memory_efficient_init'):
            # That path bypasses this class's loader on every rank but local rank 0, which would run
            # unpatched HF modules while rank 0 runs unsloth's. Fail instead of training a mismatch.
            raise NotImplementedError('UnslothModel does not support memory_efficient_init: the rank0-broadcast '
                                      'path skips unsloth on the other ranks.')
        # Construction is owned by unsloth; an explicit model_cls has nothing to act on.
        kwargs.pop('model_cls', None)

        os.environ.update(_UNSLOTH_ENV)
        self._unsloth_variant = variant or ('FastVisionModel'
                                            if is_multimodal else 'FastModel' if is_moe else 'FastLanguageModel')
        # Filtered against the variant's from_pretrained signature; a knob it lacks is logged, not raised.
        self._unsloth_options = {
            'max_seq_length': max_seq_length,
            'full_finetuning': full_finetuning,
            'load_in_4bit': load_in_4bit,
            'load_in_8bit': load_in_8bit,
            'load_in_16bit': load_in_16bit,
            'use_gradient_checkpointing': use_gradient_checkpointing,
            'offload_embedding': offload_embedding,
            'unsloth_tiled_mlp': unsloth_tiled_mlp,
            'float32_mixed_precision': float32_mixed_precision,
            'qat_scheme': qat_scheme,
        }
        self._unsloth_kwargs = dict(unsloth_kwargs or {})
        self._unsloth_lora_kwargs = dict(unsloth_lora_kwargs or {})
        self._use_gradient_checkpointing = use_gradient_checkpointing
        # Filled by the loader, which runs inside super().__init__ below.
        self._unsloth_tokenizer = None

        super().__init__(model_cls=_UnslothLoader(self), model_id=model_id, **kwargs)

        # unsloth returns its (fixed-up) tokenizer/processor next to the model. Keep it as the
        # default so save() writes a loadable checkpoint even when the recipe set no template.
        if self._unsloth_tokenizer is not None:
            self._default_tokenizer = self._unsloth_tokenizer

    # --- construction ---------------------------------------------------------

    def _unsloth_cls(self):
        try:
            import unsloth
        except ImportError as e:
            raise ImportError('UnslothModel requires the `unsloth` package. '
                              'Install it with `pip install unsloth`.') from e
        cls = getattr(unsloth, self._unsloth_variant, None)
        if cls is None:
            available = sorted(n for n in dir(unsloth) if n.startswith('Fast'))
            raise ValueError(f'unsloth has no entry class {self._unsloth_variant!r}; available: {available}')
        return cls

    def _load_unsloth_model(self, model_id: str, **load_kwargs):
        """Build the model with unsloth. Called by ``_UnslothLoader`` from the parent ``__init__``."""
        sharded = [key for key in _SHARDED_LOAD_KEYS if key in load_kwargs]
        if sharded:
            raise NotImplementedError(f'UnslothModel does not support tensor parallelism: unsloth patches '
                                      f'whole modules and cannot consume sharded weights (got {sharded}).')

        # The import is inside the context too: unsloth runs its compiled-kernel setup through
        # distributed_function at import time, not just during from_pretrained.
        with apply_context(None, UnslothDistributedFunctionPatch()):
            unsloth_cls = self._unsloth_cls()
            accepted = _accepted_params(unsloth_cls.from_pretrained)
            # twinkle forwards every leftover ctor kwarg to the construction call; keep what this
            # variant declares and report the rest, so an ignored request is visible in the log.
            candidates = {**self._unsloth_options, **load_kwargs}
            forwarded = {k: v for k, v in candidates.items() if k in accepted and v is not None}
            # Only report names that were actually *set* and are not in this variant's signature; an
            # unset knob (None) is not something the caller asked for.
            unsupported = sorted(k for k, v in candidates.items() if v is not None and k not in accepted)
            if unsupported:
                logger.info(f'unsloth.{self._unsloth_variant}.from_pretrained does not take {unsupported}; ignored.')
            # unsloth_kwargs is the explicit escape hatch, so it is not filtered away.
            model, tokenizer = unsloth_cls.from_pretrained(model_name=model_id, **forwarded, **self._unsloth_kwargs)
        self._unsloth_tokenizer = tokenizer
        logger.info(f'Loaded {model_id} through unsloth.{self._unsloth_variant} '
                    f'(4bit={self._unsloth_options["load_in_4bit"]}, '
                    f'full_finetuning={self._unsloth_options["full_finetuning"]}).')
        return model

    # --- LoRA -----------------------------------------------------------------

    def _patch_adapter(self, adapter_name: str, config_or_dir, **kwargs):
        """Install LoRA through unsloth instead of peft.

        The parent resolves ``get_peft_model`` as a module global, so swapping that one symbol keeps
        all the surrounding bookkeeping (optimizer group, adapter_config, lora dtype fixup, active
        group) byte-for-byte identical -- the alternative is copying the whole method and letting it
        drift. Loading a *saved* adapter directory is left on peft's ``PeftModel.from_pretrained``,
        which is the format unsloth itself writes.
        """
        if isinstance(config_or_dir, str):
            return super()._patch_adapter(adapter_name, config_or_dir, **kwargs)
        with apply_context(None, _UnslothPeftPatch(self)):
            return super()._patch_adapter(adapter_name, config_or_dir, **kwargs)

    def _unsloth_get_peft_model(self, model, config, adapter_name: str = _default_adapter_name, **kwargs):
        if adapter_name != _PEFT_DEFAULT_ADAPTER_NAME:
            raise NotImplementedError(f'unsloth installs a single adapter under peft\'s default name '
                                      f'{_PEFT_DEFAULT_ADAPTER_NAME!r} and takes no adapter_name, so '
                                      f'adapter_name={adapter_name!r} cannot be honoured; multi-adapter '
                                      f'serving needs the default backend.')
        lora_kwargs = _unsloth_lora_kwargs(config, self._unsloth_options['max_seq_length'],
                                           self._use_gradient_checkpointing, self._unsloth_lora_kwargs)
        logger.info(f'unsloth_config: {lora_kwargs}')
        return self._unsloth_cls().get_peft_model(model, **lora_kwargs)
