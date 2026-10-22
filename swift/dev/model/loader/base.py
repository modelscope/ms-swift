# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import importlib
import os
import platform
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Type, Union

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

from swift.dev.utils import get_logger
from swift.dev.utils.tags import validate_tags

logger = get_logger(__name__)

#: Families already warned about trust_remote_code this process: the warning states a
#: per-family fact, so once per family is the right cadence -- not once per checkpoint.
_TRC_WARNED_FAMILIES: set = set()

__all__ = [
    'ModelArch', 'ModelInfo', 'ModelLoader', 'MODEL_ALIASES', 'MODEL_MAPPING', 'get_model_loader', 'match_model_type',
    'match_model_types_by_architectures', 'register_model', 'resolve_template'
]

# Fields that accept a bare string for convenience but always read back as a list; `moe_block` is a
# module-name hint rather than a trainable part but normalises the same way.
_PART_FIELDS = ('language_model', 'vision_tower', 'aligner', 'generator')
_LIST_FIELDS = _PART_FIELDS + ('moe_block', )

MODEL_MAPPING: Dict[str, Type['ModelLoader']] = {}
# Alias -> registered model_type, so `--model_type ovis_ocr2` and old `args.json` keep working after
# two legacy model_types that differed only by template were merged into one family.
MODEL_ALIASES: Dict[str, str] = {}
# HuggingFace class name -> model_types, lazily built and invalidated on every registration.
_ARCH_MAPPING: Dict[str, List[str]] = {}


@dataclass
class ModelArch:
    """Module-name prefixes splitting a checkpoint into separately-controllable parts.

    Values are verbatim ``named_parameters()`` / ``named_modules()`` names; consumers either
    ``startswith``-match them (freeze, LR grouping, quantization exclusion) or ``deep_getattr``-
    resolve them (gradient checkpointing, rollout weight splitting). A bare string is a one-element
    list; all-empty is correct for a plain LLM ("no such part").
    """

    # The LLM incl. its output head: gradient checkpointing, rollout weight splitting, freeze_llm.
    # Legacy's ~15 layer-path fields are gone: only the unimplemented llama-pro matcher read them.
    language_model: Union[str, List[str]] = field(default_factory=list)
    # Encoder: freeze_vit, vit_gradient_checkpointing, vit_lr group; no quantization/mcore.
    vision_tower: Union[str, List[str]] = field(default_factory=list)
    # Connector between encoder and LLM (projector/merger/resampler): freeze_aligner, aligner_lr.
    aligner: Union[str, List[str]] = field(default_factory=list)
    # An output generator that is not the LM head (speech/image decoder); trained by its own recipe.
    generator: Union[str, List[str]] = field(default_factory=list)
    # Output projection name when it is not literally 'lm_head' (internlm2 'output', chatglm
    # 'transformer.output_layer'); a leaf name suffices to keep the head out of 'all-linear' LoRA.
    lm_head: Optional[str] = None
    # Leaf class name(s) of a MoE block: a ZeRO-3 leaf marker, not a trainable part. Class names
    # cover remote-code blocks; class (not attribute) matching skips dense siblings (llama4).
    moe_block: Union[str, List[str]] = field(default_factory=list)

    def __setattr__(self, key: str, value) -> None:
        # Normalise on every assignment, not just in __post_init__: the annotation says a bare
        # string is fine, so `arch.vision_tower = 'visual'` must not leave a string for startswith.
        if key in _LIST_FIELDS:
            value = [value] if isinstance(value, str) else list(value or [])
        super().__setattr__(key, value)

    @property
    def lm_head_name(self) -> str:
        """Leaf name of the output projection; ``'lm_head'`` when nothing else was registered."""
        return self.lm_head.rsplit('.', 1)[-1] if self.lm_head else 'lm_head'

    @property
    def decoder(self) -> List[str]:
        """:attr:`language_model` without its output head.

        Registrations put the head inside ``language_model`` on purpose (freeze/LR ownership), but
        callers asking "where do the decoder layers live" must not receive the head prefix.
        """
        head = self.lm_head_name
        return [name for name in self.language_model if not name.endswith(head)]


@dataclass
class ModelInfo:
    """Everything known about one concrete checkpoint directory.

    Merges legacy's ``ModelInfo`` (facts read from ``config.json`` at load time) and ``ModelMeta``
    (facts hand-declared per family) -- the boundary had already leaked (``is_multimodal`` /
    ``model_type`` / ``torch_dtype`` existed on both). Legacy's ``loader`` field is self-referential
    here (the loader class *is* the family unit) and ``model_groups`` is a registry concern.
    """

    # -- identity ------------------------------------------------------------------------------
    model_type: Optional[str] = None
    model_dir: str = ''
    # The raw PretrainedConfig -- an escape hatch, not a substitute for the fields below.
    config: Optional[PretrainedConfig] = None
    # config.architectures as found on disk (the family's declared list lives in the registry).
    architectures: List[str] = field(default_factory=list)

    # -- read off the checkpoint ---------------------------------------------------------------
    # Resolved compute dtype: the caller's request, else a family default, else the checkpoint's.
    torch_dtype: Optional[torch.dtype] = None
    max_model_len: Optional[int] = None
    # Quantization already baked into the weights (config.quantization_config) + its bit width.
    quant_method: Optional[Literal['gptq', 'awq', 'bnb', 'aqlm', 'hqq', 'fp8']] = None
    quant_bits: Optional[int] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    is_moe_model: bool = False
    # Family declaration OR config detection, merged on purpose (the bug was two objects each
    # carrying the same field).
    is_multimodal: bool = False
    # Reward family -> seq_cls head with num_labels=1.
    is_reward: bool = False

    # -- task shape ----------------------------------------------------------------------------
    # Training/inference head (causal_lm/seq_cls/embedding/reranker); follows from num_labels.
    task_type: Optional[str] = None
    num_labels: Optional[int] = None
    # Declared explicitly -- legacy set it as an undeclared attribute, invisible to asdict/editors.
    problem_type: Optional[str] = None

    # -- declared by the family ----------------------------------------------------------------
    # One of several family templates (thinking / non-thinking, vendor re-releases).
    template: Optional[str] = None
    # Key into the Megatron bridge's own registry, when a conversion exists.
    mcore_model_type: Optional[str] = None
    # Version constraints to check and warn about, e.g. ['transformers>=4.49'].
    requires: List[str] = field(default_factory=list)
    # Extra files to copy next to the weights on full-parameter save or merge-lora, for
    # checkpoints whose code lives beside the config.
    additional_saved_files: List[str] = field(default_factory=list)
    # Descriptive labels (vision/video/audio/coding/...); documentation and filtering only.
    tags: List[str] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        """Human-readable name for API responses, recovered from :attr:`model_dir`.

        A property (not a field): a pure function of ``model_dir`` must not be able to drift.
        """
        model_dir = self.model_dir
        if platform.system().lower() == 'windows':
            model_dir = model_dir.replace('\\', '/')
        model_dir = model_dir.rstrip('/')
        # Hub caches bury the real name inside `.../models--org--name/snapshots/<sha>`.
        match = re.search('/models--(?:.+?--)?(.+?)/snapshots/', model_dir) or re.search(
            '/models/(?:.+?--)?(.+?)/snapshots/', model_dir)
        if match is not None:
            return match.group(1)
        # `___` is how modelscope's snapshot_download escapes a dot in a directory name.
        return model_dir.rsplit('/', 1)[-1].replace('___', '.')


def _basename(model_id: str) -> str:
    """The trailing path component, lower-cased -- how ids and local dirs are compared."""
    return model_id.rstrip('/').rsplit('/', 1)[-1].lower()


def _iter_ids(models: Sequence[Union[str, Tuple[str, str]]]) -> Iterator[Tuple[str, str]]:
    """Yield ``(ms_id, hf_id)`` for every entry, expanding the bare-string shorthand.

    A bare string means the two hubs agree on the id (720 of the 1152 legacy entries do); a
    ``(ms_id, hf_id)`` pair is written only when they differ.
    """
    for entry in models:
        if isinstance(entry, str):
            yield entry, entry
        else:
            yield entry[0], entry[1]


def register_model(loader_cls: Type['ModelLoader'] = None, *, exist_ok: bool = False):
    """Register a family, keyed by its ``model_type``. Usable bare or with keywords as a decorator.

    Everything is read off the class, so a family is declared in one place instead of being split
    between a loader class and a separate ``ModelMeta`` literal repeating its name.
    """

    def _register(cls: Type['ModelLoader']) -> Type['ModelLoader']:
        model_type = cls.model_type
        assert model_type, f'{cls.__name__} must set `model_type`.'
        validate_tags(cls.tags, context=f'model_type `{model_type}`')
        if not exist_ok and model_type in MODEL_MAPPING:
            raise ValueError(f'model_type `{model_type}` is already registered '
                             f'by {MODEL_MAPPING[model_type].__name__}.')
        MODEL_MAPPING[model_type] = cls
        for alias in cls.aliases:
            if not exist_ok and alias in MODEL_ALIASES and MODEL_ALIASES[alias] != model_type:
                raise ValueError(f'alias `{alias}` already points at `{MODEL_ALIASES[alias]}`.')
            MODEL_ALIASES[alias] = model_type
        _ARCH_MAPPING.clear()  # invalidate the reverse-lookup cache
        return cls

    return _register if loader_cls is None else _register(loader_cls)


def get_model_loader(model_type: str) -> Type['ModelLoader']:
    """Look up a family, following aliases."""
    model_type = MODEL_ALIASES.get(model_type, model_type)
    if model_type not in MODEL_MAPPING:
        raise ValueError(f'model_type `{model_type}` is not registered. Available: {sorted(MODEL_MAPPING)}')
    return MODEL_MAPPING[model_type]


def match_model_type(model_id_or_path: str) -> Optional[str]:
    """Match a model id or local path against the registered checkpoint ids.

    Compares the trailing path component case-insensitively, so a local directory holding a
    downloaded snapshot matches too. This is the *primary* resolution path: it runs before anything
    is downloaded, and unlike ``architectures`` it cannot collide -- 33 legacy HuggingFace class
    names map to more than one family.
    """
    name = _basename(model_id_or_path)
    for model_type, cls in MODEL_MAPPING.items():
        for ms_id, hf_id in _iter_ids(cls.models):
            if name in (_basename(ms_id), _basename(hf_id)):
                return model_type
    return None


def match_model_types_by_architectures(architectures: Optional[Sequence[str]]) -> List[str]:
    """Reverse-lookup families from ``config.architectures``; the fallback when the id is unknown.

    Returns *all* candidates rather than picking one, because the mapping genuinely is many-to-many
    (``ChatGLMModel`` covered six legacy model_types); the caller asks the user when >1 comes back.
    """
    if not architectures:
        return []
    if not _ARCH_MAPPING:
        for model_type, cls in MODEL_MAPPING.items():
            for arch in cls.architectures:
                _ARCH_MAPPING.setdefault(arch, []).append(model_type)
    return list(_ARCH_MAPPING.get(architectures[0], ()))


def resolve_template(model_type: str) -> Optional[str]:
    """The template a family speaks.

    A template variant (Qwen's QVQ prompt, Xiaomi's MiMo-VL format, ...) is its own thin loader
    subclass -- it loads identically to its parent but declares a different ``template`` and its own
    ``models`` -- so the template is a plain per-family attribute; legacy's within-family
    ``ModelGroup`` layer is gone.
    """
    return get_model_loader(model_type).template


def resolve_loader_cls(model_type: Optional[str], model_id: str) -> Optional[Type['ModelLoader']]:
    """The family loader class for an explicit ``model_type``, else one inferred from ``model_id``.

    ``model_type`` is authoritative when given: a registered family key/alias, or an external plugin
    source (a local file/folder or a ``hf://`` / ``ms://`` id) resolved through the registry. With
    no explicit type the family is inferred from the checkpoint name; ``None`` means "no family
    matched", and the caller falls back to the generic transformers path.
    """
    from swift.dev.utils.naming import resolve_plugin_class
    if model_type:
        if model_type in MODEL_MAPPING or model_type in MODEL_ALIASES:
            return get_model_loader(model_type)
        return resolve_plugin_class(model_type, ModelLoader, MODEL_MAPPING, kind='model')
    matched = match_model_type(model_id)
    return get_model_loader(matched) if matched is not None else None


def build_model_loader(model_id: str, model_type: Optional[str] = None, **kwargs) -> Optional['ModelLoader']:
    """Build the family loader for one checkpoint id -- the entry a twinkle server names in its
    ``model_loader`` import spec (``swift.dev.model.loader:build_model_loader``).

    The server has only the id, so this builds a minimal ``ModelInfo``; the six hooks take
    ``model_dir`` explicitly. ``model_type`` overrides inference; ``None`` means no family matched
    and the server stays on the generic AutoConfig/AutoModel path.
    """
    loader_cls = resolve_loader_cls(model_type, model_id)
    if loader_cls is None:
        return None
    model_info = ModelInfo(model_type=loader_cls.model_type or model_type, model_dir=model_id, **kwargs)
    return loader_cls(model_info)


def _import_cls(spec: str) -> type:
    """Resolve ``'transformers:Qwen3VLForConditionalGeneration'`` lazily.

    Deferred on purpose: 52% of the legacy ``get_model`` overrides existed *only* to keep a
    ``from transformers import X`` out of module scope, because the class is absent on older
    transformers and a top-level import would break the whole registry.
    """
    module_name, _, attr = spec.partition(':')
    module = importlib.import_module(module_name)
    return getattr(module, attr)


class ModelLoader:
    """The plain transformers path, driven by declarations.

    Families declare only what differs: ``build_config`` / ``build_processor`` fall back to
    ``AutoConfig`` / ``AutoProcessor`` and the ``process_*`` hooks are no-ops. A specific class is
    a lazy ``'module:ClassName'`` string (an absent class costs nothing); anything more overrides
    a ``build_*`` / ``process_*`` hook.
    """

    model_type: Optional[str] = None
    # Checkpoint ids this family covers: a bare id, or a (ms_id, hf_id) pair when the two hubs
    # disagree. A template variant is its own subclass with its own `models`, so this stays flat.
    models: Sequence[Union[str, Tuple[str, str]]] = ()
    architectures: Sequence[str] = ()
    template: Optional[str] = None
    aliases: Sequence[str] = ()
    requires: Sequence[str] = ()
    tags: Sequence[str] = ()
    # Download skip globs. ``None`` = default skips (*.pth / consolidated* / ...); ``[]`` disables
    # them (Mistral/Qwen-Omni need those files); a non-empty list skips exactly those globs.
    ignore_patterns: Optional[List[str]] = None
    mcore_model_type: Optional[str] = None
    is_multimodal: bool = False
    # The task a family is *inherently* built for (a bge reranker is always a reranker). ``None``
    # for a generic backbone: the user's ModelConfig.task_type decides and overrides this pin.
    task_type: Optional[str] = None
    # A reward model is a seq_cls head with num_labels=1 -- kept separate from task_type because
    # legacy drives num_labels/reward-specific load handling from is_reward, not task resolution.
    is_reward: bool = False
    # External framework constructing this family ('sentence_transformers': an embedding pipeline
    # training through SentenceTransformerModel). One open-ended name + a builders dispatch branch.
    model_framework: Optional[str] = None
    # Resolved lazily; unset config_cls/processor_cls fall back to Auto*, model_cls has no default.
    config_cls: Optional[str] = None
    processor_cls: Optional[str] = None
    model_cls: Optional[str] = None
    # True for a remote-code checkpoint: the build_* hooks then pass trust_remote_code=True,
    # replacing legacy's scattered get_class_from_dynamic_module calls. In-tree families: False.
    trust_remote_code: bool = False

    def __init__(self, model_info, **kwargs):
        # `model_info` describes one concrete checkpoint, so it arrives per instance rather than
        # being a per-family constant.
        self._model_info = model_info
        self._kwargs = kwargs

    @property
    def model_arch(self) -> ModelArch:
        """Architecture partition for this family.

        All-empty is the correct default: these fields describe *multimodal* partitioning, so a
        plain LLM wants an empty :class:`ModelArch`. Multimodal families override this.
        """
        return ModelArch()

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    def _apply_trust_remote_code(self, kwargs: Dict[str, Any]) -> None:
        """Default ``trust_remote_code`` on for remote-code families, warning once per family.

        "Trusting" executes the checkpoint's own Python (modeling_*.py) inside this
        process before any validation runs -- the exact attack path the upstream
        security advisories describe -- so the default stays loud, never silent.
        An explicit ``trust_remote_code=False`` from the caller is preserved (the
        load then fails loudly inside transformers instead).
        """
        if not self.trust_remote_code:
            return
        if kwargs.setdefault('trust_remote_code', True) and type(self) not in _TRC_WARNED_FAMILIES:
            _TRC_WARNED_FAMILIES.add(type(self))
            logger.warning('%s loads checkpoints with trust_remote_code=True: the checkpoint\'s own Python '
                           'code will execute inside this process. Only use checkpoints you trust.',
                           type(self).__name__)

    def build_config(self, model_dir: str, **kwargs) -> PretrainedConfig:
        self._apply_trust_remote_code(kwargs)
        config_cls = _import_cls(self.config_cls) if self.config_cls else AutoConfig
        return config_cls.from_pretrained(model_dir, **kwargs)

    def build_processor(self, model_dir: str, config: PretrainedConfig, **kwargs):
        self._apply_trust_remote_code(kwargs)
        if self.processor_cls:
            processor_cls = _import_cls(self.processor_cls)
        elif (os.path.exists(os.path.join(model_dir, 'preprocessor_config.json'))
              or os.path.exists(os.path.join(model_dir, 'processor_config.json'))):
            # A multimodal checkpoint ships a (pre)processor config; a plain LLM ships only a
            # tokenizer. Mirror the legacy detection so a text model does not hit AutoProcessor.
            processor_cls = AutoProcessor
        else:
            processor_cls = AutoTokenizer
        return processor_cls.from_pretrained(model_dir, **kwargs)

    @classmethod
    def resolve_model_cls(cls) -> type:
        """The transformers class this family loads with, resolved from :attr:`model_cls`.

        Split out of :meth:`build_model` so a caller needing the class itself -- ``from_config`` on
        a config with no weights, an ``isinstance`` check -- does not re-resolve the string.
        """
        assert cls.model_cls is not None, f'{cls.__name__} must declare `model_cls`.'
        return _import_cls(cls.model_cls)

    def build_model(self, model_dir: str, config: PretrainedConfig, processor, **kwargs) -> PreTrainedModel:
        self._apply_trust_remote_code(kwargs)
        return self.resolve_model_cls().from_pretrained(model_dir, config=config, **kwargs)

    def process_config(self, config):
        return config

    def process_tokenizer(self, tokenizer):
        return tokenizer

    def process_model(self, model):
        return model

    @staticmethod
    @contextmanager
    def ignore_check_imports():
        """Skip transformers' remote-code dependency check for the duration of a load.

        Delegates to ``TransformersIgnoreCheckImportsPatch``: it keeps ``check_imports``'s
        real return value (``get_relative_imports``, which the module loader consumes) and drops
        only the verification loop -- a blank ``[]`` stub would starve the loader. Used by
        ``minimax_vl`` / ``florence`` (legacy ``patch_ignore_check_imports``).
        """
        from twinkle.patch import apply_context

        from swift.dev.patch.transformers_check_imports import TransformersIgnoreCheckImportsPatch
        with apply_context(None, TransformersIgnoreCheckImportsPatch()):
            yield

    @staticmethod
    def delegate_to_submodel(model, submodel_name: str, func_list: Optional[List[str]] = None) -> None:
        """Proxy a wrapper model's top-level methods to its inner LLM sub-module.

        Thin MLLM wrappers hang the real LLM off an attribute (``model.llm``, ``model.language_model``)
        with no usable top-level ``forward`` / ``generate``. Faithful half of legacy
        ``use_submodel_func``; the device-shuffling half is dropped (twinkle owns placement).
        """
        from functools import wraps
        from types import MethodType
        if func_list is None:
            func_list = ['generate', 'get_input_embeddings', 'gradient_checkpointing_enable', 'forward']
        submodel = getattr(model, submodel_name)

        def _make(func_name: str):
            _old_func = getattr(submodel, func_name).__func__

            @wraps(_old_func)
            def _new_func(self, *args, **kwargs):
                return _old_func(submodel, *args, **kwargs)

            return _new_func

        for key in func_list:
            setattr(model, key, MethodType(_make(key), model))

    @staticmethod
    def apply_z3_leaf_modules(model, moe_block: List[str]) -> None:
        """Mark a MoE block as a ZeRO-3 leaf module, or ZeRO-3 deadlocks on dark experts.

        ``moe_block`` holds leaf *class names* matched via ``type(...).__name__`` (covers remote
        code; class -- not attribute -- matching is what lets llama4 work). Call from the ZeRO-3
        strategy setup, NOT transformers' ``is_deepspeed_zero3_enabled()`` (unset without HF Trainer).
        """
        if not moe_block:
            return
        leaves = set(moe_block)
        classes = {type(module) for module in model.modules() if type(module).__name__ in leaves}
        if not classes:
            return
        from deepspeed.utils import set_z3_leaf_modules
        set_z3_leaf_modules(model, list(classes))
