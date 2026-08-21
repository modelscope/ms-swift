# Copyright (c) ModelScope Contributors. All rights reserved.
"""What a checkpoint is, and which named sub-modules make up which trainable part of it."""
from __future__ import annotations

import importlib
import os
import platform
import re
import torch
from contextlib import contextmanager
from dataclasses import dataclass, field
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Type, Union

__all__ = [
    'ModelArch', 'ModelInfo', 'ModelLoader', 'MODEL_ALIASES', 'MODEL_MAPPING', 'get_model_loader', 'match_model_type',
    'match_model_types_by_architectures', 'register_model', 'resolve_template'
]

# Fields that accept a bare string for convenience but always read back as a list.
_PART_FIELDS = ('language_model', 'vision_tower', 'aligner', 'generator')
# Fields normalised to a list of strings on assignment. Beyond the freeze/LR "parts" above, this
# also covers `moe_block`, which is a module-name hint rather than a trainable part.
_LIST_FIELDS = _PART_FIELDS + ('moe_block', )

MODEL_MAPPING: Dict[str, Type['ModelLoader']] = {}
# Alias -> registered model_type. Keeps `--model_type ovis_ocr2` and old `args.json` working after
# two legacy model_types that differed only by template were merged into one family.
MODEL_ALIASES: Dict[str, str] = {}
# HuggingFace class name -> model_types, lazily built and invalidated on every registration.
_ARCH_MAPPING: Dict[str, List[str]] = {}


@dataclass
class ModelArch:
    """The module-name prefixes that split a checkpoint into separately-controllable parts.

    Values are *module name prefixes* exactly as they appear in ``named_parameters()`` /
    ``named_modules()`` -- not dotted templates with ``{}`` placeholders. Consumers either match with
    ``name.startswith(prefix)`` (freeze, learning-rate grouping, quantization exclusion) or resolve
    them with ``deep_getattr`` (gradient checkpointing, rollout weight splitting). A bare string is
    accepted and normalised to a one-element list, because a part legitimately lives in more than one
    place on some checkpoints (``language_model=['model.language_model', 'lm_head']``). The
    ``Union[str, List[str]]`` annotation describes what may be *written*; reads always give a list.

    All-empty is the correct value for a plain LLM: these fields describe *multimodal* partitioning,
    and every consumer reads an empty list as "no such part, treat the whole model uniformly".

    Deliberately omitted: legacy swift's ``ModelKeys`` also carried ~15 layer-path fields
    (``module_list`` / ``attention`` / ``embedding`` / ``q_proj`` / ``qkv_proj`` / ``qa_proj`` / ...).
    Their only real client was the llama-pro tuner's architecture matcher, which is not implemented
    here, so they were data every new model had to fill in and nothing ever read. Whoever needs
    per-layer access should reach for the runtime accessors (``get_decoder()``,
    ``get_output_embeddings()``, module-type scans) rather than re-adding path strings.

    Args:
        language_model: The LLM, including its output head. Drives freeze_llm, the LLM learning-rate
            group, gradient checkpointing, rollout weight splitting, and scoping a quantizer's block
            walk (use :attr:`decoder` for the latter -- it strips the head).
        vision_tower: The encoder (vision, audio, ...). Drives freeze_vit / vit_lr, and is excluded
            from quantization and from mcore conversion.
        aligner: The connector between encoder and LLM -- projector / merger / resampler. Drives
            freeze_aligner / aligner_lr, excluded from quantization.
        generator: An output generator that is not the LM head (e.g. a speech or image decoder).
            Frozen by default: it is trained by its own recipe, not by text SFT.
        lm_head: Output projection name, needed only when it is not literally ``lm_head`` (internlm2
            names it ``output``, chatglm ``transformer.output_layer``). Used to keep the head out of
            ``all-linear`` LoRA targets, so a leaf name is enough.
        moe_block: Leaf *class* name(s) of a MoE sparse/expert block (Mixtral's
            ``MixtralSparseMoeBlock``, llama4's ``Llama4TextMoe``). Not a trainable "part" -- it exists
            only so the DeepSpeed ZeRO-3 setup can mark that block as a leaf module (ZeRO-3 must not
            shard inside it). Stored as a class *name* rather than legacy's imported class so it also
            covers remote-code blocks with no static import path, and matches on class (not attribute)
            so an interleaved dense sibling under the same attribute is not caught;
            :func:`apply_z3_leaf_modules` resolves the name to the live class at load time. Empty for a
            dense model.
    """

    language_model: Union[str, List[str]] = field(default_factory=list)
    vision_tower: Union[str, List[str]] = field(default_factory=list)
    aligner: Union[str, List[str]] = field(default_factory=list)
    generator: Union[str, List[str]] = field(default_factory=list)
    lm_head: Optional[str] = None
    moe_block: Union[str, List[str]] = field(default_factory=list)

    def __setattr__(self, key: str, value) -> None:
        # Normalise on every assignment, not just in __post_init__: the annotation advertises that a
        # bare string is acceptable, so `arch.vision_tower = 'visual'` must not leave a string behind
        # for `startswith` loops to iterate character by character.
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

        Registrations put the head inside ``language_model`` on purpose -- for freeze and LR purposes
        it belongs to the LLM side. But callers asking "where do the decoder layers live" (quantizer
        block scoping, mcore conversion) must not receive the head prefix, so the filtering lives here
        instead of an ``endswith('lm_head')`` check copy-pasted into each consumer.
        """
        head = self.lm_head_name
        return [name for name in self.language_model if not name.endswith(head)]


@dataclass
class ModelInfo:
    """Everything known about one concrete checkpoint directory.

    This merges legacy swift's ``ModelInfo`` (facts read out of ``config.json`` at load time) and
    ``ModelMeta`` (facts hand-declared per model family). Keeping them apart looked principled but the
    boundary had already leaked: ``is_multimodal`` / ``model_type`` / ``task_type`` / ``torch_dtype``
    existed on *both*, with the declared value silently merged into the detected one, so a reader could
    not tell which object was authoritative -- and the usage counts showed the ownership was de-facto
    split per field anyway (``torch_dtype`` 41 reads on info vs 1 on meta; ``is_multimodal`` 2 vs 31).
    One object, one owner per field.

    Two legacy ``ModelMeta`` fields deliberately have no counterpart here:

    - ``loader``: the :class:`ModelLoader` subclass *is* the per-family unit now, so a field pointing
      at a loader class would be self-referential.
    - ``model_groups`` (the ~1150 registered model ids): that is a *registry* concern -- it answers
      "which family does this name belong to" before anything is downloaded. It has no meaning on an
      object describing one already-resolved directory.

    Args:
        model_type: Family id. Resolved by the registry, either from the model id or by reverse lookup
            from ``architectures``; ``None`` when nothing matched and the checkpoint is loaded as a
            plain LLM.
        model_dir: Local directory holding the resolved snapshot.
        config: The raw ``PretrainedConfig``. An escape hatch, not a substitute for the fields below --
            reach for it only when a value genuinely has no home here.
        architectures: ``config.architectures`` as found on disk. Note this is the *actual* value, not
            the class-name list a family declares for reverse lookup; that list belongs to the registry.
        torch_dtype: Resolved compute dtype -- the caller's request, else a family default, else the
            checkpoint's own dtype.
        max_model_len: Maximum position count the checkpoint declares.
        quant_method: Quantization already baked into the weights, from ``config.quantization_config``.
        quant_bits: Bit width that goes with :attr:`quant_method`.
        rope_scaling: RoPE scaling config, read out and possibly overridden by the caller.
        is_moe_model: Whether the config exposes expert counts.
        is_multimodal: Whether non-text inputs are supported -- family declaration OR config detection.
            Merging the two is intentional here; what was wrong before was having both under one name
            on two objects.
        is_reward: Whether this family is a reward model, so ``num_labels`` defaults to 1.
        task_type: Training/inference head to build, e.g. ``causal_lm`` / ``seq_cls`` / ``embedding`` /
            ``reranker``. A family may pin it; otherwise it follows from ``num_labels``.
        num_labels: Output count for classification heads.
        problem_type: ``regression`` / ``single_label_classification`` /
            ``multi_label_classification``. Declared explicitly here -- legacy set it as an undeclared
            attribute, so it was invisible to ``asdict`` and to editors.
        template: Chat template id. A family may offer several (thinking / non-thinking, vendor
            re-releases sharing a structure); the registry picks one before this object is built.
        mcore_model_type: Key into the Megatron bridge's own registry, when a conversion exists.
        requires: Version constraints to check and warn about, e.g. ``['transformers>=4.49']``.
        additional_saved_files: Extra files to copy next to the weights on full-parameter save or
            merge-lora, for checkpoints whose code lives beside the config.
        tags: Descriptive labels (``vision`` / ``video`` / ``audio`` / ``coding`` / ...). Documentation
            and filtering only; no code branches on them.
    """

    # -- identity ------------------------------------------------------------------------------
    model_type: Optional[str] = None
    model_dir: str = ''
    config: Optional[PretrainedConfig] = None
    architectures: List[str] = field(default_factory=list)

    # -- read off the checkpoint ---------------------------------------------------------------
    torch_dtype: Optional[torch.dtype] = None
    max_model_len: Optional[int] = None
    quant_method: Optional[Literal['gptq', 'awq', 'bnb', 'aqlm', 'hqq', 'fp8']] = None
    quant_bits: Optional[int] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    is_moe_model: bool = False
    is_multimodal: bool = False
    is_reward: bool = False

    # -- task shape ----------------------------------------------------------------------------
    task_type: Optional[str] = None
    num_labels: Optional[int] = None
    problem_type: Optional[str] = None

    # -- declared by the family ----------------------------------------------------------------
    template: Optional[str] = None
    mcore_model_type: Optional[str] = None
    requires: List[str] = field(default_factory=list)
    additional_saved_files: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        """Human-readable name for API responses, recovered from :attr:`model_dir`.

        A property rather than a field set in ``__post_init__``: it is a pure function of
        ``model_dir`` and must not be able to drift away from it.
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

    A bare string means ModelScope and HuggingFace agree on the id (720 of the 1152 legacy entries
    do); a ``(ms_id, hf_id)`` pair is written only when they differ.
    """
    for entry in models:
        if isinstance(entry, str):
            yield entry, entry
        else:
            yield entry[0], entry[1]


def register_model(loader_cls: Type['ModelLoader'] = None, *, exist_ok: bool = False):
    """Register a family, keyed by its ``model_type``. Usable bare or with keywords as a decorator.

    Everything is read off the class, so a family is declared in one place instead of being split
    between a loader class and a separate ``ModelMeta`` literal that has to repeat its name.
    """

    def _register(cls: Type['ModelLoader']) -> Type['ModelLoader']:
        model_type = cls.model_type
        assert model_type, f'{cls.__name__} must set `model_type`.'
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
    -- ``ChatGLMModel`` covered six legacy model_types. The caller is expected to ask the user when
    more than one comes back.
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

    A template variant (Qwen's QVQ prompt, Xiaomi's MiMo-VL format, ...) is its own thin
    :class:`ModelLoader` subclass -- it loads identically to its parent but declares a different
    ``template`` and its own ``models``. So the template is a plain per-family attribute; the
    within-family ``ModelGroup`` layer that legacy needed for this is gone.
    """
    return get_model_loader(model_type).template


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

    The happy path lives here so a family declares only what differs: ``build_config`` /
    ``build_processor`` fall back to ``AutoConfig`` / ``AutoProcessor``, and the ``process_*`` hooks
    are no-ops. A family that needs a specific transformers class points ``config_cls`` /
    ``processor_cls`` / ``model_cls`` at it as a ``'module:ClassName'`` string, resolved lazily so an
    absent class costs nothing; anything beyond that overrides a ``build_*`` / ``process_*`` hook.
    ``model_cls`` is the one declaration with no safe default -- see :meth:`build_model`.
    """

    # -- registration ---------------------------------------------------------------------------
    model_type: Optional[str] = None
    # Checkpoint ids this family covers: a bare id, or a (ms_id, hf_id) pair when the two hubs
    # disagree. A template variant is its own subclass with its own `models`, so this stays flat.
    models: Sequence[Union[str, Tuple[str, str]]] = ()
    architectures: Sequence[str] = ()
    template: Optional[str] = None
    aliases: Sequence[str] = ()
    requires: Sequence[str] = ()
    tags: Sequence[str] = ()
    # Glob patterns of files to skip at download. ``None`` lets the download layer apply its default
    # skips (``*.pth`` / ``consolidated*`` / ``onnx/*`` / ...). An empty list ``[]`` deliberately
    # DISABLES those defaults to fetch everything -- required by families whose real weights live in
    # otherwise-skipped files (Mistral ships ``consolidated*``; Qwen-Omni ships extra assets). A
    # non-empty list skips exactly those globs (e.g. gpt-oss's ``metal/`` / ``original/`` dirs).
    ignore_patterns: Optional[List[str]] = None
    mcore_model_type: Optional[str] = None
    is_multimodal: bool = False
    # The task a family is *inherently* built for, when that is not a user choice: a bge reranker is
    # always a reranker, a gte checkpoint always embedding. Left ``None`` for a generic backbone
    # (Llama/Qwen), which can serve any task -- the user's ``ModelConfig.task_type`` decides, and it
    # overrides this pin whenever set. ``None`` resolves to ``causal_lm``.
    task_type: Optional[str] = None
    # A reward model is a seq_cls head with ``num_labels=1``. Declared as a flag rather than pinning
    # ``task_type='seq_cls'`` because legacy keeps the two separate: ``is_reward`` drives the
    # num_labels=1 default and reward-specific load handling, while task resolution still lands on
    # seq_cls. Left ``False`` for every non-reward family.
    is_reward: bool = False

    # Declared per family; resolved lazily so an absent transformers class costs nothing. Leave
    # `config_cls` / `processor_cls` unset to fall back to Auto*; `model_cls` has no safe default.
    config_cls: Optional[str] = None
    processor_cls: Optional[str] = None
    model_cls: Optional[str] = None
    # Set True for a remote-code checkpoint (a custom `modeling_*.py` loaded through HF's
    # dynamic-module path). It makes `build_config` / `build_processor` / `build_model` pass
    # `trust_remote_code=True`, so an `AutoConfig` / `AutoProcessor` / `AutoModel*` resolves the
    # checkpoint's own classes instead of failing to find an in-tree one. In-tree families leave it
    # False (the default) -- passing it would be harmless, but the flag also documents, per family,
    # exactly which checkpoints ship code we execute. Injected via `setdefault` so an explicit caller
    # kwarg always wins, and it replaces legacy's scattered `get_class_from_dynamic_module` calls.
    trust_remote_code: bool = False

    def __init__(self, model_info, **kwargs):
        # `model_info` describes one concrete checkpoint, so it arrives per instance rather than
        # being a per-family constant.
        self._model_info = model_info
        self._kwargs = kwargs

    @property
    def model_arch(self) -> ModelArch:
        """Architecture partition for this model family.

        All-empty is the correct default: these fields describe *multimodal* partitioning, so a
        plain LLM wants an empty :class:`ModelArch` (every consumer reads an empty part as "no such
        part, treat the whole model uniformly"). Multimodal families override this.
        """
        return ModelArch()

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    def build_config(self, model_dir: str, **kwargs) -> PretrainedConfig:
        if self.trust_remote_code:
            kwargs.setdefault('trust_remote_code', True)
        config_cls = _import_cls(self.config_cls) if self.config_cls else AutoConfig
        return config_cls.from_pretrained(model_dir, **kwargs)

    def build_processor(self, model_dir: str, config: PretrainedConfig, **kwargs):
        if self.trust_remote_code:
            kwargs.setdefault('trust_remote_code', True)
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

    def build_model(self, model_dir: str, config: PretrainedConfig, processor, **kwargs) -> PreTrainedModel:
        assert self.model_cls is not None, f'{type(self).__name__} must declare `model_cls`.'
        if self.trust_remote_code:
            kwargs.setdefault('trust_remote_code', True)
        return _import_cls(self.model_cls).from_pretrained(model_dir, config=config, **kwargs)

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

        ``check_imports`` scans a downloaded ``modeling_*.py`` for third-party imports and raises if
        one is not installed. Some remote-code checkpoints declare imports that are never reached on
        the paths we execute, and without relaxing the check the dynamic import fails outright -- so
        this gates loading rather than papering over behaviour. Used by ``minimax_vl`` and
        ``florence`` (legacy's ``patch_ignore_check_imports``).
        """
        import transformers.dynamic_module_utils as dynamic_module_utils
        _origin_check = dynamic_module_utils.check_imports
        dynamic_module_utils.check_imports = lambda filename: []
        try:
            yield
        finally:
            dynamic_module_utils.check_imports = _origin_check

    @staticmethod
    def delegate_to_submodel(model, submodel_name: str, func_list: Optional[List[str]] = None) -> None:
        """Proxy a wrapper model's top-level methods to its inner LLM sub-module.

        Many MLLM checkpoints are a thin wrapper whose real language model hangs off an attribute
        (``model.llm`` for Ovis, ``model.language_model`` elsewhere): the wrapper itself has no usable
        ``forward`` / ``generate`` / ``get_input_embeddings``, so those calls must be forwarded to the
        sub-module. This is the faithful half of legacy ``use_submodel_func``; the device-shuffling
        half (moving ``res.logits`` / ``loss`` back onto the input device, and the ``fix device_map``
        branch) is dropped -- it patched HF ``device_map`` sharding, which dev does not use (twinkle
        owns placement). Called from a subclass ``process_model``; ``func_list`` defaults to the
        legacy set.
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
        """Mark a MoE sparse/expert block as a DeepSpeed ZeRO-3 leaf module.

        ZeRO-3 partitions parameters across ranks and re-gathers them per submodule forward; for a
        MoE block only a token-dependent subset of experts runs, so ZeRO-3 must treat the whole block
        as one leaf (gather all experts up front) or it deadlocks on the experts that stay dark.
        ``moe_block`` holds the block's *leaf class name(s)* (Mixtral ``MixtralSparseMoeBlock``); we
        match them against the live modules' ``type(...).__name__`` rather than importing, so a
        remote-code block with no static import path works too (legacy needed
        ``get_class_from_dynamic_module`` per such model). Matching on the class name -- not the
        attribute name -- is also what lets llama4 work: it names both its sparse ``Llama4TextMoe``
        and its dense ``Llama4TextMLP`` ``self.feed_forward``, so an attribute-name match would
        wrongly mark the dense MLP as a leaf.

        Call this from the DeepSpeed ZeRO-3 *strategy setup* -- only when ZeRO-3 is active, and before
        ``deepspeed.initialize`` partitions the parameters. The "is this ZeRO-3" decision is the
        strategy's (it reads the planned ``DistributedConfig.deepspeed``); it is deliberately NOT
        taken from transformers' ``is_deepspeed_zero3_enabled()``, which only reports True after HF
        ``TrainingArguments`` has instantiated transformers' ``HfDeepSpeedConfig``. dev drives
        DeepSpeed through twinkle/accelerate with no HF Trainer, so that global stays unset and the
        probe would silently return False -- skipping the leaf setting and re-introducing the very
        deadlock this prevents.
        """
        if not moe_block:
            return
        leaves = set(moe_block)
        classes = {type(module) for module in model.modules() if type(module).__name__ in leaves}
        if not classes:
            return
        from deepspeed.utils import set_z3_leaf_modules
        set_z3_leaf_modules(model, list(classes))
