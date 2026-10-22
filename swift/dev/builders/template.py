"""Build a public dev Template from the dev-native family registry."""

from __future__ import annotations
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, FrozenSet, Literal, Optional

from swift.dev.template_contract import ModelInfo as SpecModelInfo
from swift.dev.template_contract import ModelRef, TemplateSpec

if TYPE_CHECKING:
    from swift.dev.config import TemplateConfig
    from swift.dev.template import Template


def _build_meta_model(config, model_type: Optional[str], model_dir: str):
    """Rebuild a process-local shape/position helper, never a checkpoint or a training model."""
    import torch
    from transformers import AutoModelForCausalLM

    from swift.dev.model.loader import (
        MODEL_MAPPING,
        get_model_loader,
        match_model_type,
        match_model_types_by_architectures,
    )
    from swift.dev.model.loader.base import expose_nested_multimodal_attrs

    if config is None:
        raise RuntimeError("A template meta model needs processor.model_info.config; bind a real model instead.")
    # Use only the dev registry and the already loaded HF config. In particular, do not resolve
    # plugin URLs or call build_model/from_pretrained: those paths can fetch code or weights.
    model_type = model_type or match_model_type(model_dir)
    if model_type is None:
        candidates = match_model_types_by_architectures(getattr(config, "architectures", None))
        if len(candidates) == 1:
            model_type = candidates[0]
        elif config.model_type in MODEL_MAPPING:
            model_type = config.model_type
    model_cls = get_model_loader(model_type).resolve_model_cls() if model_type else AutoModelForCausalLM
    config = deepcopy(config)  # HF construction may mutate attention/dtype fields.
    with torch.device("meta"):
        if hasattr(model_cls, "from_config"):  # HF Auto* classes (e.g. GLM text models).
            model = model_cls.from_config(config, trust_remote_code=False, attn_implementation="eager")
        else:
            model = model_cls._from_config(config, attn_implementation="eager")
    expose_nested_multimodal_attrs(model)
    return model


def _multimodal_graph_adapter(processor: Any):
    from swift.dev.model.loader import get_model_loader

    if not _media_types(processor):
        return None
    model_info = processor.model_info
    model_type = model_info.model_type or getattr(processor.model_meta, "model_type", None)
    if model_type is None:
        return None
    loader = get_model_loader(model_type)(model_info)
    return loader.build_multimodal_graph_adapter(processor, model_info.config)


def _media_types(processor: Any) -> FrozenSet[Literal["image", "video", "audio"]]:
    model_info = processor.model_info
    tags = set(getattr(model_info, "tags", ()) or ())
    model_meta = processor.model_meta
    tags.update(getattr(model_meta, "tags", ()) or ())
    media_types = {"image" if tag == "vision" else tag for tag in tags if tag in {"vision", "image", "video", "audio"}}
    is_multimodal = model_info.is_multimodal or bool(getattr(model_meta, "is_multimodal", False))
    if is_multimodal and not media_types:
        media_types.add("image")
    return frozenset(media_types)


def build_processor_from_ref(model: ModelRef, model_info: SpecModelInfo):
    from swift.dev.model.loader.base import LoaderModelInfo, ModelLoader, resolve_loader_cls
    from swift.dev.utils import safe_snapshot_download

    model_dir = safe_snapshot_download(model.model_id_or_dir, revision=model.revision, download_model=False)
    loader_cls = resolve_loader_cls(model_info.model_type, model.model_id_or_dir) or ModelLoader
    info = LoaderModelInfo(
        model_type=model_info.model_type,
        model_dir=model_dir,
        torch_dtype=model_info.torch_dtype,
        max_model_len=model_info.max_model_len,
        task_type=model_info.task_type,
        is_multimodal=model_info.is_multimodal,
        template=loader_cls.template,
        tags=list(loader_cls.tags),
    )
    loader = loader_cls(info)
    config_kwargs = {"trust_remote_code": model.trust_remote_code}
    config = loader.process_config(loader.build_config(model_dir, **config_kwargs))
    info.config = config
    info.architectures = list(getattr(config, "architectures", None) or ())
    processor_kwargs = dict(model.processor_kwargs)
    processor_kwargs.setdefault("trust_remote_code", model.trust_remote_code)
    processor = loader.process_tokenizer(loader.build_processor(model_dir, config, **processor_kwargs))
    processor.model_info = info
    processor.model_meta = SimpleNamespace(
        template=loader_cls.template,
        candidate_templates=[loader_cls.template] if loader_cls.template else [],
        model_type=loader_cls.model_type,
        is_multimodal=loader_cls.is_multimodal,
        tags=list(loader_cls.tags),
    )
    return processor


def _make_template_spec(
    template_config, model: ModelRef, model_info: SpecModelInfo, template_type: str, media_types=()
) -> TemplateSpec:
    return TemplateSpec(
        model=model,
        model_info=model_info,
        template_type=template_type,
        agent_template=template_config.agent_template,
        system=template_config.system,
        max_length=template_config.max_length,
        truncation_strategy=template_config.truncation_strategy,
        max_pixels=template_config.max_pixels,
        norm_bbox=template_config.norm_bbox,
        use_chat_template=template_config.use_chat_template,
        padding_side=template_config.padding_side,
        padding_free=template_config.padding_free,
        loss_scale=template_config.loss_scale,
        sequence_parallel_size=template_config.sequence_parallel_size,
        is_binary_loss_scale=template_config.is_binary_loss_scale,
        template_backend=template_config.template_backend,
        response_prefix=template_config.response_prefix,
        enable_thinking=template_config.enable_thinking,
        preserve_thinking=template_config.preserve_thinking,
        add_non_thinking_prefix=template_config.add_non_thinking_prefix,
        disable_ignore_empty_think=template_config.disable_ignore_empty_think,
        media_types=tuple(sorted(media_types)),
    )


def build_template_spec(template_config: "TemplateConfig", model_config) -> TemplateSpec:
    from swift.dev.model.loader.base import resolve_loader_cls

    if model_config.model is None:
        raise ValueError("ModelConfig.model is required to build a TemplateSpec.")
    loader_cls = resolve_loader_cls(model_config.model_type, model_config.model)
    model_type = model_config.model_type or (loader_cls.model_type if loader_cls is not None else None)
    template_type = template_config.template or (loader_cls.template if loader_cls is not None else None)
    if template_type is None:
        raise ValueError("TemplateConfig.template or a model loader default is required to build a TemplateSpec.")
    tags = set(getattr(loader_cls, "tags", ()) or ())
    media_types = {"image" if tag == "vision" else tag for tag in tags if tag in {"vision", "image", "video", "audio"}}
    is_multimodal = bool(getattr(loader_cls, "is_multimodal", False))
    if is_multimodal and not media_types:
        media_types.add("image")
    return _make_template_spec(
        template_config,
        ModelRef(model_config.model, revision=model_config.model_revision),
        SpecModelInfo(
            torch_dtype_name=model_config.torch_dtype or "float32",
            task_type=model_config.task_type or "causal_lm",
            max_model_len=model_config.max_model_len,
            model_type=model_type,
            is_multimodal=is_multimodal,
        ),
        template_type,
        media_types,
    )


def build_template_from_spec(spec: TemplateSpec, processor=None):
    from swift.dev.config import TemplateConfig

    if processor is None:
        processor = build_processor_from_ref(spec.model, spec.model_info)
    config = TemplateConfig(
        template=spec.template_type,
        agent_template=spec.agent_template,
        system=spec.system,
        max_length=spec.max_length,
        truncation_strategy=spec.truncation_strategy,
        max_pixels=spec.max_pixels,
        norm_bbox=spec.norm_bbox,
        use_chat_template=spec.use_chat_template,
        padding_side=spec.padding_side,
        padding_free=spec.padding_free,
        loss_scale=spec.loss_scale,
        sequence_parallel_size=spec.sequence_parallel_size,
        is_binary_loss_scale=spec.is_binary_loss_scale,
        template_backend=spec.template_backend,
        response_prefix=spec.response_prefix,
        enable_thinking=spec.enable_thinking,
        preserve_thinking=spec.preserve_thinking,
        add_non_thinking_prefix=spec.add_non_thinking_prefix,
        disable_ignore_empty_think=spec.disable_ignore_empty_think,
    )
    return build_template(config, processor, task_type=spec.model_info.task_type)


def build_template(template_config: TemplateConfig, processor: Any, *, task_type: Optional[str] = None) -> Template:
    """TemplateConfig + processor -> :class:`swift.dev.template.Template`.

    Family resolution and implementation both live under ``swift.dev.template``. ``task_type`` stays
    explicit because a processor loaded without model weights may not carry the requested training task.
    """
    from swift.dev.template.register import get_template, get_template_meta

    # ``delete`` is realised dataset-side (EncodePreprocessor drops rows that raise MaxLengthError),
    # so the encoder itself always raises.
    truncation_strategy = template_config.truncation_strategy
    if truncation_strategy in (None, "delete"):
        truncation_strategy = "raise"

    # None means "use the registry factory's default", not a false chat-template flag.
    kwargs = {}
    if template_config.use_chat_template is not None:
        kwargs["use_chat_template"] = template_config.use_chat_template
    template_meta = get_template_meta(processor.model_info, processor.model_meta, template_config.template)
    loss_scale = template_config.loss_scale
    auto_ignore_empty_think = template_meta.is_thinking and template_meta.non_thinking_prefix
    # Disable only automatic masking; an explicitly composed loss_scale remains authoritative.
    if (
        auto_ignore_empty_think
        and not template_config.disable_ignore_empty_think
        and "ignore_empty_think" not in loss_scale.split("+")
    ):
        loss_scale += "+ignore_empty_think"

    model_info = processor.model_info
    model_factory = partial(
        _build_meta_model,
        deepcopy(model_info.config),
        getattr(model_info, "model_type", None) or getattr(processor.model_meta, "model_type", None),
        model_info.model_dir,
    )
    template = get_template(
        processor,
        default_system=template_config.system,
        agent_template=template_config.agent_template,
        max_length=template_config.max_length,
        template_type=template_meta.template_type,
        truncation_strategy=truncation_strategy,
        max_pixels=template_config.max_pixels,
        norm_bbox=template_config.norm_bbox,
        loss_scale=loss_scale,
        is_binary_loss_scale=template_config.is_binary_loss_scale,
        padding_free=template_config.padding_free,
        padding_side=template_config.padding_side,
        sequence_parallel_size=template_config.sequence_parallel_size,
        template_backend=template_config.template_backend,
        response_prefix=template_config.response_prefix,
        enable_thinking=template_config.enable_thinking,
        preserve_thinking=template_config.preserve_thinking,
        add_non_thinking_prefix=template_config.add_non_thinking_prefix,
        media_types=_media_types(processor),
        task_type=task_type,
        model_factory=model_factory,
        multimodal_graph_adapter=_multimodal_graph_adapter(processor),
        **kwargs,
    )
    template._append_truncation_strategy = template_config.truncation_strategy or "raise"

    model_type = getattr(model_info, "model_type", None) or getattr(processor.model_meta, "model_type", None)
    dtype_name = getattr(model_info, "torch_dtype", None)
    if dtype_name is not None and hasattr(dtype_name, "__str__"):
        dtype_name = str(dtype_name).replace("torch.", "")
    template._spec = _make_template_spec(
        template_config,
        getattr(processor, "_dev_model_ref", None)
        or ModelRef(
            model_id_or_dir=getattr(model_info.config, "_name_or_path", None) or model_info.model_dir,
            revision=(getattr(processor, "init_kwargs", None) or {}).get("revision"),
            trust_remote_code=bool((getattr(processor, "init_kwargs", None) or {}).get("trust_remote_code", True)),
        ),
        SpecModelInfo(
            torch_dtype_name=dtype_name or "float32",
            task_type=template.task_type,
            max_model_len=model_info.max_model_len,
            model_type=model_type,
            is_multimodal=template.is_multimodal,
        ),
        template_meta.template_type,
        template.media_types,
    )
    return template
