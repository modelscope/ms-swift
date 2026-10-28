# Copyright (c) ModelScope Contributors. All rights reserved.
# ruff: noqa: C901
from __future__ import annotations
import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import json

from swift.dev.utils.logger import get_logger
from swift.dev.utils.naming import resolve_plugin_class
from swift.dev.utils.plugin import PluginKind, PluginRegistry

if TYPE_CHECKING:
    from swift.dev.config import (
        GenerationConfig,
        InferConfig,
        ModelConfig,
        QuantizeConfig,
        RolloutConfig,
        TemplateConfig,
    )

logger = get_logger()

#: Built-in engine names a ``--sampler`` resolves to with no external source; anything else is a
#: custom sampler, reached by class or by source (see :func:`_resolve_sampler`). 'twinkle' is the
#: client-side sampler over a twinkle-server (mode='remote'); it resolves in :func:`build_sampler`
#: before the engine table, because it is not an engine at all.
SamplerBackend = Literal["vllm", "sglang", "transformers"]

#: ModelConfig knobs the engines accept, under the name each uses for them; everything else an
#: engine takes goes verbatim through ``engine_args`` -- dev does not mirror engine flags.
_MODEL_KNOB_NAMES = {
    "vllm": {"torch_dtype": "dtype", "max_model_len": "max_model_len"},
    "sglang": {"torch_dtype": "dtype", "max_model_len": "context_length"},
    "transformers": {"torch_dtype": "dtype", "max_model_len": "max_model_len", "attn_impl": "attn_implementation"},
}

# The 'sampler' plugin registry
_SAMPLERS: Dict[str, type] = {}


def _sampler_kind() -> PluginKind:
    """Declare (once) and seed the 'sampler' plugin kind, then return it.

    Lazy: seeding needs twinkle's Sampler base (whose package pulls in vLLM), so it runs here where
    the engine stack is imported anyway; engines are seeded by name into the table plugins share.
    """
    if "sampler" not in PluginRegistry.KINDS:
        import twinkle.sampler as twinkle_sampler

        PluginRegistry.register_kind("sampler", twinkle_sampler.Sampler, config_field="sampler", entries=_SAMPLERS)
        for name, cls in (
            ("vllm", twinkle_sampler.vLLMSampler),
            ("sglang", twinkle_sampler.SGLangSampler),
            ("transformers", twinkle_sampler.TransformersSampler),
        ):
            _SAMPLERS[name] = cls
    return PluginRegistry.kind("sampler")


def _resolve_sampler(spec: Any) -> Tuple[type, Optional[str]]:
    """A ``--sampler`` spec -> ``(sampler class, engine key)``.

    engine key ('vllm'/'sglang'/'transformers'; 'pt' aliases the last) selects the knob mapping /
    LoRA / pooling rules; None = a custom sampler built with engine_args verbatim.
    """
    if spec == "pt":
        spec = "transformers"
    if spec in ("client", "no"):
        # legacy's remote-generation sampler; without this guard it falls into plugin resolution
        # and dies with an unrelated "plugin not found" error.
        raise ValueError(
            f"sampler={spec!r} (legacy remote generation) is not migrated: v5 samples with the local "
            "vllm/sglang/transformers backends. Serve the remote side with `swift deploy` and point an "
            "http(s) URL at it (an API-judge reward channel) instead of a client sampler.")
    kind = _sampler_kind()
    sampler_cls = resolve_plugin_class(spec, kind.base, kind.entries, kind="sampler")
    engine = next((name for name, cls in kind.entries.items() if cls is sampler_cls), None)
    return sampler_cls, engine


#: dev task_types that run a pooling forward, and the head each runs: embedding -> embed (a sentence
#: vector), seq_cls/reranker -> classify. Pooling needs a backend with a pooling head (vLLM/SGLang).
_POOLING_TASK_MAP = {
    "embedding": "embed",
    "seq_cls": "classify",
    "reranker": "classify",
}

#: dev task_types served by generation rather than a pooling head: causal_lm is plain generation;
#: generative_reranker is decoder-only, scored off its first generated token (run_infer's own math).
_GENERATIVE_TASK_TYPES = ("causal_lm", "generative_reranker")


def build_engine_args(backend: str, infer_config: "InferConfig", rollout_config: "RolloutConfig") -> Dict[str, Any]:
    """Map inference and rollout Config fields to sampler engine arguments."""
    if backend == "pt":
        backend = "transformers"
    if backend == "transformers":
        return {"max_batch_size": infer_config.max_batch_size}
    prefix = f"{backend}_"
    #: Not engine args: ``engine_kwargs`` is merged verbatim below, and the LoRA knobs are owned by
    #: _enable_lora/_max_adapter_rank -- a config default here would squat the key and defeat them.
    excluded = {"engine_kwargs"}
    # sglang spells its parallel sizes tp_size/dp_size on sglang.Engine, but SGLangEngine takes them
    # as tensor_parallel_size/data_parallel_size; forwarding the short names would collide with the
    # engine's own explicit keywords (``dict(tp_size=..., **engine_kwargs)`` -> TypeError).
    sglang_rename = {"tp_size": "tensor_parallel_size", "dp_size": "data_parallel_size"}
    result = {}
    for name, value in asdict(rollout_config).items():
        if name.startswith(prefix) and value is not None:
            key = name[len(prefix) :]
            if key in excluded:
                continue
            if backend == "sglang":
                key = sglang_rename.get(key, key)
            result[key] = value
    result.update(getattr(rollout_config, f"{prefix}engine_kwargs", None) or {})
    return result


def resolve_twinkle_template(
    template_config: Optional["TemplateConfig"], model_config: "ModelConfig"
) -> Dict[str, Any]:
    """dev Configs -> a serializable twinkle template spec, for a sampler in a Ray actor.

    Returns ``{'template_cls': <name>, **kwargs}`` -- a class NAME plus kwargs, not the DevMixin
    instance (it cannot cross the actor boundary); the name comes from twinkle's own template map.
    """
    from twinkle.server.utils import get_template_for_model

    # model_id is left out: the sampler fills it from its own (the same model).
    spec: Dict[str, Any] = {"template_cls": get_template_for_model(str(model_config.model or ""))}
    if template_config is None:
        return spec
    # Only values the config actually sets are carried, so twinkle's own defaults
    # (use_chat_template=True, max_length=8192, enable_thinking=True) stand for the rest.
    for spec_key, value in (
        ("use_chat_template", template_config.use_chat_template),
        ("max_length", template_config.max_length),
        ("truncation_strategy", template_config.truncation_strategy),
        ("default_system", template_config.system),
        ("enable_thinking", template_config.enable_thinking),
    ):
        if value is not None:
            spec[spec_key] = value
    return spec


def _derive_sampler_type(backend: str, enable_data_plane: bool = False) -> str:
    """Map a dev backend (+ the data-plane flag) to a twinkle-server ``sampler_type``.

    The async variants give the data plane a non-blocking engine to ride (its rollout submit hands
    back a Future), so enable_data_plane selects them; transformers has none and refuses.
    """
    if backend == "pt":
        backend = "transformers"
    if backend == "vllm":
        return "vllm_async" if enable_data_plane else "vllm"
    if backend == "sglang":
        return "sglang_async" if enable_data_plane else "sglang"
    if backend == "transformers":
        if enable_data_plane:
            raise ValueError(
                'backend="transformers" cannot serve the data plane: sample_to_data_plane needs '
                "the sampler's non-blocking submit, which only the vllm_async / "
                'sglang_async backends provide. Use backend="vllm" or "sglang" with '
                "enable_data_plane, or drop enable_data_plane."
            )
        return "torch"
    raise ValueError(
        f"backend={backend!r} has no twinkle-server sampler; `swift deploy` serves vllm, sglang or transformers (pt)."
    )


def build_sampler(
    model_config: ModelConfig,
    *,
    backend: Any = "vllm",
    engine_args: Optional[Dict[str, Any]] = None,
    device_mesh: Any = None,
    template: Any = None,
    adapters: Optional[List[str]] = None,
    remote_group: Optional[str] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    instance_id: Optional[str] = None,
) -> Any:
    """ModelConfig -> a twinkle Sampler, with the template already set.

    Built-in backends map ModelConfig knobs + LoRA/pooling rules (custom samplers get engine_args
    verbatim); engine_args win over mapped knobs; adapters must arrive at construction time.
    """
    if model_config.model is None:
        raise ValueError("ModelConfig.model is required to build a sampler (it is the model id/path).")
    task_type = model_config.task_type or "causal_lm"
    is_pooling = task_type in _POOLING_TASK_MAP
    if task_type not in _GENERATIVE_TASK_TYPES and not is_pooling:
        raise ValueError(
            f"build_sampler got task_type={task_type!r}; expected one of the generation tasks "
            f"{list(_GENERATIVE_TASK_TYPES)} or the pooling tasks {sorted(_POOLING_TASK_MAP)}."
        )
    if backend == "twinkle":
        return _build_client_sampler(model_config, template, adapters, is_pooling)
    sampler_cls, engine = _resolve_sampler(backend)
    if is_pooling and engine == "transformers":
        raise ValueError(
            f'backend="transformers" cannot serve the pooling task_type={task_type!r}: it has '
            "no pooling head. Build it with build_model(..) and pass task= instead, or use "
            'backend="vllm"/"sglang".'
        )

    kwargs = dict(engine_args or {})
    quant_method = getattr(quantize_config, "quant_method", None)
    if engine is None:
        # A custom sampler owns its engine's knob vocabulary: engine_args pass through verbatim, none
        # of the built-ins' mapping applies; quant_method (transformers load-time) is refused, not dropped.
        if quant_method is not None:
            raise ValueError(
                f"quant_method={quant_method!r} is a Transformers load-time setting and cannot be applied to the "
                f"custom sampler {sampler_cls.__name__}. Quantize inside the sampler, or pass its own engine_args."
            )
    else:
        if quant_method is not None:
            if engine != "transformers":
                engine_option = "vllm_quantization" if engine == "vllm" else "sglang_quantization"
                raise ValueError(
                    f"quant_method={quant_method!r} is a Transformers load-time setting and cannot be translated to "
                    f"backend={engine!r}. Use --{engine_option} (or backend-specific engine_args), or rely on the "
                    "quantization metadata embedded in an AWQ/GPTQ checkpoint."
                )
            from swift.dev.builders.quantization import build_load_quantization_config

            quantization_config = build_load_quantization_config(quantize_config, torch_dtype=model_config.torch_dtype)
            if quantization_config is not None:
                kwargs.setdefault("quantization_config", quantization_config)

        for cfg_name, engine_name in _MODEL_KNOB_NAMES[engine].items():
            value = getattr(model_config, cfg_name, None)
            # setdefault, not assignment: an explicit engine_args entry is the caller's override.
            if value is not None:
                kwargs.setdefault(engine_name, value)
        if is_pooling:
            # Route the model to the backend's pooling forward: vLLM spells it ``runner='pooling'``
            # (swaps the LM head for the pooling head), sglang ``is_embedding=True``.
            if engine == "vllm":
                kwargs.setdefault("runner", "pooling")
            elif engine == "sglang":
                kwargs.setdefault("is_embedding", True)
        if adapters:
            _enable_lora(kwargs, engine, adapters)

        if engine == "transformers":
            # AutoModelForCausalLM cannot load a VL checkpoint; hand the family loader's declared
            # model_cls ('module:ClassName': engine-resolvable, Ray-safe); setdefault keeps overrides.
            from swift.dev.builders.model import _resolve_model_loader

            loader = _resolve_model_loader(model_config)
            if loader is not None:
                kwargs.setdefault("model_cls", loader.model_cls)

    validate_capabilities = getattr(template, "require_capabilities", None)
    if validate_capabilities is not None:
        from swift.dev.template import TemplateCapability

        required = [TemplateCapability.OUTPUT_POSTPROCESS]
        if engine == "transformers":
            required.extend((TemplateCapability.HF_GENERATE, TemplateCapability.GENERATE_KWARGS))
        elif engine == "vllm":
            required.append(TemplateCapability.ENGINE_KWARGS)
        validate_capabilities(*required)

    logger.info(f"Building {engine or sampler_cls.__name__} sampler for {model_config.model} with engine_args={kwargs}")
    extra: Dict[str, Any] = {"remote_group": remote_group} if remote_group else {}
    if instance_id is not None:
        if not instance_id:
            raise ValueError("instance_id must be non-empty when provided.")
        extra["instance_id"] = instance_id
    sampler = sampler_cls(model_config.model, engine_args=kwargs, device_mesh=device_mesh, **extra)
    try:
        if template is not None:
            sampler.set_template(template)
    except BaseException:
        try:
            sampler.close_workers()
        except Exception:
            logger.warning("Failed to close sampler workers after template init failed.", exc_info=True)
        raise
    return sampler


def _build_client_sampler(model_config: "ModelConfig", template: Any,
                          adapters: Optional[List[str]], is_pooling: bool) -> Any:
    """backend='twinkle': a client-side vLLMSampler handle over a SERVED sampler.

    The transport must already be bound (init_twinkle_client, done by the mode='remote'
    initialization), and ``model`` names the server-side served model. Only the template travels
    (as a class name, the same cross-boundary form Ray placement uses); pooling forwards ride the
    sampler's ``encode`` face. Local adapters cannot attach to a remote engine -- a remote LoRA
    rides ``adapter_uri`` (a twinkle:// checkpoint), whose infer wiring is future work.
    """
    from twinkle_client.sampler import vLLMSampler as ClientSampler

    if is_pooling:
        raise ValueError(
            'backend="twinkle" cannot serve the pooling task: the client sampler face is '
            'sample()/encode() over a served generation engine. Run pooling locally, or serve it '
            'through the gateway\'s OpenAI-compatible /embeddings route.')
    if adapters:
        raise ValueError(
            f'backend="twinkle" received local adapters {adapters!r}: a local adapter directory '
            'cannot load into a remote engine. Merge the adapter first (`swift merge`), or wait '
            'for the adapter_uri wiring (a twinkle:// checkpoint served as a sampler LoRA).')
    sampler = ClientSampler(model_config.model)
    if template is not None:
        spec = resolve_twinkle_template(None, model_config)
        template_kwargs = {key: value for key, value in spec.items() if key != "template_cls"}
        sampler.set_template(spec["template_cls"], **template_kwargs)
    return sampler


def _enable_lora(kwargs: Dict[str, Any], backend: str, adapters: List[str]) -> None:
    """Turn on the engine's LoRA machinery, sized for ``adapters``.

    vLLM allocates max_loras slots and rejects ranks above max_lora_rank at request time, so both
    must be right before the first request; the rank is read from each adapter_config.json.
    """
    if backend == "transformers":
        return  # peft loads adapters into the live module; nothing to reserve.
    if backend == "vllm":
        kwargs.setdefault("enable_lora", True)
        kwargs.setdefault("max_loras", len(adapters))
        kwargs.setdefault("max_lora_rank", _max_adapter_rank(adapters))
    elif backend == "sglang":
        kwargs.setdefault("enable_lora", True)
        kwargs.setdefault("max_loras_per_batch", len(adapters))


def _max_adapter_rank(adapters: List[str], default: int = 16) -> int:
    """Largest ``r`` across the adapters, from their configs; ``default`` when none can be read."""
    ranks = []
    for adapter in adapters:
        config_path = os.path.join(adapter, "adapter_config.json")
        if not os.path.isfile(config_path):
            # Hub ids are only resolved later by the sampler, so the rank is not knowable here.
            logger.warning(f"No adapter_config.json under {adapter}; falling back to max_lora_rank={default}.")
            continue
        with open(config_path, encoding="utf-8") as f:
            ranks.append(int(json.load(f).get("r", default)))
    return max(ranks) if ranks else default


def apply_generation_baseline(generation_config: Optional[GenerationConfig], model_id: Optional[str]) -> None:
    """Fill unset sampling knobs from the model's own ``generation_config.json``; explicit knobs win.

    legacy loaded the model-side file as the sampling baseline, so a model's declared defaults (a low
    temperature, gemma's top_k/top_p) survive when the caller sets nothing; without this the samplers'
    own defaults (temperature=1.0) silently replace them. ``GenerationConfig.generation_config`` -- a
    path, directory, or HF GenerationConfig instance -- overrides the model id as the source.
    """
    if generation_config is None:
        return
    source = generation_config.generation_config
    try:
        if source is None:
            if not model_id:
                return
            from transformers import GenerationConfig as HfGenerationConfig

            source = HfGenerationConfig.from_pretrained(model_id)
        elif isinstance(source, str):
            from transformers import GenerationConfig as HfGenerationConfig

            source = HfGenerationConfig.from_pretrained(source)
    except OSError:
        # No generation_config.json beside the model -- the common case for a plain LLM.
        return
    for name in ("max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty"):
        value = getattr(source, name, None)
        if value is None or getattr(generation_config, name, None) is not None:
            continue
        if name == "top_k" and value == 0:
            value = -1  # HF spells "disabled" as 0; the sampler side spells it -1 (legacy mapping)
        setattr(generation_config, name, value)


def to_sampling_params(generation_config: Optional[GenerationConfig] = None, **overrides) -> Any:
    """GenerationConfig -> twinkle SamplingParams.

    Only set fields are carried (twinkle's defaults stand); names map max_new_tokens -> max_tokens,
    stop_words -> stop, logprobs+top_logprobs -> one int (vLLM semantics). ``overrides`` win.
    """
    from twinkle.data_format import SamplingParams

    params: Dict[str, Any] = {}
    if generation_config is not None:
        _reject_unsupported(generation_config)
        if generation_config.max_new_tokens is not None:
            params["max_tokens"] = generation_config.max_new_tokens
        for name in ("temperature", "top_k", "top_p", "repetition_penalty"):
            value = getattr(generation_config, name)
            if value is not None:
                params[name] = value
        if generation_config.stop_words:
            params["stop"] = list(generation_config.stop_words)
        if generation_config.logprobs:
            params["logprobs"] = generation_config.top_logprobs or 0
    params.update(overrides)
    return SamplingParams(**params)


def _reject_unsupported(generation_config: GenerationConfig) -> None:
    """Fail loudly on GenerationConfig knobs twinkle's SamplingParams has no field for."""
    if generation_config.num_beams and generation_config.num_beams > 1:
        raise ValueError(
            f"num_beams={generation_config.num_beams} is not supported: twinkle SamplingParams "
            "has no beam-search field. Use num_samples for parallel sampling instead."
        )
    if generation_config.structured_outputs_regex:
        raise ValueError(
            "structured_outputs_regex is not supported: twinkle SamplingParams has no "
            "guided-decoding field. Pass the engine its own guided-decoding args via "
            "engine_args if the backend supports them."
        )


def sampled_texts(responses: List[Any]) -> List[List[str]]:
    """SampleResponse list -> the decoded text of each sequence, grouped per input."""
    return [[seq.decoded for seq in response.sequences] for response in responses]


def pooling_task_for(task_type: Optional[str]) -> str:
    """dev ``task_type`` -> the twinkle/vLLM pooling head name, or ``None`` for a generation task."""
    return _POOLING_TASK_MAP.get(task_type or "causal_lm")


def is_pooling_task(task_type: Optional[str]) -> bool:
    """Whether ``task_type`` is a pooling forward rather than generation."""
    return (task_type or "causal_lm") in _POOLING_TASK_MAP


def to_pooling_params(task_type: Optional[str] = None, **overrides) -> Any:
    """dev ``task_type`` (+ overrides) -> twinkle ``PoolingParams``.

    task_type selects the head (default embed); reranker also sets is_cross_encoder (vLLM infers it
    from the model, sglang routes on it). ``overrides`` carry the post-processing knobs.
    """
    from twinkle.data_format import PoolingParams

    task_type = task_type or "causal_lm"
    params: Dict[str, Any] = {"task": pooling_task_for(task_type) or "embed"}
    if task_type == "reranker":
        params["is_cross_encoder"] = True
    params.update(overrides)
    return PoolingParams(**params)


def pooled_data(responses: List[Any]) -> List[List[float]]:
    """PoolingResponse list -> the pooled floats of each, one entry per input."""
    return [list(response.data) for response in responses]
