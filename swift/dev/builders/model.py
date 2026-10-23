# Copyright (c) ModelScope Contributors. All rights reserved.
# ruff: noqa: C901
from __future__ import annotations
import copy
import dataclasses
import math
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from twinkle import DeviceMesh

    from swift.dev.config import (
        DistributedConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        TrainConfig,
        TunerConfig,
    )
    from swift.dev.model import TrainableModel

logger = get_logger()

# Global, version-gated: torch 2.9.x ships a broken nn.Conv3d for video/3D ViT patch embeds; the
# patch is a no-op on every other torch. Mirrors legacy swift/model/utils.py's import-time apply.
from twinkle.patch import apply_patch  # noqa: E402

from swift.dev.patch.torch_conv3d import TorchConv3dPatch  # noqa: E402

apply_patch(None, TorchConv3dPatch())


def load_model_processor(model_config: ModelConfig, *, load_model: bool = False, **overrides):
    """Load a plain HF model and its processor with every relevant ``ModelConfig`` option.

    The dev-registry twin of the training build path: a matching family loader owns
    config/processor/model construction, the plain ``Auto*`` classes cover everything else.
    Template, export and weight-transform paths share this one seam so call sites cannot each
    forward a different subset of the contract. Returns ``(model_or_None, processor)`` with
    ``model.model_dir`` / ``model.model_info`` / ``model.model_meta`` / ``processor.model_info``
    / ``processor.model_meta`` attached -- the attributes the consumers read.
    """
    import torch

    from swift.dev.model.loader import resolve_loader_cls, resolve_model_info
    from swift.dev.utils import HfConfigFactory

    # Vision preprocessing overrides go through env so the loader's patch_qwen_vl_utils (read via
    # get_env_args) picks them up inside the worker's process_tokenizer across the Ray boundary.
    if model_config.vision_process_kwargs:
        for key, value in model_config.vision_process_kwargs.items():
            os.environ[key.upper()] = str(value)

    return_dummy_model = bool(overrides.pop("return_dummy_model", False))
    download_model = overrides.pop("download_model", None)
    if download_model is None:
        download_model = load_model and not return_dummy_model
    device_map = overrides.pop("device_map", None)
    quantization_config = overrides.pop("quantization_config", None)
    auto_model_cls = overrides.pop("auto_model_cls", None)

    info = resolve_model_info(
        model_config.model,
        revision=model_config.model_revision,
        download_model=download_model,
        model_type=model_config.model_type,
        task_type=model_config.task_type,
        num_labels=model_config.num_labels,
        problem_type=model_config.problem_type,
        torch_dtype=(getattr(torch, model_config.torch_dtype) if model_config.torch_dtype else None),
    )
    model_dir = info.model_dir
    loader_cls = resolve_loader_cls(model_config.model_type or info.model_type, model_config.model)
    loader = loader_cls(info) if loader_cls is not None else None

    from twinkle.patch import apply_context

    from swift.dev.patch.transformers_dynamic_module import TransformersDynamicModulePatch

    with apply_context(None, TransformersDynamicModulePatch()):
        # A copy of ModelConfig pointed at the resolved local directory keeps _build_hf_config's
        # reads off the (possibly remote) original id.
        load_config = copy.copy(model_config)
        load_config.model = model_dir
        config = _build_hf_config(load_config, loader)
        _postprocess_load_config(config, info, model_config.attn_impl)
        info.config = config

        if loader is not None:
            processor = loader.process_tokenizer(loader.build_processor(model_dir, config))
        elif os.path.exists(os.path.join(model_dir, "preprocessor_config.json")) or os.path.exists(
            os.path.join(model_dir, "processor_config.json")
        ):
            # A multimodal checkpoint ships a (pre)processor config; a plain LLM ships only a
            # tokenizer. The generic path trusts remote code, as legacy's default loader did.
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        else:
            from transformers import AutoTokenizer

            processor = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        pad_token_id = _normalize_pad_eos(processor)
        processor.model_info = info
        candidate_templates = [template for template in (info.template,) if template]
        if not candidate_templates:
            # Unregistered checkpoint: the plain chat-template fallback (legacy's dummy ModelMeta).
            candidate_templates = ["dummy"]
        from types import SimpleNamespace

        processor.model_meta = SimpleNamespace(
            template=info.template or "dummy",
            candidate_templates=candidate_templates,
            model_type=info.model_type,
            is_multimodal=info.is_multimodal,
            tags=info.tags,
        )

        model = None
        if load_model:
            if info.task_type in ("seq_cls", "reranker"):
                _prepare_seq_cls_config(config, model_config, info, pad_token_id)
            load_kwargs = dict(model_config.model_kwargs or {})
            # device_map has no default: it reaches from_pretrained only when a caller passes it
            # explicitly (merge's CPU load, quantize's 'auto' guard) or via model_kwargs.
            if device_map is not None:
                _set_load_kwarg(load_kwargs, "device_map", device_map)
            if quantization_config is not None:
                _set_load_kwarg(load_kwargs, "quantization_config", quantization_config)
            if model_config.max_memory:
                _set_load_kwarg(load_kwargs, "max_memory", model_config.max_memory)
            # Legacy always forwards a resolved dtype; an fp8 checkpoint wants transformers' 'auto'.
            _set_load_kwarg(load_kwargs, _dtype_kwarg_name(), "auto" if info.quant_method == "fp8" else info.torch_dtype)
            _forward_experts_impl(load_kwargs, model_config.experts_impl)
            for name, value in overrides.items():
                _set_load_kwarg(load_kwargs, name, value)

            if auto_model_cls is not None:
                model_cls = auto_model_cls
            elif info.task_type in ("seq_cls", "reranker"):
                from transformers import AutoModelForSequenceClassification

                model_cls = AutoModelForSequenceClassification
            elif loader is not None:
                model_cls = loader.resolve_model_cls()
            else:
                from transformers import AutoModelForCausalLM

                model_cls = AutoModelForCausalLM
            if hasattr(model_cls, "_tp_plan"):  # a None tp_plan breaks from_pretrained (legacy fix)
                model_cls._tp_plan = model_cls._tp_plan or {}
            if return_dummy_model:
                # cls(config) without reading the checkpoint; the caller's init_empty_weights
                # context (merge's streaming path) lands every parameter on meta. The default-dtype
                # dance keeps fresh parameters in the checkpoint's dtype.
                origin_dtype = torch.get_default_dtype()
                torch.set_default_dtype(getattr(config, "torch_dtype", None) or torch.float32)
                model = model_cls(copy.deepcopy(config))
                torch.set_default_dtype(origin_dtype)
            elif loader is not None and auto_model_cls is None and info.task_type not in ("seq_cls", "reranker"):
                # Full family takeover: model_cls, trust_remote_code warning and the transformers-5
                # multimodal attribute re-exposure all live in the loader's build_model.
                with _patch_tp_plan():
                    model = loader.build_model(model_dir, config, processor, **load_kwargs)
            else:
                with _patch_tp_plan():
                    trust = {} if auto_model_cls is not None else {"trust_remote_code": True}
                    model = model_cls.from_pretrained(model_dir, config=config, **trust, **load_kwargs)
            if loader is not None:
                model = loader.process_model(model)
            if model_config.init_strategy is not None:
                _apply_raw_init_strategy(model, model_config.init_strategy)
            model.model_info = info
            model.model_meta = info
            model.model_dir = model_dir
            _init_generation_config(model, model_dir)
            HfConfigFactory.set_config_attr(model.config, "pad_token_id", pad_token_id)
        return model, processor


def _postprocess_load_config(config, info, attn_impl) -> None:
    """The raw-load half of legacy ``ModelLoader._postprocess_config`` that ``_build_hf_config``
    does not already cover: inference-ignore keys, the resolved dtype (vision sub-configs
    included, so a tower built from the config picks it up), ZeRO-3 compat and the attention
    implementation write."""
    from swift.dev.utils import HfConfigFactory

    if not hasattr(config, "keys_to_ignore_at_inference"):
        config.keys_to_ignore_at_inference = []
    if "past_key_values" not in config.keys_to_ignore_at_inference:
        config.keys_to_ignore_at_inference.append("past_key_values")
    HfConfigFactory.set_config_attr(config, "torch_dtype", info.torch_dtype, include_vit=True)
    HfConfigFactory.compat_zero3(config)
    if attn_impl:
        use_flash_attn = attn_impl in {"flash_attn", "flash_attention_2"}
        resolved = "flash_attention_2" if use_flash_attn else attn_impl
        for key in ("_attn_implementation", "attn_implementation", "llm_attn_implementation"):
            HfConfigFactory.set_config_attr(config, key, resolved, include_vit=True, ensure_set=False)
        for key in ("_flash_attn_2_enabled", "use_flash_attn", "_use_flash_attention_2"):
            HfConfigFactory.set_config_attr(config, key, use_flash_attn, include_vit=True, ensure_set=False)


def _normalize_pad_eos(processor) -> int:
    """Make pad and eos ids both exist (pad defaults to eos and vice versa); returns the pad id.

    The fixed-format batch contract needs a pad id; a tokenizer declaring neither cannot produce
    one, which is a hard error rather than a silent ``None`` written into every batch.
    """
    tokenizer = processor if not hasattr(processor, "tokenizer") else processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token_id
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError(
            "the tokenizer declares neither pad_token_id nor eos_token_id; every fixed-format batch "
            "needs a pad id. Register the special tokens on the checkpoint or pass a tokenizer that "
            "declares them."
        )
    return pad_token_id


def _prepare_seq_cls_config(config, model_config: ModelConfig, info, pad_token_id) -> None:
    """Route a seq_cls/reranker load onto a num_labels-wide SequenceClassification head.

    Classification attrs live ON the config (HF only auto-parses num_labels when config is None);
    the head needs a pad id (eos fallback) to find the last token.
    """
    num_labels = model_config.num_labels if model_config.num_labels is not None else info.num_labels
    if num_labels is None:
        if info.task_type == "reranker":
            num_labels = 1
        else:
            raise ValueError('ModelConfig.num_labels is required for task_type="seq_cls".')
    config.num_labels = num_labels
    if model_config.problem_type is not None:
        config.problem_type = model_config.problem_type
    elif info.problem_type is not None:
        config.problem_type = info.problem_type
    config.tie_word_embeddings = False
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = pad_token_id


def _dtype_kwarg_name() -> str:
    """from_pretrained's dtype kwarg name for the installed transformers (``dtype`` since 4.56)."""
    import transformers
    from packaging import version

    return "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"


def _forward_experts_impl(load_kwargs: dict, experts_impl) -> None:
    """Forward experts_impl as from_pretrained's ``experts_implementation`` (transformers>=5 only)."""
    if experts_impl is None:
        return
    import transformers
    from packaging import version

    if version.parse(transformers.__version__) < version.parse("5.0.0"):
        if experts_impl != "eager":
            raise ValueError('experts_impl is only supported in "transformers>=5.0.0".')
        return
    _set_load_kwarg(load_kwargs, "experts_implementation", experts_impl)


@contextmanager
def _patch_tp_plan():
    """Hide ``WORLD_SIZE`` from transformers for the duration of a from_pretrained call.

    Under torchrun with several devices per process, transformers infers a tp_plan from
    ``WORLD_SIZE`` and fails the load; swapping the variable for the duration avoids the
    inference. dev's ``get_dist_setting`` reads the swapped name, so rank detection is unaffected.
    """
    import transformers
    from packaging import version

    world_size = os.environ.get("WORLD_SIZE")
    if (world_size is None or not _is_multi_gpu_per_process()
            or version.parse(transformers.__version__) < version.parse("4.50")):
        yield
        return
    logger.info("Patch tp_plan.")
    os.environ["_PATCH_WORLD_SIZE"] = world_size
    os.environ.pop("WORLD_SIZE")
    try:
        yield
    finally:
        os.environ["WORLD_SIZE"] = world_size


def _visible_device_count() -> int:
    """How many accelerator devices this process sees (cuda > npu > mps > 0)."""
    import torch
    from transformers.utils import is_torch_mps_available, is_torch_npu_available

    if is_torch_npu_available():
        return torch.npu.device_count()
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if is_torch_mps_available():
        return 1
    return 0


def _is_multi_gpu_per_process() -> bool:
    """Whether each process sees more than one device -- the shape that makes transformers
    infer a tp_plan from ``WORLD_SIZE`` during from_pretrained (legacy ``is_mp``)."""
    if os.environ.get("SWIFT_SINGLE_DEVICE_MODE", "0") == "1":
        return False
    from swift.dev.utils import get_dist_setting

    _, _, _, local_world_size = get_dist_setting()
    return _visible_device_count() // max(local_world_size, 1) >= 2


def _apply_raw_init_strategy(model, init_strategy: str) -> None:
    """Re-initialize the weights a blank / partially-loaded config left uninitialized.

    Only parameters whose scale is non-finite or absurdly large (> 1e7) are touched, so a properly
    loaded checkpoint is left intact. The training path applies the same kernel inside twinkle's
    TransformersModel; this is the raw-HF twin for the transform loads.
    """
    import torch
    from torch import nn

    def high_dim(param, init):
        if param.dim() > 1:
            init(param)
        elif param.dim() == 1 and param.numel() > 0:
            nn.init.zeros_(param)

    initializers = {
        "zero": nn.init.zeros_,
        "uniform": lambda param: nn.init.uniform_(param, -0.1, 0.1),
        "normal": lambda param: nn.init.normal_(param, 0.0, 0.01),
        "xavier_uniform": lambda param: high_dim(param, nn.init.xavier_uniform_),
        "xavier_normal": lambda param: high_dim(param, nn.init.xavier_normal_),
        "kaiming_uniform": lambda param: high_dim(
            param, lambda value: nn.init.kaiming_uniform_(value, mode="fan_out", nonlinearity="leaky_relu", a=0.1)),
        "kaiming_normal": lambda param: high_dim(
            param, lambda value: nn.init.kaiming_normal_(value, mode="fan_in", nonlinearity="relu")),
        "orthogonal": lambda param: high_dim(param, nn.init.orthogonal_),
    }
    if init_strategy not in initializers:
        raise ValueError(f"Unknown init_strategy={init_strategy!r}; expected one of {sorted(initializers)}.")
    initialize = initializers[init_strategy]
    with torch.no_grad():
        for param in model.parameters():
            if param.numel() == 0:
                continue
            mean_abs = param.abs().mean()
            std = param.std()
            if not torch.isfinite(mean_abs) or not torch.isfinite(std) or mean_abs > 1e7 or std > 1e7:
                initialize(param)


def _init_generation_config(model, model_dir: str) -> None:
    """Load generation_config.json when the model has none; fix the do_sample/temperature clash."""
    from transformers import GenerationConfig

    path = os.path.join(model_dir, "generation_config.json")
    if getattr(model, "generation_config", None) is None:
        model.generation_config = GenerationConfig.from_pretrained(model_dir) if os.path.isfile(path) else None
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "do_sample"):
        # A zero temperature with do_sample=True is contradictory; align the trio the way legacy's
        # fix_do_sample_warning did so greedy decoding does not warn on every generate call.
        if generation_config.temperature == 0:
            generation_config.do_sample = False
        if generation_config.do_sample is False:
            generation_config.temperature = 1.0
            generation_config.top_p = 1.0
            generation_config.top_k = 50


def build_dp_mesh(distributed_config: DistributedConfig, sequence_parallel_size: int = 1) -> Optional["DeviceMesh"]:
    """The transformers-backend data-parallel DeviceMesh (dp x ulysses), for local AND ray modes.

    Local reads Platform.get_world_size(), ray reads nproc_per_node; local + sp=1 with no spec
    returns None. dp spans ALL ranks, so SP peers share one data rank (identical samples).
    """
    from twinkle import DeviceMesh

    spec = distributed_config.parallel_spec
    is_ray = distributed_config.mode == "ray"

    if is_ray:
        world = distributed_config.nproc_per_node
        if world is None:
            raise ValueError(
                "DistributedConfig.nproc_per_node is required in mode='ray' (it sizes the 'model' "
                "DeviceGroup and its data-parallel mesh). Pass it explicitly -- there is no default."
            )
    else:
        from twinkle.utils import Platform

        world = Platform.get_world_size()

    if spec:
        return DeviceMesh.from_spec(spec, world_size=world)

    if not is_ray and sequence_parallel_size <= 1:
        return None

    if sequence_parallel_size > 1:
        if world < 2:
            raise ValueError(
                f"sequence_parallel_size={sequence_parallel_size} requires WORLD_SIZE>=2; "
                "a single process has no ranks to split a sequence across."
            )
        if world % sequence_parallel_size != 0:
            raise ValueError(f"world_size={world} is not divisible by sequence_parallel_size={sequence_parallel_size}.")
    return DeviceMesh.from_sizes(world_size=world, dp_size=world, ulysses_size=sequence_parallel_size)


def build_model(
    model_config: ModelConfig,
    distributed_config: DistributedConfig,
    train_config: Optional[TrainConfig] = None,
    tuner_config: Optional[TunerConfig] = None,
    device_mesh: Optional["DeviceMesh"] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    megatron_config: Optional[MegatronConfig] = None,
    moe_config: Optional["MoEConfig"] = None,
    remote_group: Optional[str] = None,
    instance_id: Optional[str] = None,
    enable_router_replay: bool = False,
    torch_dtype: Optional[str] = None,
) -> TrainableModel:
    """ModelConfig + DistributedConfig -> a twinkle-native Model (no loss/optim yet).

    megatron -> MegatronModel (bridge), else TransformersModel; device_mesh is transformers-only.
    remote_group/instance_id name the ray group; enable_router_replay is the MoE replay flag.
    ``torch_dtype`` overrides ModelConfig.torch_dtype on the transformers branch only (RL's
    trainable actor pins float32 there; the megatron branch ignores it).
    """
    # Aux models (teacher/reference/reward) arrive as bare hub ids that never pass _resolve_model;
    # resolve here so callers may hand in either form -- a local path short-circuits the download.
    if model_config.model:
        from swift.dev.utils.hub import safe_snapshot_download

        model_config.model = safe_snapshot_download(model_config.model, revision=model_config.model_revision)
    if is_megatron_backend(distributed_config):
        return _build_megatron_model(
            model_config,
            distributed_config,
            train_config,
            megatron_config=megatron_config,
            moe_config=moe_config,
            remote_group=remote_group,
            instance_id=instance_id,
            enable_router_replay=enable_router_replay,
        )
    if torch_dtype is not None:
        model_config = copy.copy(model_config)
        model_config.torch_dtype = torch_dtype
    return _build_transformers_model(
        model_config,
        distributed_config,
        train_config,
        tuner_config,
        device_mesh,
        quantize_config=quantize_config,
        remote_group=remote_group,
        instance_id=instance_id,
        enable_router_replay=enable_router_replay,
    )


def apply_full_param_freeze(model: TrainableModel, train_config: Optional[TrainConfig]) -> None:
    """Consume TrainConfig's freeze/trainable knobs on a full-parameter model.

    Full-param path only (adapters select via target_modules); must run BEFORE configure_optimizer
    (param groups build from requires_grad). No-op when no knob is set.
    """
    if train_config is None:
        return
    freeze_names = train_config.freeze_parameters
    trainable_names = train_config.trainable_parameters
    freeze_regex = train_config.freeze_parameters_regex
    trainable_regex = train_config.trainable_parameters_regex
    freeze_ratio = train_config.freeze_parameters_ratio
    if not (freeze_ratio or freeze_names or freeze_regex or trainable_names or trainable_regex):
        return
    model.freeze_parameters(
        freeze_ratio=freeze_ratio,
        freeze_names=freeze_names,
        freeze_regex=freeze_regex,
        trainable_names=trainable_names,
        trainable_regex=trainable_regex,
    )


def build_ref_model(
    *,
    name: str,
    model_id: Optional[str],
    model_config: ModelConfig,
    template,
    distributed_config: DistributedConfig,
    device_mesh: Any = None,
    parallel_spec: Optional[str] = None,
    adapters: Sequence[str] = (),
    policy_model: Any = None,
    disable_adapter: bool = False,
    model_type: Optional[str] = None,
    model_revision: Optional[str] = None,
    instance_id: Optional[str] = None,
    deepspeed: Optional[str] = None,
) -> Optional[Any]:
    """One frozen auxiliary (reference / teacher), as the BARE twinkle model (cookbook shape).

    ``disable_adapter`` returns the policy model itself with the caller-side contract "forward it
    with ``disable_lora=True``" (the LoRA KL anchor -- same weights, adapters off); ``model_id``
    empty returns ``None`` (the channel is off). Otherwise: build the frozen model in its own
    remote group ``name`` sized by the GIVEN ``device_mesh`` (the caller derives it from its plan
    or spec -- a None mesh is a single-rank build) and install the single frozen adapter. The
    model IS the reference -- twinkle cookbooks forward it directly (``ref_model.forward_only``);
    ``remote_group=name`` already encodes its placement. ``model_type`` / ``model_revision`` let an
    architecturally distinct teacher override the policy's metadata; ``instance_id`` disambiguates
    K same-group actors (MOPD's packed teachers); ``deepspeed`` is the teacher-only engine config.
    """
    if disable_adapter:
        if policy_model is None:
            raise ValueError(f"{name}: disable_adapter=True needs the policy model (the shared-weights view).")
        return policy_model
    if not model_id:
        return None
    from swift.dev.processor import InputProcessor, processor_kwargs

    config = copy.copy(model_config)
    config.model = model_id
    if model_type is not None:
        config.model_type = model_type
    if model_revision is not None:
        config.model_revision = model_revision
    dist = frozen_auxiliary_distributed_config(
        distributed_config,
        device_mesh.world_size if device_mesh else 1,
        parallel_spec=parallel_spec,
        deepspeed=deepspeed,
    )
    frozen = build_model(config, dist, device_mesh=device_mesh, remote_group=name, instance_id=instance_id)
    if len(adapters) > 1:
        raise ValueError(f"{name} currently supports exactly one frozen adapter per auxiliary model.")
    if adapters:
        frozen.add_adapter_to_model(f"{name}_adapter", adapters[0], is_trainable=False)
        frozen.set_processor(InputProcessor, adapter_name=f"{name}_adapter", **processor_kwargs(template))
        frozen.set_template(template, adapter_name=f"{name}_adapter")
    else:
        frozen.set_processor(InputProcessor, **processor_kwargs(template))
        frozen.set_template(template)
    return frozen


def _mixed_precision_for(torch_dtype: Optional[str]) -> str:
    """torch_dtype -> twinkle's mixed_precision mode. Shared by both backends so they cannot drift.

    float32 -> 'no' is a DELIBERATE divergence: legacy sets fp16=True for a float32 run, so "full
    precision" there still autocasts to fp16; dev takes float32 at face value instead.
    """
    if torch_dtype == "float32":
        return "no"
    if torch_dtype == "float16":
        return "fp16"
    return "bf16"


def is_megatron_backend(distributed_config: DistributedConfig) -> bool:
    backend = distributed_config.backend
    if backend in (None, "hf", "unsloth"):
        # 'unsloth' rides the transformers family: it swaps the construction class, not the
        # strategy/parallel machinery, so every megatron-only guard keeps rejecting it.
        return False
    if backend == "megatron":
        return True
    raise ValueError(f"DistributedConfig.backend must be one of {{'megatron', 'hf', 'unsloth'}}, got {backend!r}.")


def frozen_auxiliary_distributed_config(
    distributed_config: DistributedConfig,
    world_size: int,
    *,
    parallel_spec: Optional[str] = None,
    deepspeed: Optional[str] = None,
) -> DistributedConfig:
    """The DistributedConfig a FROZEN auxiliary model (reference / teacher / reward) is built with.

    Inherits only the run's backend identity and forces mode='ray' (RL never runs local); NOT the
    trainer's parallel layout/deepspeed -- a forward-only auxiliary is sized by world_size alone.
    """
    from swift.dev.config import DistributedConfig as _DistributedConfig

    return _DistributedConfig(
        mode="ray",
        backend=distributed_config.backend,
        bridge_backend=distributed_config.bridge_backend,
        nproc_per_node=world_size,
        parallel_spec=parallel_spec,
        deepspeed=deepspeed,
    )


def _apply_seq_cls_head(kwargs: dict, model_config: ModelConfig, config, model_loader=None) -> None:
    """Route a seq_cls/reranker model to a num_labels-wide SequenceClassification head.

    Classification attrs live ON the config (HF only auto-parses num_labels when config is None);
    reranker defaults num_labels=1; the head needs a pad id (eos fallback) to find the last token.
    """
    num_labels = model_config.num_labels
    if num_labels is None:
        if model_config.task_type == "reranker":
            num_labels = 1
        else:
            raise ValueError('ModelConfig.num_labels is required for task_type="seq_cls".')
    config.num_labels = num_labels
    if model_config.problem_type is not None:
        config.problem_type = model_config.problem_type
    config.tie_word_embeddings = False

    if getattr(config, "pad_token_id", None) is None:
        # Source the tokenizer from the resolved dev loader when there is one, else AutoTokenizer.
        if model_loader is not None:
            processor = model_loader.build_processor(model_config.model, config, revision=model_config.model_revision)
        else:
            from transformers import AutoTokenizer

            processor = AutoTokenizer.from_pretrained(
                model_config.model, revision=model_config.model_revision, trust_remote_code=True
            )
        tokenizer = processor if not hasattr(processor, "tokenizer") else processor.tokenizer
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    kwargs["model_cls"] = "AutoModelForSequenceClassification"
    kwargs["config"] = config


def _apply_unsloth_kwargs(
    kwargs: dict,
    model_config: ModelConfig,
    tuner_config: Optional[TunerConfig],
    train_config: Optional[TrainConfig],
    quantize_config: Optional[QuantizeConfig],
) -> None:
    """Add the UnslothModel-only kwargs to an otherwise unchanged TransformersModel kwargs dict.

    unsloth rebuilds the module graph around a causal-LM checkpoint: seq_cls/reranker heads have
    nowhere to land, BNB rides load_in_4bit/8bit, None tuner_config means full_finetuning.
    """
    if model_config.task_type in ("seq_cls", "reranker", "generative_reranker"):
        raise NotImplementedError(
            f'backend="unsloth" supports causal_lm only; task_type='
            f"{model_config.task_type!r} needs a head unsloth does not build."
        )
    kwargs["full_finetuning"] = tuner_config is None
    if quantize_config is not None and quantize_config.quant_method is not None:
        if quantize_config.quant_method != "bnb":
            raise NotImplementedError(
                f'backend="unsloth" only supports BNB load-time quantization, got '
                f"{quantize_config.quant_method!r}. Use the default backend for this method."
            )
        if quantize_config.quant_bits == 4:
            kwargs.update(load_in_4bit=True, load_in_8bit=False)
        elif quantize_config.quant_bits == 8:
            kwargs.update(load_in_4bit=False, load_in_8bit=True)
        else:
            raise ValueError(f"Unsloth BNB supports quant_bits 4 or 8, got {quantize_config.quant_bits!r}.")
    # unsloth compiles kernels for a fixed length; leave its own 2048 default when unset.
    if model_config.max_model_len:
        kwargs["max_seq_length"] = model_config.max_model_len
    # unsloth installs its offloaded checkpointing inside get_peft_model -- use_gradient_checkpointing
    # is the only lever; it never sees twinkle's gradient_checkpointing kwarg.
    if train_config is not None and not train_config.gradient_checkpointing:
        kwargs["use_gradient_checkpointing"] = False
    # The plain transformers post-load surgery has no seam in unsloth's rebuilt graph: reject rather
    # than let these constructor kwargs silently no-op.
    if model_config.new_special_tokens:
        raise NotImplementedError(
            'new_special_tokens is not supported with backend="unsloth": unsloth owns construction and '
            "may load a quantized base whose embeddings cannot be resized after the fact. Add the tokens to "
            "the checkpoint upstream, or drop --backend unsloth."
        )
    if model_config.init_strategy is not None:
        raise NotImplementedError(
            'init_strategy is not supported with backend="unsloth": unsloth always loads a real '
            "checkpoint (no from-config blank init), so there are no uninitialized weights to re-init, and "
            "re-initializing a quantized base would corrupt it. Drop init_strategy for this backend."
        )
    if train_config is not None and train_config.use_liger_kernel:
        raise NotImplementedError(
            'use_liger_kernel is not composed with backend="unsloth": unsloth already patches the module '
            "graph with its own Triton kernels, so Liger kernelize would double-patch the same ops. Disable "
            "use_liger_kernel for this backend."
        )


def _apply_ray_placement(
    kwargs: dict,
    distributed_config: DistributedConfig,
    remote_group: str = "model",
    device_mesh: Optional["DeviceMesh"] = None,
    instance_id: Optional[str] = None,
) -> None:
    """Place the transformers model in a remote DeviceGroup under mode='ray'.

    A caller-supplied mesh (a frozen reward model's parallel_spec) is installed as-is; only when
    there is none is the default pure-DP mesh synthesized. Local mode leaves both unset.
    """
    if distributed_config.mode == "local":
        return
    if device_mesh is None:
        device_mesh = build_dp_mesh(distributed_config)
    kwargs["device_mesh"] = device_mesh
    kwargs["remote_group"] = remote_group
    if instance_id is not None:
        kwargs["instance_id"] = instance_id


def _resolve_model_loader(model_config: ModelConfig):
    """Resolve the explicitly requested or inferred dev model family loader.

    ``model_type`` may be a family key, a ModelLoader subclass, or a plugin source; a resolved
    loader is authoritative and an unknown explicit value fails instead of re-matching the name.
    """
    from swift.dev.model.loader import ModelInfo, resolve_loader_cls

    loader_cls = resolve_loader_cls(model_config.model_type, model_config.model)
    if loader_cls is None:
        return None
    model_info = ModelInfo(
        model_type=loader_cls.model_type or model_config.model_type,
        model_dir=model_config.model,
        max_model_len=model_config.max_model_len,
        rope_scaling=model_config.rope_scaling if isinstance(model_config.rope_scaling, dict) else None,
        task_type=model_config.task_type,
        num_labels=model_config.num_labels,
        problem_type=model_config.problem_type,
    )
    return loader_cls(model_info)


def _build_hf_config(model_config: ModelConfig, model_loader=None):
    """Build the one HF config shared by Transformers and Megatron model construction.

    rope_scaling mirrors legacy ``_init_rope_scaling``: names map to {'type': name}; a factor-less
    rope derives it from max_model_len / origin, and the budget (origin * factor) is served length.
    """
    if model_loader is not None:
        config = model_loader.build_config(model_config.model, revision=model_config.model_revision)
        config = model_loader.process_config(config)
    else:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_config.model, revision=model_config.model_revision, trust_remote_code=True
        )

    from swift.dev.utils import HfConfigFactory

    rope_scaling = model_config.rope_scaling
    if isinstance(rope_scaling, str):
        if rope_scaling not in ("linear", "dynamic", "yarn"):
            raise ValueError("ModelConfig.rope_scaling must be a JSON object or one of: linear, dynamic, yarn.")
        rope_scaling = {"type": rope_scaling}
    elif rope_scaling is not None:
        if not isinstance(rope_scaling, dict):
            raise TypeError("ModelConfig.rope_scaling must resolve to a dict or a supported strategy name.")
        rope_scaling = copy.deepcopy(rope_scaling)
    current_rope = HfConfigFactory.get_config_attr(config, "rope_scaling")
    # Adopt the checkpoint's rope only when a max_model_len is given to re-derive its factor against
    # (legacy adopts + pops unconditionally but then has nothing to rebuild the factor from).
    if rope_scaling is None and current_rope and model_config.max_model_len is not None:
        rope_scaling = copy.deepcopy(current_rope)
        rope_scaling.pop("factor", None)

    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if "factor" not in rope_scaling and model_config.max_model_len is None and rope_type == "default":
            # Legacy's "fix megatron qwen2_5_vl" short-circuit: write such a rope through verbatim.
            HfConfigFactory.set_config_attr(config, "rope_scaling", rope_scaling)
        else:
            origin_max_len = rope_scaling.get("original_max_position_embeddings")
            if origin_max_len is None and current_rope:
                origin_max_len = current_rope.get("original_max_position_embeddings")
                if origin_max_len is None and current_rope.get("factor"):
                    current_max_len = HfConfigFactory.get_max_model_len(config)
                    if current_max_len is not None:
                        origin_max_len = int(current_max_len / current_rope["factor"])
            if origin_max_len is None:
                origin_max_len = HfConfigFactory.get_max_model_len(config)
            # The early-return above handled the factor-less default case, so reaching here with no
            # factor means a max_model_len to derive it from is mandatory (legacy asserts the same).
            if "factor" not in rope_scaling:
                if model_config.max_model_len is None:
                    raise ValueError("ModelConfig.max_model_len is required when rope_scaling has no factor.")
                if origin_max_len is None:
                    raise ValueError("The checkpoint does not declare a maximum length needed to derive rope_scaling.")
                rope_scaling["factor"] = max(float(math.ceil(model_config.max_model_len / origin_max_len)), 1.0)
            if origin_max_len is not None:
                rope_scaling.setdefault("original_max_position_embeddings", origin_max_len)
            HfConfigFactory.set_config_attr(config, "rope_scaling", rope_scaling)

            # The rope budget is origin * factor: adopt it as the served context when no max_model_len
            # was requested; warn when the request exceeds the scaling's trained range.
            factor = rope_scaling.get("factor")
            if factor is not None and origin_max_len is not None:
                rope_model_len = int(origin_max_len * factor)
                if model_config.max_model_len is None:
                    HfConfigFactory.set_max_model_len(config, rope_model_len)
                elif model_config.max_model_len > rope_model_len:
                    logger.warning(
                        f"rope_scaling budget ({rope_model_len} = {factor} * {origin_max_len}) is smaller than "
                        f"max_model_len ({model_config.max_model_len}); positions beyond the budget are not "
                        "covered by the scaling."
                    )

    if model_config.max_model_len is not None:
        HfConfigFactory.set_max_model_len(config, model_config.max_model_len)
    return config


def _set_load_kwarg(kwargs: dict, name: str, value) -> None:
    """Set a named load option, rejecting disagreement with ``model_kwargs``."""
    if value is None:
        return
    if name in kwargs and kwargs[name] != value:
        raise ValueError(f"ModelConfig.model_kwargs[{name!r}]={kwargs[name]!r} conflicts with {name}={value!r}.")
    kwargs[name] = value


def resolve_new_special_tokens(model_config: ModelConfig) -> List[str]:
    """Expand ``new_special_tokens`` into a flat token list on the driver.

    A ``.txt`` entry is a file of whitespace-separated tokens; any other entry is a literal --
    resolved driver-side so only serializable strings cross the Ray boundary.
    """
    special_tokens: List[str] = []
    for token in model_config.new_special_tokens or []:
        if token.endswith(".txt"):
            if not os.path.isfile(token):
                raise FileNotFoundError(f"new_special_tokens file does not exist: {token}")
            with open(token, "r", encoding="utf-8") as file:
                special_tokens.extend(file.read().split())
        else:
            special_tokens.append(token)
    return special_tokens


def apply_new_special_tokens(model, processor, config, new_special_tokens: List[str]) -> None:
    """Grow the tokenizer's special set and resize a raw-loaded model's embeddings to match.

    The training side owns the canonical application (twinkle's TransformersModel / MegatronModel
    take ``new_special_tokens`` as a constructor argument); this is the twin for models loaded
    outside those constructors (weight transforms), called explicitly by whoever needs it --
    ``load_model_processor`` deliberately does not apply it on its own.
    """
    from swift.dev.utils import HfConfigFactory

    if not new_special_tokens:
        return
    tokenizer = processor if not hasattr(processor, "tokenizer") else processor.tokenizer
    num_new = tokenizer.add_special_tokens({"additional_special_tokens": list(new_special_tokens)})
    if num_new <= 0:
        return
    logger.info("Added %s new special tokens.", num_new)
    origin_vocab_size = HfConfigFactory.get_config_attr(config, "vocab_size")
    if origin_vocab_size is None or origin_vocab_size >= len(tokenizer):
        return
    # 128-alignment matches the checkpoint's embedding stride; the head must cover every token.
    vocab_size = math.ceil(len(tokenizer) / 128) * 128
    HfConfigFactory.set_config_attr(config, "vocab_size", vocab_size)
    model.resize_token_embeddings(vocab_size)


def _build_transformers_model(
    model_config: ModelConfig,
    distributed_config: DistributedConfig,
    train_config: Optional[TrainConfig] = None,
    tuner_config: Optional[TunerConfig] = None,
    device_mesh: Optional["DeviceMesh"] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    remote_group: Optional[str] = None,
    instance_id: Optional[str] = None,
    enable_router_replay: bool = False,
) -> TrainableModel:
    import torch

    from swift.dev.model import TransformersModel

    if not model_config.model:
        raise ValueError("ModelConfig.model (path/id) is required")

    kwargs: dict = dict(model_config.model_kwargs or {})
    kwargs["model_id"] = model_config.model
    # torch_dtype is a Literal, so getattr always resolves; forward to from_pretrained.
    if model_config.torch_dtype:
        _set_load_kwarg(kwargs, "dtype", getattr(torch, model_config.torch_dtype))
    _set_load_kwarg(kwargs, "attn_implementation", model_config.attn_impl)
    _set_load_kwarg(kwargs, "experts_implementation", model_config.experts_impl)
    _set_load_kwarg(kwargs, "revision", model_config.model_revision)
    _set_load_kwarg(kwargs, "max_memory", model_config.max_memory)

    # Resolve a dev family loader and build the config once so caller overrides reach custom loaders.
    model_loader = _resolve_model_loader(model_config)
    hf_config = _build_hf_config(model_config, model_loader)
    kwargs["config"] = hf_config
    # MoE aux loss: twinkle adds any `aux_loss` it finds to the training loss; HF only emits it with
    # output_router_logits on. Opt-in (0.0 default leaves the checkpoint's own config untouched).
    if train_config is not None and train_config.router_aux_loss_coef:
        hf_config.router_aux_loss_coef = train_config.router_aux_loss_coef
        hf_config.output_router_logits = True

    # seq_cls / reranker ride a num_labels-wide SequenceClassification head (reranker = num_labels=1;
    # generative_reranker keeps CausalLM and is NOT here; PPO's value critic is seq_cls+num_labels=1).
    if model_config.task_type in ("seq_cls", "reranker"):
        _apply_seq_cls_head(kwargs, model_config, hf_config, model_loader)

    # strategy: deepspeed/fsdp select the twinkle strategy (default: accelerate).
    from swift.dev.utils.naming import resolve_strategy

    strategy = "accelerate"
    if distributed_config.deepspeed:
        strategy = "deepspeed"
        kwargs["deepspeed_config"] = _resolve_deepspeed_config(distributed_config)
    elif distributed_config.fsdp:
        strategy = "native_fsdp"
    kwargs["strategy"] = resolve_strategy(strategy)
    # The transformers strategies read the process-group timeout from this env var (not a ctor arg),
    # so --ddp_timeout reaches them through it; None leaves twinkle's own default.
    if distributed_config.ddp_timeout is not None:
        os.environ["TWINKLE_DIST_TIMEOUT_SECONDS"] = str(distributed_config.ddp_timeout)
    kwargs["mixed_precision"] = _mixed_precision_for(model_config.torch_dtype)

    # DDP find_unused_parameters, mirroring HF Trainer: an adapter run needs it (unused-param grads);
    # a full run with gradient checkpointing turns it off.
    find_unused = distributed_config.ddp_find_unused_parameters
    if find_unused is None:
        is_peft = tuner_config is not None
        if is_peft:
            find_unused = True
        else:
            gc_on = train_config.gradient_checkpointing if train_config is not None else False
            find_unused = not gc_on
    kwargs["ddp_config"] = {"find_unused_parameters": bool(find_unused)}
    if distributed_config.ddp_broadcast_buffers is not None:
        kwargs["ddp_config"]["broadcast_buffers"] = distributed_config.ddp_broadcast_buffers
    if distributed_config.ddp_bucket_cap_mb is not None:
        kwargs["ddp_config"]["bucket_cap_mb"] = distributed_config.ddp_bucket_cap_mb
    if distributed_config.ddp_static_graph is not None:
        kwargs["ddp_config"]["static_graph"] = distributed_config.ddp_static_graph

    # No mesh by default: twinkle's local mode assigns its pure-DP mesh (load-bearing for token-weighted
    # loss); the opt-in exception is Ulysses SP (local mode only; validate rejects ray).
    if device_mesh is not None and distributed_config.mode == "local":
        kwargs["device_mesh"] = device_mesh
    # Ray placement (RL): the model lives in a remote DeviceGroup (separate from a vLLMSampler, so
    # CheckpointEngineManager can weight-sync); a caller mesh is honored, a missing one defaults.
    _apply_ray_placement(kwargs, distributed_config, remote_group or "model", device_mesh, instance_id=instance_id)

    # MoE routing replay: a construction-time flag twinkle's TransformersModel.__init__ pops; only
    # the trainable policy is built with it True.
    if enable_router_replay:
        kwargs["enable_router_replay"] = True

    # Three exclusive paths: sentence_transformers (its own pipeline) / unsloth (owns construction
    # + LoRA install) / plain TransformersModel with the family loader handed over.
    model_framework = _resolve_model_framework(model_config, model_loader)
    if model_framework == "sentence_transformers":
        model = _build_sentence_transformer_model(
            kwargs, model_config, distributed_config, tuner_config, quantize_config, train_config
        )
    elif model_framework is not None:
        # A declared framework with no builder here must fail loudly, not fall through as plain HF.
        raise ValueError(f"No builder is wired for model_framework={model_framework!r}; add a branch here.")
    elif distributed_config.backend == "unsloth":
        from swift.dev.model import UnslothModel

        _apply_unsloth_kwargs(kwargs, model_config, tuner_config, train_config, quantize_config)
        model = UnslothModel(**kwargs)
    else:
        from swift.dev.builders.quantization import build_load_quantization_config

        quantization_config = build_load_quantization_config(
            quantize_config, model_loader=model_loader, torch_dtype=model_config.torch_dtype
        )
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        # Full takeover: hand the resolved family loader to twinkle (config/processor/model through
        # it). seq_cls/reranker are excluded -- their num_labels head overrides model_cls.
        if model_loader is not None and model_config.task_type not in ("seq_cls", "reranker"):
            kwargs["model_loader"] = model_loader
        # gradient_checkpointing is a ctor kwarg twinkle gates worker-side, so --gradient_checkpointing
        # false is honored under Ray too (unsloth/ST manage their own).
        if train_config is not None:
            kwargs["gradient_checkpointing"] = train_config.gradient_checkpointing
            if train_config.gradient_checkpointing_kwargs:
                kwargs["gradient_checkpointing_kwargs"] = train_config.gradient_checkpointing_kwargs
        if model_loader is not None:
            arch = model_loader.model_arch
            model_arch = {
                part: list(getattr(arch, part, None) or []) for part in ("language_model", "vision_tower", "aligner")
            }
            lm_head = getattr(arch, "lm_head", None)
            if lm_head:
                model_arch["lm_head"] = lm_head
            kwargs["model_arch"] = model_arch
        if (
            train_config is not None
            and train_config.vit_gradient_checkpointing is not None
            and kwargs.get("model_arch", {}).get("vision_tower")
        ):
            kwargs["vit_gradient_checkpointing"] = train_config.vit_gradient_checkpointing
            if train_config.vit_gradient_checkpointing_kwargs:
                kwargs["vit_gradient_checkpointing_kwargs"] = train_config.vit_gradient_checkpointing_kwargs
        # NEFTune + acc_strategy are transformers-only ctor kwargs (megatron rejects both in _HF_ONLY).
        if train_config is not None and train_config.neftune_noise_alpha:
            kwargs["neftune_noise_alpha"] = train_config.neftune_noise_alpha
        if train_config is not None:
            kwargs["acc_strategy"] = train_config.acc_strategy
        # Post-load module surgery rides the constructor too (applied worker-side; a driver-side
        # reach into model.model only touches the Ray proxy stub and silently no-ops).
        if model_config.new_special_tokens:
            kwargs["new_special_tokens"] = resolve_new_special_tokens(model_config)
        if model_config.init_strategy is not None:
            kwargs["init_strategy"] = model_config.init_strategy
        if train_config is not None and train_config.use_liger_kernel:
            kwargs["use_liger_kernel"] = True
            kwargs["liger_kernel_config"] = train_config.liger_kernel_config
        model = TransformersModel(**kwargs)

        if model_config.enable_kernel:
            from twinkle.kernel import kernelize
            kernelize(model.model)

    # twinkle's TransformersModel.__init__ calls gradient_checkpointing_enable() unconditionally
    # (model/transformers/transformers.py), so the user's --gradient_checkpointing false was silently
    # ignored -- and the find_unused_parameters derivation above already assumes the flag is honored,
    # so the two disagreed. Turn it back off here when the config says so.
    # NOTE: this is a post-hoc undo, not the right shape. The fix belongs upstream as a twinkle
    # constructor argument (gradient_checkpointing=...); switch to passing it once that lands, so the
    # model is never built in a state the caller did not ask for.
    if train_config is not None and not train_config.gradient_checkpointing:
        model.model.gradient_checkpointing_disable()

    return model


def _resolve_model_framework(model_config: ModelConfig, model_loader) -> Optional[str]:
    """The external framework that must construct this embedding model, or None for plain HF.

    A resolved family loader is authoritative (qwen2_gte/gemma_emb declare sentence_transformers;
    qwen3_emb deliberately does not); with no loader, an on-disk ST layout routes local checkpoints.
    """
    if model_config.task_type != "embedding":
        return None
    if model_loader is not None:
        return getattr(model_loader, "model_framework", None)
    return "sentence_transformers" if _has_sentence_transformer_layout(model_config.model) else None


def _has_sentence_transformer_layout(model_dir) -> bool:
    """True when a LOCAL checkpoint directory carries the sentence-transformers pipeline files.

    A hub id (not yet downloaded) is not a directory, so False there -- the loader marker is the
    signal for registered families; this only catches an unregistered local ST checkpoint.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return False
    return any(
        os.path.exists(os.path.join(model_dir, name))
        for name in ("modules.json", "config_sentence_transformers.json", "1_Pooling")
    )


def _build_sentence_transformer_model(
    kwargs: dict,
    model_config: ModelConfig,
    distributed_config: DistributedConfig,
    tuner_config: Optional[TunerConfig],
    quantize_config: Optional[QuantizeConfig],
    train_config: Optional[TrainConfig],
) -> TrainableModel:
    """Construct a SentenceTransformerModel from the shared transformers-build kwargs.

    Only the plumbing ST shares with the base is forwarded (ST builds its own backbone via
    SentenceTransformer(model_id)); features the pipeline cannot honor are rejected HERE.
    """
    if distributed_config.deepspeed:
        raise NotImplementedError(
            "A sentence-transformers embedding model cannot train under DeepSpeed: SentenceTransformerModel "
            "builds its own module pipeline and wires no deepspeed_config into its strategy, so ZeRO would be "
            "silently ignored. Use --fsdp or plain DDP for this model."
        )
    if distributed_config.backend == "unsloth":
        raise NotImplementedError(
            'backend="unsloth" cannot wrap a sentence-transformers embedding model: unsloth rebuilds a '
            "causal-LM module graph, while this checkpoint trains as an ST pooling pipeline. Use "
            "the default backend (LoRA or full fine-tuning)."
        )
    if quantize_config is not None and quantize_config.quant_method is not None:
        raise NotImplementedError(
            f"Load-time quantization ({quantize_config.quant_method!r}) is not applied to a sentence-transformers "
            "embedding model: ST builds its backbone via SentenceTransformer(model_id), which takes no "
            "transformers quantization_config. Train this model unquantized."
        )
    if train_config is not None and train_config.use_liger_kernel:
        raise NotImplementedError(
            "use_liger_kernel is not composed with a sentence-transformers embedding model: Liger replaces "
            "causal-LM decoder ops / the fused-linear-CE loss path, while this checkpoint trains an ST pooling "
            "pipeline with an embedding loss. Disable use_liger_kernel for this model."
        )
    if model_config.new_special_tokens:
        raise NotImplementedError(
            "new_special_tokens is not composed with a sentence-transformers embedding model: ST builds its "
            "backbone via SentenceTransformer(model_id), which exposes no resize_token_embeddings seam, so a "
            "grown vocabulary could not be matched by resized embeddings. Add the tokens to the checkpoint "
            "upstream, or train this model on the plain transformers path."
        )
    from swift.dev.model import SentenceTransformerModel

    st_kwargs = {
        "model_id": kwargs["model_id"],
        "strategy": kwargs["strategy"],
        "mixed_precision": kwargs["mixed_precision"],
        "ddp_config": kwargs.get("ddp_config"),
    }
    # ST never sees the shared kwargs, so thread the gradient-checkpointing flag straight through.
    if train_config is not None:
        st_kwargs["gradient_checkpointing"] = train_config.gradient_checkpointing
        if train_config.gradient_checkpointing_kwargs:
            st_kwargs["gradient_checkpointing_kwargs"] = train_config.gradient_checkpointing_kwargs
    # init_strategy re-initializes uninitialized weights worker-side via the inherited
    # TransformersModel._apply_init_strategy (ST's pipeline is an nn.Module, so parameters() reaches it).
    if model_config.init_strategy is not None:
        st_kwargs["init_strategy"] = model_config.init_strategy
    # SP mesh (local) / DeviceGroup placement (ray) ride the shared kwargs; forward them so ST lands
    # on the right ranks (ST's own forward guard rejects an SP mesh).
    for key in ("device_mesh", "remote_group"):
        if key in kwargs:
            st_kwargs[key] = kwargs[key]
    return SentenceTransformerModel(**st_kwargs)


def _resolve_deepspeed_config(distributed_config: DistributedConfig) -> Optional[dict]:
    """Resolve ``DistributedConfig.deepspeed`` (preset / JSON file / JSON string) into a dict.

    The preset mapping lives in swift.dev.utils (shared with config validation); only the ZeRO++
    zero_hpz_partition_size injection -- a strategy-construction concern -- stays here.
    """
    from swift.dev.utils import resolve_deepspeed_config

    config = resolve_deepspeed_config(distributed_config.deepspeed)
    if config is None:
        return None
    if distributed_config.zero_hpz_partition_size is not None:
        # ZeRO++ hpz shards within a node and only means something inside zero_optimization.
        if "zero_optimization" not in config:
            raise ValueError(
                "zero_hpz_partition_size (ZeRO++) needs a deepspeed config with a zero_optimization "
                "section; the resolved config has none."
            )
        config["zero_optimization"]["zero_hpz_partition_size"] = distributed_config.zero_hpz_partition_size
    return config


def _apply_mtp_kwargs(kwargs: dict, model_config: ModelConfig, strict: set) -> None:
    """Forward the Multi-Token Prediction knobs into mcore-bridge's ModelConfig.

    ``mtp_num_layers`` gates the rest; each knob is forwarded only when set so an unset
    mtp_loss_scaling_factor keeps mcore's own default (None would override it).
    """
    if model_config.mtp_num_layers is None:
        return
    values = {
        "mtp_num_layers": model_config.mtp_num_layers,
        "mtp_loss_scaling_factor": model_config.mtp_loss_scaling_factor,
        "enable_mtp_training": model_config.enable_mtp_training,
        "mtp_freeze": model_config.mtp_freeze,
        "mtp_decoder_input_detach": model_config.mtp_decoder_input_detach,
        "mtp_shared_weights": model_config.mtp_shared_weights,
    }
    for name, value in values.items():
        if value is not None:
            kwargs[name] = value
        if _field_was_requested(model_config, name):
            strict.add(name)


def _apply_fp4_kwargs(kwargs: dict, model_config: ModelConfig, strict: set) -> None:
    """Forward the FP4 knobs into mcore-bridge's ModelConfig, under megatron's names for them.

    A rename (fp4_format->fp4, fp4_param_gather->fp4_param: twinkle derives the DDP flag itself).
    Gated on fp4_format (fp4_recipe has a non-None default on both sides).
    """
    if model_config.fp4_format is None:
        return
    kwargs["fp4"] = model_config.fp4_format
    kwargs["fp4_recipe"] = model_config.fp4_recipe
    if model_config.fp4_param_gather:
        kwargs["fp4_param"] = True
    for source, target in (
        ("fp4_format", "fp4"),
        ("fp4_recipe", "fp4_recipe"),
        ("fp4_param_gather", "fp4_param"),
    ):
        if _field_was_requested(model_config, source):
            strict.add(target)


def _apply_fp8_kwargs(kwargs: dict, model_config: ModelConfig, strict: set) -> None:
    """Forward the FP8 knobs into mcore-bridge's ModelConfig, under megatron's names for them.

    A sibling of _apply_fp4_kwargs on purpose: FP8's amax knobs' dev defaults differ from
    megatron's (1024/'max' vs 1/'most_recent', following legacy), so they are forwarded EXPLICITLY.
    """
    if model_config.fp8_format is None:
        return
    kwargs["fp8"] = model_config.fp8_format
    kwargs["fp8_recipe"] = model_config.fp8_recipe
    kwargs["fp8_amax_history_len"] = model_config.fp8_amax_history_len
    kwargs["fp8_amax_compute_algo"] = model_config.fp8_amax_compute_algo
    if model_config.fp8_param_gather:
        kwargs["fp8_param"] = True
    for source, target in (
        ("fp8_format", "fp8"),
        ("fp8_recipe", "fp8_recipe"),
        ("fp8_amax_history_len", "fp8_amax_history_len"),
        ("fp8_amax_compute_algo", "fp8_amax_compute_algo"),
        ("fp8_param_gather", "fp8_param"),
    ):
        if _field_was_requested(model_config, source):
            strict.add(target)


def build_parallel_mesh(distributed_config: Optional[DistributedConfig], *, none_if_single_rank: bool = False):
    """DistributedConfig -> DeviceMesh from the declared parallel sizes (tp/pp/cp/ep/etp/vpp/sp).

    Backend-agnostic (transformers degenerates to a pure DP mesh) and a pure function of the config
    (no torch.distributed/mpu reads). ``none_if_single_rank=True`` -> None for dw<=1 / no world.
    """
    if none_if_single_rank:
        if distributed_config is None:
            return None
        if distributed_config.nproc_per_node is None and not distributed_config.parallel_spec:
            return None
    from twinkle import DeviceMesh

    spec = distributed_config.parallel_spec
    if spec:
        mesh = DeviceMesh.from_spec(spec, world_size=distributed_config.nproc_per_node)
        if none_if_single_rank and getattr(mesh, "data_world_size", 1) <= 1:
            return None
        return mesh

    tp = distributed_config.tensor_model_parallel_size
    pp = distributed_config.pipeline_model_parallel_size
    cp = distributed_config.context_parallel_size
    ep = distributed_config.expert_model_parallel_size

    world_size = distributed_config.nproc_per_node
    if world_size is None:
        raise ValueError(
            "DistributedConfig.nproc_per_node is required for the Megatron backend (it sets the "
            "DeviceMesh world size). Pass it explicitly -- there is no default, since a wrong "
            "world size silently builds the wrong data-parallel layout."
        )
    model_parallel = tp * pp * cp
    if world_size % model_parallel != 0:
        raise ValueError(
            f"nproc_per_node={world_size} is not divisible by tp*pp*cp={model_parallel} (tp={tp}, pp={pp}, cp={cp})."
        )
    dp_size = world_size // model_parallel

    mesh_kwargs = {"world_size": world_size, "dp_size": dp_size}
    if tp > 1:
        mesh_kwargs["tp_size"] = tp
    if pp > 1:
        mesh_kwargs["pp_size"] = pp
    if cp > 1:
        mesh_kwargs["cp_size"] = cp
    if ep > 1:
        mesh_kwargs["ep_size"] = ep
    if distributed_config.expert_tensor_parallel_size > 1:
        mesh_kwargs["etp_size"] = distributed_config.expert_tensor_parallel_size
    if distributed_config.virtual_pipeline_model_parallel_size is not None:
        mesh_kwargs["vpp_size"] = distributed_config.virtual_pipeline_model_parallel_size
    # Megatron TP sequence-parallelism rides on the DeviceMesh (twinkle reads it via
    # strategy.sequence_parallel -> device_mesh.sequence_parallel); only meaningful with tp > 1.
    if distributed_config.sequence_parallel:
        mesh_kwargs["sequence_parallel"] = True
    mesh = DeviceMesh.from_sizes(**mesh_kwargs)
    if none_if_single_rank and getattr(mesh, "data_world_size", 1) <= 1:
        return None
    return mesh


_NON_MODEL_MEGATRON_FIELDS = {
    "data_parallel_random_init",
    "skip_megatron_init",
    "manual_gc",
    "manual_gc_eval",
    "manual_gc_steps",
    "megatron_extra_kwargs",
}
_MEGATRON_FIELD_ALIASES = {"te_rng_tracker": "use_te_rng_tracker"}


def _field_was_requested(config, field_name: str) -> bool:
    """Whether a non-default Config value was requested explicitly or programmatically."""
    explicit = getattr(config, "_explicit_fields", None)
    if explicit is not None:
        return field_name in explicit
    config_field = next(field for field in dataclasses.fields(config) if field.name == field_name)
    if config_field.default is not dataclasses.MISSING:
        default = config_field.default
    elif config_field.default_factory is not dataclasses.MISSING:
        default = config_field.default_factory()
    else:
        return True
    return getattr(config, field_name) != default


def _megatron_model_kwargs(megatron_config: Optional[MegatronConfig], moe_config: Optional[MoEConfig]) -> dict:
    """Translate backend-only Configs into model kwargs and retain explicit-field provenance.

    Bridge providers vary by version: defaults an older provider lacks may fall back to its own,
    but an explicitly requested option must never disappear (_strict_model_kwargs carries that).
    """
    kwargs = {}
    strict = set()
    if megatron_config is not None:
        for config_field in dataclasses.fields(megatron_config):
            name = config_field.name
            if name in _NON_MODEL_MEGATRON_FIELDS:
                continue
            value = getattr(megatron_config, name)
            if value is not None:
                target = _MEGATRON_FIELD_ALIASES.get(name, name)
                kwargs[target] = value
                if _field_was_requested(megatron_config, name):
                    strict.add(target)

        extra = megatron_config.megatron_extra_kwargs or {}
        if not isinstance(extra, dict):
            raise TypeError("MegatronConfig.megatron_extra_kwargs must be a dict after process_configs().")
        for name, value in extra.items():
            if name in kwargs and kwargs[name] != value:
                raise ValueError(f"megatron_extra_kwargs[{name!r}]={value!r} conflicts with {name}={kwargs[name]!r}.")
            kwargs[name] = value
            strict.add(name)

    if moe_config is not None:
        for config_field in dataclasses.fields(moe_config):
            value = getattr(moe_config, config_field.name)
            if value is not None:
                kwargs[config_field.name] = value
                if _field_was_requested(moe_config, config_field.name):
                    strict.add(config_field.name)
    if strict:
        kwargs["_strict_model_kwargs"] = tuple(sorted(strict))
    return kwargs


def _build_megatron_model(
    model_config: ModelConfig,
    distributed_config: DistributedConfig,
    train_config: Optional[TrainConfig] = None,
    *,
    megatron_config: Optional[MegatronConfig] = None,
    moe_config: Optional[MoEConfig] = None,
    remote_group: Optional[str] = None,
    instance_id: Optional[str] = None,
    enable_router_replay: bool = False,
) -> TrainableModel:
    """Build a MegatronModel via the selected bridge backend.

    twinkle must already be initialized in Ray mode so the 'model' DeviceGroup exists. The world
    size is nproc_per_node (== the DeviceGroup size); the driver is not part of the process group.
    """
    from twinkle.model.megatron import MegatronModel

    if not model_config.model:
        raise ValueError("ModelConfig.model (path/id) is required")

    device_mesh = build_parallel_mesh(distributed_config)
    model_loader = _resolve_model_loader(model_config)
    hf_config = _build_hf_config(model_config, model_loader)

    mixed_precision = _mixed_precision_for(model_config.torch_dtype)

    # High-frequency knobs flow straight into MegatronModel.__init__; forwarded only when set so the
    # bit-exact SFT baseline is unchanged unless opted in (use_distributed_optimizer has a real default).
    extra_kwargs: dict = {
        "use_distributed_optimizer": distributed_config.use_distributed_optimizer,
        "align_grad_reduce": distributed_config.align_grad_reduce,
        "nccl_comm_warmup": distributed_config.nccl_comm_warmup,
        **_megatron_model_kwargs(megatron_config, moe_config),
    }
    strict_model_kwargs = set(extra_kwargs.pop("_strict_model_kwargs", ()))
    # ModelParallelConfig/TransformerConfig fields owned by DistributedConfig; the spelling
    # translation stays next to the DeviceMesh translation so a parsed field reaches the worker.
    model_field_aliases = {
        "decoder_first_pipeline_num_layers": "num_layers_in_first_pipeline_stage",
        "decoder_last_pipeline_num_layers": "num_layers_in_last_pipeline_stage",
    }
    model_fields = (
        "recompute_granularity",
        "recompute_method",
        "recompute_num_layers",
        "recompute_modules",
        "cp_comm_type",
        "pipeline_model_parallel_layout",
        "decoder_first_pipeline_num_layers",
        "decoder_last_pipeline_num_layers",
        "account_for_embedding_in_pipeline_split",
        "account_for_loss_in_pipeline_split",
        "overlap_p2p_comm",
        "batch_p2p_comm",
        "tp_comm_overlap",
    )
    for name in model_fields:
        value = getattr(distributed_config, name)
        if value is not None:
            target = model_field_aliases.get(name, name)
            extra_kwargs[target] = value
            if _field_was_requested(distributed_config, name):
                strict_model_kwargs.add(target)

    if train_config is not None:
        if train_config.calculate_per_token_loss is not None:
            extra_kwargs["calculate_per_token_loss"] = train_config.calculate_per_token_loss
            strict_model_kwargs.add("calculate_per_token_loss")
        if train_config.microbatch_group_size_per_vp_stage is not None:
            extra_kwargs["microbatch_group_size_per_vp_stage"] = train_config.microbatch_group_size_per_vp_stage
            strict_model_kwargs.add("microbatch_group_size_per_vp_stage")
        # Per-tower vision checkpointing (the megatron counterpart of the transformers kwarg);
        # consumed worker-side against the live vision tower, so no strict check and no model_arch.
        if train_config.vit_gradient_checkpointing is not None:
            extra_kwargs["vit_gradient_checkpointing"] = train_config.vit_gradient_checkpointing
            if train_config.vit_gradient_checkpointing_kwargs:
                extra_kwargs["vit_gradient_checkpointing_kwargs"] = train_config.vit_gradient_checkpointing_kwargs

    if extra_kwargs.get("recompute_granularity") == "selective":
        extra_kwargs["recompute_num_layers"] = None
        extra_kwargs["recompute_method"] = None

    ddp_config = {
        "grad_reduce_in_fp32": bool(train_config and train_config.accumulate_allreduce_grads_in_fp32),
        "overlap_grad_reduce": distributed_config.overlap_grad_reduce,
        "overlap_param_gather": distributed_config.overlap_param_gather,
        "align_param_gather": distributed_config.align_param_gather,
        "data_parallel_sharding_strategy": distributed_config.data_parallel_sharding_strategy,
    }
    if distributed_config.use_megatron_fsdp:
        ddp_config["use_megatron_fsdp"] = True
    extra_kwargs["ddp_config"] = ddp_config

    # The attention kernel is ALWAYS forwarded: the meaningful default is legacy's 'flash', not
    # mcore's auto (which picks a different kernel for a Qwen2.5 bf16 causal THD forward).
    from swift.dev.utils.naming import resolve_megatron_attn_backend

    extra_kwargs["attention_backend"] = resolve_megatron_attn_backend(model_config.attn_impl)
    if _field_was_requested(model_config, "attn_impl"):
        strict_model_kwargs.add("attention_backend")
    # A flash_N value also pins the FA VERSION by mutating transformer_engine module globals -- a
    # per-process side effect that must run worker-side, so forward the raw string for MegatronStrategy.
    extra_kwargs["attn_impl"] = model_config.attn_impl

    # task_type/num_labels flow into mcore-bridge's ModelConfig, which builds the head; the bridge
    # has no plain 'reranker' task, so map it to seq_cls with num_labels=1 (legacy's same head).
    task_type = model_config.task_type
    if task_type == "reranker":
        extra_kwargs["task_type"] = "seq_cls"
        extra_kwargs["num_labels"] = model_config.num_labels or 1
    elif task_type in ("seq_cls", "embedding", "generative_reranker"):
        extra_kwargs["task_type"] = task_type
        if task_type == "seq_cls":
            if model_config.num_labels is None:
                raise ValueError('ModelConfig.num_labels is required for task_type="seq_cls".')
            extra_kwargs["num_labels"] = model_config.num_labels

    for name in ("vit_attn_impl", "language_model_only"):
        value = getattr(model_config, name)
        if value:
            extra_kwargs[name] = value
        if _field_was_requested(model_config, name):
            strict_model_kwargs.add(name)

    _apply_mtp_kwargs(extra_kwargs, model_config, strict_model_kwargs)
    _apply_fp4_kwargs(extra_kwargs, model_config, strict_model_kwargs)
    _apply_fp8_kwargs(extra_kwargs, model_config, strict_model_kwargs)
    # new_special_tokens: resolved driver-side (serializable strings only), consumed worker-side by
    # MegatronModel (extend tokenizer, raise vocab_size, re-init the new embedding rows).
    if model_config.new_special_tokens:
        extra_kwargs["new_special_tokens"] = resolve_new_special_tokens(model_config)
    # MoE routing replay: MegatronModel.__init__ pops enable_router_replay and translates it to the
    # mcore moe_enable_routing_replay field -- the counterpart of the transformers flag.
    if enable_router_replay:
        extra_kwargs["enable_router_replay"] = True
    if strict_model_kwargs:
        extra_kwargs["_strict_model_kwargs"] = tuple(sorted(strict_model_kwargs))

    # Ray mode places the model in a remote DeviceGroup; local (torchrun) builds in-process per rank.
    model_kwargs = dict(
        model_id=model_config.model,
        config=hf_config,
        revision=model_config.model_revision,
        device_mesh=device_mesh,
        mixed_precision=mixed_precision,
        # A NAME, not a built instance: only a string survives the Ray worker boundary
        # (@remote_class forwards constructor kwargs, not live objects).
        bridge_backend=distributed_config.bridge_backend,
        **extra_kwargs,
    )
    if distributed_config.mode != "local":
        model_kwargs["remote_group"] = remote_group or "model"
        # Same-group actor disambiguator, mirroring the transformers path's _apply_ray_placement.
        if instance_id is not None:
            model_kwargs["instance_id"] = instance_id
    return MegatronModel(**model_kwargs)
