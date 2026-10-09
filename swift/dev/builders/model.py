"""build_model: ModelConfig + DistributedConfig -> twinkle-native TransformersModel / MegatronModel."""
from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any, Optional

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


def load_model_processor(model_config: ModelConfig, *, load_model: bool = False, **overrides):
    """Load a legacy-compatible processor/model with every relevant ``ModelConfig`` option.

    The dev model registry is used by Twinkle model construction, while template/export paths still
    use ``swift.model`` for its mature processor catalogue. Keeping this translation here prevents
    those call sites from each forwarding a different subset of the same model-loading contract.
    """
    import torch

    from swift.model import get_model_processor

    kwargs = dict(model_config.model_kwargs or {})
    values = {
        'model_type': model_config.model_type,
        'revision': model_config.model_revision,
        'experts_impl': model_config.experts_impl,
        'new_special_tokens': model_config.new_special_tokens,
        'rope_scaling': model_config.rope_scaling,
        'max_model_len': model_config.max_model_len,
        'device_map': model_config.device_map,
        'max_memory': model_config.max_memory,
        'local_repo_path': model_config.local_repo_path,
        'task_type': model_config.task_type,
        'num_labels': model_config.num_labels,
        'problem_type': model_config.problem_type,
        'init_strategy': model_config.init_strategy,
    }
    if model_config.torch_dtype:
        values['torch_dtype'] = getattr(torch, model_config.torch_dtype)
    if model_config.attn_impl:
        values['attn_impl'] = model_config.attn_impl
    for name, value in values.items():
        if value is not None and value != []:
            _set_load_kwarg(kwargs, name, value)
    for name, value in overrides.items():
        _set_load_kwarg(kwargs, name, value)
    return get_model_processor(model_config.model, load_model=load_model, **kwargs)


def build_hf_device_mesh(distributed_config: DistributedConfig,
                         sequence_parallel_size: int) -> Optional['DeviceMesh']:
    """HF-backend local-mode DeviceMesh for Ulysses sequence parallelism.

    Returns None unless SP is actually on in local (torchrun) mode -- an sp=1 run must reach
    TransformersModel with no mesh, exactly as before (see the load-bearing note in
    _build_transformers_model on why the default-mesh substitution is deliberate there).

    Unlike build_device_mesh (Megatron: a pure function of the config, nproc_per_node-driven),
    this reads the world size the same place twinkle's local mode does -- Platform.get_world_size()
    (the torchrun WORLD_SIZE env), NOT torch.distributed, which twinkle initializes lazily and which
    is therefore not up yet when run_sft builds the mesh.

    Distinct from the Megatron mesh on purpose: no tp/pp/cp, just dp x ulysses. Note ulysses is
    NOT a mesh dim in twinkle -- the dp dim spans ALL ranks (dp_size=world) and ``data_world_size``
    derives world/ulysses from it (utils/device_mesh.py), so SP peers share one data rank and
    receive identical samples.

    With ``parallel_spec`` set, the string is parsed and handed straight to DeviceMesh over the torchrun
    world instead -- from_sizes raises if the dims do not fill it and TransformersModel rejects any
    dimension it does not implement, so there is no validation layer here.
    """
    spec = distributed_config.parallel_spec
    if spec:
        from twinkle import DeviceMesh
        from twinkle.utils import Platform
        return DeviceMesh.from_spec(spec, world_size=Platform.get_world_size())

    if sequence_parallel_size <= 1 or distributed_config.mode != 'local':
        return None

    from twinkle import DeviceMesh
    from twinkle.utils import Platform

    world = Platform.get_world_size()
    if world < 2:
        raise ValueError(f'sequence_parallel_size={sequence_parallel_size} requires torchrun (WORLD_SIZE>=2); '
                         'a single process has no ranks to split a sequence across.')
    if world % sequence_parallel_size != 0:
        raise ValueError(f'world_size={world} is not divisible by sequence_parallel_size={sequence_parallel_size}.')
    return DeviceMesh.from_sizes(world_size=world, dp_size=world, ulysses_size=sequence_parallel_size)


def build_model(model_config: ModelConfig,
                distributed_config: DistributedConfig,
                train_config: Optional[TrainConfig] = None,
                tuner_config: Optional[TunerConfig] = None,
                device_mesh: Optional['DeviceMesh'] = None,
                quantize_config: Optional[QuantizeConfig] = None,
                megatron_config: Optional[MegatronConfig] = None,
                moe_config: Optional[MoEConfig] = None,
                remote_group: Optional[str] = None,
                instance_id: Optional[str] = None,
                enable_router_replay: bool = False) -> TrainableModel:
    """ModelConfig + DistributedConfig -> twinkle-native Model (no loss/optim yet).

    ``remote_group`` overrides the Ray DeviceGroup the model is placed in (mode='ray' only);
    it defaults to 'model' so the training path is unchanged, and a frozen reward model can
    target its own dedicated group instead of colliding with the trainable one.

    ``instance_id`` disambiguates two models that share ONE ``remote_group`` (mode='ray' only). twinkle
    names a Ray actor ``{group}-{class}-{instance_id}{caller_file}_{caller_line}-{rank}``, and every model
    built here shares the same caller line, so two same-group builds collide on ``ActorAlreadyExistsError``
    unless they carry distinct ``instance_id``\\ s. PPO's trainable critic (shares the policy's 'model'
    group) and MOPD's K frozen teachers (share the 'teacher' group) each pass one; a lone model in its group
    leaves it None and is unaffected.

    Thin mapping (no Registry/Factory): model_config fields -> twinkle __init__ kwargs.
    DistributedConfig.backend=='megatron' builds a MegatronModel (via the selected bridge
    backend); otherwise a TransformersModel. twinkle self-builds weights on the correct rank.

    ``device_mesh`` is the HF-local Ulysses SP mesh from :func:`build_hf_device_mesh` (None when
    SP is off); it is forwarded to the TransformersModel build only and must NOT be set for the
    Megatron backend, which derives its own mesh from DistributedConfig.

    ``enable_router_replay`` turns on MoE routing replay in the built model (RLHFConfig.router_replay_mode
    != 'disabled'). It is a construction-time flag on BOTH backends -- TransformersModel takes it directly,
    MegatronModel translates it to the mcore ``moe_enable_routing_replay`` TransformerConfig field -- so the
    policy can later RECORD (R2) or REPLAY (R3) expert routing during the RL update. Only the trainable
    policy sets it; frozen auxiliaries and the sampler do not.

    PPO's value critic is NOT a special build flag: it is a ``task_type='seq_cls', num_labels=1`` model
    forwarded with ``task='value'`` (which keeps the head's per-token output instead of pooling), so it
    goes through the ordinary seq_cls build path on both backends.
    """
    # Auxiliary models (teacher / reference / reward / value) are built inside recipes from a bare hub id
    # via ``ModelConfig(model=<id>)`` and never pass through ``process._resolve_model`` (which resolves the
    # PRIMARY model's id to a local snapshot dir before its build). Resolve here so any caller may hand in
    # either a hub id or a local dir: ``safe_snapshot_download`` short-circuits an existing local path, so
    # the already-resolved primary model is a cheap no-op and only bare-id aux models actually download.
    # It follows the ``USE_HF`` env and needs no token by default; a gated aux model would need the primary
    # path's explicit ``use_hf``/``hub_token`` threading, which the aux sites do not carry (known limit).
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
            enable_router_replay=enable_router_replay)
    return _build_transformers_model(
        model_config,
        distributed_config,
        train_config,
        tuner_config,
        device_mesh,
        quantize_config=quantize_config,
        remote_group=remote_group,
        instance_id=instance_id,
        enable_router_replay=enable_router_replay)


def apply_full_param_freeze(model: TrainableModel, train_config: Optional[TrainConfig]) -> None:
    """Consume TrainConfig's freeze_parameters / trainable_parameters knobs on a full-parameter model.

    Called ONLY on the full-param path (``tuner_config is None``): an adapter run selects what trains
    through target_modules / modules_to_save, not ``requires_grad`` on the base weights. It must run
    BEFORE ``configure_optimizer`` -- the optimizer builds its param groups from ``requires_grad``, so
    a param frozen after that point would still be stepped. It forwards to twinkle's
    ``model.freeze_parameters`` remote seam (dispatch='all'), which is what reaches the workers under
    Ray; freeze order and the trainable-wins rule live there.

    A no-op when no knob is set, so an ordinary full run pays no remote round-trip. The
    ``freeze_parameters_ratio`` + Megatron PP>1 mutual exclusion is rejected earlier, in
    ``validate._check_freeze_ratio_pp`` (config-only), so this just forwards resolved values.
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
        trainable_regex=trainable_regex)


def _mixed_precision_for(torch_dtype: Optional[str]) -> str:
    """torch_dtype -> twinkle's mixed_precision mode. Shared by both backends so they cannot drift.

    float16 -> 'fp16' and bfloat16 -> 'bf16' mirror legacy, whose TrainingArguments carry fp16/bf16
    for those dtypes.

    float32 -> 'no' is a DELIBERATE divergence: legacy sets fp16=True for a float32 run (measured:
    fp16=True, bf16=False, torch_dtype=float32), so "full precision" there still autocasts to fp16.
    dev takes float32 at face value instead of quietly training in half precision. A float32 dev vs
    legacy loss comparison is therefore NOT expected to match -- they are different objectives, not a
    bug. bf16 remains the aligned baseline for numerical comparisons.
    """
    if torch_dtype == 'float32':
        return 'no'
    if torch_dtype == 'float16':
        return 'fp16'
    return 'bf16'


def is_megatron_backend(distributed_config: DistributedConfig) -> bool:
    backend = distributed_config.backend
    if backend in (None, 'hf'):
        return False
    if backend == 'megatron':
        return True
    raise ValueError(f"DistributedConfig.backend must be one of {{'megatron', 'hf'}}, got {backend!r}.")


def frozen_auxiliary_distributed_config(distributed_config: DistributedConfig,
                                        world_size: int,
                                        *,
                                        parallel_spec: Optional[str] = None,
                                        deepspeed: Optional[str] = None) -> DistributedConfig:
    """The DistributedConfig a FROZEN auxiliary model (reference / teacher / reward scorer) is built with.

    Every RL auxiliary model is built by :func:`build_model` from its OWN DistributedConfig, so this is the
    single place that fixes two invariants the auxiliary build sites used to get wrong:

    1. **Backend-agnostic** (RL_PLAN basic principle 1): it inherits ONLY the run's backend identity
       (``backend`` + ``bridge_backend``), so a megatron policy gets megatron auxiliaries and a transformers
       policy gets transformers ones -- an auxiliary only ever ``forward_only``s, which both backends
       implement identically. Building one with a backend-less config (the old ``mode='local'`` sites did)
       silently pinned it to transformers even under a megatron run.
    2. **Always a Ray actor** (basic principle 2): ``mode`` is forced to ``'ray'``. RL never runs
       ``mode='local'``; under Ray the driver holds no GPU, so a ``mode='local'`` auxiliary would be a
       driver-process model with no device -- it must be an actor on its own planned DeviceGroup.

    It deliberately does NOT inherit the trainer's parallel layout / deepspeed / fsdp: those size and shard
    the TRAINABLE model's optimizer states. A frozen forward-only auxiliary is sized by ``world_size`` (which
    MUST equal the rank count ``plan_rl_device_groups`` allocated to its DeviceGroup, since build_model reads
    ``nproc_per_node`` as the group size) and laid out by its own optional ``parallel_spec``; it needs no
    optimizer sharding, so ``deepspeed`` stays None unless a caller passes an auxiliary-specific one.
    """
    from swift.dev.config import DistributedConfig as _DistributedConfig
    return _DistributedConfig(
        mode='ray',
        backend=distributed_config.backend,
        bridge_backend=distributed_config.bridge_backend,
        nproc_per_node=world_size,
        parallel_spec=parallel_spec,
        deepspeed=deepspeed)


def _apply_seq_cls_head(kwargs: dict, model_config: ModelConfig, config, model_loader=None) -> None:
    """Route a seq_cls/reranker model to a num_labels-wide SequenceClassification head.

    twinkle's TransformersModel forwards ``model_cls`` + ``config`` to ``from_pretrained``. We build
    the config here and set the classification attrs ON IT (not as from_pretrained kwargs): HF only
    auto-parses ``num_labels`` when ``config`` is None, so with an explicit config those kwargs are
    rejected -- they must live on the config object.

    - num_labels: reranker scores one relevance value, so it defaults to 1; seq_cls must pass its N.
    - problem_type: recorded on the config for HF/legacy inference parity; the training loss is
      chosen explicitly by the recipe (configure_seq_cls_loss), not inferred here.
    - pad_token_id: the SequenceClassification head locates the last non-pad token by it; without it
      HF raises for batch>1. Fall back to eos when the tokenizer has no pad.
    - tie_word_embeddings=False: the LM head is dropped for a fresh score head, mirroring legacy
      (register.py sets this for seq_cls/reranker).
    """
    num_labels = model_config.num_labels
    if num_labels is None:
        if model_config.task_type == 'reranker':
            num_labels = 1
        else:
            raise ValueError('ModelConfig.num_labels is required for task_type="seq_cls".')
    config.num_labels = num_labels
    if model_config.problem_type is not None:
        config.problem_type = model_config.problem_type
    config.tie_word_embeddings = False

    if getattr(config, 'pad_token_id', None) is None:
        # The SequenceClassification head needs a pad id to locate the last non-pad token. Source the
        # tokenizer from the resolved dev loader (build_processor) when there is one, else a plain
        # AutoTokenizer -- which is why dev no longer imports swift.model.get_model_processor here.
        if model_loader is not None:
            processor = model_loader.build_processor(
                model_config.model, config, revision=model_config.model_revision)
        else:
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(
                model_config.model, revision=model_config.model_revision, trust_remote_code=True)
        tokenizer = processor if not hasattr(processor, 'tokenizer') else processor.tokenizer
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    kwargs['model_cls'] = 'AutoModelForSequenceClassification'
    kwargs['config'] = config


def _apply_unsloth_kwargs(kwargs: dict, model_config: ModelConfig, tuner_config: TunerConfig,
                          train_config: Optional[TrainConfig],
                          quantize_config: Optional[QuantizeConfig]) -> None:
    """Add the UnslothModel-only kwargs to an otherwise unchanged TransformersModel kwargs dict.

    unsloth rebuilds the module graph around a causal-LM checkpoint, so the num_labels head built by
    _apply_seq_cls_head has nowhere to land -- reject those task types instead of silently training a
    plain causal LM against a classification loss. Its load API represents BNB quantization as
    ``load_in_4bit`` / ``load_in_8bit`` rather than a Transformers ``quantization_config``.
    """
    if model_config.task_type in ('seq_cls', 'reranker', 'generative_reranker'):
        raise NotImplementedError(f'tuner_backend="unsloth" supports causal_lm only; task_type='
                                  f'{model_config.task_type!r} needs a head unsloth does not build.')
    kwargs['full_finetuning'] = tuner_config.tuner == 'full'
    if quantize_config is not None and quantize_config.quant_method is not None:
        if quantize_config.quant_method != 'bnb':
            raise NotImplementedError(
                f'tuner_backend="unsloth" only supports BNB load-time quantization, got '
                f'{quantize_config.quant_method!r}. Use tuner_backend="peft" for this method.')
        if quantize_config.quant_bits == 4:
            kwargs.update(load_in_4bit=True, load_in_8bit=False)
        elif quantize_config.quant_bits == 8:
            kwargs.update(load_in_4bit=False, load_in_8bit=True)
        else:
            raise ValueError(f'Unsloth BNB supports quant_bits 4 or 8, got {quantize_config.quant_bits!r}.')
    # unsloth compiles its kernels and RoPE cache for a fixed length; leave its own 2048 default in
    # place when the config says nothing.
    if model_config.max_model_len:
        kwargs['max_seq_length'] = model_config.max_model_len
    if model_config.device_map:
        kwargs['device_map'] = model_config.device_map
    # unsloth installs its offloaded checkpointing inside get_peft_model, which would silently undo
    # the gradient_checkpointing_disable() below.
    if train_config is not None and not train_config.gradient_checkpointing:
        kwargs['use_gradient_checkpointing'] = False


def _apply_hf_sp_mesh(kwargs: dict, device_mesh: Optional['DeviceMesh'],
                      distributed_config: DistributedConfig) -> None:
    """Install the Ulysses SP mesh (from build_hf_device_mesh) on a local-mode transformers build.

    A no-op for sp=1 (device_mesh is None) and for mode='ray', where validate_configs has already
    rejected SP>1 and placement is _apply_ray_placement's business.
    """
    if device_mesh is not None and distributed_config.mode == 'local':
        kwargs['device_mesh'] = device_mesh


def build_ray_dp_mesh(distributed_config: DistributedConfig, sequence_parallel_size: int = 1) -> 'DeviceMesh':
    """The data-parallel DeviceMesh a transformers model group spans under mode='ray'.

    The transformers backend has no TP/PP weight sharding, so its Ray DeviceGroup is data parallel
    over every rank. Both the training model (see :func:`_apply_ray_placement`) and any engine
    colocated in the same 'model' group -- the vLLM/SGLang generative-eval sampler -- must carry this
    SAME mesh: a Ray-remote twinkle object built with ``device_mesh=None`` and no global default is
    rejected (infra raises "Set device_mesh=DeviceMesh(...) to enable ray"), and a colocated engine
    that disagreed with the model's mesh would slice its ``sample`` inputs across a different rank
    layout than the weights it syncs from that model.

    ``sequence_parallel_size`` is the HF Ulysses degree (TemplateConfig.sequence_parallel_size), the ray
    counterpart of what :func:`build_hf_device_mesh` does for local/torchrun. Ulysses is NOT an extra mesh
    dim: the dp dim still spans ALL ranks (``dp_size=nproc``) and ``data_world_size`` derives
    ``nproc/ulysses`` from it, so SP peers share one data rank and receive identical samples (each computes
    a slice of every sequence). Every consumer of this mesh for one run -- model placement, the colocated
    sampler, the global default mesh, and the ``train_batch_size = per_device * data_world_size`` formula --
    must pass the SAME ``sequence_parallel_size`` or the layouts disagree; default 1 keeps the pure-DP
    callers (SP off) bit-for-bit unchanged.

    Distinct from :func:`build_device_mesh_if_dp`, which deliberately returns None for a single DP
    rank so an in-process (non-Ray) engine stays mesh-free; a Ray group needs an explicit mesh even
    at dp==1, which is exactly the nproc_per_node==1 colocate-eval case.
    """
    from twinkle import DeviceMesh
    nproc = distributed_config.nproc_per_node
    if nproc is None:
        raise ValueError("DistributedConfig.nproc_per_node is required in mode='ray' (it sizes the 'model' "
                         'DeviceGroup and its data-parallel mesh). Pass it explicitly -- there is no default.')
    if sequence_parallel_size > 1:
        if nproc % sequence_parallel_size != 0:
            raise ValueError(f'nproc_per_node={nproc} is not divisible by sequence_parallel_size='
                             f'{sequence_parallel_size}: SP peers must evenly partition the data-parallel ranks.')
    return DeviceMesh.from_sizes(world_size=nproc, dp_size=nproc, ulysses_size=sequence_parallel_size)


def _apply_ray_placement(kwargs: dict, distributed_config: DistributedConfig,
                         remote_group: str = 'model',
                         device_mesh: Optional['DeviceMesh'] = None,
                         instance_id: Optional[str] = None) -> None:
    """Place the transformers model in a remote DeviceGroup under mode='ray'.

    ``device_mesh`` is the caller's mesh when it has one (a frozen reward model built from a
    ``parallel_spec``); it is installed as-is so the caller's layout is honored. Only when the caller
    passes none is the default mesh synthesized -- pure data parallel over nproc_per_node, since the
    transformers backend has no TP/PP weight sharding. Local (torchrun) mode leaves both device_mesh and
    remote_group unset, exactly as before -- see the note in _build_transformers_model on why a None mesh
    is the correct (and load-bearing) choice there. ``remote_group`` defaults to 'model' (the trainable
    group); a frozen reward model passes its own group name so it lands on dedicated GPUs instead of the
    trainable ones.
    """
    if distributed_config.mode == 'local':
        return
    if device_mesh is None:
        device_mesh = build_ray_dp_mesh(distributed_config)
    kwargs['device_mesh'] = device_mesh
    kwargs['remote_group'] = remote_group
    # A per-actor disambiguator for two models that share ONE remote_group (PPO's policy+critic, MOPD's K
    # teachers): twinkle folds it into the Ray actor name so they do not collide on ActorAlreadyExistsError.
    # Ray-only by construction -- the local branch returned above, so a torchrun build never carries an
    # instance_id kwarg its in-process model cannot consume.
    if instance_id is not None:
        kwargs['instance_id'] = instance_id


def _resolve_model_loader(model_config: ModelConfig):
    """Resolve the explicitly requested or inferred dev model family loader.

    ``model_type`` is authoritative when supplied. It is normally a registered family key, but it may
    also be a :class:`ModelLoader` subclass or an external plugin source (a local file/folder or a
    ``hf://`` / ``ms://`` id): a complex model that implements the loader hooks plugs in that way,
    without joining the family registry. ``model`` stays the checkpoint id/path the weights load from.
    Falling back to basename matching keeps the zero-config path convenient, while an unknown explicit
    value that is neither a family key nor a loadable plugin fails immediately instead of silently
    selecting a different family from the checkpoint name.
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
        problem_type=model_config.problem_type)
    return loader_cls(model_info)


def _build_hf_config(model_config: ModelConfig, model_loader=None):
    """Build the one HF config shared by Transformers and Megatron model construction."""
    import copy
    import math

    if model_loader is not None:
        config = model_loader.build_config(model_config.model, revision=model_config.model_revision)
        config = model_loader.process_config(config)
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_config.model, revision=model_config.model_revision, trust_remote_code=True)

    from swift.dev.utils import HfConfigFactory

    rope_scaling = model_config.rope_scaling
    if isinstance(rope_scaling, str):
        if rope_scaling not in ('linear', 'dynamic', 'yarn'):
            raise ValueError('ModelConfig.rope_scaling must be a JSON object or one of: linear, dynamic, yarn.')
        rope_scaling = {'type': rope_scaling}
    elif rope_scaling is not None:
        if not isinstance(rope_scaling, dict):
            raise TypeError('ModelConfig.rope_scaling must resolve to a dict or a supported strategy name.')
        rope_scaling = copy.deepcopy(rope_scaling)

    current_rope = HfConfigFactory.get_config_attr(config, 'rope_scaling')
    if rope_scaling is None and current_rope and model_config.max_model_len is not None:
        rope_scaling = copy.deepcopy(current_rope)
        rope_scaling.pop('factor', None)

    if rope_scaling is not None:
        rope_type = rope_scaling.get('rope_type', rope_scaling.get('type', 'default'))
        origin_max_len = rope_scaling.get('original_max_position_embeddings')
        if origin_max_len is None and current_rope:
            origin_max_len = current_rope.get('original_max_position_embeddings')
            if origin_max_len is None and current_rope.get('factor'):
                current_max_len = HfConfigFactory.get_max_model_len(config)
                if current_max_len is not None:
                    origin_max_len = int(current_max_len / current_rope['factor'])
        if origin_max_len is None:
            origin_max_len = HfConfigFactory.get_max_model_len(config)
        if 'factor' not in rope_scaling and not (model_config.max_model_len is None and rope_type == 'default'):
            if model_config.max_model_len is None:
                raise ValueError('ModelConfig.max_model_len is required when rope_scaling has no factor.')
            if origin_max_len is None:
                raise ValueError('The checkpoint does not declare a maximum length needed to derive rope_scaling.')
            rope_scaling['factor'] = max(float(math.ceil(model_config.max_model_len / origin_max_len)), 1.0)
        if origin_max_len is not None:
            rope_scaling.setdefault('original_max_position_embeddings', origin_max_len)
        HfConfigFactory.set_config_attr(config, 'rope_scaling', rope_scaling)

    if model_config.max_model_len is not None:
        HfConfigFactory.set_max_model_len(config, model_config.max_model_len)
    return config


def _set_load_kwarg(kwargs: dict, name: str, value) -> None:
    """Set a named load option, rejecting disagreement with ``model_kwargs``."""
    if value is None:
        return
    if name in kwargs and kwargs[name] != value:
        raise ValueError(f'ModelConfig.model_kwargs[{name!r}]={kwargs[name]!r} conflicts with {name}={value!r}.')
    kwargs[name] = value


def _apply_model_post_load(model, model_config: ModelConfig) -> None:
    """Apply ModelConfig operations that require the live Transformers module."""
    if model_config.new_special_tokens:
        import os

        special_tokens = []
        for token in model_config.new_special_tokens:
            if token.endswith('.txt'):
                if not os.path.isfile(token):
                    raise FileNotFoundError(f'new_special_tokens file does not exist: {token}')
                with open(token, 'r', encoding='utf-8') as file:
                    special_tokens.extend(file.read().split())
            else:
                special_tokens.append(token)

        processor = getattr(model, '_default_tokenizer', None)
        if processor is None:
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(
                model_config.model, revision=model_config.model_revision, trust_remote_code=True)
            model._default_tokenizer = processor
        tokenizer = getattr(processor, 'tokenizer', processor)
        added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        if added:
            vocab_size = ((len(tokenizer) + 127) // 128) * 128
            model.model.resize_token_embeddings(vocab_size)

    if model_config.init_strategy is not None:
        import torch
        from torch import nn

        def high_dim(param, init):
            if param.dim() > 1:
                init(param)
            elif param.dim() == 1 and param.numel() > 0:
                nn.init.zeros_(param)

        initializers = {
            'zero': nn.init.zeros_,
            'uniform': lambda param: nn.init.uniform_(param, -0.1, 0.1),
            'normal': lambda param: nn.init.normal_(param, 0.0, 0.01),
            'xavier_uniform': lambda param: high_dim(param, nn.init.xavier_uniform_),
            'xavier_normal': lambda param: high_dim(param, nn.init.xavier_normal_),
            'kaiming_uniform': lambda param: high_dim(
                param, lambda value: nn.init.kaiming_uniform_(value, mode='fan_out', nonlinearity='leaky_relu', a=0.1)),
            'kaiming_normal': lambda param: high_dim(
                param, lambda value: nn.init.kaiming_normal_(value, mode='fan_in', nonlinearity='relu')),
            'orthogonal': lambda param: high_dim(param, nn.init.orthogonal_),
        }
        initialize = initializers[model_config.init_strategy]
        with torch.no_grad():
            for param in model.model.parameters():
                if param.numel() == 0:
                    continue
                mean_abs = param.abs().mean()
                std = param.std()
                if not torch.isfinite(mean_abs) or not torch.isfinite(std) or mean_abs > 1e7 or std > 1e7:
                    initialize(param)


def _build_transformers_model(model_config: ModelConfig,
                              distributed_config: DistributedConfig,
                              train_config: Optional[TrainConfig] = None,
                              tuner_config: Optional[TunerConfig] = None,
                              device_mesh: Optional['DeviceMesh'] = None,
                              quantize_config: Optional[QuantizeConfig] = None,
                              megatron_config: Optional[MegatronConfig] = None,
                              moe_config: Optional[MoEConfig] = None,
                              remote_group: Optional[str] = None,
                              instance_id: Optional[str] = None,
                              enable_router_replay: bool = False) -> TrainableModel:
    import torch

    from swift.dev.model import TransformersModel

    if not model_config.model:
        raise ValueError('ModelConfig.model (path/id) is required')

    kwargs: dict = dict(model_config.model_kwargs or {})
    kwargs['model_id'] = model_config.model
    # dtype: ModelConfig.torch_dtype is a string ('bfloat16'); forward to from_pretrained.
    if model_config.torch_dtype:
        dt = getattr(torch, model_config.torch_dtype, None)
        if dt is not None:
            _set_load_kwarg(kwargs, 'dtype', dt)
    _set_load_kwarg(kwargs, 'attn_implementation', model_config.attn_impl)
    _set_load_kwarg(kwargs, 'experts_implementation', model_config.experts_impl)
    _set_load_kwarg(kwargs, 'revision', model_config.model_revision)
    _set_load_kwarg(kwargs, 'device_map', model_config.device_map)
    _set_load_kwarg(kwargs, 'max_memory', model_config.max_memory)

    # Resolve a dev family loader and build the config once so caller overrides also reach custom loaders.
    model_loader = _resolve_model_loader(model_config)
    hf_config = _build_hf_config(model_config, model_loader)
    kwargs['config'] = hf_config

    # seq_cls / reranker ride a num_labels-wide SequenceClassification head instead of the LM head.
    # (reranker = num_labels=1; a plain reranker maps to this same head with a reranker loss.)
    # generative_reranker keeps the CausalLM + a forward-time lm_head patch, so it is NOT here.
    # PPO's value critic also rides this head (task_type='seq_cls', num_labels=1) and is forwarded with
    # task='value' to keep the per-token output.
    if model_config.task_type in ('seq_cls', 'reranker'):
        _apply_seq_cls_head(kwargs, model_config, hf_config, model_loader)

    # strategy: deepspeed/fsdp config selects the twinkle strategy (default: accelerate).
    from swift.dev.naming import resolve_strategy
    strategy = 'accelerate'
    if distributed_config.deepspeed:
        strategy = 'deepspeed'
        # twinkle's DeepSpeedStrategy wants a config dict; the dev field is the same string the legacy
        # CLI accepts (preset name / JSON file / inline JSON), so resolve it here.
        kwargs['deepspeed_config'] = _resolve_deepspeed_config(distributed_config)
    elif distributed_config.fsdp:
        strategy = 'native_fsdp'
    kwargs['strategy'] = resolve_strategy(strategy)
    # twinkle's transformers strategies (accelerate / deepspeed) take the process-group timeout from
    # this env var, not a constructor arg (see AccelerateStrategy.__init__), so the configured
    # --ddp_timeout reaches them through it. None leaves twinkle's own default in place.
    if distributed_config.ddp_timeout is not None:
        os.environ['TWINKLE_DIST_TIMEOUT_SECONDS'] = str(distributed_config.ddp_timeout)
    # Derived from torch_dtype, exactly like the Megatron branch below. This used to be hardcoded to
    # 'bf16', so --torch_dtype float16 silently trained in bf16 and --torch_dtype float32 did too --
    # the flag reached from_pretrained but never the autocast mode.
    kwargs['mixed_precision'] = _mixed_precision_for(model_config.torch_dtype)

    # DDP find_unused_parameters: mirror HF Trainer's three-way derivation
    find_unused = distributed_config.ddp_find_unused_parameters
    if find_unused is None:
        # A None TunerConfig means full-parameter training (select_tuner maps tuner='full' to None);
        # any config here is an adapter, whose unused-parameter gradients need find_unused_parameters.
        is_peft = tuner_config is not None
        if is_peft:
            find_unused = True
        else:
            gc_on = train_config.gradient_checkpointing if train_config is not None else False
            find_unused = not gc_on
    kwargs['ddp_config'] = {'find_unused_parameters': bool(find_unused)}

    # No device_mesh passed by default: twinkle's local mode assigns its default one (pure data
    # parallel over WORLD_SIZE, infra/__init__.py:538-541), which is exactly the transformers layout --
    # this backend has no TP/PP/CP (validate_configs rejects those sizes here). It is NOT optional
    # bookkeeping though: a None mesh silently changes the training objective to an avg-of-avg, so it
    # is load-bearing that run_sft calls twinkle.initialize first (see _initialize_twinkle, and the
    # 2-GPU aggregation test that pins the result).
    #
    # The one opt-in exception is Ulysses sequence parallelism: run_sft passes an explicit mesh from
    # build_hf_device_mesh (dp x ulysses over the torchrun world) when sequence_parallel_size > 1,
    # which switches twinkle's TransformersModel into its SP path (_enable_sp, transformers.py).
    _apply_hf_sp_mesh(kwargs, device_mesh, distributed_config)
    #
    # Ray placement (RL): under mode='ray' the model lives in a remote DeviceGroup named 'model' --
    # the same group _initialize_twinkle builds -- mirroring the Megatron branch below. This is what
    # an online RL recipe needs so the trainer and a vLLMSampler are SEPARATE Ray actors that
    # CheckpointEngineManager can weight-sync between (it asserts both have `_actors`+`device_mesh`).
    # A caller-supplied device_mesh (a frozen reward model's parallel_spec) is threaded through and
    # honored; only when there is none does placement synthesize the default pure-DP mesh.
    _apply_ray_placement(kwargs, distributed_config, remote_group or 'model', device_mesh, instance_id=instance_id)

    # MoE routing replay (RLHFConfig.router_replay_mode != 'disabled'): a construction-time flag twinkle's
    # TransformersModel.__init__ pops (enable_router_replay). Set it in the common kwargs so all three
    # construction paths below carry it; only the trainable policy is built with it True.
    if enable_router_replay:
        kwargs['enable_router_replay'] = True

    # Three mutually exclusive construction paths:
    #   - a family naming an external ``model_framework`` builds through that framework's own pipeline --
    #     today only ``sentence_transformers``, whose embedding model builds Transformer -> Pooling ->
    #     Normalize and pools per sentence, so it must NOT ride the plain HF embedding task (which re-pools
    #     per-token features) -- see _resolve_model_framework;
    #   - tuner_backend='unsloth' swaps the class: unsloth owns both construction (its Triton kernels /
    #     optional 4bit base) and LoRA installation -- see swift/dev/model/unsloth_model.py. Everything
    #     derived above (dtype, strategy, mixed_precision, ddp_config) is passed through unchanged;
    #   - otherwise the plain twinkle TransformersModel, with the resolved family loader handed over.
    model_framework = _resolve_model_framework(model_config, model_loader)
    if model_framework == 'sentence_transformers':
        model = _build_sentence_transformer_model(kwargs, model_config, distributed_config, tuner_config,
                                                  quantize_config, train_config)
    elif model_framework is not None:
        # A declared framework with no builder wired here would otherwise fall through and build as a plain
        # TransformersModel, silently ignoring the framework -- fail loudly instead, so a typo in a loader's
        # ``model_framework`` cannot train a different model than the family declares.
        raise ValueError(f'No builder is wired for model_framework={model_framework!r}; add a branch here.')
    elif tuner_config is not None and tuner_config.tuner_backend == 'unsloth':
        from swift.dev.model import UnslothModel
        _apply_unsloth_kwargs(kwargs, model_config, tuner_config, train_config, quantize_config)
        model = UnslothModel(**kwargs)
    else:
        from swift.dev.builders.quantization import build_load_quantization_config
        quantization_config = build_load_quantization_config(
            quantize_config, model_loader=model_loader, torch_dtype=model_config.torch_dtype)
        if quantization_config is not None:
            kwargs['quantization_config'] = quantization_config
        # Full takeover: hand the resolved family loader to twinkle, which then builds config/
        # processor/model through it. seq_cls/reranker are excluded -- their num_labels head overrides
        # model_cls, which a family (causal-LM) loader would not build; they keep the
        # AutoModelForSequenceClassification path applied above.
        if model_loader is not None and model_config.task_type not in ('seq_cls', 'reranker'):
            kwargs['model_loader'] = model_loader
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
        _disable_gradient_checkpointing(model)

    # Liger op-level kernels (rms_norm / rotary / swiglu / ...) replace layers on the unwrapped HF
    # module, so this runs after construction and before the loss/optimizer are set. The
    # fused-linear-CE op is deliberately NOT here -- it skips the lm_head GEMM rather than swapping a
    # layer, so it is a loss-path selection (see swift.dev.loss.configure_loss), not a kernelize target.
    if train_config is not None and train_config.use_liger_kernel:
        _apply_liger_kernel(model, train_config)

    _apply_model_post_load(model, model_config)
    return model


def _resolve_model_framework(model_config: ModelConfig, model_loader) -> Optional[str]:
    """The external framework that must construct this embedding model, or None for the plain HF path
    (TransformersModel + task='embedding'). The returned name keys the dispatch in ``build_model``.

    Only an embedding task can ride a special framework today (``sentence_transformers`` is the sole one
    wired). A resolved family loader is authoritative: qwen2_gte / gemma_emb declare
    ``model_framework='sentence_transformers'`` (legacy loaded them via ``SentenceTransformer``), while
    e.g. qwen3_emb deliberately does not -- its checkpoint ships an ST layout, but there that layout is an
    *export* artifact and training still rides the HF embedding task. With no loader resolved (an
    unregistered checkpoint), fall back to the on-disk ST layout so a re-trained/exported
    sentence-transformers directory still routes correctly.
    """
    if model_config.task_type != 'embedding':
        return None
    if model_loader is not None:
        return getattr(model_loader, 'model_framework', None)
    return 'sentence_transformers' if _has_sentence_transformer_layout(model_config.model) else None


def _has_sentence_transformer_layout(model_dir) -> bool:
    """True when a local checkpoint directory carries the sentence-transformers pipeline files.

    A hub id (not yet downloaded) is not a directory, so this is False there -- the loader marker is the
    signal for registered families; this only catches an unregistered *local* ST checkpoint.
    """
    import os
    if not model_dir or not os.path.isdir(model_dir):
        return False
    return any(
        os.path.exists(os.path.join(model_dir, name))
        for name in ('modules.json', 'config_sentence_transformers.json', '1_Pooling'))


def _build_sentence_transformer_model(kwargs: dict, model_config: ModelConfig,
                                      distributed_config: DistributedConfig,
                                      tuner_config: Optional[TunerConfig],
                                      quantize_config: Optional[QuantizeConfig],
                                      train_config: Optional[TrainConfig]) -> TrainableModel:
    """Construct a SentenceTransformerModel from the shared transformers-build kwargs.

    Only the plumbing ST shares with the base transformers model is forwarded (strategy / precision /
    ddp / mesh / ray placement); the ``from_pretrained`` load kwargs and the family ``model_loader`` are
    dropped because ST builds its backbone through ``SentenceTransformer(model_id)``. ``config`` is left
    unset so ST exposes the *backbone's* config (its own documented intent) rather than the dev-built
    one, which the ST backbone never consumed.

    The features the plain transformers build honors but the ST pipeline cannot are rejected here rather
    than silently dropped: this is where the embedding model is diverted off the TransformersModel path,
    and knowing it is ST needs the resolved loader / on-disk layout, which only exist at build time -- so
    validate_configs (config-only) cannot catch these combinations.
    """
    if distributed_config.deepspeed:
        raise NotImplementedError(
            'A sentence-transformers embedding model cannot train under DeepSpeed: SentenceTransformerModel '
            'builds its own module pipeline and wires no deepspeed_config into its strategy, so ZeRO would be '
            'silently ignored. Use --fsdp or plain DDP for this model.')
    if tuner_config is not None and tuner_config.tuner_backend == 'unsloth':
        raise NotImplementedError(
            'tuner_backend="unsloth" cannot wrap a sentence-transformers embedding model: unsloth rebuilds a '
            'causal-LM module graph, while this checkpoint trains as an ST pooling pipeline. Use '
            'tuner_backend="peft" (LoRA) or full fine-tuning.')
    if quantize_config is not None and quantize_config.quant_method is not None:
        raise NotImplementedError(
            f'Load-time quantization ({quantize_config.quant_method!r}) is not applied to a sentence-transformers '
            'embedding model: ST builds its backbone via SentenceTransformer(model_id), which takes no '
            'transformers quantization_config. Train this model unquantized.')
    if train_config is not None and train_config.use_liger_kernel:
        raise NotImplementedError(
            'use_liger_kernel is not composed with a sentence-transformers embedding model: Liger replaces '
            'causal-LM decoder ops / the fused-linear-CE loss path, while this checkpoint trains an ST pooling '
            'pipeline with an embedding loss. Disable use_liger_kernel for this model.')
    from swift.dev.model import SentenceTransformerModel
    st_kwargs = {
        'model_id': kwargs['model_id'],
        'strategy': kwargs['strategy'],
        'mixed_precision': kwargs['mixed_precision'],
        'ddp_config': kwargs.get('ddp_config'),
    }
    # SP mesh (local) / DeviceGroup placement (ray) were installed on the shared kwargs by
    # _apply_hf_sp_mesh / _apply_ray_placement; forward them so ST lands on the right ranks. ST's own
    # forward guard rejects an SP mesh (a whole-sentence pooling head cannot see a sequence shard).
    for key in ('device_mesh', 'remote_group'):
        if key in kwargs:
            st_kwargs[key] = kwargs[key]
    return SentenceTransformerModel(**st_kwargs)


def _disable_gradient_checkpointing(model: TrainableModel) -> None:
    """Undo twinkle's unconditional gradient_checkpointing_enable() when the config turned it off.

    A plain transformers build holds an HF backbone at ``model.model``; a SentenceTransformerModel holds a
    ``SentenceTransformer`` pipeline there instead, so it exposes its own disable that drives the leading
    Transformer module's backbone.
    """
    disable = getattr(model, 'gradient_checkpointing_disable', None)
    if callable(disable):
        disable()
    else:
        model.model.gradient_checkpointing_disable()


def _apply_liger_kernel(model: TrainableModel, train_config: TrainConfig) -> None:
    """Patch the HF module's per-layer ops with Liger kernels via twinkle's mapping-driven kernelize.

    ``liger_kernel_config`` (an ``{op: bool}`` dict) switches individual ops off on top of Liger's
    default set; None keeps the default. kernelize mutates the module in place (class / forward
    replacement), which is why it belongs at the end of model construction.
    """
    from twinkle.kernel import kernelize
    kernelize(model.model, _liger_kernel_mapping(train_config.liger_kernel_config))


def _liger_kernel_mapping(liger_kernel_config: Optional[dict]) -> Optional[dict]:
    """DEFAULT_KERNEL_CONFIG minus any op the config switches off; None keeps the built-in default.

    Returning None rather than a copy of the default matters: kernelize logs family-skips at DEBUG on
    the default path but raises them to WARNING for any explicit mapping. 'fused_linear_cross_entropy'
    is not a kernelize op, so it matches no DEFAULT_KERNEL_CONFIG entry and is ignored here (it is
    consumed on the loss side instead).
    """
    if not liger_kernel_config:
        return None
    from twinkle.kernel.config import DEFAULT_KERNEL_CONFIG
    disabled = {op for op, enabled in liger_kernel_config.items() if not enabled}
    return {
        target: choice
        for target, choice in DEFAULT_KERNEL_CONFIG.items()
        if getattr(choice, 'op', None) not in disabled
    }


def _resolve_deepspeed_config(distributed_config: DistributedConfig) -> Optional[dict]:
    """Resolve ``DistributedConfig.deepspeed`` (preset name / JSON file / JSON string) into a dict.

    twinkle's DeepSpeedStrategy takes a config dict, while the dev field carries the same string the
    legacy CLI accepts. The preset-name -> bundled-JSON mapping lives in ``swift.dev.utils`` (shared with
    config validation, which reads the ZeRO stage from the same resolution); only the ZeRO++
    ``zero_hpz_partition_size`` injection -- a strategy-construction concern -- stays here. Returns None
    when DeepSpeed is not requested.
    """
    from swift.dev.utils import resolve_deepspeed_config

    config = resolve_deepspeed_config(distributed_config.deepspeed)
    if config is None:
        return None
    if distributed_config.zero_hpz_partition_size is not None:
        # ZeRO++ hpz shards within a node and replicates across nodes; it only means something inside a
        # zero_optimization section, so refuse rather than drop it when the resolved config has none.
        if 'zero_optimization' not in config:
            raise ValueError('zero_hpz_partition_size (ZeRO++) needs a deepspeed config with a zero_optimization '
                             'section; the resolved config has none.')
        config['zero_optimization']['zero_hpz_partition_size'] = distributed_config.zero_hpz_partition_size
    return config


def _apply_mtp_kwargs(kwargs: dict, model_config: ModelConfig, strict: set) -> None:
    """Forward the Multi-Token Prediction knobs into mcore-bridge's ModelConfig.

    All six land on the same object (``get_model_config`` forwards **kwargs verbatim), so they are
    grouped here rather than mixed into the recompute/attention block above.

    ``mtp_num_layers`` gates the rest: without it the bridge builds no MTP block at all, and
    mcore-bridge's own ``__post_init__`` rejects the other knobs rather than ignoring them. Each is
    forwarded only when set, so an MTP-free run reaches the bridge with exactly the kwargs it had
    before this existed -- an unset ``mtp_loss_scaling_factor`` has to stay unset for mcore's own
    default (0.1) to apply, and passing None would override it with None.
    """
    if model_config.mtp_num_layers is None:
        return
    values = {
        'mtp_num_layers': model_config.mtp_num_layers,
        'mtp_loss_scaling_factor': model_config.mtp_loss_scaling_factor,
        'enable_mtp_training': model_config.enable_mtp_training,
        'mtp_freeze': model_config.mtp_freeze,
        'mtp_decoder_input_detach': model_config.mtp_decoder_input_detach,
        'mtp_shared_weights': model_config.mtp_shared_weights,
    }
    for name, value in values.items():
        if value is not None:
            kwargs[name] = value
        if _field_was_requested(model_config, name):
            strict.add(name)


def _apply_fp4_kwargs(kwargs: dict, model_config: ModelConfig, strict: set) -> None:
    """Forward the FP4 knobs into mcore-bridge's ModelConfig, under megatron's names for them.

    A rename, not a copy: dev's fields are named after the legacy CLI flags (``--fp4-format``,
    ``--fp4-param-gather``) so the Megatron CLI bridge picks them up by same-name copy, while
    megatron's TransformerConfig calls the same two things ``fp4`` and ``fp4_param``.

    ``fp4_param_gather`` maps onto ``fp4_param`` ALONE even though megatron has a same-named DDP
    field: the two must agree or the run silently does not train, so twinkle derives the DDP flag
    from ``fp4_param`` itself (MegatronStrategy._finalize_quantized_param_config). Setting it here as
    well would create the second, independent source of truth that derivation exists to remove.

    Gated on ``fp4_format`` so an FP4-free run reaches the bridge with exactly the kwargs it had
    before this existed -- ``fp4_recipe`` in particular has a non-None default on both sides, so
    forwarding it unconditionally would be indistinguishable from the user asking for it.
    """
    if model_config.fp4_format is None:
        return
    kwargs['fp4'] = model_config.fp4_format
    kwargs['fp4_recipe'] = model_config.fp4_recipe
    if model_config.fp4_param_gather:
        kwargs['fp4_param'] = True
    for source, target in (
        ('fp4_format', 'fp4'),
        ('fp4_recipe', 'fp4_recipe'),
        ('fp4_param_gather', 'fp4_param'),
    ):
        if _field_was_requested(model_config, source):
            strict.add(target)


def _apply_fp8_kwargs(kwargs: dict, model_config: ModelConfig, strict: set) -> None:
    """Forward the FP8 knobs into mcore-bridge's ModelConfig, under megatron's names for them.

    Deliberately a sibling of ``_apply_fp4_kwargs`` rather than a shared loop: the two formats look
    symmetric in the config but are not here. FP8 carries the delayed-scaling amax knobs, whose dev
    defaults intentionally differ from megatron's (1024 / 'max' vs 1 / 'most_recent', following
    legacy Megatron-SWIFT), which means they must be forwarded EXPLICITLY -- leaving them out would
    silently hand the run megatron's defaults and change its numerics against legacy. FP4 has no
    equivalent, so a shared implementation would need a per-format exception table to say so.

    ``fp8_param_gather`` maps to ``fp8_param`` alone, for the same reason as the FP4 case: twinkle
    derives the DDP flag from it, and a second writer would be a second source of truth.
    """
    if model_config.fp8_format is None:
        return
    kwargs['fp8'] = model_config.fp8_format
    kwargs['fp8_recipe'] = model_config.fp8_recipe
    kwargs['fp8_amax_history_len'] = model_config.fp8_amax_history_len
    kwargs['fp8_amax_compute_algo'] = model_config.fp8_amax_compute_algo
    if model_config.fp8_param_gather:
        kwargs['fp8_param'] = True
    for source, target in (
        ('fp8_format', 'fp8'),
        ('fp8_recipe', 'fp8_recipe'),
        ('fp8_amax_history_len', 'fp8_amax_history_len'),
        ('fp8_amax_compute_algo', 'fp8_amax_compute_algo'),
        ('fp8_param_gather', 'fp8_param'),
    ):
        if _field_was_requested(model_config, source):
            strict.add(target)


def build_device_mesh(distributed_config: DistributedConfig):
    """DistributedConfig -> the Megatron DeviceMesh (parallel layout).

    A pure function of the config: it reads only the declared sizes, never torch.distributed or
    Megatron's mpu, so it can be called before anything is initialized.
    """
    from twinkle import DeviceMesh

    spec = distributed_config.parallel_spec
    if spec:
        # A parallel_spec string is a lossless alternative to the integer size fields. Parse it and hand
        # the dims straight to DeviceMesh: from_sizes raises if they do not fill nproc_per_node exactly,
        # and each model backend rejects any dimension it does not implement. No validation layer here.
        return DeviceMesh.from_spec(spec, world_size=distributed_config.nproc_per_node)

    tp = distributed_config.tensor_model_parallel_size
    pp = distributed_config.pipeline_model_parallel_size
    cp = distributed_config.context_parallel_size
    ep = distributed_config.expert_model_parallel_size

    world_size = distributed_config.nproc_per_node
    if world_size is None:
        raise ValueError('DistributedConfig.nproc_per_node is required for the Megatron backend (it sets the '
                         'DeviceMesh world size). Pass it explicitly -- there is no default, since a wrong '
                         'world size silently builds the wrong data-parallel layout.')
    model_parallel = tp * pp * cp
    if world_size % model_parallel != 0:
        raise ValueError(f'nproc_per_node={world_size} is not divisible by tp*pp*cp={model_parallel} '
                         f'(tp={tp}, pp={pp}, cp={cp}).')
    dp_size = world_size // model_parallel

    mesh_kwargs = dict(world_size=world_size, dp_size=dp_size)
    if tp > 1:
        mesh_kwargs['tp_size'] = tp
    if pp > 1:
        mesh_kwargs['pp_size'] = pp
    if cp > 1:
        mesh_kwargs['cp_size'] = cp
    if ep > 1:
        mesh_kwargs['ep_size'] = ep
    if distributed_config.expert_tensor_parallel_size > 1:
        mesh_kwargs['etp_size'] = distributed_config.expert_tensor_parallel_size
    if distributed_config.virtual_pipeline_model_parallel_size is not None:
        mesh_kwargs['vpp_size'] = distributed_config.virtual_pipeline_model_parallel_size
    # Megatron TP sequence-parallelism rides on the DeviceMesh (twinkle reads it via
    # strategy.sequence_parallel -> device_mesh.sequence_parallel). Only meaningful with tp > 1.
    if distributed_config.sequence_parallel:
        mesh_kwargs['sequence_parallel'] = True
    return DeviceMesh.from_sizes(**mesh_kwargs)


def build_device_mesh_if_dp(distributed_config: Optional[DistributedConfig]) -> Any:
    """A DeviceMesh only when DP > 1: a single-process run wants a plain in-process engine.

    The inference/sampling recipes share this so "when does a sampler get a device mesh" has one
    answer: ``None`` for no config or a single data-parallel rank (the engine stays in-process), the
    mesh only when there are actually multiple DP ranks to slice across.
    """
    if distributed_config is None:
        return None
    # No world size and no explicit layout means the run was not launched distributed: a single
    # in-process engine, which is definitionally not multi-rank DP. Return None here rather than let
    # build_device_mesh raise for the missing nproc_per_node -- the recipes call this on a default
    # DistributedConfig (e.g. a programmatic run_deploy/run_infer that never went through
    # process_and_validate_configs), and every caller already reads None as "no mesh, dp_size 1".
    if distributed_config.nproc_per_node is None and not distributed_config.parallel_spec:
        return None
    mesh = build_device_mesh(distributed_config)
    return mesh if mesh is not None and getattr(mesh, 'data_world_size', 1) > 1 else None


_NON_MODEL_MEGATRON_FIELDS = {
    'attention_backend',
    'data_parallel_random_init',
    'skip_megatron_init',
    'manual_gc',
    'manual_gc_eval',
    'manual_gc_steps',
    'megatron_extra_kwargs',
}
_MEGATRON_FIELD_ALIASES = {'te_rng_tracker': 'use_te_rng_tracker'}
_UNSUPPORTED_MEGATRON_MODEL_FIELDS = {
    'apply_dsa_kernel_fusion',
    'csa_dense_mode',
    'sequence_packing_scheduler',
    'use_fused_mhc',
}


def _field_was_requested(config, field_name: str) -> bool:
    """Whether a non-default Config value was requested explicitly or programmatically."""
    import dataclasses

    explicit = getattr(config, '_explicit_fields', None)
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

    Megatron-Bridge providers vary by installed version. Defaults which an older provider does not
    expose may safely fall back to that provider's own default, but an explicitly requested option
    must never disappear. ``_strict_model_kwargs`` carries that distinction to the worker-side
    bridge backend and is removed before constructing an mcore ``ModelConfig``.
    """
    import dataclasses

    kwargs = {}
    strict = set()
    if megatron_config is not None:
        for config_field in dataclasses.fields(megatron_config):
            name = config_field.name
            if name in _UNSUPPORTED_MEGATRON_MODEL_FIELDS:
                if _field_was_requested(megatron_config, name):
                    raise NotImplementedError(
                        f'MegatronConfig.{name} is not exposed by the installed mcore-bridge ModelConfig. '
                        'Upgrade mcore-bridge/Megatron-LM or remove this option.')
                continue
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
            raise TypeError('MegatronConfig.megatron_extra_kwargs must be a dict after process_configs().')
        for name, value in extra.items():
            if name in kwargs and kwargs[name] != value:
                raise ValueError(f'megatron_extra_kwargs[{name!r}]={value!r} conflicts with {name}={kwargs[name]!r}.')
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
        kwargs['_strict_model_kwargs'] = tuple(sorted(strict))
    return kwargs


def _build_megatron_model(model_config: ModelConfig,
                           distributed_config: DistributedConfig,
                           train_config: Optional[TrainConfig] = None,
                           *,
                           megatron_config: Optional[MegatronConfig] = None,
                           moe_config: Optional[MoEConfig] = None,
                           remote_group: Optional[str] = None,
                           instance_id: Optional[str] = None,
                           enable_router_replay: bool = False) -> TrainableModel:
    """Build a MegatronModel via the selected bridge backend.

    twinkle must already be initialized in Ray mode (run_sft does this) so the 'model' DeviceGroup
    exists. The world size is DistributedConfig.nproc_per_node (== the DeviceGroup size); dp_size is
    derived from it and the model-parallel sizes. Driver-side dist.get_world_size() is NOT used --
    in Ray mode the driver is not part of the model process group (its world size is 1).
    """
    from twinkle.model.megatron import MegatronModel

    if not model_config.model:
        raise ValueError('ModelConfig.model (path/id) is required')

    device_mesh = build_device_mesh(distributed_config)
    model_loader = _resolve_model_loader(model_config)
    hf_config = _build_hf_config(model_config, model_loader)

    mixed_precision = _mixed_precision_for(model_config.torch_dtype)

    # A few high-frequency Megatron knobs flow straight into MegatronModel.__init__. Forward
    # only when set (None -> twinkle's own default), so the bit-exact SFT baseline is unchanged
    # unless the user opts in. use_distributed_optimizer has a real default (True) so pass it.
    extra_kwargs: dict = {
        'use_distributed_optimizer': distributed_config.use_distributed_optimizer,
        'align_grad_reduce': distributed_config.align_grad_reduce,
        'nccl_comm_warmup': distributed_config.nccl_comm_warmup,
        **_megatron_model_kwargs(megatron_config, moe_config),
    }
    strict_model_kwargs = set(extra_kwargs.pop('_strict_model_kwargs', ()))
    # ModelParallelConfig/TransformerConfig fields owned by DistributedConfig. Keep the spelling
    # translation here, next to the DeviceMesh translation, so a parsed field cannot stop at the
    # dataclass without reaching the worker-side Megatron config.
    model_field_aliases = {
        'decoder_first_pipeline_num_layers': 'num_layers_in_first_pipeline_stage',
        'decoder_last_pipeline_num_layers': 'num_layers_in_last_pipeline_stage',
    }
    model_fields = (
        'recompute_granularity',
        'recompute_method',
        'recompute_num_layers',
        'recompute_modules',
        'cp_comm_type',
        'pipeline_model_parallel_layout',
        'decoder_first_pipeline_num_layers',
        'decoder_last_pipeline_num_layers',
        'account_for_embedding_in_pipeline_split',
        'account_for_loss_in_pipeline_split',
        'overlap_p2p_comm',
        'batch_p2p_comm',
        'tp_comm_overlap',
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
            extra_kwargs['calculate_per_token_loss'] = train_config.calculate_per_token_loss
            strict_model_kwargs.add('calculate_per_token_loss')
        if train_config.microbatch_group_size_per_vp_stage is not None:
            extra_kwargs['microbatch_group_size_per_vp_stage'] = train_config.microbatch_group_size_per_vp_stage
            strict_model_kwargs.add('microbatch_group_size_per_vp_stage')

    if extra_kwargs.get('recompute_granularity') == 'selective':
        extra_kwargs['recompute_num_layers'] = None
        extra_kwargs['recompute_method'] = None

    ddp_config = {
        'grad_reduce_in_fp32': bool(train_config and train_config.accumulate_allreduce_grads_in_fp32),
        'overlap_grad_reduce': distributed_config.overlap_grad_reduce,
        'overlap_param_gather': distributed_config.overlap_param_gather,
        'align_param_gather': distributed_config.align_param_gather,
        'data_parallel_sharding_strategy': distributed_config.data_parallel_sharding_strategy,
    }
    if distributed_config.use_megatron_fsdp:
        ddp_config['use_megatron_fsdp'] = True
    extra_kwargs['ddp_config'] = ddp_config

    # Attention kernel. Always forwarded (unlike the recompute knobs above) because the meaningful
    # default is legacy's 'flash', not mcore's AttnBackend.auto -- under auto TE picks per shape and
    # selects the FUSED cuDNN kernel for a Qwen2.5 bf16 causal THD forward, so leaving it unset makes
    # dev and legacy run different attention kernels on identical config. resolve_* returns the enum
    # (mcore compares by identity), and both bridge backends now read this one value instead of the
    # megatron-bridge path hardcoding its own.
    from swift.dev.naming import resolve_megatron_attn_backend
    extra_kwargs['attention_backend'] = resolve_megatron_attn_backend(model_config.attn_impl)
    if _field_was_requested(model_config, 'attn_impl'):
        strict_model_kwargs.add('attention_backend')
    # A flash_N / flash_attention_N value also pins the FA VERSION, which is enforced by mutating
    # transformer_engine module globals -- a per-process side effect, so it CANNOT be applied here:
    # in Ray mode build_model runs on the driver, which is not where the model is built. The raw
    # string is forwarded so MegatronStrategy (worker-side) can apply the pin itself.
    extra_kwargs['attn_impl'] = model_config.attn_impl

    # task_type / num_labels flow straight into mcore-bridge's ModelConfig (get_model_config forwards
    # **kwargs), which builds the head: seq_cls -> OutputLayerLinear(hidden, num_labels),
    # generative_reranker -> yes/no-diff vocab head. The bridge has no plain 'reranker' task, so map
    # it to seq_cls with num_labels=1 (the reranker loss, set later, makes it a reranker); this is
    # the same head legacy Megatron uses. embedding/causal_lm pass through untouched.
    task_type = model_config.task_type
    if task_type == 'reranker':
        extra_kwargs['task_type'] = 'seq_cls'
        extra_kwargs['num_labels'] = model_config.num_labels or 1
    elif task_type in ('seq_cls', 'embedding', 'generative_reranker'):
        extra_kwargs['task_type'] = task_type
        if task_type == 'seq_cls':
            if model_config.num_labels is None:
                raise ValueError('ModelConfig.num_labels is required for task_type="seq_cls".')
            extra_kwargs['num_labels'] = model_config.num_labels

    for name in ('vit_attn_impl', 'language_model_only'):
        value = getattr(model_config, name)
        if value:
            extra_kwargs[name] = value
        if _field_was_requested(model_config, name):
            strict_model_kwargs.add(name)

    _apply_mtp_kwargs(extra_kwargs, model_config, strict_model_kwargs)
    _apply_fp4_kwargs(extra_kwargs, model_config, strict_model_kwargs)
    _apply_fp8_kwargs(extra_kwargs, model_config, strict_model_kwargs)
    # MoE routing replay (RLHFConfig.router_replay_mode != 'disabled'): MegatronModel.__init__ pops
    # enable_router_replay and translates it to the mcore ``moe_enable_routing_replay`` TransformerConfig
    # field -- the Megatron counterpart of TransformersModel's flag (basic principle 1). Only the trainable
    # policy is built with it True.
    if enable_router_replay:
        extra_kwargs['enable_router_replay'] = True
    if strict_model_kwargs:
        extra_kwargs['_strict_model_kwargs'] = tuple(sorted(strict_model_kwargs))

    # In Ray mode the model lives in a remote DeviceGroup named 'model'; in local (torchrun) mode
    # each rank builds the model in-process, so there is no remote group to target.
    model_kwargs = dict(
        model_id=model_config.model,
        config=hf_config,
        revision=model_config.model_revision,
        device_mesh=device_mesh,
        mixed_precision=mixed_precision,
        # A NAME, not a built instance: twinkle's strategy resolves it to a backend, and only a
        # string survives the Ray worker boundary (@remote_class forwards constructor kwargs, not
        # live objects -- which is exactly why the old dev subclass + mock.patch could not reach it).
        bridge_backend=distributed_config.bridge_backend,
        **extra_kwargs)
    if distributed_config.mode != 'local':
        model_kwargs['remote_group'] = remote_group or 'model'
        # Same-group actor disambiguator, mirroring the transformers path's _apply_ray_placement (backend
        # equivalence: a megatron PPO critic / MOPD teacher needs it exactly as a transformers one does).
        if instance_id is not None:
            model_kwargs['instance_id'] = instance_id
    return MegatronModel(**model_kwargs)
