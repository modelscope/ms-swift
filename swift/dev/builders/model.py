"""build_model: ModelConfig + DistributedConfig -> twinkle-native TransformersModel / MegatronModel."""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from swift.dev.config import DistributedConfig, ModelConfig, TrainConfig, TunerConfig
    from swift.dev.model import TrainableModel


def build_model(model_config: ModelConfig,
                distributed_config: DistributedConfig,
                train_config: Optional[TrainConfig] = None,
                tuner_config: Optional[TunerConfig] = None) -> TrainableModel:
    """ModelConfig + DistributedConfig -> twinkle-native Model (no loss/optim yet).

    Thin mapping (no Registry/Factory): model_config fields -> twinkle __init__ kwargs.
    DistributedConfig.backend=='megatron' builds a MegatronModel (via the selected bridge
    backend); otherwise a TransformersModel. twinkle self-builds weights on the correct rank.

    PPO's value critic is NOT a special build flag: it is a ``task_type='seq_cls', num_labels=1`` model
    forwarded with ``task='value'`` (which keeps the head's per-token output instead of pooling), so it
    goes through the ordinary seq_cls build path on both backends.
    """
    if is_megatron_backend(distributed_config):
        return _build_megatron_model(model_config, distributed_config)
    return _build_transformers_model(model_config, distributed_config, train_config, tuner_config)


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


def _apply_seq_cls_head(kwargs: dict, model_config: ModelConfig, model_loader=None) -> None:
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
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_config.model, trust_remote_code=True)

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
            processor = model_loader.build_processor(model_config.model, config)
        else:
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(model_config.model, trust_remote_code=True)
        tokenizer = processor if not hasattr(processor, 'tokenizer') else processor.tokenizer
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    kwargs['model_cls'] = 'AutoModelForSequenceClassification'
    kwargs['config'] = config


def _apply_unsloth_kwargs(kwargs: dict, model_config: ModelConfig, tuner_config: TunerConfig,
                          train_config: Optional[TrainConfig]) -> None:
    """Add the UnslothModel-only kwargs to an otherwise unchanged TransformersModel kwargs dict.

    unsloth rebuilds the module graph around a causal-LM checkpoint, so the num_labels head built by
    _apply_seq_cls_head has nowhere to land -- reject those task types instead of silently training a
    plain causal LM against a classification loss.

    QLoRA is NOT wired here: build_model never receives a QuantizeConfig, so a 4bit base has to be
    requested by constructing UnslothModel(load_in_4bit=True) directly until that config is plumbed
    through this builder.
    """
    if model_config.task_type in ('seq_cls', 'reranker', 'generative_reranker'):
        raise NotImplementedError(f'tuner_backend="unsloth" supports causal_lm only; task_type='
                                  f'{model_config.task_type!r} needs a head unsloth does not build.')
    kwargs['full_finetuning'] = tuner_config.tuner_type == 'full'
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


def _apply_ray_placement(kwargs: dict, distributed_config: DistributedConfig) -> None:
    """Place the transformers model in the remote 'model' DeviceGroup under mode='ray'.

    The transformers backend has no TP/PP, so the mesh is pure data parallel over nproc_per_node.
    Local (torchrun) mode leaves both device_mesh and remote_group unset, exactly as before -- see
    the note in _build_transformers_model on why a None mesh is the correct (and load-bearing)
    choice there.
    """
    if distributed_config.mode == 'local':
        return
    from twinkle import DeviceMesh
    nproc = distributed_config.nproc_per_node
    if nproc is None:
        raise ValueError("DistributedConfig.nproc_per_node is required in mode='ray' (it sizes the 'model' "
                         'DeviceGroup and its data-parallel mesh). Pass it explicitly -- there is no default.')
    kwargs['device_mesh'] = DeviceMesh.from_sizes(world_size=nproc, dp_size=nproc)
    kwargs['remote_group'] = 'model'


def _resolve_model_loader(model_config: ModelConfig):
    """Resolve a dev :class:`ModelLoader` instance for ``model_config.model``, or None.

    Matches the checkpoint basename against the registered families (a pre-download match that,
    unlike ``architectures``, cannot collide). A miss returns None so an unregistered checkpoint
    still loads through twinkle's default AutoModel path. When non-None the loader is handed to
    ``TransformersModel(model_loader=...)`` and fully owns config/processor/model construction.
    """
    from swift.dev.model.loader import ModelInfo, get_model_loader, match_model_type
    model_type = match_model_type(model_config.model)
    if model_type is None:
        return None
    model_info = ModelInfo(
        model_type=model_type,
        model_dir=model_config.model,
        task_type=model_config.task_type,
        num_labels=model_config.num_labels)
    return get_model_loader(model_type)(model_info)


def _build_transformers_model(model_config: ModelConfig,
                              distributed_config: DistributedConfig,
                              train_config: Optional[TrainConfig] = None,
                              tuner_config: Optional[TunerConfig] = None) -> TrainableModel:
    import torch

    from swift.dev.model import TransformersModel

    if not model_config.model:
        raise ValueError('ModelConfig.model (path/id) is required')

    kwargs: dict = {'model_id': model_config.model}
    # dtype: ModelConfig.torch_dtype is a string ('bfloat16'); forward to from_pretrained.
    if model_config.torch_dtype:
        dt = getattr(torch, model_config.torch_dtype, None)
        if dt is not None:
            kwargs['dtype'] = dt
    if model_config.attn_impl:
        kwargs['attn_implementation'] = model_config.attn_impl
    if model_config.model_revision:
        kwargs['revision'] = model_config.model_revision

    # Resolve a dev family loader from the checkpoint id (None if unregistered -> twinkle default).
    model_loader = _resolve_model_loader(model_config)

    # seq_cls / reranker ride a num_labels-wide SequenceClassification head instead of the LM head.
    # (reranker = num_labels=1; a plain reranker maps to this same head with a reranker loss.)
    # generative_reranker keeps the CausalLM + a forward-time lm_head patch, so it is NOT here.
    # PPO's value critic also rides this head (task_type='seq_cls', num_labels=1) and is forwarded with
    # task='value' to keep the per-token output.
    if model_config.task_type in ('seq_cls', 'reranker'):
        _apply_seq_cls_head(kwargs, model_config, model_loader)

    # strategy: deepspeed/fsdp config selects the twinkle strategy (default: accelerate).
    from swift.dev.naming import resolve_strategy
    strategy = 'accelerate'
    if distributed_config.deepspeed:
        strategy = 'deepspeed'
        raise NotImplementedError('DeepSpeed is not supported yet')
    elif distributed_config.fsdp:
        strategy = 'native_fsdp'
    kwargs['strategy'] = resolve_strategy(strategy)
    # Derived from torch_dtype, exactly like the Megatron branch below. This used to be hardcoded to
    # 'bf16', so --torch_dtype float16 silently trained in bf16 and --torch_dtype float32 did too --
    # the flag reached from_pretrained but never the autocast mode.
    kwargs['mixed_precision'] = _mixed_precision_for(model_config.torch_dtype)

    # DDP find_unused_parameters: mirror HF Trainer's three-way derivation
    find_unused = distributed_config.ddp_find_unused_parameters
    if find_unused is None:
        is_peft = tuner_config is not None
        if is_peft:
            find_unused = True
        else:
            gc_on = train_config.gradient_checkpointing if train_config is not None else False
            find_unused = not gc_on
    kwargs['ddp_config'] = {'find_unused_parameters': bool(find_unused)}

    # No device_mesh passed on purpose: twinkle's local mode assigns its default one (pure data
    # parallel over WORLD_SIZE, infra/__init__.py:538-541), which is exactly the transformers layout --
    # this backend has no TP/PP/CP (validate_configs rejects those sizes here). It is NOT optional
    # bookkeeping though: a None mesh silently changes the training objective to an avg-of-avg, so it
    # is load-bearing that run_sft calls twinkle.initialize first (see _initialize_twinkle, and the
    # 2-GPU aggregation test that pins the result).
    #
    # Ray placement (RL): under mode='ray' the model lives in a remote DeviceGroup named 'model' --
    # the same group _initialize_twinkle builds -- mirroring the Megatron branch below. This is what
    # an online RL recipe needs so the trainer and a vLLMSampler are SEPARATE Ray actors that
    # CheckpointEngineManager can weight-sync between (it asserts both have `_actors`+`device_mesh`).
    _apply_ray_placement(kwargs, distributed_config)

    # tuner_backend='unsloth' swaps the class: unsloth owns both construction (its Triton kernels /
    # optional 4bit base) and LoRA installation -- see swift/dev/model/unsloth_model.py. Everything
    # derived above (dtype, strategy, mixed_precision, ddp_config) is passed through unchanged.
    if tuner_config is not None and tuner_config.tuner_backend == 'unsloth':
        from swift.dev.model import UnslothModel
        _apply_unsloth_kwargs(kwargs, model_config, tuner_config, train_config)
        model = UnslothModel(**kwargs)
    else:
        # Full takeover: hand the resolved family loader to twinkle, which then builds config/
        # processor/model through it. seq_cls/reranker are excluded -- their num_labels head overrides
        # model_cls, which a family (causal-LM) loader would not build; they keep the
        # AutoModelForSequenceClassification path applied above.
        if model_loader is not None and model_config.task_type not in ('seq_cls', 'reranker'):
            kwargs['model_loader'] = model_loader
        model = TransformersModel(**kwargs)

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


def _apply_mtp_kwargs(kwargs: dict, model_config: ModelConfig) -> None:
    """Forward the Multi-Token Prediction knobs into mcore-bridge's ModelConfig.

    All five land on the same object (``get_model_config`` forwards **kwargs verbatim), so they are
    grouped here rather than mixed into the recompute/attention block above.

    ``mtp_num_layers`` gates the rest: without it the bridge builds no MTP block at all, and
    mcore-bridge's own ``__post_init__`` rejects the other knobs rather than ignoring them. Each is
    forwarded only when set, so an MTP-free run reaches the bridge with exactly the kwargs it had
    before this existed -- an unset ``mtp_loss_scaling_factor`` has to stay unset for mcore's own
    default (0.1) to apply, and passing None would override it with None.
    """
    if model_config.mtp_num_layers is None:
        return
    kwargs['mtp_num_layers'] = model_config.mtp_num_layers
    if model_config.mtp_loss_scaling_factor is not None:
        kwargs['mtp_loss_scaling_factor'] = model_config.mtp_loss_scaling_factor
    if model_config.enable_mtp_training:
        kwargs['enable_mtp_training'] = True
    if model_config.mtp_freeze:
        kwargs['mtp_freeze'] = True
    if model_config.mtp_decoder_input_detach:
        kwargs['mtp_decoder_input_detach'] = True


def _apply_fp4_kwargs(kwargs: dict, model_config: ModelConfig) -> None:
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


def _apply_fp8_kwargs(kwargs: dict, model_config: ModelConfig) -> None:
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


def _apply_fsdp_kwargs(kwargs: dict, distributed_config: DistributedConfig) -> None:
    """Forward the Megatron-FSDP switch, which travels inside ddp_config rather than on its own.

    Unlike the other knobs here this is not a MegatronModel argument: twinkle reads
    ``ddp_config['use_megatron_fsdp']`` to pick WHICH data-parallel class wraps the model, and then
    hands the same dict to megatron's DistributedDataParallelConfig, which declares a field of that
    name. One key, two readers -- which is why it cannot simply be passed as a top-level kwarg.

    Only set when enabled, so a DDP run reaches twinkle with no ddp_config at all, exactly as it did
    before this existed.
    """
    if not distributed_config.use_megatron_fsdp:
        return
    kwargs['ddp_config'] = {'use_megatron_fsdp': True}


def _resolve_bridge_backend(name: str):
    """bridge_backend name (DistributedConfig default: 'mcore-bridge') -> a BridgeBackend instance."""
    from swift.dev.model.megatron.bridge import MCoreBridgeBackend, MegatronBridgeBackend
    key = name.lower()
    if key == 'mcore-bridge':
        return MCoreBridgeBackend()
    if key == 'megatron-bridge':
        return MegatronBridgeBackend()
    raise NotImplementedError(f"Unknown bridge_backend {name!r}. Known: 'mcore-bridge', 'megatron-bridge'.")


def build_device_mesh(distributed_config: DistributedConfig):
    """DistributedConfig -> the Megatron DeviceMesh (parallel layout).

    A pure function of the config: it reads only the declared sizes, never torch.distributed or
    Megatron's mpu, so it can be called before anything is initialized.
    """
    from twinkle import DeviceMesh

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
    # Megatron TP sequence-parallelism rides on the DeviceMesh (twinkle reads it via
    # strategy.sequence_parallel -> device_mesh.sequence_parallel). Only meaningful with tp > 1.
    if distributed_config.sequence_parallel:
        mesh_kwargs['sequence_parallel'] = True
    return DeviceMesh.from_sizes(**mesh_kwargs)


def _build_megatron_model(model_config: ModelConfig, distributed_config: DistributedConfig) -> TrainableModel:
    """Build a MegatronModel via the selected bridge backend.

    twinkle must already be initialized in Ray mode (run_sft does this) so the 'model' DeviceGroup
    exists. The world size is DistributedConfig.nproc_per_node (== the DeviceGroup size); dp_size is
    derived from it and the model-parallel sizes. Driver-side dist.get_world_size() is NOT used --
    in Ray mode the driver is not part of the model process group (its world size is 1).
    """
    from swift.dev.model.megatron.model import MegatronModel

    if not model_config.model:
        raise ValueError('ModelConfig.model (path/id) is required')

    device_mesh = build_device_mesh(distributed_config)

    mixed_precision = _mixed_precision_for(model_config.torch_dtype)

    # A few high-frequency Megatron knobs flow straight into MegatronModel.__init__. Forward
    # only when set (None -> twinkle's own default), so the bit-exact SFT baseline is unchanged
    # unless the user opts in. use_distributed_optimizer has a real default (True) so pass it.
    extra_kwargs: dict = {'use_distributed_optimizer': distributed_config.use_distributed_optimizer}
    # name is a dynamic attribute -> getattr is required here (not defensive over-protection).
    for name in ('recompute_granularity', 'recompute_method', 'recompute_num_layers'):
        value = getattr(distributed_config, name)
        if value is not None:
            extra_kwargs[name] = value

    if extra_kwargs.get('recompute_granularity') == 'selective':
        extra_kwargs['recompute_num_layers'] = None
        extra_kwargs['recompute_method'] = None

    _apply_fsdp_kwargs(extra_kwargs, distributed_config)

    # Attention kernel. Always forwarded (unlike the recompute knobs above) because the meaningful
    # default is legacy's 'flash', not mcore's AttnBackend.auto -- under auto TE picks per shape and
    # selects the FUSED cuDNN kernel for a Qwen2.5 bf16 causal THD forward, so leaving it unset makes
    # dev and legacy run different attention kernels on identical config. resolve_* returns the enum
    # (mcore compares by identity), and both bridge backends now read this one value instead of the
    # megatron-bridge path hardcoding its own.
    from swift.dev.naming import resolve_megatron_attn_backend
    extra_kwargs['attention_backend'] = resolve_megatron_attn_backend(model_config.attn_impl)
    # A flash_N / flash_attention_N value also pins the FA VERSION, which is enforced by mutating
    # transformer_engine module globals -- a per-process side effect, so it CANNOT be applied here:
    # in Ray mode build_model runs on the driver, which is not where the model is built. The raw
    # string is forwarded so DevMegatronStrategy (worker-side) can apply the pin itself.
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

    _apply_mtp_kwargs(extra_kwargs, model_config)
    _apply_fp4_kwargs(extra_kwargs, model_config)
    _apply_fp8_kwargs(extra_kwargs, model_config)

    backend = _resolve_bridge_backend(distributed_config.bridge_backend)
    # In Ray mode the model lives in a remote DeviceGroup named 'model'; in local (torchrun) mode
    # each rank builds the model in-process, so there is no remote group to target.
    model_kwargs = dict(
        model_id=model_config.model,
        device_mesh=device_mesh,
        mixed_precision=mixed_precision,
        backend=backend,
        **extra_kwargs)
    if distributed_config.mode != 'local':
        model_kwargs['remote_group'] = 'model'
    return MegatronModel(**model_kwargs)
