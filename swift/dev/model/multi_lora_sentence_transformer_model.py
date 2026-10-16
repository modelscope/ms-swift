from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import torch.distributed as dist
from peft.utils import load_peft_weights
from safetensors.torch import save_file
from transformers import PretrainedConfig, PreTrainedModel

from twinkle import DeviceMesh, Platform, remote_class, remote_function
from twinkle.hub import HubOperation
from twinkle.infra import get_device_mesh
from twinkle.model.base import ModelLoaderProtocol, copy_checkpoint_args, rotate_checkpoints
from twinkle.model.multi_lora import MultiLora
from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel
from twinkle.model.transformers.transformers import OptimizerGroup

from .sentence_transformer_model import SentenceTransformerModel


@remote_class()
class MultiLoraSentenceTransformerModel(MultiLoraTransformersModel, SentenceTransformerModel):
    """A sentence-transformers encoder served to several LoRA tenants at once.

    Both parents are needed, and neither alone is enough:
      - ``SentenceTransformerModel`` (the single-adapter embedding model) builds the ST pipeline and runs
        it: the ``Transformer -> Pooling -> Normalize`` chain produces the sentence embeddings itself,
        and its checkpointing is ST's own whole-pipeline format;
      - ``MultiLoraTransformersModel`` owns the multi-tenant LoRA machinery: pre-allocated PEFT slots,
        per-tenant optimizer groups and adapter checkpoints, tenant validation, then wraps the training
        methods (``forward``/``forward_only``/``calculate_loss``/``backward``/``step``...) in an adapter
        context so one base model serves many tenants.

    Layout chosen here (differs from both parents):
      - ``self.model`` stays the ST pipeline, which is the contract the inherited ``forward`` and
        ``forward_only`` rely on when they call ``SentenceTransformerModel._encode_sentence_embeddings``;
      - the LoRA slots are installed on the pipeline's HF backbone, and the resulting ``PeftModel`` is
        written back into the leading ``Transformer`` module (``pipeline[0].model``) so the ST module
        chain routes through the LoRA layers. ``MultiLora`` holds that same object as
        ``self.multi_adapter.module``. The backbone is what gets patched because ``peft.get_peft_model``
        only accepts a transformers ``PreTrainedModel``, while the pipeline is an ``nn.Sequential``.

    Only the members the layout above invalidates are overridden: ``__init__`` (build the ST pipeline,
    then patch the backbone), ``save``/``load`` (the inherited bodies look for the ``PeftModel`` on
    ``self.model``, which holds the pipeline here), and ``register_global_mm_forward_hook``. Everything
    else is inherited unchanged -- the tenant lifecycle (``add_adapter_to_model``/``remove_adapter``),
    the adapter-context wrappers, the optimizer/scheduler/step surface, and the ST encoding path.

    Scope: the standard (non-sequence-parallel) deployment of a text or multimodal ST model. Sequence
    parallel is rejected by ``SentenceTransformerModel`` because the pooling head reads whole sentences.

    Args:
        model_id: A sentence-transformers model id or local directory. Required: ST models are loaded by
            ``SentenceTransformer(model_id)``, never built from a config.
        config: Optional pretrained config, taken as ``hf_config`` as-is. Without it, ``hf_config`` comes
            from ``model_loader`` when one is given, otherwise from the loaded backbone.
        model_loader: The loader seam ``MultiLoraTransformersModel`` uses. Its config/processor hooks
            stay wired the same way, but ``build_model`` is not called: the ST pipeline owns the model
            construction.
        sentence_transformer_kwargs: Extra keyword arguments forwarded to ``SentenceTransformer``.
        gradient_checkpointing / gradient_checkpointing_kwargs: Applied to the backbone through
            ``SentenceTransformerModel._enable_gradient_checkpointing`` (the ST pipeline has no top-level
            gradient-checkpointing entry point).
        init_strategy: Re-initialize the weights a partial checkpoint left uninitialized; applied to the
            pipeline before the LoRA slots are installed.
        The remaining arguments (``device_mesh``, ``mixed_precision``, ``strategy``, ``ddp_config``,
        ``fsdp_config``, ``lora_config``, ``grad_scaler_config``, ``memory_efficient_init``,
        ``max_loras``, ``max_r``, ``max_length``, ``target_modules``, ``acc_strategy``, ``model_arch``)
        keep the meaning ``MultiLoraTransformersModel`` gives them.
    """

    def __init__(self,
                 model_id: Optional[str] = None,
                 config: Optional[PretrainedConfig] = None,
                 device_mesh: Optional[DeviceMesh] = None,
                 mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
                 strategy: Literal['accelerate', 'native_fsdp'] = 'accelerate',
                 ddp_config: Optional[Dict[str, Any]] = None,
                 fsdp_config: Optional[Dict[str, Any]] = None,
                 lora_config: Optional[Dict[str, Any]] = None,
                 grad_scaler_config: Optional[Dict[str, Any]] = None,
                 memory_efficient_init: bool = False,
                 max_loras: int = 5,
                 max_r: int = 32,
                 max_length: int = 8192,
                 target_modules: Union[List[str], str] = 'all-linear',
                 model_loader: Optional[ModelLoaderProtocol] = None,
                 sentence_transformer_kwargs: Optional[Dict[str, Any]] = None,
                 gradient_checkpointing: bool = True,
                 gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None,
                 init_strategy: Optional[str] = None,
                 acc_strategy: str = 'token',
                 model_arch: Optional[Dict[str, Any]] = None,
                 **kwargs):
        # This __init__ overrides both parents' (the ST one builds the pipeline, the multi-LoRA one
        # patches an AutoModel backbone), so it carries the union of the state they set up: the strategy
        # plumbing and slot bookkeeping of MultiLoraTransformersModel, plus the ST construction helpers
        # of SentenceTransformerModel, in an order where each piece sees what it needs.
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self._try_init_process_group()
        super(PreTrainedModel, self).__init__()
        self._default_tokenizer = None
        # This __init__ overrides the @remote_class-wrapped base, so it does NOT receive the base's
        # automatic device_mesh injection; resolve the global default here to uphold the same invariant
        # (a constructed model holds a real mesh) the wrapped TransformersModel.__init__ guarantees.
        self.device_mesh = device_mesh if device_mesh is not None else get_device_mesh()
        self.mixed_precision = mixed_precision
        self._fsdp_config = dict(fsdp_config or {})
        self._ddp_config = ddp_config or {}
        self._memory_efficient_init = memory_efficient_init
        # Router replay is a MoE-causal-LM concept; irrelevant to sentence encoders.
        self._router_replay_enabled = False
        self._router_replay_applied = False
        # ``_acc_strategy`` / ``model_arch`` are read by inherited TransformersModel code: the former by
        # ``_construct_default_optimizer_group`` (reached from ``_register_adapter``, i.e. every
        # ``add_adapter_to_model``), the latter by the Muon/GaLore param-group builder. Neither parent's
        # fork forwards them, so this one sets both itself, with the base's role normalization.
        self._acc_strategy = acc_strategy
        self.model_arch = dict(model_arch or {})
        for _role in ('language_model', 'vision_tower', 'aligner'):
            _value = self.model_arch.get(_role)
            self.model_arch[_role] = [_value] if isinstance(_value, str) else list(_value or [])
        self._decide_strategy(strategy)
        self.grad_scaler_config = grad_scaler_config
        self.lora_config = lora_config
        if model_id is not None:
            model_id = HubOperation.download_model(model_id)
        self.model_id = model_id
        # The ST pipeline owns construction, so ``model_loader.build_model`` is skipped; its config and
        # processor hooks stay wired the same way MultiLoraTransformersModel uses them.
        self.model = self._build_sentence_transformer(model_id, sentence_transformer_kwargs or {})
        if config is not None:
            self.hf_config = config
        elif model_loader is not None and model_id is not None:
            self.hf_config = model_loader.process_config(model_loader.build_config(model_id))
        else:
            self.hf_config = self._backbone_config()
        if model_loader is not None and model_id is not None:
            self._default_tokenizer = model_loader.process_tokenizer(
                model_loader.build_processor(model_id, self.hf_config))
        if gradient_checkpointing:
            self._enable_gradient_checkpointing(gradient_checkpointing_kwargs)
        if init_strategy is not None:
            self._apply_init_strategy(init_strategy)

        self.tokenizer_id = kwargs.get('tokenizer_id', self.model_id)
        self.sp_strategy = None
        self._model_wrapped = False
        self.optimizer_group: Dict[str, OptimizerGroup] = {}
        self.multi_adapter = MultiLora(max_loras=max_loras, max_r=max_r, max_length=max_length)
        # Patch the backbone in place inside the pipeline and write it back: ``Transformer.forward``
        # dispatches through this attribute (its ``auto_model`` property is the same object), so the ST
        # module chain keeps working and now runs the LoRA-augmented backbone.
        transformer_module = self._first_module()
        transformer_module.model = self.multi_adapter.patch(
            transformer_module.model, target_modules=target_modules, lora_config=self.lora_config)
        # PEFT creates LoRA parameters in FP32 even when the base model uses a lower-precision dtype.
        # Align the preallocated slots before FSDP2 records their parameter and gradient dtypes.
        self._ensure_lora_dtype(self.model)
        self.multi_adapter.save_initial_weights()
        # Active group for compatibility with single adapter
        self.active_group = None
        self.handler = self.register_global_mm_forward_hook()
        self.multi_adapter.reset_adapter_status()

    def register_global_mm_forward_hook(self):
        """Do not install MultiLoraTransformersModel's global multimodal pre-forward hook.

        That hook reads ``model.active_adapters[0]`` off the base model to find the tenant whose template
        should prepare the batch. Here the base model is the ST pipeline, and the pipeline's PEFT API
        raises instead of answering: sentence-transformers' ``active_adapters`` delegates to the wrapped
        backbone only when it is a transformers ``PeftAdapterMixin``, which the ``peft`` ``PeftModel`` in
        the ``Transformer`` module is not. The hook would also only ever see HF-forward-shaped calls,
        while the pipeline forwards a features dict. Per-tenant templates keep working through the
        template lookup inside ``forward``. A multimodal ST deployment that needs a tenant-aware
        pre-forward hook needs a pipeline-shaped one, which is deliberately left out for now.
        """
        return None

    # lazy_collect=False: see TransformersModel.save -- a checkpoint write the caller acts on immediately
    # must block until it lands, or the run's final save is torn down with the actor before writing.
    @remote_function(collect='first', lazy_collect=False)
    def save(self, name, output_dir: Optional[str] = None, interval: int = 1, **kwargs):
        """Save one tenant's adapter checkpoint.

        Why the body is spelled out instead of calling ``super().save``: the inherited chain is
        ``MultiLoraTransformersModel.save`` -> ``SentenceTransformerModel.save`` (the ST class is next in
        this class's MRO), and the ST ``save`` dumps the whole pipeline in sentence-transformers' own
        format -- single-adapter behaviour, not a tenant archive. What follows is
        ``TransformersModel.save``'s PEFT branch, with the ``PeftModel`` read from
        ``multi_adapter.module`` (on ``MultiLoraTransformersModel`` that object is ``self.model``; here
        ``self.model`` is the ST pipeline). ``save_context`` swaps the tenant's ``LoraConfig`` into
        ``peft_config`` and restores the slot bookkeeping afterwards.

        Args:
            name: The name of checkpoint to save.
            output_dir: An output_dir to save the model.
            interval: Save each interval steps.
            **kwargs:
                adapter_name: The tenant to save. Required: a multi-LoRA deployment only ever saves the
                    adapters registered under tenant names.
                save_optimizer: Whether to save optimizer state.
                safe_serialization: Save weights with safetensors instead of torch serialization.
                max_shard_size: Rejected for adapter checkpoints (see below).
                save_total_limit: Maximum number of completed checkpoint directories to retain.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        self._check_adapter_valid(adapter_name)
        safe_serialization = kwargs.pop('safe_serialization', True)
        max_shard_size = kwargs.pop('max_shard_size', '5GB')
        save_total_limit = kwargs.pop('save_total_limit', None)
        with self.multi_adapter.save_context(adapter_name):
            optimizer_config = self.optimizer_group[adapter_name]
            if name is None:
                name = f'checkpoint-step-{optimizer_config.cur_step}'
            if output_dir is None:
                output_dir = 'output'
            checkpoint_dir = os.path.join(output_dir, name)
            if optimizer_config.cur_step % interval != 0:
                return
            # ``TransformersModel.save``'s ``adapter_name == _default_adapter_name`` full-model branch is
            # unreachable here: this class only ever registers tenant adapters, and
            # ``_check_adapter_valid`` rejects every other name.
            processed_state_dict = self._get_adapter_state_dict_for_save(adapter_name)
            if max_shard_size != '5GB':
                raise NotImplementedError(
                    'max_shard_size is not supported for PEFT adapter checkpoints; remove the option or save a '
                    'full-model checkpoint.')
            if Platform.is_master():
                self.multi_adapter.module.peft_config[adapter_name].save_pretrained(checkpoint_dir)
                contiguous_dict = {k: v.contiguous() for k, v in processed_state_dict.items()}
                if safe_serialization:
                    save_file(contiguous_dict, os.path.join(checkpoint_dir, 'adapter_model.safetensors'))
                else:
                    torch.save(contiguous_dict, os.path.join(checkpoint_dir, 'adapter_model.bin'))

            self._save_tokenizer(checkpoint_dir, adapter_name=adapter_name)

            if kwargs.get('save_optimizer', False):
                self._save_training_state(
                    checkpoint_dir,
                    adapter_name=adapter_name,
                    consumed_train_samples=kwargs.get('consumed_train_samples', 0),
                    component_states=kwargs.get('component_states', {}),
                )

            copy_checkpoint_args(output_dir, checkpoint_dir)
            rotate_checkpoints(output_dir, checkpoint_dir, save_total_limit)
        if dist.is_initialized():
            dist.barrier()
        return checkpoint_dir

    @remote_function()
    def load(self, name: str, output_dir: Optional[str] = None, **kwargs):
        """Load one tenant's LoRA weights and, on request, its training state.

        Same shape as ``MultiLoraTransformersModel.load``, except that the
        ``isinstance(self.model, PeftModel)`` gate is dropped: the ``PeftModel`` lives on
        ``multi_adapter.module`` (``self.model`` is the ST pipeline), and ``MultiLora.set_state_dict``
        resolves the tenant's slot itself.

        Args:
            name: The name of checkpoint to load.
            output_dir: An output_dir to load the model.
            **kwargs:
                adapter_name: The tenant to load into.
                load_optimizer: Whether to load optimizer and scheduler states.
        """
        adapter_name = kwargs.get('adapter_name')
        self._check_adapter_valid(adapter_name)
        with self.multi_adapter.save_context(adapter_name):
            load_optimizer = kwargs.get('load_optimizer', False)
            if output_dir is None:
                if os.path.exists(name):
                    checkpoint_dir = name
                else:
                    token = kwargs.pop('token', None)
                    checkpoint_dir = HubOperation.download_model(name, token=token)
            else:
                checkpoint_dir = os.path.join(output_dir, name)
            # Load to CPU to avoid safetensors device issues in Ray environment
            adapter_weights = load_peft_weights(checkpoint_dir, device='cpu')
            self.multi_adapter.set_state_dict(adapter_name, adapter_weights)

            if load_optimizer:
                self._restore_training_state(checkpoint_dir, adapter_name=adapter_name)
        if dist.is_initialized():
            dist.barrier()
