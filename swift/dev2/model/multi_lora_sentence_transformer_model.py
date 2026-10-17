# Copyright (c) ModelScope Contributors. All rights reserved.
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

    ``SentenceTransformerModel`` builds and runs the ST pipeline; ``MultiLoraTransformersModel``
    owns the multi-tenant LoRA machinery. ``self.model`` stays the ST pipeline; the LoRA slots go on
    its HF backbone (``get_peft_model`` rejects the ``nn.Sequential`` pipeline) with the ``PeftModel``
    written back into ``pipeline[0].model``. Only ``__init__``/``save``/``load``/the mm hook differ.
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
        # Overrides both parents' __init__ (the ST one builds the pipeline, the multi-LoRA one
        # patches an AutoModel backbone), so it carries the union of the state they set up.
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self._try_init_process_group()
        super(PreTrainedModel, self).__init__()
        self._default_tokenizer = None
        # Overrides the @remote_class-wrapped base, so no automatic device_mesh injection; resolve
        # the global default to uphold the "a constructed model holds a real mesh" invariant.
        self.device_mesh = device_mesh if device_mesh is not None else get_device_mesh()
        self.mixed_precision = mixed_precision
        self._fsdp_config = dict(fsdp_config or {})
        self._ddp_config = ddp_config or {}
        self._memory_efficient_init = memory_efficient_init
        # Router replay is a MoE-causal-LM concept; irrelevant to sentence encoders.
        self._router_replay_enabled = False
        self._router_replay_applied = False
        # ``_acc_strategy`` / ``model_arch`` are read by inherited TransformersModel code but
        # forwarded by neither parent's fork; set both with the base's role normalization.
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
        # Patch the backbone in place inside the pipeline and write it back: Transformer.forward
        # dispatches through this attribute, so the ST chain now runs the LoRA-augmented backbone.
        transformer_module = self._first_module()
        transformer_module.model = self.multi_adapter.patch(
            transformer_module.model, target_modules=target_modules, lora_config=self.lora_config)
        # PEFT creates LoRA parameters in FP32 even when the base model uses a lower-precision dtype;
        # align the preallocated slots before FSDP2 records their parameter and gradient dtypes.
        self._ensure_lora_dtype(self.model)
        self.multi_adapter.save_initial_weights()
        # Active group for compatibility with single adapter
        self.active_group = None
        self.handler = self.register_global_mm_forward_hook()
        self.multi_adapter.reset_adapter_status()

    def register_global_mm_forward_hook(self):
        """Do not install MultiLoraTransformersModel's global multimodal pre-forward hook.

        That hook reads ``model.active_adapters[0]`` off the base model; here the base model is the
        ST pipeline, whose PEFT API raises instead of answering, and the pipeline forwards a
        features dict the HF-shaped hook would never match. Per-tenant templates keep working
        through the template lookup inside ``forward``.
        """
        return None

    # lazy_collect=False: see TransformersModel.save -- a checkpoint write the caller acts on
    # immediately must block until it lands, or the run's final save is torn down with the actor.
    @remote_function(collect='first', lazy_collect=False)
    def save(self, name, output_dir: Optional[str] = None, interval: int = 1, **kwargs):
        """Save one tenant's adapter checkpoint (``TransformersModel.save``'s PEFT branch, mirrored).

        The inherited chain (MLTFM.save -> ST.save, next in this class's MRO) would dump the whole
        pipeline in ST's own format -- single-adapter behaviour, not a tenant archive. The tenant
        ``LoraConfig`` is captured BY REFERENCE inside ``save_context`` (the async writer must not
        re-read ``peft_config`` after the context has exited).
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        self._check_adapter_valid(adapter_name)
        safe_serialization = kwargs.pop('safe_serialization', True)
        max_shard_size = kwargs.pop('max_shard_size', '5GB')
        save_total_limit = kwargs.pop('save_total_limit', None)
        async_save = kwargs.pop('async_save', False)
        with self.multi_adapter.save_context(adapter_name):
            optimizer_config = self.optimizer_group[adapter_name]
            if name is None:
                name = f'checkpoint-step-{optimizer_config.cur_step}'
            if output_dir is None:
                output_dir = 'output'
            checkpoint_dir = os.path.join(output_dir, name)
            if optimizer_config.cur_step % interval != 0:
                return
            # The full-model branch is unreachable here: this class only ever registers tenant
            # adapters, and ``_check_adapter_valid`` rejects every other name.
            processed_state_dict = self._get_adapter_state_dict_for_save(adapter_name)
            if max_shard_size != '5GB':
                raise NotImplementedError(
                    'max_shard_size is not supported for PEFT adapter checkpoints; remove the option or save a '
                    'full-model checkpoint.')
            tenant_peft_config = self.multi_adapter.module.peft_config[adapter_name]
            self._deferred_trainer_state_writes = []
            if kwargs.get('save_optimizer', False):
                self._save_training_state(
                    checkpoint_dir,
                    adapter_name=adapter_name,
                    consumed_train_samples=kwargs.get('consumed_train_samples', 0),
                    component_states=kwargs.get('component_states', {}),
                    defer_trainer_state=async_save,
                )

            def _write_weights():
                if Platform.is_master():
                    tenant_peft_config.save_pretrained(checkpoint_dir)
                    contiguous_dict = {k: v.contiguous() for k, v in processed_state_dict.items()}
                    if safe_serialization:
                        save_file(contiguous_dict, os.path.join(checkpoint_dir, 'adapter_model.safetensors'))
                    else:
                        torch.save(contiguous_dict, os.path.join(checkpoint_dir, 'adapter_model.bin'))

            def _write_tail():
                _write_weights()
                self._save_tokenizer(checkpoint_dir, adapter_name=adapter_name)
                copy_checkpoint_args(output_dir, checkpoint_dir)
                rotate_checkpoints(output_dir, checkpoint_dir, save_total_limit)
                for write in self._deferred_trainer_state_writes:
                    write()

            if async_save:
                self.submit_async_write(_write_tail)
            else:
                _write_tail()
        if dist.is_initialized():
            dist.barrier()
        return checkpoint_dir

    @remote_function()
    def load(self, name: str, output_dir: Optional[str] = None, **kwargs):
        """Load one tenant's LoRA weights and, on request, its training state.

        Same shape as ``MultiLoraTransformersModel.load``, except the ``isinstance(self.model,
        PeftModel)`` gate is dropped: the ``PeftModel`` lives on ``multi_adapter.module`` (here
        ``self.model`` is the ST pipeline), and ``MultiLora.set_state_dict`` resolves the slot.
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
