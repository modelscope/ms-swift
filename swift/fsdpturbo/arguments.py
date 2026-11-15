# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import dataclass
from typing import Literal, Optional

from swift.arguments import SftArguments


@dataclass
class FSDPTurboArguments:
    """Topology owned by the independent FSDPTurbo training backend."""

    fsdp_size: int = 1
    tp_size: int = 1
    cp_size: int = 1
    cp_implementation: Literal['auto', 'ulysses', 'kv_allgather', 'kcp'] = 'auto'
    ep_size: int = 1
    efsdp_size: int = 1
    pp_size: int = 1
    ep_dispatcher: Literal['eager', 'fused', 'mc2', 'domino'] = 'eager'
    fsdp_implementation: Literal['native', 'custom'] = 'native'
    offload_params: bool = False
    offload_activations: bool = False
    pin_memory: bool = True
    host_pin_memory: Optional[bool] = None
    forward_prefetch: int = 1
    backward_prefetch: int = 1

    def validate_fsdpturbo(self, world_size: Optional[int] = None) -> None:
        sizes = {
            'fsdp_size': self.fsdp_size,
            'tp_size': self.tp_size,
            'cp_size': self.cp_size,
            'ep_size': self.ep_size,
            'efsdp_size': self.efsdp_size,
            'pp_size': self.pp_size,
        }
        for name, size in sizes.items():
            if size < 1:
                raise ValueError(f'{name} must be positive, got {size}.')
        if self.cp_implementation not in ('auto', 'ulysses', 'kv_allgather', 'kcp'):
            raise ValueError(f'Unknown FSDPTurbo context-parallel implementation: {self.cp_implementation!r}.')
        if self.pp_size != 1:
            raise ValueError('FSDPTurbo main does not implement pipeline parallelism; pp_size must remain 1. '
                             'Adding PP outside the backend would build FSDP/EP meshes across pipeline stages.')
        if self.efsdp_size > 1 and self.ep_size == 1:
            raise ValueError('efsdp_size > 1 requires ep_size > 1 in FSDPTurbo.')
        if self.tp_size > 1 and self.fsdp_implementation != 'native':
            raise ValueError('FSDPTurbo TP requires the native composable FSDP implementation.')
        if self.forward_prefetch < 0 or self.backward_prefetch < 0:
            raise ValueError('FSDPTurbo prefetch counts must be non-negative.')

        if world_size is None and 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
        if world_size is None:
            return
        if world_size < 1:
            raise ValueError(f'world_size must be positive, got {world_size}.')

        dense_product = self.fsdp_size * self.tp_size
        expert_product = self.efsdp_size * self.ep_size
        if world_size % dense_product:
            raise ValueError(f'world_size ({world_size}) must be divisible by fsdp_size * tp_size ({dense_product}).')
        if world_size % expert_product:
            raise ValueError(f'world_size ({world_size}) must be divisible by efsdp_size * ep_size ({expert_product}).')
        if world_size % (self.cp_size * self.tp_size):
            raise ValueError('world_size must be divisible by cp_size * tp_size.')


@dataclass
class FSDPTurboSftArguments(FSDPTurboArguments, SftArguments):
    tuner_type: Literal['full', 'lora'] = 'full'
    add_version: bool = False
    optim: str = 'adamw_torch'

    def _validate_model_scope(self):
        if self.model_type == 'deepseek_v4':
            raise ValueError('DeepSeek V4 is no longer supported by the FSDPTurbo backend.')
        if self.model_type == 'deepseek_v41':
            if self.resume_from_checkpoint and self.tuner_type != 'lora':
                raise ValueError('Native V4.1 checkpoint resume is not supported.')
            if self.save_only_model and self.tuner_type == 'full':
                raise ValueError('Native V4.1 save_only_model is not supported: Host Engram requires its '
                                 'model-owned checkpoint state.')

    def _init_device(self):
        self._validate_model_scope()
        super()._init_device()
        from swift.utils import init_process_group, is_dist

        if not self.offload_params or not is_dist():
            return
        import torch
        import torch.distributed as dist

        device = torch.accelerator.current_accelerator()
        if device is None or device.type == 'cpu':
            return
        accelerator_backend = dist.get_default_backend_for_device(device)
        if self.ddp_backend not in (None, accelerator_backend):
            raise ValueError('FSDPTurbo CPU offload requires ddp_backend to be unset or use '
                             f'the accelerator backend ({accelerator_backend}).')
        if dist.is_initialized():
            backends = dict(item.split(':') for item in dist.get_backend_config().split(','))
            if 'cpu' not in backends or device.type not in backends:
                raise ValueError('FSDPTurbo CPU offload requires a process group with both CPU and accelerator '
                                 'backends; initialize that process group before constructing training arguments.')
            return
        # Accelerate otherwise initializes an accelerator-only group. Mesh
        # subgroups must also reduce CPU gradient norms after FSDP offload.
        init_process_group(backend=f'cpu:gloo,{device.type}:{accelerator_backend}', timeout=self.ddp_timeout)

    def __post_init__(self) -> None:
        self.validate_fsdpturbo()
        self._validate_model_scope()
        if self.enable_npu_model_patch:
            raise ValueError('FSDPTurbo requires `--enable_npu_model_patch false` before Swift model imports.')
        if self.add_version:
            raise ValueError('FSDPTurbo requires a shared explicit output_dir; `--add_version` must be false.')
        if self.resume_only_model:
            raise ValueError('FSDPTurbo resumes model and optimizer state; resume_only_model is not supported.')
        if self.resume_from_checkpoint and self.ignore_data_skip:
            raise ValueError('FSDPTurbo resume restores the data cursor; ignore_data_skip is not supported.')
        if self.fsdp:
            raise ValueError('`--fsdp` selects the HF/Accelerate backend and cannot be combined with FSDPTurbo.')
        if self.deepspeed:
            raise ValueError('DeepSpeed cannot be combined with the independent FSDPTurbo backend.')
        if self.tuner_type not in ('full', 'lora'):
            raise ValueError('FSDPTurbo supports full SFT or vanilla PEFT LoRA.')
        if self.sequence_parallel_size > 1:
            raise ValueError('Use `--tp_size` for FSDPTurbo; Swift sequence parallel is a different backend.')
        if self.padding_free or self.packing:
            raise ValueError('FSDPTurbo padding-free and packing paths have not been validated yet.')
        super().__post_init__()
        if self.tuner_type == 'lora':
            if self.tuner_backend != 'peft' or self.use_swift_lora:
                raise ValueError('FSDPTurbo LoRA uses the PEFT backend.')
            if self.resume_from_checkpoint and self.model_type != 'deepseek_v41':
                raise ValueError('FSDPTurbo LoRA optimizer resume is not supported; use --adapters for a warm start.')
            if (self.target_parameters or self.modules_to_save or self.use_dora or self.lorap_lr_ratio is not None
                    or self.lora_bias != 'none' or self.init_weights not in (True, 'gaussian')):
                raise ValueError('FSDPTurbo LoRA supports normally initialized linear adapters with bias=none; '
                                 'target_parameters, modules_to_save, DoRA and LoRA+ are not supported.')
        self._validate_model_scope()
        self.validate_fsdpturbo()
        if self.task_type != 'causal_lm':
            raise ValueError('The initial FSDPTurbo backend supports causal-LM SFT only.')
        if self.max_steps <= 0:
            raise ValueError('The initial FSDPTurbo backend requires `--max_steps` to be a positive integer.')
        if self.gradient_accumulation_steps < 1:
            raise ValueError('gradient_accumulation_steps must be positive.')
        native_adapter_accumulation = self.tuner_type == 'lora' and self.model_type == 'deepseek_v41'
        if self.gradient_accumulation_steps > 1 and self.offload_params and not native_adapter_accumulation:
            raise ValueError('FSDPTurbo CPU parameter offload requires gradient_accumulation_steps=1, except '
                             'native V4.1 LoRA whose adapter gradients remain replicated on the accelerator.')
        if self.streaming:
            raise ValueError('The initial FSDPTurbo backend requires a map-style dataset, not streaming data.')
        save_strategy = getattr(self.training_args.save_strategy, 'value', self.training_args.save_strategy)
        if save_strategy not in ('no', 'steps', 'epoch'):
            raise ValueError('FSDPTurbo supports save_strategy=no, steps or epoch.')
        if self.optimizer is not None or self.use_galore:
            raise ValueError('The initial FSDPTurbo backend supports its native AdamW optimizer only.')
        optim = getattr(self.training_args.optim, 'value', self.training_args.optim)
        if optim != 'adamw_torch':
            raise ValueError(f'FSDPTurbo supports native AdamW (`--optim adamw_torch`) only, got {optim!r}.')
        if self.loss_type is not None or self.enable_dft_loss or self.enable_channel_loss:
            raise ValueError('Swift custom loss plugins are not supported by the initial FSDPTurbo backend.')
        if self.label_smoothing_factor:
            raise ValueError('FSDPTurbo label smoothing has not been validated yet.')
        if self.cp_size > 1 and self.router_aux_loss_coef:
            raise ValueError('FSDPTurbo CP does not yet support router auxiliary loss masks.')
        if self.use_flash_ckpt:
            raise ValueError('DLRover flash checkpoints cannot be combined with FSDPTurbo checkpoints.')
        if self.quant_method is not None:
            raise ValueError('Do not set quant_method for FSDPTurbo; FP8/FP4 checkpoints are dequantized on rank 0 '
                             'and trained as floating-point weights.')
        if self.experts_impl is not None:
            raise ValueError('FSDPTurbo owns the expert implementation; leave `--experts_impl` unset.')
        if self.group_by_length:
            raise ValueError('FSDPTurbo group-by-length sampling has not been implemented yet.')
        if self.vit_gradient_checkpointing != self.gradient_checkpointing:
            raise ValueError('FSDPTurbo currently uses one recompute setting for both language and vision blocks.')
        if self.attn_impl is None:
            self.attn_impl = 'eager'
        elif self.attn_impl != 'eager':
            raise ValueError('The FSDPTurbo model plans require `--attn_impl eager`.')
