# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import os
import torch
import torch.distributed as dist
from dataclasses import replace
from fsdp_turbo.training.trainer import BaseTrainer
from functools import partial
from torch.utils.data import DataLoader, DistributedSampler
from types import SimpleNamespace

from swift.pipelines.train.tuner import TunerMixin
from swift.utils import get_logger, seed_worker
from .model_specs import FSDPTurboModelSpec, get_model_spec

logger = get_logger()


def _validate_dtype(args) -> None:
    if not (args.bf16 or args.fp16 or args.torch_dtype in (torch.bfloat16, torch.float16)):
        raise ValueError('FSDPTurbo currently requires bf16 or fp16 training.')


def _save_steps(train_args) -> int:
    if getattr(train_args.save_strategy, 'value', train_args.save_strategy) != 'steps':
        return 2**31 - 1
    if isinstance(train_args.save_steps, float) and train_args.save_steps < 1:
        return max(1, math.ceil(train_args.max_steps * train_args.save_steps))
    return max(1, int(train_args.save_steps))


def _scheduler_name(train_args) -> str:
    return getattr(train_args.lr_scheduler_type, 'value', train_args.lr_scheduler_type)


def _cp_implementation(args, spec):
    requested = getattr(args, 'cp_implementation', 'auto')
    selected = spec.cp_implementation if requested == 'auto' else requested
    if args.cp_size > 1 and selected != spec.cp_implementation:
        raise ValueError(f'{spec.model_type} supports {spec.cp_implementation} CP, not {selected}.')
    return selected


def build_fsdpturbo_config(args, spec: FSDPTurboModelSpec):
    from fsdp_turbo.fsdp_turbo_config import (CheckpointConfig, CPPlanConfig, DataConfig, DistributedConfig,
                                              EPPlanConfig, FSDPPlanConfig, FSDPTurboConfig, MemoryConfig, ModelConfig,
                                              OptimizerConfig, SwapActivationPlanConfig, TPPlanConfig, TrainRunConfig)

    train_args = args.training_args
    _validate_dtype(train_args)
    cp_size = args.cp_size
    cp_implementation = _cp_implementation(args, spec)
    native_cp = spec.native_model_factory and cp_implementation == 'kv_allgather'
    if native_cp and cp_size > 1 and args.tp_size > 1:
        raise ValueError('Native V4.1 KV-AllGather requires Ulysses=KCP=TP=1.')
    if cp_size > 1 and not (native_cp or spec.cp_function_patches):
        raise ValueError(f'FSDPTurbo CP is not implemented for {spec.model_type}. Use cp_size=1.')
    module_patches = spec.module_patches + (spec.cp_module_patches if cp_size > 1 else ())
    cp_patches = [{
        'target_functions': [target],
        'type': patch_type
    } for target, patch_type in spec.cp_function_patches] if cp_size > 1 else []
    return FSDPTurboConfig(
        module_patches=[{
            'target': target,
            'replacement': replacement
        } for target, replacement in module_patches],
        model=ModelConfig(
            model_name_or_path=args.model,
            tokenizer_name_or_path=args.model,
            torch_dtype=args.torch_dtype,
        ),
        optimizer=OptimizerConfig(
            optimizer_type='AdamW',
            weight_decay=train_args.weight_decay,
            adam_beta1=train_args.adam_beta1,
            adam_beta2=train_args.adam_beta2,
            adam_epsilon=train_args.adam_epsilon,
            lr=train_args.learning_rate,
            warmup_ratio=train_args.warmup_ratio,
            lr_scheduler_type=_scheduler_name(train_args),
            clip_grad=train_args.max_grad_norm,
        ),
        data=DataConfig(
            dataset_path=','.join((args.dataset or []) + (args.cached_dataset or [])),
            batch_size=train_args.per_device_train_batch_size,
            max_seq_length=args.max_length,
            num_workers=train_args.dataloader_num_workers,
            pin_memory=train_args.dataloader_pin_memory,
            shuffle=train_args.train_dataloader_shuffle,
        ),
        run=TrainRunConfig(
            seed=train_args.seed,
            max_steps=train_args.max_steps,
            # A positive max_steps owns termination, even for one-batch epochs.
            num_train_epochs=train_args.max_steps * train_args.gradient_accumulation_steps,
            gradient_accumulation_steps=train_args.gradient_accumulation_steps,
            logging_steps=train_args.logging_steps,
        ),
        checkpoint=CheckpointConfig(
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            save_steps=_save_steps(train_args),
            save_optim=not train_args.save_only_model,
            load_optim=not args.resume_only_model,
        ),
        distributed=DistributedConfig(
            fully_shard_parallel_size=args.fsdp_size,
            tensor_parallel_size=args.tp_size,
            ulysses_parallel_size=cp_size if cp_implementation == 'ulysses' else 1,
            kv_allgather_parallel_size=cp_size if cp_implementation == 'kv_allgather' else 1,
            kcp_parallel_size=cp_size if cp_implementation == 'kcp' else 1,
            cp_plan=CPPlanConfig(
                **{f'{cp_implementation}_function_patches': cp_patches},
                loss_function_patches=[{
                    'target_functions': ['transformers.loss.loss_utils.ForCausalLMLoss'],
                    'type': 'causal_lm_loss'
                }] if cp_size > 1 and not native_cp else [],
            ),
            expert_parallel_size=args.ep_size,
            expert_fully_shard_parallel_size=args.efsdp_size,
            fsdp_plan=FSDPPlanConfig(
                ignored_modules=[],
                apply_modules={pattern: {}
                               for pattern in spec.fsdp_modules},
                # Swift's loader already materializes ordinary weights in the
                # requested dtype while preserving numerically sensitive
                # checkpoint parameters (for example Qwen3.5 A_log) in fp32.
                # A global FSDP param cast would erase that mixed-dtype policy.
                param_dtype=None,
                reduce_dtype='fp32',
                # Preserve HF's fp32 CE/auxiliary loss; forcing the entire
                # ModelOutput to bf16 quantizes scalar losses and metrics.
                output_dtype=None,
                num_to_forward_prefetch=args.forward_prefetch,
                num_to_backward_prefetch=args.backward_prefetch,
                hook_modules=list(spec.fsdp_hook_modules),
                cpu_offload=args.offload_params,
                pin_memory=args.pin_memory,
                fsdp_implementation=args.fsdp_implementation,
            ),
            tp_plan=TPPlanConfig(
                colwise_parallel=list(spec.tp_colwise_modules),
                rowwise_parallel=list(spec.tp_rowwise_modules),
                sequence_parallel=[],
            ),
            ep_plan=EPPlanConfig(
                apply_modules=list(spec.ep_modules),
                apply_efsdp_modules=list(spec.efsdp_modules),
                dispatcher=(spec.eager_dispatcher or 'all2all') if args.ep_dispatcher == 'eager' else
                ('all2all' if args.ep_dispatcher == 'fused' else args.ep_dispatcher),
                use_fused_ops=args.ep_dispatcher != 'eager',
            ),
        ),
        memory=MemoryConfig(
            recompute=args.gradient_checkpointing,
            recompute_plan=list(spec.recompute_modules),
            swap_activation=getattr(args, 'offload_activations', False),
            swap_activation_plan=(SwapActivationPlanConfig(
                apply_modules=list(spec.recompute_modules)) if getattr(args, 'offload_activations', False) else None),
        ),
    )


class FSDPTurboTrainer(BaseTrainer):
    """A narrow Swift data/model adapter around FSDPTurbo's training lifecycle."""

    def __init__(self, args, template, processor, train_dataset, eval_dataset=None):
        self.args = args
        self.train_args = args.training_args
        self.template = template
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._eval_history = []
        self.spec = get_model_spec(args.model_type)
        self._adapter_resume = args.resume_from_checkpoint if args.tuner_type == 'lora' else None
        if self._adapter_resume:
            from .adapter_checkpoint import adapter_run_signature, read_adapter_metadata

            read_adapter_metadata(
                self._adapter_resume, topology=self._adapter_topology(), run_signature=adapter_run_signature(args))
            args.adapters = [self._adapter_resume]
        if self.spec.native_model_factory:
            if args.gradient_checkpointing and _cp_implementation(args, self.spec) != 'kv_allgather':
                raise ValueError('Native V4.1 ordinary shared runtime does not support recompute; '
                                 'use KV-AllGather stateless prefill for block recomputation.')
            if args.ep_size < 2 or args.ep_dispatcher != 'eager' or (args.resume_from_checkpoint
                                                                     and not self._adapter_resume):
                raise ValueError('Native V4.1 currently requires EP > 1, eager dispatch, and no checkpoint resume.')
            if args.router_aux_loss_coef:
                raise ValueError('Native V4.1 router auxiliary loss is not implemented.')
            if self.train_args.save_only_model and args.tuner_type == 'full':
                raise ValueError('Native V4.1 save_only_model is not supported: Host Engram requires its '
                                 'model-owned checkpoint state.')
            if (args.tuner_type == 'full' and args.offload_params and not args.pin_memory
                    and getattr(self.train_args.device, 'type', None) == 'npu'):
                raise ValueError('Native V4.1 Full SFT with NPU CPU offload requires pin_memory=true; '
                                 'unpinned full-model backward synchronization is not supported.')
        if args.tp_size > 1 and not (self.spec.tp_colwise_modules and self.spec.tp_rowwise_modules):
            raise ValueError(f'FSDPTurbo TP is not implemented for {args.model_type}. Use tp_size=1.')
        evaluation = getattr(self.train_args, 'eval_strategy', 'no')
        if getattr(evaluation, 'value', evaluation) != 'no' and eval_dataset is None:
            raise ValueError('FSDPTurbo evaluation requires a non-empty validation dataset.')
        self._replicated_parameters = []
        self._adapter_model = None
        super().__init__(config=build_fsdpturbo_config(args, self.spec))
        self.processor = processor

    def setup(self):
        super().setup()
        self._initialize_loss_group()
        steps_per_epoch = len(self.dataloader)
        if steps_per_epoch < 1:
            raise ValueError('FSDPTurbo requires at least one local training batch per epoch.')

    def _initialize_loss_group(self):
        # HCCL allocates communicator buffers lazily. Initialize the separate
        # metric-reduction group before backward fills the allocator cache.
        parallel_state = self.model.parallel_state
        if parallel_state.get_data_group_size() > 1:
            device = self.train_args.device
            probe = torch.zeros((), device=device)
            dist.all_reduce(probe, group=parallel_state.get_data_group())
            probe.item()  # Complete communicator allocation before training.

    def _initialize_native_groups(self):
        # CANN 9.2 AICPU communicator initialization can fail after large
        # pinned Host Engram allocations. Initialize every mesh group first.
        from fsdp_turbo.distributed.parallel_state import init_parallel_state

        self._native_load_group = dist.new_group(backend='gloo')
        probe = torch.zeros((), device=self.train_args.device)
        dist.all_reduce(probe)
        probe.item()
        parallel_state = init_parallel_state(self.config)
        initialized = set()
        for mesh in parallel_state.device_mesh_map.values():
            for group in mesh.get_all_groups():
                if group in initialized or dist.get_world_size(group) <= 1:
                    continue
                initialized.add(group)
                probe = torch.zeros((), device=self.train_args.device)
                dist.all_reduce(probe, group=group)
                probe.item()

    def _init_distributed(self):
        from fsdp_turbo.utils.log import set_log_level

        # Swift may already own the process group, so BaseTrainer can return
        # before configuring the backend topology and per-step metric logs.
        set_log_level('INFO')
        super()._init_distributed()
        self.args.validate_fsdpturbo(dist.get_world_size())
        if self.args.offload_params and not self.spec.native_model_factory:
            self._offload_status_group = dist.new_group(backend='gloo')

    def build_monitor(self):
        monitor = super().build_monitor()
        monitor.host_sync_group = getattr(self, '_native_load_group', getattr(self, '_offload_status_group', None))
        return monitor

    def build_tokenizer(self):
        return self.template.tokenizer

    def build_processor(self):
        return self.processor

    def build_model(self):
        from .loading import create_meta_model, load_rank0_model, materialize_resume_model, materialize_sharded_model

        native = self.spec.native_model_factory is not None
        if not native:
            source_model, dtypes = load_rank0_model(self.args)
        if native:
            # Follow the upstream native lifecycle: build, normalize dtype and
            # place modules, wrap parallelism, then stream pretrained shards.
            model = create_meta_model(self.args)
            from fsdp_turbo.models.deepseek.deepseek_v41.adapter import prepare_deepseek_v41_model_for_fsdp

            # Normalize only the base before PEFT creates adapter parameters.
            # Meta buffers stay on CPU until wrapping selects local shards.
            model = prepare_deepseek_v41_model_for_fsdp(
                model,
                device=torch.device('cpu'),
                parameter_dtype=torch.bfloat16,
                initialize_host_weights=False,
                pin_host_memory=(self.args.pin_memory
                                 if self.args.host_pin_memory is None else self.args.host_pin_memory))
        else:
            from accelerate import init_empty_weights
            with init_empty_weights(include_buffers=False):
                model = create_meta_model(self.args)
        if not native:
            for name, param in model.named_parameters():
                param.data = param.data.to(dtypes[name])
        lora = self.args.tuner_type == 'lora'
        if lora:
            from peft import LoraConfig, PeftConfig

            from swift.pipelines.train.tuner import get_target_modules

            targets = get_target_modules(self.args, model)
            adapter_config = (
                PeftConfig.from_pretrained(self.args.adapters[0]) if self.args.adapters else LoraConfig(
                    target_modules=targets))
            self.spec.validate_lora_config(model, adapter_config, tp_size=self.args.tp_size)
        model = TunerMixin.prepare_model(self.args, model, template=self.template, train_dataset=self.train_dataset)
        if lora:
            self._adapter_model = model
            model = model.get_base_model()
            if not native and source_model is not None:
                source_model = TunerMixin.prepare_model(
                    self.args, source_model, template=self.template, train_dataset=self.train_dataset).get_base_model()
        for names in self.spec.matching_modules(model, self.spec.frozen_modules).values():
            for name in names:
                model.get_submodule(name).requires_grad_(False)
        self.spec.validate_model(
            model,
            require_tp=self.args.tp_size > 1,
            require_ep=self.args.ep_size > 1,
            require_recompute=self.args.gradient_checkpointing,
        )
        parameter_names = {
            name
            for name, parameter in model.named_parameters()
            if (parameter.dtype == torch.float32 and self.args.torch_dtype != torch.float32) or (
                native and lora and parameter.requires_grad)
        }
        if parameter_names:
            self.config = replace(
                self.config,
                distributed=replace(
                    self.config.distributed,
                    fsdp_plan=replace(
                        self.config.distributed.fsdp_plan,
                        ignored_modules=list(self.config.distributed.fsdp_plan.ignored_modules),
                        ignored_params=sorted(parameter_names))))
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        if self.args.router_aux_loss_coef:
            text_config = getattr(model.config, 'text_config', model.config)
            text_config.output_router_logits = True
            text_config.router_aux_loss_coef = self.args.router_aux_loss_coef
            if hasattr(model, 'router_aux_loss_coef'):
                model.router_aux_loss_coef = self.args.router_aux_loss_coef
        self.template.model = model

        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        logger.info('Applying the independent FSDPTurbo parallel backend.')
        if native:
            self._initialize_native_groups()
        model = FSDPTurbo(self.config, model)
        if native and self.args.cp_size == 1 and self.args.gradient_checkpointing:
            model.model.set_context_parallel_group(model.parallel_state.get_cp_group(), stateless_prefill=True)
        if lora:
            unexpected = [
                name for name, parameter in model.model.named_parameters()
                if parameter.requires_grad and name not in parameter_names
            ]
            if unexpected:
                raise RuntimeError(f'Parallel wrapping unfroze base parameters during LoRA: {unexpected}')
        if native:
            from fsdp_turbo.models.deepseek.deepseek_v41.adapter import materialize_deepseek_v41_model

            device = torch.device('cpu') if self.args.offload_params else self.train_args.device
            materialize_deepseek_v41_model(model.model, device=device)
            model.model.from_pretrained(self.args.model, strict=True, engram_rows_per_chunk=65_536)
        elif self.args.resume_from_checkpoint:
            device = 'cpu' if self.args.offload_params else self.train_args.device
            materialize_resume_model(model.model, device=device)
        else:
            materialize_sharded_model(model.model, source_model, offload=self.args.offload_params)
        # DCP cpu_offload includes persistent buffers. FSDP only offloads
        # parameters; routing lookup tables and rotary buffers stay on device.
        moved_buffers = {}
        for module in model.model.modules():
            for name, buffer in module.named_buffers(recurse=False):
                if id(buffer) not in moved_buffers:
                    moved_buffers[id(buffer)] = buffer.to(self.train_args.device)
                setattr(module, name, moved_buffers[id(buffer)])
        self._replicated_parameters = [
            parameter for name, parameter in model.model.named_parameters() if name in parameter_names
        ]
        for parameter in self._replicated_parameters:
            parameter.data = parameter.data.to(self.train_args.device)
        if native and dist.is_initialized():
            # Checkpoint readers can finish minutes apart. Wait on Gloo before
            # scheduling any HCCL broadcast/loss probe, whose connect timeout
            # is shorter than a full single-host V4.1 checkpoint read.
            dist.barrier(group=self._native_load_group)
        if native and lora:
            from .loading import initialize_native_lora

            initialize_native_lora(self._adapter_model, self.args.adapters)
            # Swift seeds ranks separately; initialized adapter replicas agree.
            for parameter in self._replicated_parameters:
                dist.broadcast(parameter.data, src=0)
        self.template.model = model.model
        return model

    def _sync_replicated_gradients(self):
        if not self._replicated_parameters or not dist.is_initialized() or dist.get_world_size() == 1:
            return
        parameters = [parameter for parameter in self._replicated_parameters if parameter.requires_grad]
        if not parameters:
            return
        # All ranks participate even when a local branch was unused. Globally
        # unused replicas retain grad=None, so AdamW cannot decay them.
        present = torch.tensor([parameter.grad is not None for parameter in parameters],
                               device=parameters[0].device,
                               dtype=torch.int32)
        gradients = torch.cat([
            (parameter.grad.detach() if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1)
            for parameter in parameters
        ])
        dist.all_reduce(gradients, op=dist.ReduceOp.SUM)
        dist.all_reduce(present, op=dist.ReduceOp.SUM)
        gradients.div_(dist.get_world_size())
        for parameter, gradient, active in zip(parameters, gradients.split([p.numel() for p in parameters]),
                                               present.tolist()):
            parameter.grad = gradient.view_as(parameter) if active else None

    def _on_optimizer_step(self):
        self._sync_replicated_gradients()
        return super()._on_optimizer_step()

    def build_dataloader(self):
        if self.train_dataset is None:
            raise ValueError('FSDPTurbo requires a training dataset.')
        parallel_state = self.model.parallel_state
        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=parallel_state.get_data_group_size(),
            rank=parallel_state.get_data_rank(),
            shuffle=self.train_args.train_dataloader_shuffle,
            seed=self.args.data_seed,
            drop_last=self.train_args.dataloader_drop_last,
        )
        workers = self.train_args.dataloader_num_workers
        collator = partial(self.template.data_collator, padding_to=None)
        if self.spec.model_type == 'deepseek_v41':
            from .data import _native_cp_alignment, _native_text_collator

            collator = partial(
                _native_text_collator,
                self.template.data_collator,
                sequence_alignment=_native_cp_alignment(self.model.model.model_args,
                                                        parallel_state.get_cp_group_size()),
                pad_token_id=self.template.tokenizer.pad_token_id or 0)
        kwargs = {
            'dataset': self.train_dataset,
            'batch_size': self.train_args.per_device_train_batch_size,
            'sampler': sampler,
            'collate_fn': collator,
            'num_workers': workers,
            'pin_memory': self.train_args.dataloader_pin_memory,
            'drop_last': self.train_args.dataloader_drop_last,
            'worker_init_fn': partial(seed_worker, num_workers=workers, rank=parallel_state.get_data_rank()),
            # Iterator worker seeding must not consume the model RNG on resume.
            'generator': torch.Generator().manual_seed(self.args.data_seed),
        }
        if workers > 0:
            kwargs.update({
                'persistent_workers': self.train_args.dataloader_persistent_workers,
                'prefetch_factor': self.train_args.dataloader_prefetch_factor,
                'multiprocessing_context': self.train_args.dataloader_multiprocessing_context,
            })
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return DataLoader(**kwargs)

    def _split_batch_for_context_parallel(self, batch):
        if self.spec.native_model_factory and self.model.parallel_state.get_cp_group_size() > 1:
            from fsdp_turbo.distributed.context_parallel.batch import split_sequence_batch

            # HF config carries a vision tower even for a text-only batch.
            # Split native text explicitly instead of taking BaseTrainer's VLM bypass.
            return split_sequence_batch(
                batch,
                group=self.model.parallel_state.get_cp_group(),
                sequence_keys=('input_ids', 'attention_mask'),
                build_position_ids=True)
        return super()._split_batch_for_context_parallel(batch)

    def build_scheduler(self):
        if self.optimizer is None:
            return None
        from transformers import get_scheduler

        num_training_steps = self._resolve_max_steps()
        num_warmup_steps = self.train_args.get_warmup_steps(num_training_steps)
        factory = partial(
            get_scheduler,
            _scheduler_name(self.train_args),
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=self.train_args.lr_scheduler_kwargs or {},
        )
        optimizers = getattr(self.optimizer, 'checkpoint_optimizers', None)
        if optimizers is not None:
            from fsdp_turbo.models.deepseek.deepseek_v41.engram.engram_optimizer import EngramSchedulerContainer

            return EngramSchedulerContainer([factory(optimizer=optimizer) for optimizer in optimizers])
        return factory(optimizer=self.optimizer)

    def build_optimizer(self):
        from fsdp_turbo.quantization.cache import hook_optimizer_step
        from fsdp_turbo.training.factories import OptimizerFactory
        from transformers import Trainer

        tables = []
        if self.spec.model_type == 'deepseek_v41':
            from fsdp_turbo.models.deepseek.deepseek_v41.engram.engram_optimizer import register_engram_optimizers
            from fsdp_turbo.models.deepseek.deepseek_v41.engram.host_offload_engram import host_engram_tables

            tables = [table for table in host_engram_tables(self.model) if table.weight.requires_grad]
            register_engram_optimizers(OptimizerFactory)
        sparse_ids = {id(table.weight) for table in tables}
        decay_names = set(Trainer.get_decay_parameter_names(None, self.model))
        config = self.config.optimizer
        groups = [{
            'params': [
                parameter for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and id(parameter) not in sparse_ids and (name in decay_names) == decay
            ],
            'weight_decay':
            config.weight_decay if decay else 0.0,
        } for decay in (True, False)]
        kwargs = dict(lr=config.lr, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon)
        if tables:

            def create_optimizer(name, params, **options):
                return OptimizerFactory.create(name, groups if name == config.optimizer_type else params, **options)

            optimizer = OptimizerFactory.create(
                'EngramOptimizer',
                self.model,
                tables,
                dense_optimizer_type=config.optimizer_type,
                optimizer_factory=SimpleNamespace(create=create_optimizer),
                **kwargs)
        else:
            optimizer = OptimizerFactory.create(config.optimizer_type, groups, **kwargs)
        hook_optimizer_step(self.model, optimizer.dense_optimizer if tables else optimizer)
        return optimizer

    def _save_checkpoint(self):
        if self._adapter_model is None:
            return super()._save_checkpoint()
        from pathlib import Path
        from torch.distributed.tensor import DTensor

        # Do not traverse the base state dict: Host tables use shard-specific
        # checkpoint keys. Only trainable adapters belong in a PEFT export.
        state = {}
        for name, parameter in self._adapter_model.named_parameters():
            if parameter.requires_grad:
                value = parameter.detach()
                state[name] = (value.full_tensor() if isinstance(value, DTensor) else value).cpu()
        if dist.get_rank() == 0:
            output = Path(self.args.output_dir) / f'checkpoint-{self._global_step}'
            (output / 'adapter_training_state.json').unlink(missing_ok=True)
            self._adapter_model.save_pretrained(
                output, state_dict=state, safe_serialization=True, save_embedding_layers=False)
            self.args.save_args(output)
        dist.barrier()
        if self.spec.native_model_factory and not self.train_args.save_only_model:
            from pathlib import Path

            from .adapter_checkpoint import (adapter_run_signature, capture_rng, write_adapter_metadata,
                                             write_tensor_state)

            output = Path(self.args.output_dir) / f'checkpoint-{self._global_step}'
            signature = self._adapter_parameter_signature()
            write_tensor_state(output / f'rng-rank-{dist.get_rank()}.pt', capture_rng(self.train_args.device))
            if dist.get_rank() == 0:
                write_tensor_state(output / 'optimizer.pt', self.optimizer.state_dict())
                write_tensor_state(output / 'scheduler.pt', self.lr_scheduler.state_dict())
            dist.barrier()
            if dist.get_rank() == 0:
                write_adapter_metadata(
                    output,
                    dict(
                        topology=self._adapter_topology(),
                        dataset_fingerprint=self._adapter_dataset_fingerprint(),
                        global_step=self._global_step,
                        consumed_samples=self._consumed_samples,
                        parameters=signature,
                        run_signature=adapter_run_signature(self.args)))
            dist.barrier()

    def _adapter_topology(self):
        world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get('WORLD_SIZE', '1'))
        return dict(
            world_size=world_size,
            fsdp_size=self.args.fsdp_size,
            cp_size=self.args.cp_size,
            ep_size=self.args.ep_size,
            tp_size=self.args.tp_size,
            efsdp_size=self.args.efsdp_size)

    def _adapter_dataset_fingerprint(self):
        from fsdp_turbo.training.checkpoint import dataset_fingerprint

        return dict(
            config=dataset_fingerprint(self.config.data), content=getattr(self.train_dataset, '_fingerprint', None))

    def _adapter_parameter_signature(self):
        from torch.distributed.tensor import DTensor

        result = {}
        for name, parameter in self._adapter_model.named_parameters():
            if parameter.requires_grad:
                if isinstance(parameter, DTensor):
                    raise ValueError('Portable adapter optimizer checkpoints require replicated parameters.')
                result[name] = dict(shape=list(parameter.shape), dtype=str(parameter.dtype))
        return result

    def _load_checkpoint(self, checkpoint_path):
        if not self._adapter_resume:
            return super()._load_checkpoint(checkpoint_path)
        from pathlib import Path

        from .adapter_checkpoint import adapter_run_signature, read_adapter_metadata, restore_rng

        path = Path(checkpoint_path)
        metadata = read_adapter_metadata(
            path,
            topology=self._adapter_topology(),
            dataset_fingerprint=self._adapter_dataset_fingerprint(),
            run_signature=adapter_run_signature(self.args))
        if metadata.get('parameters') != self._adapter_parameter_signature():
            raise ValueError('Adapter checkpoint trainable parameter names, shapes or dtypes changed.')
        self.optimizer.load_state_dict(torch.load(path / 'optimizer.pt', map_location='cpu', weights_only=True))
        self.lr_scheduler.load_state_dict(torch.load(path / 'scheduler.pt', map_location='cpu', weights_only=True))
        self._global_step, self._consumed_samples = metadata['global_step'], metadata['consumed_samples']
        restore_rng(
            torch.load(path / f'rng-rank-{dist.get_rank()}.pt', map_location='cpu', weights_only=True),
            self.train_args.device)
        logger.info(f'Restored adapter optimizer at step {self._global_step}, samples {self._consumed_samples}.')

    def _prepare_model_inputs(self, batch):
        loss_scale = batch.pop('loss_scale', None)
        batch.pop('channel', None)
        batch.pop('text_position_ids', None)
        batch.pop('_extra_kwargs', None)
        if loss_scale is not None and 'labels' in batch:
            active_scale = loss_scale.masked_select(batch['labels'].ne(-100))
            if active_scale.numel() and not torch.all(active_scale.eq(1)):
                raise NotImplementedError('Non-uniform Swift loss scaling is not supported by FSDPTurbo yet.')
        batch['use_cache'] = False
        if self.args.router_aux_loss_coef:
            batch['output_router_logits'] = True
        return batch

    def train_step(self, batch):
        outputs = self.model(**self._prepare_model_inputs(batch))
        if outputs.loss is None:
            raise RuntimeError('Model did not return a loss; ensure the batch contains labels.')
        loss = outputs.loss / self.config.run.gradient_accumulation_steps
        loss.backward()
        return loss, getattr(outputs, 'aux_loss', None)

    def evaluate(self):
        """Evaluate token-weighted loss, excluding sampler padding duplicates."""
        if self.eval_dataset is None or not len(self.eval_dataset):
            raise ValueError('FSDPTurbo evaluation requires a non-empty validation dataset.')
        state = self.model.parallel_state
        size, rank = state.get_data_group_size(), state.get_data_rank()
        sampler = DistributedSampler(self.eval_dataset, num_replicas=size, rank=rank, shuffle=False)
        batch_size = self.train_args.per_device_eval_batch_size
        # Equal numbers of forwards preserve FSDP collective ordering. The
        # padded sampler items contribute no targets to the reported metric.
        loader = DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.dataloader.collate_fn,
            num_workers=0)
        loss_total = torch.zeros((), dtype=torch.float32, device=self.train_args.device)
        token_total = torch.zeros((), dtype=torch.int64, device=self.train_args.device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for step, batch in enumerate(loader):
                    labels = batch['labels'].clone()
                    for row in range(labels.shape[0]):
                        if ((step * batch_size + row) * size + rank) >= len(self.eval_dataset):
                            labels[row].fill_(-100)
                    batch['labels'] = labels
                    count = labels[:, 1:].ne(-100).sum().to(self.train_args.device)
                    prepared = self._prepare_model_inputs(self.prepare_batch(batch))
                    loss = self.model(**prepared).loss
                    if loss is None:
                        raise RuntimeError('Validation forward did not return a loss.')
                    loss_total += torch.where(count > 0, loss.float() * count, 0)
                    token_total += count
            if size > 1:
                dist.all_reduce(loss_total, group=state.get_data_group())
                dist.all_reduce(token_total, group=state.get_data_group())
            metrics = {
                'step': self._global_step,
                'eval_loss': (loss_total / token_total.clamp_min(1)).item(),
                'eval_tokens': int(token_total.item())
            }
            self._eval_history.append(metrics)
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info(f'Evaluation: {metrics}')
            return metrics
        finally:
            self.model.train(was_training)

    def _on_checkpoint_step(self):
        super()._on_checkpoint_step()
        if getattr(self, '_save_at_next_step', False):
            self._save_checkpoint()
            self._save_at_next_step = False
        strategy = getattr(self.train_args.eval_strategy, 'value', self.train_args.eval_strategy)
        if strategy == 'steps':
            interval = self.train_args.eval_steps or self.train_args.logging_steps
            if isinstance(interval, float) and interval < 1:
                interval = math.ceil(self.train_args.max_steps * interval)
            if self._global_step % max(1, int(interval)) == 0:
                self.evaluate()

    def _on_epoch_end(self):
        strategy = getattr(self.train_args.eval_strategy, 'value', self.train_args.eval_strategy)
        if strategy == 'epoch':
            self.evaluate()
        save_strategy = getattr(self.train_args.save_strategy, 'value', self.train_args.save_strategy)
        if save_strategy == 'epoch':
            if self._accumulated_losses['loss']:
                self._save_at_next_step = True
            elif self._global_step > 0:
                self._save_checkpoint()

    @property
    def result(self):
        history = []
        if self._monitor is not None:
            history = [{
                'step': step,
                'loss': loss,
                'step_time': step_time,
                'grad_norm': float(grad_norm) if grad_norm is not None else None,
            } for step, loss, step_time, grad_norm in self._monitor.loss_history]
        return {'global_step': self._global_step, 'log_history': history, 'eval_history': self._eval_history}
