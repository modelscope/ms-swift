# Copyright (c) ModelScope Contributors. All rights reserved.
import dataclasses
import logging
import megatron.core
import operator
import os
import shutil
import torch
import torch.nn
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from functools import partial
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from modelscope import check_local_model_is_latest
from packaging import version
from pathlib import Path
from typing import Callable, Dict, List, Optional

from swift.dataset import RowPreprocessor
from swift.megatron.callbacks import megatron_callbacks_map
from swift.megatron.model import get_mcore_model
from swift.megatron.tuners import LoraParallelLinear
from swift.megatron.utils import (copy_original_module_weight, disable_forward_pre_hook, enable_forward_pre_hook,
                                  get_optimizer_param_scheduler, get_padding_to, init_persistent_async_worker,
                                  initialize_tp_communicators, load_mcore_checkpoint,
                                  logical_and_across_model_parallel_group, maybe_finalize_async_save,
                                  prepare_mcore_model, reduce_max_stat_across_model_parallel_group,
                                  save_mcore_checkpoint, should_disable_forward_pre_hook, warmup_jit_function,
                                  wrap_model)
from swift.template import Template
from swift.trainers import dynamic_gradient_checkpointing
from swift.trainers.utils import patch_modelscope_hub_timeout
from swift.utils import deep_getattr, get_last_valid_indices, get_logger, is_last_rank, is_master, ms_logger_context
from .batch_sampler import MegatronPretrainingRandomSampler, MegatronPretrainingSampler
from .utils import (TrainerState, build_streaming_dataloader, get_batch_on_this_cp_rank, get_batch_on_this_pp_rank,
                    get_packed_seq_params)

try:
    from megatron.core.optimizer import param_group_identifier_keys
except ImportError:
    param_group_identifier_keys = None

logger = get_logger()


class BaseMegatronTrainer(ABC):

    def __init__(self, args, template: Template):
        self.args = args
        self.template = template
        self.bridge = args.megatron_model_meta.bridge_cls(args)
        self.prepare_model()
        self.config = self.unwrapped_models[0].config
        self.optimizer, self.opt_param_scheduler = self.get_optimizer_and_scheduler()
        self.data_collator = self._get_data_collator()

        self.state = TrainerState(max_steps=args.train_iters)
        initialize_embedding = args.new_special_tokens or args.task_type == 'seq_cls'
        if initialize_embedding:
            for m in self.unwrapped_models:
                self._initialize_embedding(m)
        if args.tuner_type != 'full' and args.modules_to_save:
            for m in self.unwrapped_models:
                copy_original_module_weight(m)
        self._load_checkpoint()

        self.eval_metrics = None
        if args.check_model and hasattr(args, 'model_dir'):
            with ms_logger_context(logging.CRITICAL), patch_modelscope_hub_timeout():
                config_info = self._collect_config_info()
                config_info.update({
                    'invoked_by': 'local_trainer',
                    'third_party': 'swift',
                    'trainer_class': self.__class__.__name__,
                    'trainer_backend': 'megatron',
                })
                check_local_model_is_latest(args.model_info.model_dir, user_agent=config_info)

        self.mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
        self.mcore_014 = version.parse(megatron.core.__version__) >= version.parse('0.14.0rc0')
        self.callbacks = []
        for callback in args.callbacks:
            self.callbacks.append(megatron_callbacks_map[callback](self))

        if args.tp_comm_overlap:
            initialize_tp_communicators(args, self.config)

        warmup_jit_function(self.config, args)

        if args.async_save and args.use_persistent_ckpt_worker:
            init_persistent_async_worker()

    def _load_checkpoint(self):
        args = self.args
        if not args.finetune:
            self.state.iteration = self._load_iteration()
        if args.mcore_model is not None:
            self.state.iteration = load_mcore_checkpoint(
                args, self.wrapped_models, self.optimizer, self.opt_param_scheduler, load_arg='mcore_model')
        if args.mcore_adapter is not None:
            self.state.iteration = load_mcore_checkpoint(
                args, self.wrapped_models, self.optimizer, self.opt_param_scheduler, load_arg='mcore_adapter')
        self.state.consumed_train_samples = getattr(args, 'consumed_train_samples', 0)

    def call_event(self, event, **kwargs):
        for callback in self.callbacks:
            getattr(callback, event)(**kwargs)

    def on_log(self, logs, prefix=''):
        n_steps = logs.pop('n_steps')
        self._log_callback(logs, n_steps)
        if prefix:
            logs = {f'{prefix}{k}': v for k, v in logs.items()}
        self.call_event('on_log', logs=logs)

    def _log_callback(self, logs, n_steps):
        args = self.args
        config = self.config
        if config.num_moe_experts is not None:
            moe_loss_scale = 1 / args.num_microbatches / n_steps
            track_names = []
            if config.moe_router_load_balancing_type == 'aux_loss':
                track_names.append('load_balancing_loss')
            elif config.moe_router_load_balancing_type == 'seq_aux_loss':
                if self.mcore_014:
                    track_names.append('seq_load_balancing_loss')
                else:
                    track_names.append('load_balancing_loss')
            elif config.moe_router_load_balancing_type == 'global_aux_loss':
                track_names.append('global_load_balancing_loss')
            if config.moe_z_loss_coeff is not None:
                track_names.append('z_loss')
            track_moe_kwargs = {'mtp_num_layers': args.mtp_num_layers} if self.mcore_013 else {}
            track_moe_metrics(
                loss_scale=moe_loss_scale,
                iteration=self.state.iteration,
                writer=None,
                total_loss_dict=logs,
                force_initialize=True,
                track_names=track_names,
                num_layers=config.num_layers,
                moe_layer_freq=config.moe_layer_freq,
                **track_moe_kwargs)
        if args.mtp_num_layers is not None:
            mtp_loss_scale = 1 / args.num_microbatches / n_steps
            mtp_logs = {}
            MTPLossLoggingHelper.track_mtp_metrics(mtp_loss_scale, self.state.iteration, None, None, mtp_logs)
            logs.update({k.replace(' ', '_'): v for k, v in mtp_logs.items()})
        for k, v in logs.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 2:
                    v = v[0] / v[1]
                v = v.item()
            logs[k] = v

    def prepare_model(self):
        args = self.args
        self.wrapped_models = []
        self.unwrapped_models = get_mcore_model(args, self.template.config)
        self.peft_models = self._prepare_peft_model(self.unwrapped_models)
        self.wrapped_models = wrap_model(args, self.unwrapped_models)

    def _prepare_peft_model(self, models):
        args = self.args
        if args.mcore_model is None:
            self.bridge.load_weights(models, args.model_dir)
        peft_models = [prepare_mcore_model(args, model) for model in models]
        if args.tuner_type == 'lora' and args.adapters and args.mcore_adapter is None:
            assert len(args.adapters) == 1, 'Currently only support one adapter.'
            self.bridge.load_weights(models, args.adapters[0], is_peft_format=True, adapter_name='default')
        return peft_models

    def get_optimizer_and_scheduler(self):
        args = self.args
        kwargs = {}
        for f in dataclasses.fields(OptimizerConfig):
            if hasattr(args, f.name) and f.name != 'loss_scale':
                kwargs[f.name] = getattr(args, f.name)
        config = OptimizerConfig(**kwargs)
        if args.apply_wd_to_qk_layernorm or self.args.vit_lr is not None or self.args.aligner_lr is not None:
            param_groups_context = self._patch_get_param_groups()
        else:
            param_groups_context = nullcontext()
        with param_groups_context:
            optimizer = get_megatron_optimizer(config, self.wrapped_models)
        opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
        return optimizer, opt_param_scheduler

    def _get_data_collator(self):
        data_collator = self.template.data_collator
        padding_to = get_padding_to(self.args)
        logger.info(f'padding_to: {padding_to}')
        data_collator = partial(data_collator, padding_to=padding_to)
        return data_collator

    def cyclic_iter(self, iterable, use_origin_cyclic: bool = False):
        training = self.unwrapped_models[0].training
        if not training or use_origin_cyclic:
            while True:
                for x in iterable:
                    yield x
            return

        args = self.args
        state = self.state
        is_finished = False
        while True:
            if not is_finished:
                logger.info(f'The training of Epoch {state.epoch} starts...')
            for x in iterable:
                yield x
            # streaming
            if training and args.num_train_epochs and state.epoch >= args.num_train_epochs - 1:
                is_finished = True
            state.epoch += 1
            if is_finished:
                # Note that this approach will train for one additional step.
                logger.info(f'Training of {state.epoch} epochs has been completed, the training has finished.')
                args.train_iters = state.iteration + 1

    # Code borrowed from Megatron-LM
    def _get_param_groups(
        self,
        model_chunks: List[MegatronModule],
        no_weight_decay_cond: Optional[Callable],
        scale_lr_cond: Optional[Callable],
        lr_mult: float,
        lr: float,
        min_lr: float,
        decoupled_lr: Optional[float],
        decoupled_min_lr: Optional[float],
        default_skip_embedding_weight_decay: bool = False,
    ) -> List[Dict]:
        """Create parameter groups for optimizer.

        Creates parameter groups based on weight decay condition (regularized vs
        non regularized), learning rate scale condition (lr vs lr_mult * lr),
        and whether it is expert parameters. scale_lr_cond is used during finetuning
        where head of the network requires a scaled version of the base learning rate.

        Args:
            model_chunks (List[MegatronModule]): model chunks to create parameter
                groups for.
            no_weight_decay_cond (func, optional): function to determine whether a
                parameter should not perform weight decay.
            scale_lr_cond (func, optional): function to determine whether a parameter
                should have a scaled learning rate.
            lr_mult (float): learning rate multiplier for parameters that
                satisfy scale_lr_cond.
            lr (float): learning rate.
            min_lr (float): minimum learning rate.
            decoupled_lr (Optional[float]): optional decoupled learning rate.
            decoupled_min_lr (Optional[float]): optional decoupled minimum learning rate.
            default_skip_embedding_weight_decay (bool): whether to skip weight decay for embedding
                parameters by default, if no_weight_decay_cond is not provided.

        Returns:
            List of parameter groups.
        """
        from megatron.core.optimizer import _update_min_and_max_lr_in_param_groups
        args = self.args
        is_multimodal = args.megatron_model_meta.is_multimodal
        if args.vit_lr is not None or args.aligner_lr is not None:
            assert is_multimodal, 'vit_lr and aligner_lr are only supported for multimodal models.'
            vit_lr = args.vit_lr if args.vit_lr is not None else args.lr
            aligner_lr = args.aligner_lr if args.aligner_lr is not None else args.lr
            logger.info_once(f'vit_lr: {vit_lr}, aligner_lr: {aligner_lr}, llm_lr: {args.lr}')
        use_decoupled_learning_rate = decoupled_lr is not None

        # Map (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr) to params.
        params_map = {}
        for model_chunk in model_chunks:
            visual = model_chunk.module.module.visual if is_multimodal else None
            for name, param in model_chunk.named_parameters():
                if not param.requires_grad:
                    continue

                is_expert_parallel = not getattr(param, 'allreduce', True)

                if no_weight_decay_cond is not None:
                    no_wd: bool = no_weight_decay_cond(name, param)
                elif args.apply_wd_to_qk_layernorm and any(
                        name.endswith(k) for k in ['q_layernorm.weight', 'k_layernorm.weight']):
                    no_wd = False
                else:
                    # Do not regularize biases and norm parameters.
                    #  optionally, also skip weight decay for embedding parameters if requested
                    #  (useful if you do not want embeddings to shrink to zero in training
                    #  https://arxiv.org/abs/2312.16903)
                    no_wd = (
                        name.endswith('.bias') or len(param.shape) == 1
                        or (default_skip_embedding_weight_decay and 'embedding' in name))
                _lr_mult = lr_mult
                if scale_lr_cond is not None:
                    scale_lr = scale_lr_cond(name, param)
                else:
                    scale_lr = False
                    # Handling multimodal models: vit_lr, aligner_lr
                    unwrapped_name = name.removeprefix('module.').removeprefix('module.')
                    if visual is not None:
                        is_aligner = any(unwrapped_name.startswith(f'visual.{k}') for k in visual._aligner or [])
                        is_vit = any(unwrapped_name.startswith(f'visual.{k}')
                                     for k in visual._vision_tower) and not is_aligner
                    else:
                        is_aligner, is_vit = False, False
                    if is_vit and args.vit_lr:
                        scale_lr = True
                        _lr_mult = args.vit_lr / lr
                    elif is_aligner and args.aligner_lr:
                        scale_lr = True
                        _lr_mult = args.aligner_lr / lr

                if not no_wd and not scale_lr:
                    wd_mult, _lr_mult = 1.0, 1.0
                elif not no_wd and scale_lr:
                    wd_mult, _lr_mult = 1.0, _lr_mult
                elif no_wd and not scale_lr:
                    wd_mult, _lr_mult = 0.0, 1.0
                else:
                    wd_mult, _lr_mult = 0.0, _lr_mult

                is_decoupled_lr = False
                # For input/embedding and output layer: embedding.word_embeddings.weight /
                # output_layer.weight.
                if use_decoupled_learning_rate and getattr(param, 'is_embedding_or_output_parameter', False):
                    is_decoupled_lr = True

                key = (wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr)
                if key not in params_map:
                    params_map[key] = []
                params_map[key].append(param)

        # Distributed checkpoint requires all ranks to have the same param groups,
        # so we need to align the param groups across ranks, otherwise we may have
        # runtime error when loading the checkpoint or numerical error when resuming training.
        params_key = list(params_map.keys())
        gathered_params_key = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_params_key, params_key)
        for keys in gathered_params_key:
            for key in keys:
                if key not in params_key:
                    params_key.append(key)

        param_groups = []
        for key in params_key:
            wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr = key
            params = params_map[key] if key in params_map else []
            param_group = {
                'params': params,
                'wd_mult': wd_mult,
                'lr_mult': _lr_mult,
                'is_expert_parallel': is_expert_parallel,
                'is_decoupled_lr': is_decoupled_lr,
            }
            # Ensure param_group has required keys for matching when loading optimizer state
            # See MegatronOptimizer._filter_and_reorder_param_groups.
            if param_group_identifier_keys is not None:
                assert set(param_group.keys()) - set(param_group_identifier_keys) == {'params'}
            param_groups.append(param_group)

        param_groups = _update_min_and_max_lr_in_param_groups(
            param_groups,
            lr=lr,
            min_lr=min_lr,
            decoupled_lr=decoupled_lr,
            decoupled_min_lr=decoupled_min_lr,
        )

        return param_groups

    @contextmanager
    def _patch_get_param_groups(self):
        from megatron.core import optimizer

        _get_param_groups = optimizer._get_param_groups
        optimizer._get_param_groups = self._get_param_groups
        try:
            yield
        finally:
            optimizer._get_param_groups = _get_param_groups

    def _load_iteration(self):
        args = self.args
        ckpt_dir = None
        if args.tuner_type == 'full':
            ckpt_dir = args.model
        elif args.tuner_type == 'lora':
            ckpt_dir = args.adapters[0] if args.adapters else args.model
        if ckpt_dir is None:
            return 0
        logger.info(f'checkpoint_dir: {ckpt_dir}')
        tracker_path = os.path.join(ckpt_dir, 'latest_checkpointed_iteration.txt')
        if not os.path.exists(tracker_path):
            return 0
        with open(tracker_path, 'r') as f:
            iteration = int(f.read())

        common_path = os.path.join(ckpt_dir, f'iter_{iteration:07d}', 'common.pt')
        if not os.path.exists(common_path):
            return iteration

        state_dict = torch.load(common_path)
        if 'args' not in state_dict:
            return iteration
        self.args.consumed_train_samples = getattr(state_dict['args'], 'consumed_train_samples', 0)

        return iteration

    def _prepare_vit_gradient_checkpointing(self, model):
        visual = model.visual
        if visual is None:
            return
        for vision_tower in visual._vision_tower:
            module = deep_getattr(visual, vision_tower)
            if self.args.vit_gradient_checkpointing:
                dynamic_gradient_checkpointing(module, False)
                try:
                    module.gradient_checkpointing_enable(**(self.args.gradient_checkpointing_kwargs or {}))
                    module.enable_input_require_grads()
                except AttributeError:
                    pass

    @staticmethod
    def _initialize_embedding(model):
        # compat new_special_tokens
        init_method = model.config.init_method
        if hasattr(model, 'language_model'):
            model = model.language_model
        for key in ['embedding.word_embeddings', 'output_layer']:
            if key == 'output_layer' and model.share_embeddings_and_output_weights:
                continue
            module = deep_getattr(model, key)
            if module is None:
                continue
            initialize_mask = (module.weight == 0).all(dim=-1)
            num_to_initialize = initialize_mask.sum().item()
            if num_to_initialize == 0:
                continue
            logger.info_if(f'num_to_initialize: {num_to_initialize}', cond=mpu.get_data_parallel_rank() == 0)
            tensor = module.weight.new_empty(num_to_initialize, module.weight.shape[1])
            module.weight.data[initialize_mask] = init_method(tensor)
            if getattr(module.weight, 'main_param', None) is not None:
                module.weight.main_param.copy_(module.weight.view(-1))

    def _all_reduce_metric(self,
                           metric: Dict[str, torch.Tensor],
                           reduction=torch.distributed.ReduceOp.AVG,
                           group=None) -> Dict[str, torch.Tensor]:
        if group is None:
            group = mpu.get_data_parallel_group()
        reporting_metric = torch.stack(list(metric.values()), dim=0)
        torch.distributed.all_reduce(reporting_metric, reduction, group=group)
        return {k: reporting_metric[i] for i, k in enumerate(metric.keys())}

    def merge_lora_adapters(self, adapter_name='default'):
        """Merge LoRA adapters into base model weights for vLLM inference."""
        with torch.no_grad():
            for model in self.unwrapped_models:
                for module in model.modules():
                    if isinstance(module, LoraParallelLinear):
                        # Merge all active adapters
                        module.merge(adapter_names=[adapter_name])

    def unmerge_lora_adapters(self):
        """Unmerge LoRA adapters to restore training state."""
        with torch.no_grad():
            for model in self.unwrapped_models:
                for module in model.modules():
                    if isinstance(module, LoraParallelLinear):
                        # Unmerge to restore separate LoRA weights for training
                        module.unmerge()

    @staticmethod
    def copy_path(src_path: str, tgt_path: str):
        if not is_master():
            return
        if not os.path.exists(src_path):
            raise FileNotFoundError(f'Source path does not exist: {src_path}')

        if os.path.isfile(src_path):
            os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            shutil.copy(src_path, tgt_path)
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, tgt_path, dirs_exist_ok=True)
        else:
            raise ValueError(f'Source path is neither a file nor a directory: {src_path}')

    def _prepare_data_iterator(self, train_dataset, val_dataset=None, use_origin_cyclic: bool = False):
        train_dataloader, val_dataloader = self._prepare_dataloader(train_dataset, val_dataset)
        train_data_iterator = iter(self.cyclic_iter(train_dataloader, use_origin_cyclic=use_origin_cyclic))
        val_data_iterator = None
        if val_dataset is not None:
            val_data_iterator = iter(self.cyclic_iter(val_dataloader, use_origin_cyclic=use_origin_cyclic))
        return train_data_iterator, val_data_iterator

    def train(self, train_dataset, val_dataset):
        args = self.args
        config = self.config
        state = self.state
        for m in self.wrapped_models:
            m.train()

        if args.is_multimodal:
            for m in self.unwrapped_models:
                self._prepare_vit_gradient_checkpointing(m)

        config.grad_scale_func = self.optimizer.scale_loss
        if isinstance(self.wrapped_models[0], DDP) and args.overlap_grad_reduce:
            assert config.no_sync_func is None, ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
                                                 'a custom no_sync_func is not supported when overlapping grad-reduce')
            config.no_sync_func = [model_chunk.no_sync for model_chunk in self.wrapped_models]
            if len(self.wrapped_models) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if args.align_grad_reduce:
                config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in self.wrapped_models]
                if len(self.wrapped_models) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        if args.overlap_param_gather and args.align_param_gather:
            config.param_sync_func = [model_chunk.start_param_sync for model_chunk in self.wrapped_models]
            if len(self.wrapped_models) == 1:
                config.param_sync_func = config.param_sync_func[0]
        config.finalize_model_grads_func = finalize_model_grads
        start_iteration = state.iteration
        pre_hook_enabled = False
        # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
        # or random initialization don't propagate to all ranks in first all-gather (which is a
        # no-op if things work correctly).
        if should_disable_forward_pre_hook(args):
            disable_forward_pre_hook(self.wrapped_models, param_sync=False)
            # Also remove param_sync_func temporarily so that sync calls made in
            # `forward_backward_func` are no-ops.
            param_sync_func = config.param_sync_func
            config.param_sync_func = None
            pre_hook_enabled = False

        self.call_event('on_train_begin')
        train_metrics = {}
        if args.virtual_pipeline_model_parallel_size is not None:
            train_data_iterator, val_data_iterator = [], []
            for _ in range(args.virtual_pipeline_model_parallel_size):
                train_it, val_it = self._prepare_data_iterator(train_dataset, val_dataset)
                train_data_iterator.append(train_it)
                val_data_iterator.append(train_it)
        else:
            train_data_iterator, val_data_iterator = self._prepare_data_iterator(train_dataset, val_dataset)
        while state.iteration < args.train_iters:
            self.call_event('on_step_begin')
            maybe_finalize_async_save(args, blocking=False)
            metrics, grad_norm, update_successful = self.train_step(train_data_iterator)
            if state.iteration == start_iteration:
                if update_successful:
                    # Enable forward pre-hook after training step has successfully run. All subsequent
                    # forward passes will use the forward pre-hook / `param_sync_func` in
                    # `forward_backward_func`.
                    if should_disable_forward_pre_hook(args):
                        enable_forward_pre_hook(self.wrapped_models)
                        config.param_sync_func = param_sync_func
                        pre_hook_enabled = True
                else:
                    start_iteration = state.iteration + 1

            state.iteration += 1
            self.call_event('on_step_end')
            self._aggregated_metrics(metrics, train_metrics)
            train_metrics['grad_norm'] = grad_norm
            learning_rate = None
            for param_group in self.optimizer.param_groups:
                if len(param_group['params']) == 0:
                    continue
                learning_rate = param_group['lr']
            if learning_rate is not None:
                train_metrics['learning_rate'] = learning_rate
            if state.should_log:
                state.should_log = False
                self.on_log(logs=train_metrics)
                train_metrics = {}

            eval_metrics = None
            if state.should_eval:
                state.should_eval = False
                if should_disable_forward_pre_hook(args):
                    disable_forward_pre_hook(self.wrapped_models)
                    pre_hook_enabled = False
                eval_metrics = self.evaluate(val_data_iterator)
                for m in self.wrapped_models:
                    m.train()
                if should_disable_forward_pre_hook(args):
                    enable_forward_pre_hook(self.wrapped_models)
                    pre_hook_enabled = True

            if state.should_save:
                self._determine_best_metric(eval_metrics)
                if should_disable_forward_pre_hook(args):
                    disable_forward_pre_hook(self.wrapped_models)
                state.should_save = False
                self.save_checkpoint()
                if should_disable_forward_pre_hook(args):
                    enable_forward_pre_hook(self.wrapped_models)

        self.call_event('on_train_end')
        # Close out pre-hooks if using distributed optimizer and overlapped param gather.
        if pre_hook_enabled:
            disable_forward_pre_hook(self.wrapped_models)
        maybe_finalize_async_save(args, blocking=True, terminate=True)

    def _determine_best_metric(self, metrics) -> bool:
        args = self.args
        state = self.state
        if (args.metric_for_best_model is None or metrics is None or not is_last_rank()
                or args.metric_for_best_model not in metrics):
            return False
        metric_value = metrics[args.metric_for_best_model]
        op = operator.ge if args.greater_is_better else operator.le
        if state.best_metric is None:
            state.best_metric = float('-inf') if args.greater_is_better else float('inf')

        is_new_best_metric = False
        if op(metric_value, state.best_metric):
            state.best_metric = metric_value
            state.best_global_step = state.global_step
            is_new_best_metric = True
        return is_new_best_metric

    def save_checkpoint(self):
        args = self.args
        state = self.state
        args.consumed_train_samples = state.consumed_train_samples
        iteration = state.iteration
        output_dir = os.path.join(args.output_dir, f'checkpoint-{iteration}')
        os.makedirs(output_dir, exist_ok=True)
        args_path = os.path.join(os.path.dirname(output_dir), 'args.json')
        self.copy_path(args_path, os.path.join(output_dir, 'args.json'))
        save_peft_format = args.tuner_type == 'lora' and not args.merge_lora
        if args.save_safetensors and args.no_save_optim:
            model = []
        else:
            model = self.wrapped_models
        save_mcore_checkpoint(
            args,
            model,
            self.optimizer,
            self.opt_param_scheduler,
            iteration=iteration,
            is_peft_format=args.tuner_type == 'lora',
            output_dir=output_dir)
        state.last_model_checkpoint = output_dir
        if state.best_global_step:
            best_model_checkpoint = os.path.join(args.output_dir, f'checkpoint-{state.best_global_step}')
            if os.path.exists(best_model_checkpoint):
                state.best_model_checkpoint = best_model_checkpoint
        # safetensors
        if args.save_safetensors:
            # merge-lora does not store lora, lora saving may report an error (Qwen3-VL-Moe)
            if args.tuner_type == 'lora' and args.merge_lora:
                self.merge_lora_adapters()
                origin_output_dir = output_dir
                output_dir = f'{output_dir}-merged'
                os.makedirs(output_dir, exist_ok=True)
                for fname in ['latest_checkpointed_iteration.txt', 'args.json']:
                    src_path = os.path.join(origin_output_dir, fname)
                    self.copy_path(src_path, os.path.join(output_dir, fname))
                # common.pt
                common_path = os.path.join(origin_output_dir, f'iter_{iteration:07d}', 'common.pt')
                tgt_common_path = os.path.join(output_dir, f'iter_{iteration:07d}', 'common.pt')
                os.makedirs(os.path.dirname(tgt_common_path), exist_ok=True)
                self.copy_path(common_path, tgt_common_path)
            self.bridge.save_weights(
                self.unwrapped_models,
                output_dir,
                is_peft_format=save_peft_format,
                processor=self.template.processor,
                hf_config=self.template.config)
            if args.tuner_type == 'lora' and args.merge_lora:
                self.unmerge_lora_adapters()

        if is_master():
            self._rotate_checkpoints(args.output_dir)

    def _rotate_checkpoints(self, output_dir: str):
        # Code borrowed from huggingface/transformers
        args = self.args
        if args.save_total_limit is None or args.save_total_limit <= 0:
            return

        checkpoints_sorted = self._sorted_checkpoints(output_dir)
        if len(checkpoints_sorted) <= args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            shutil.rmtree(checkpoint, ignore_errors=True)
            if os.path.exists(f'{checkpoint}-merged'):
                shutil.rmtree(f'{checkpoint}-merged', ignore_errors=True)

    def _sorted_checkpoints(self, output_dir: str):
        # Code borrowed from huggingface/transformers
        state = self.state
        glob_checkpoints = [
            str(p) for p in Path(output_dir).glob('checkpoint-*') if p.is_dir() and not p.name.endswith('-merged')
        ]
        # Sort by modification time
        checkpoints_sorted = sorted(glob_checkpoints, key=os.path.getmtime)

        # Make sure we don't delete the best model.
        if state.best_model_checkpoint is not None and state.best_model_checkpoint in checkpoints_sorted:
            best_model_index = checkpoints_sorted.index(state.best_model_checkpoint)
            checkpoints_sorted.pop(best_model_index)
            checkpoints_sorted.append(state.best_model_checkpoint)
        return checkpoints_sorted

    def training_log(self, metrics, grad_norm):
        learning_rate = None
        for param_group in self.optimizer.param_groups:
            if len(param_group['params']) == 0:
                continue
            learning_rate = param_group['lr']
        logger.info(f'metrics: {metrics}, grad_norm: {grad_norm}, learning_rate: {learning_rate}')

    def evaluate(self, val_data_iterator):
        args = self.args
        for m in self.wrapped_models:
            m.eval()
        eval_metrics = {}
        forward_backward_func = get_forward_backward_func()
        self.call_event('on_eval_begin')
        with torch.no_grad():
            for _ in range(args.eval_iters):
                data_iterator = self._replace_data_iterator(val_data_iterator)
                metrics = forward_backward_func(
                    forward_step_func=self.forward_step,
                    data_iterator=data_iterator,
                    model=self.wrapped_models,
                    num_microbatches=self.args.num_microbatches,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    forward_only=True,
                )
                self.call_event('on_eval_step')
                self._aggregated_metrics(metrics, eval_metrics)
        self.compute_eval_metrics(eval_metrics)
        self.on_log(logs=eval_metrics, prefix='eval_')
        self.call_event('on_eval_end')
        return eval_metrics

    def compute_eval_metrics(self, metrics):
        if self.eval_metrics is not None:
            metric = self.eval_metrics.compute()
            for k, v in metric.items():
                metrics[k] = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            self.eval_metrics.reset()

    def _replace_data_iterator(self, data_iterator):
        return data_iterator

    def train_step(self, train_data_iterator):
        args = self.args
        forward_backward_func = get_forward_backward_func()
        for m in self.wrapped_models:
            m.zero_grad_buffer()
        self.optimizer.zero_grad()
        # TODO: refactor _replace_data_iterator
        data_iterator = self._replace_data_iterator(train_data_iterator)
        metrics = forward_backward_func(
            forward_step_func=self.forward_step,
            data_iterator=data_iterator,
            model=self.wrapped_models,
            num_microbatches=args.num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=False,
        )

        update_successful, grad_norm, _ = self.optimizer.step()
        update_successful = logical_and_across_model_parallel_group(update_successful)
        grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
        if update_successful:
            self.opt_param_scheduler.step(increment=args.global_batch_size)

        return metrics, grad_norm, update_successful

    def _aggregated_metrics(self, metrics, total_metrics):
        if 'n_steps' not in total_metrics:
            total_metrics['n_steps'] = 0
        total_metrics['n_steps'] += 1
        if not metrics:
            return
        metrics = RowPreprocessor.rows_to_batched(metrics)
        for key, val in metrics.items():
            val = torch.stack([v for v in val if v is not None], dim=0)
            if val[0].numel() == 2:
                val = val.sum(dim=0)
                if val[1] == 0:
                    continue
            elif val[0].numel() == 1:
                val = val.new_tensor([val.sum(), val.shape[0]])
            else:
                raise ValueError(f'Invalid value shape: {val[0].shape} for key {key}')
            if key not in total_metrics:
                total_metrics[key] = torch.tensor([0.0, 0.0], dtype=torch.float32, device=torch.cuda.current_device())
            total_metrics[key] += val

    def _prepare_dataloader(self, train_dataset, val_dataset=None):
        args = self.args
        val_dataloader = None
        if args.streaming:
            train_dataloader = build_streaming_dataloader(args, train_dataset, self.data_collator)
            if val_dataset is not None:
                val_dataloader = build_streaming_dataloader(args, val_dataset, self.data_collator)
            return train_dataloader, val_dataloader
        train_batch_sampler = MegatronPretrainingRandomSampler(
            train_dataset,
            total_samples=len(train_dataset),
            consumed_samples=self.state.consumed_train_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
            shuffle=args.train_dataloader_shuffle,
            group_by_length=args.group_by_length,
        )
        train_dataloader = self._create_dataloader(train_dataset, train_batch_sampler)
        if val_dataset is not None:
            val_batch_sampler = MegatronPretrainingSampler(
                total_samples=len(val_dataset),
                consumed_samples=0,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
            )
            val_dataloader = self._create_dataloader(val_dataset, val_batch_sampler)
        return train_dataloader, val_dataloader

    def _create_dataloader(self, dataset, batch_sampler):
        args = self.args

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
            persistent_workers=args.dataloader_persistent_workers if args.dataloader_num_workers > 0 else False,
            prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None,
            collate_fn=self.data_collator,
        )
        return dataloader

    @abstractmethod
    def forward_step(self, data_iterator, model):
        pass

    def _prepare_batch(self, data, vp_stage=None, num_samples=None):
        batch = get_batch_on_this_pp_rank(self.args, data, vp_stage=vp_stage)
        if num_samples is None:
            num_samples = batch.pop('num_samples')
        args = self.args
        text_position_ids = batch.pop('text_position_ids', None)
        batch.pop('attention_mask_2d', None)
        if text_position_ids is None:
            text_position_ids = batch.get('position_ids')
        if args.padding_free and text_position_ids is not None:
            batch['packed_seq_params'] = get_packed_seq_params(text_position_ids)
            batch['packed_seq_params'].num_samples = num_samples
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(args, batch)
        return batch

    def get_batch(self, data_iterator, vp_stage=None):
        """Generate a batch."""
        return self._prepare_batch(next(data_iterator), vp_stage)

    def _collect_config_info(self) -> Dict[str, str]:
        """
        Collects trainer-specific configuration details.

        Subclasses can override this method to provide additional configuration
        information for model compatibility verification.

        Returns:
            Dict[str, str]: Configuration parameters as key-value pairs.
        """
        if self.__class__.__name__ == 'MegatronTrainer':
            if not self.template.use_chat_template:
                return {
                    'seq2seq_mode': 'pt',
                }
            else:
                return {
                    'seq2seq_mode': 'sft',
                }
        return {}

    def get_last_tokens(self, output_tensor, packed_seq_params=None, attention_mask=None, num_samples=None):
        if packed_seq_params is None:
            last_token_idx = get_last_valid_indices((~attention_mask[:, 0, -1]).long())
            last_tokens = output_tensor[torch.arange(output_tensor.shape[0]), last_token_idx]
        else:
            num_samples = num_samples or packed_seq_params.num_samples
            last_token_idx = packed_seq_params.cu_seqlens_q[1:num_samples + 1] - 1
            last_tokens = output_tensor[0, last_token_idx]
        return last_tokens
