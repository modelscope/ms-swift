# Copyright (c) ModelScope Contributors. All rights reserved.
import collections
import dataclasses
import logging
import os
import shutil
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Callable, Dict, List, Literal, Optional

import megatron.core
import torch
import torch.nn
from megatron.core import mpu, tensor_parallel
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig, _update_min_and_max_lr_in_param_groups, get_megatron_optimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer.module import Float16Module, MegatronModule
# from megatron.training import (checkpointing, ft_integration, get_args, get_model, get_tensorboard_writer,
#                                get_wandb_writer, initialize, is_last_rank, one_logger_utils, pretrain, print_rank_0,
#                                print_rank_last, training)
# from megatron.training.checkpointing import check_checkpoint_args, set_checkpoint_version
from modelscope import check_local_model_is_latest
from packaging import version

from swift.megatron.callbacks import megatron_callbacks_map
from swift.megatron.model import get_mcore_model
from swift.megatron.tuners import LoraParallelLinear
from swift.megatron.utils import (copy_original_module_weight, get_optimizer_param_scheduler, get_padding_to,
                                  load_mcore_checkpoint, patch_merge_fn, prepare_mcore_model, save_mcore_checkpoint,
                                  wrap_model)
from swift.metrics import MeanMetric
from swift.template import Template
from swift.trainers import SwiftMixin, dynamic_gradient_checkpointing
from swift.trainers.utils import patch_modelscope_hub_timeout
from swift.utils import (JsonlWriter, deep_getattr, format_time, get_last_valid_indices, get_logger, is_last_rank,
                         ms_logger_context)
from .batch_sampler import MegatronPretrainingRandomSampler, MegatronPretrainingSampler
from .utils import (TrainerState, get_batch_on_this_cp_rank, get_batch_on_this_tp_rank, get_packed_seq_params,
                    logical_and_across_model_parallel_group, reduce_max_stat_across_model_parallel_group)

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
        # TODO: resume_from_checkpoint
        self.state = TrainerState()
        if args.initialize_embedding:
            for m in self.unwrapped_models:
                self._initialize_embedding(m)
        if args.tuner_type != 'full' and args.modules_to_save:
            for m in self.unwrapped_models:
                copy_original_module_weight(m)
        self._load_checkpoint()

        self.eval_metrics = None
        logging_path = os.path.join(args.output_dir, 'logging.jsonl')
        logger.info(f'logging_path: {logging_path}')

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

        def _get_mean_metric():
            return MeanMetric(nan_value=None, group=mpu.get_data_parallel_group(with_context_parallel=True))

        self.custom_metrics = {
            'train': collections.defaultdict(_get_mean_metric),
            'eval': collections.defaultdict(_get_mean_metric)
        }
        self.mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
        self.callbacks = []
        for callback in self.args.callbacks:
            self.callbacks.append(megatron_callbacks_map[callback](self))

    def _load_checkpoint(self):
        args = self.args
        if args.mcore_model is not None:
            self.state.iteration = load_mcore_checkpoint(
                args, self.wrapped_models, self.optimizer, self.opt_param_scheduler, load_arg='mcore_model')
        if args.mcore_adapter is not None:
            self.state.iteration = load_mcore_checkpoint(
                args, self.wrapped_models, self.optimizer, self.opt_param_scheduler, load_arg='mcore_adapter')
        if not args.finetune:
            # TODO: check
            self.state.iteration = self._load_iteration()

    def call_event(self, event, *args, **kwargs):
        if event == 'on_log':
            self._log_callback(*args, **kwargs)
        for callback in self.callbacks:
            getattr(callback, event)(*args, **kwargs)

    def _log_callback(self, logs):
        """This function is used to normalize logs for easier use with wandb/swanlab callbacks."""
        n_iters = logs.pop('n_iters', None)
        if n_iters is None:
            n_iters = logs.pop('eval_n_iters', None)
        n_iters = n_iters.item()
        for k, v in logs.items():
            v = v / n_iters
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, float):
                v = round(v, 8)
            logs[k] = v

    def prepare_model(self):
        args = self.args
        self.peft_models = []
        self.wrapped_models = []
        self.unwrapped_models = get_mcore_model(args, self.template.config)
        for model in self.unwrapped_models:
            peft_model = self._prepare_peft_model(model)
            self.peft_models.append(peft_model)
        self.wrapped_models = wrap_model(args, self.unwrapped_models)

    def _prepare_peft_model(self, model):
        args = self.args
        if args.mcore_model is None:
            self.bridge.load_weights(model, args.model_dir)
        peft_model = prepare_mcore_model(args, model)
        if args.tuner_type == 'lora' and args.adapters and args.mcore_adapter is None:
            assert len(args.adapters) == 1, 'Currently only support one adapter.'
            self.bridge.load_weights(model, args.adapters[0], is_peft_format=True, adapter_name='default')
        return peft_model

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

    def cyclic_iter(self, iterable):
        training = self.unwrapped_models[0].training
        assert training, 'training must be True'

        args = self.args
        state = self.state
        is_finished = False
        while True:
            if not is_finished:
                logger.info(f'The training of Epoch {state.epoch} starts...')
            for x in iterable:
                yield x
            # streaming
            if training and args.max_epochs and state.epoch >= args.max_epochs - 1:
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
        args = self.args
        is_multimodal = args.megatron_model_meta.is_multimodal
        if args.vit_lr is not None or args.aligner_lr is not None:
            assert is_multimodal, 'vit_lr and aligner_lr are only supported for multimodal models.'
            vit_lr = args.vit_lr if args.vit_lr is not None else args.lr
            aligner_lr = args.aligner_lr if args.aligner_lr is not None else args.lr
            logger.info(f'vit_lr: {vit_lr}, aligner_lr: {aligner_lr}, llm_lr: {args.lr}')
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
        elif args.tuner_type == 'lora' and args.adapters:
            ckpt_dir = args.adapters[0]
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
        set_checkpoint_version(state_dict.get('checkpoint_version', 0))
        if 'args' in state_dict and not args.finetune:
            checkpoint_args = state_dict['args']
            check_checkpoint_args(checkpoint_args)
            args.consumed_train_samples = getattr(checkpoint_args, 'consumed_train_samples', 0)
        else:
            print_rank_0('could not find arguments in the checkpoint ...')

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
                           reduction=torch.distributed.ReduceOp.AVG) -> Dict[str, torch.Tensor]:
        reporting_metric = torch.stack(list(metric.values()), dim=0)
        torch.distributed.all_reduce(reporting_metric, reduction, group=mpu.get_data_parallel_group())
        return {k: reporting_metric[i] for i, k in enumerate(metric.keys())}

    def _get_metrics(self, total_loss_dict, mode):
        advanced_iters = total_loss_dict['advanced iterations'] if mode == 'train' else 1
        return {
            k: torch.tensor([v * advanced_iters], device='cuda')
            for k, v in SwiftMixin.compute_custom_metrics(self.custom_metrics[mode]).items()
        }

    # def _remove_log(self, total_loss_dict):
    #     pass

    # def custom_log(self, total_loss_dict, mode: Literal['train', 'eval'], iteration=None) -> None:
    #     writer = get_tensorboard_writer()
    #     wandb_writer = get_wandb_writer()
    #     metrics = self._get_metrics(total_loss_dict, mode)
    #     total_loss_dict.update(metrics)
    #     self._remove_log(total_loss_dict)
    #     if iteration is None:
    #         args = self.args
    #         iteration = state.iteration + 1
    #     if writer:
    #         for k, v in metrics.items():
    #             writer.add_scalar(k, v, iteration)
    #     if wandb_writer:
    #         wandb_writer.log(metrics, iteration)

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
        if not is_last_rank():
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

    def train(self, train_dataset, val_dataset):
        args = self.args
        train_dataloader, val_dataloader = self.prepare_dataloader(train_dataset, val_dataset)
        for m in self.wrapped_models:
            m.train()

        if args.is_multimodal:
            for m in self.unwrapped_models:
                self._prepare_vit_gradient_checkpointing(m)

        self.config.finalize_model_grads_func = finalize_model_grads
        # TODO: manual_gc
        self.call_event('on_train_begin')
        train_metrics = {}
        train_data_iterator = iter(self.cyclic_iter(train_dataloader))
        state = self.state
        while state.iteration < args.train_iters:
            self.call_event('on_step_begin')
            metrics, grad_norm = self.train_step(train_data_iterator)
            self.call_event('on_step_end')
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                self.aggregated_metrics(metrics, train_metrics)
                train_metrics['grad_norm'] = grad_norm
                # TODO: check vit_lr
                learning_rate = None
                for param_group in self.optimizer.param_groups:
                    if len(param_group['params']) == 0:
                        continue
                    learning_rate = param_group['lr']
                if learning_rate is not None:
                    train_metrics['learning_rate'] = learning_rate
            if state.should_log:
                state.should_log = False
                self.call_event('on_log', train_metrics)
                train_metrics = {}

            if state.should_eval:
                state.should_eval = False
                self.evaluate(val_dataloader)
                for m in self.wrapped_models:
                    m.train()

            if state.should_save:
                state.should_save = False
                self.save_checkpoint()

        self.call_event('on_train_end')

    def save_checkpoint(self):
        args = self.args
        iteration = self.state.iteration
        output_dir = os.path.join(args.output_dir, f'checkpoint-{iteration}')
        os.makedirs(output_dir, exist_ok=True)
        origin_output_dir = args.output_dir
        args.output_dir = output_dir
        args_path = os.path.join(os.path.dirname(output_dir), 'args.json')
        self.copy_path(args_path, os.path.join(output_dir, 'args.json'))
        save_peft_format = args.tuner_type == 'lora' and not args.merge_lora
        save_mcore_checkpoint(
            self.args, self.wrapped_models, self.optimizer, self.opt_param_scheduler, iteration=iteration)
        args.output_dir = origin_output_dir
        # safetensors
        if args.save_safetensors:
            # merge-lora does not store lora, lora saving may report an error (Qwen3-VL-Moe)
            if args.tuner_type == 'lora' and args.merge_lora:
                self.merge_lora_adapters()
                origin_output_dir = output_dir
                output_dir = f'{output_dir}-merged'
                os.makedirs(output_dir, exist_ok=True)
                # for fname in ['latest_checkpointed_iteration.txt', 'args.json']:
                #     src_path = os.path.join(origin_output_dir, fname)
                #     self.copy_path(src_path, os.path.join(output_dir, fname))
                # # common.pt
                # common_path = os.path.join(origin_output_dir, f'iter_{iteration:07d}', 'common.pt')
                # tgt_common_path = os.path.join(output_dir, f'iter_{iteration:07d}', 'common.pt')
                # os.makedirs(os.path.dirname(tgt_common_path), exist_ok=True)
                # self.copy_path(common_path, tgt_common_path)
            self.bridge.save_weights(
                self.unwrapped_models,
                output_dir,
                is_peft_format=save_peft_format,
                processor=self.template.processor,
                hf_config=self.template.config)
            if args.tuner_type == 'lora' and args.merge_lora:
                self.unmerge_lora_adapters()

    def training_log(self, metrics, grad_norm):
        learning_rate = None
        for param_group in self.optimizer.param_groups:
            if len(param_group['params']) == 0:
                continue
            learning_rate = param_group['lr']
        logger.info(f'metrics: {metrics}, grad_norm: {grad_norm}, learning_rate: {learning_rate}')

    def evaluate(self, val_dataloader):
        # TODO: 兼容transformers callback, eval_metrics等
        args = self.args
        for m in self.wrapped_models:
            m.eval()
        eval_metrics = {}
        forward_backward_func = get_forward_backward_func()
        val_data_iterator = iter(val_dataloader)

        self.call_event('on_eval_begin')
        with torch.no_grad():
            while self.state.eval_iteration < args.eval_iters:
                metrics = forward_backward_func(
                    forward_step_func=self.forward_step,
                    data_iterator=val_data_iterator,
                    model=self.wrapped_models,
                    num_microbatches=self.args.num_micro_batches,
                    seq_length=args.max_length,
                    micro_batch_size=args.micro_batch_size,
                    forward_only=True,
                )
                self.call_event('on_eval_step')
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    self.aggregated_metrics(metrics, eval_metrics)
        eval_metrics = {f'eval_{k}': v for k, v in eval_metrics.items()}
        self.call_event('on_log', eval_metrics)
        self.call_event('on_eval_end')

    def train_step(self, train_data_iterator):
        args = self.args
        forward_backward_func = get_forward_backward_func()
        for m in self.wrapped_models:
            m.zero_grad_buffer()
        self.optimizer.zero_grad()
        metrics = forward_backward_func(
            forward_step_func=self.forward_step,
            data_iterator=train_data_iterator,
            model=self.wrapped_models,
            num_microbatches=args.num_micro_batches,
            seq_length=args.max_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=False,
        )

        _, grad_norm, _ = self.optimizer.step()
        grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
        self.opt_param_scheduler.step(increment=args.global_batch_size)

        return metrics, grad_norm

    def aggregated_metrics(self, metrics, total_metrics):
        if 'n_iters' not in total_metrics:
            total_metrics['n_iters'] = torch.tensor([0], dtype=torch.int64, device=torch.cuda.current_device())
        total_metrics['n_iters'] += 1
        for key in metrics[0].keys():
            if key not in total_metrics:
                total_metrics[key] = torch.tensor([0.0], dtype=torch.float32, device=torch.cuda.current_device())
            val = [x[key].view(-1) for x in metrics]
            val = torch.stack(val, dim=0)
            if val[0].numel() == 2:
                val = val.sum(dim=0)
                total_metrics[key] += val[0] / val[1]
            elif val[0].numel() == 1:
                total_metrics[key] += val.sum()
            else:
                raise ValueError(f'Invalid value shape: {val[0].shape} for key {key}')

    def prepare_dataloader(self, train_dataset, val_dataset):
        args = self.args

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
        val_dataloader = None
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
        batch = get_batch_on_this_tp_rank(self.args, data, vp_stage=vp_stage)
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
