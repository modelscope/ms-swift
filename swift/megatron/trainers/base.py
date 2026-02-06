# Copyright (c) ModelScope Contributors. All rights reserved.
import collections
import dataclasses
import logging
import math
import os
import shutil
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Literal, Optional

import megatron.core
import torch
import torch.nn
from megatron.core import mpu, tensor_parallel
from megatron.core.datasets.utils import Split
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches, update_num_microbatches
from megatron.core.optimizer import OptimizerConfig, _update_min_and_max_lr_in_param_groups, get_megatron_optimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
# from megatron.training import (checkpointing, ft_integration, get_args, get_model, get_tensorboard_writer,
#                                get_wandb_writer, initialize, is_last_rank, one_logger_utils, pretrain, print_rank_0,
#                                print_rank_last, training)
# from megatron.training.checkpointing import check_checkpoint_args, set_checkpoint_version
# from megatron.training.theoretical_memory_usage import report_theoretical_memory
# from megatron.training.utils import reduce_max_stat_across_model_parallel_group, report_memory
from modelscope import check_local_model_is_latest
from packaging import version
from tqdm.auto import tqdm

from swift.megatron.model import get_mcore_model
from swift.megatron.tuners import LoraParallelLinear
from swift.megatron.utils import (adapter_state_dict_context, copy_original_module_weight,
                                  get_optimizer_param_scheduler, get_padding_to, load_mcore_checkpoint, patch_merge_fn,
                                  prepare_mcore_model, wrap_model)
from swift.metrics import MeanMetric
from swift.template import Template
from swift.trainers import SwiftMixin, dynamic_gradient_checkpointing
from swift.trainers.utils import patch_modelscope_hub_timeout
from swift.utils import (JsonlWriter, deep_getattr, format_time, get_last_valid_indices, get_logger, is_last_rank,
                         ms_logger_context)
from .batch_sampler import MegatronPretrainingRandomSampler, MegatronPretrainingSampler
from .utils import (get_batch_on_this_cp_rank, get_batch_on_this_tp_rank, get_packed_seq_params,
                    logical_and_across_model_parallel_group, reduce_max_stat_across_model_parallel_group)

# try:
#     from megatron.training.datasets.data_samplers import MegatronPretrainingSampler
# except ImportError:
#     from megatron.legacy.data.data_samplers import MegatronPretrainingSampler

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
        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler()
        self.data_collator = self._get_data_collator()
        # TODO: resume_from_checkpoint
        args.iteration = 0
        if args.initialize_embedding:
            for m in self.unwrapped_models:
                self._initialize_embedding(m)
        if args.tuner_type != 'full' and args.modules_to_save:
            for m in self.unwrapped_models:
                copy_original_module_weight(m)
        if args.ref_adapter_load is not None:
            with self._patch_load_state_dict(self._load_adapter_base_checkpoint):
                load_mcore_checkpoint(args, self.wrapped_models, load_arg='ref_adapter_load')
        if args.adapter_load is not None:
            with adapter_state_dict_context():
                args.iteration = load_mcore_checkpoint(
                    args, self.wrapped_models, self.optimizer, self.scheduler, load_arg='adapter_load')

        self.eval_metrics = None
        logging_path = os.path.join(args.save, 'logging.jsonl')
        logger.info(f'logging_path: {logging_path}')
        self.jsonl_writer = JsonlWriter(logging_path, enable_async=True, write_on_rank='last')  # for evaluate

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

    def prepare_model(self):
        args = self.args
        hf_config = self.template.config
        self.peft_models = []
        self.wrapped_models = []
        self.unwrapped_models = get_mcore_model(args, hf_config)
        for model in self.unwrapped_models:
            if args.load is None:
                self.bridge.load_weights(model, args.model_dir)

            model = prepare_mcore_model(args, model)
            if args.tuner_type == 'lora':
                if args.adapters and args.adapter_load is None:
                    assert len(args.adapters) == 1, 'Currently only support one adapter.'
                    self.bridge.load_weights(model, args.adapters[0], is_peft_format=True, adapter_name='default')
                if args.ref_adapters and args.ref_adapter_load is None:
                    assert len(args.ref_adapters) == 1, 'Currently only support one adapter.'
                    self.bridge.load_weights(
                        model, args.ref_adapters[0], is_peft_format=True, adapter_name='ref_adapter')
            self.peft_models.append(model)
        self.wrapped_models = wrap_model(args, self.peft_models)

    def get_optimizer_and_scheduler(self):
        args = self.args
        kwargs = {}
        for f in dataclasses.fields(OptimizerConfig):
            if hasattr(args, f.name) and f.name != 'loss_scale':
                kwargs[f.name] = getattr(args, f.name)
        config = OptimizerConfig(**kwargs)
        optimizer = get_megatron_optimizer(
            config,
            self.wrapped_models,
        )
        scheduler = get_optimizer_param_scheduler(args, optimizer)
        return optimizer, scheduler

    def _get_data_collator(self):
        data_collator = self.template.data_collator
        padding_to = get_padding_to(self.args)
        logger.info(f'padding_to: {padding_to}')
        data_collator = partial(data_collator, padding_to=padding_to)
        return data_collator

    def new_cyclic_iter(self, iterable):
        training = self.unwrapped_models[0].training
        if not training:
            yield from self._origin_cyclic_iter(iterable)
            return

        args = self.args
        n_epoch = 0
        is_finished = False
        while True:
            if not is_finished:
                logger.info(f'The training of Epoch {n_epoch} starts...')
            for x in iterable:
                yield x
            if training and args.max_epochs and n_epoch >= args.max_epochs - 1:
                is_finished = True
            n_epoch += 1
            if is_finished:
                # streaming
                # Note that this approach will train for one additional step.
                logger.info(f'Training of {n_epoch} epochs has been completed, the training has finished.')
                args.train_iters = args.curr_iteration + 1

    def _replace_data_iterator(self, data_iterator, model):
        return data_iterator

    def _load_adapter_base_checkpoint(self, *_args, **kwargs):
        adapter_name = kwargs.pop('adapter_name', None) or 'ref_adapter'
        sharded_state_dict = kwargs.get('sharded_state_dict')
        if sharded_state_dict is None:
            return checkpointing.origin__load_base_checkpoint(*_args, **kwargs)
        model_keys = [k for k in sharded_state_dict.keys() if k.startswith('model')]
        mapping = {}
        for model_k in model_keys:
            mapping[model_k] = {}
            state_dict_model = {}
            for k, v in sharded_state_dict[model_k].items():
                if adapter_name not in k:
                    continue
                # lora
                origin_k = k
                k = k.replace(f'.{adapter_name}.', '.default.')
                mapping[model_k][k] = origin_k
                v.key = v.key.replace(f'.{adapter_name}.', '.default.')
                state_dict_model[k] = v
            sharded_state_dict[model_k] = state_dict_model
            patch_merge_fn(state_dict_model)
        res = checkpointing.origin__load_base_checkpoint(*_args, **kwargs)
        for model_k in model_keys:
            state_dict = res[0][model_k]
            for k, origin_k in mapping[model_k].items():
                v = state_dict.pop(k)
                state_dict[origin_k] = v
        return res

    def _load_base_checkpoint(self, *_args, **kwargs):
        sharded_state_dict = kwargs.get('sharded_state_dict')
        if sharded_state_dict is None:
            return checkpointing.origin__load_base_checkpoint(*_args, **kwargs)
        model_keys = [k for k in sharded_state_dict.keys() if k.startswith('model')]
        if self.args.tuner_type == 'full':
            for k in model_keys:
                patch_merge_fn(sharded_state_dict[k])
            return checkpointing.origin__load_base_checkpoint(*_args, **kwargs)
        mapping = {}
        for model_k in model_keys:
            mapping[model_k] = {}
            state_dict_model = {}
            for k, v in sharded_state_dict[model_k].items():
                if 'lora_A' in k or 'lora_B' in k or 'original_module' in k:
                    continue
                # lora
                if '.base_layer' in k:
                    origin_k = k
                    k = k.replace('.base_layer', '')
                    mapping[model_k][k] = origin_k
                    v.key = v.key.replace('.base_layer', '')
                elif '.modules_to_save' in k:
                    if '.modules_to_save.default' not in k:
                        # e.g. ref_adapter
                        continue
                    # modules to save
                    origin_k = k
                    k = k.replace('.modules_to_save.default', '')
                    mapping[model_k][k] = origin_k
                    v.key = v.key.replace('.modules_to_save.default', '')
                state_dict_model[k] = v
            sharded_state_dict[model_k] = state_dict_model
            patch_merge_fn(state_dict_model)
        res = checkpointing.origin__load_base_checkpoint(*_args, **kwargs)
        for model_k in model_keys:
            state_dict = res[0][model_k]
            for k, origin_k in mapping[model_k].items():
                v = state_dict.pop(k)
                state_dict[origin_k] = v
        return res

    @contextmanager
    def _patch_load_state_dict(self, load_base_checkpoint):
        checkpointing.origin__load_base_checkpoint = checkpointing._load_base_checkpoint
        checkpointing._load_base_checkpoint = load_base_checkpoint

        args = self.args
        origin_load_state_dict = torch.nn.Module.load_state_dict
        origin_no_load_optim = args.no_load_optim
        origin_no_load_rng = args.no_load_rng
        origin_finetune = args.finetune

        def load_state_dict(self, state_dict, strict: bool = True, *args, **kwargs):
            strict = False
            return origin_load_state_dict(self, state_dict, strict, *args, **kwargs)

        if args.tuner_type != 'full':
            torch.nn.Module.load_state_dict = load_state_dict
            args.no_load_optim = True
            args.no_load_rng = True
            args.finetune = True
        try:
            yield
        finally:
            checkpointing._load_base_checkpoint = checkpointing.origin__load_base_checkpoint
            torch.nn.Module.load_state_dict = origin_load_state_dict
            args.no_load_optim = origin_no_load_optim
            args.no_load_rng = origin_no_load_rng
            args.finetune = origin_finetune

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
        is_multimodal = self.args.megatron_model_meta.is_multimodal
        if self.args.vit_lr is not None or self.args.aligner_lr is not None:
            assert is_multimodal, 'vit_lr and aligner_lr are only supported for multimodal models.'
            vit_lr = self.args.vit_lr if self.args.vit_lr is not None else self.args.lr
            aligner_lr = self.args.aligner_lr if self.args.aligner_lr is not None else self.args.lr
            logger.info(f'vit_lr: {vit_lr}, aligner_lr: {aligner_lr}, llm_lr: {self.args.lr}')
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
                    if is_vit and self.args.vit_lr:
                        scale_lr = True
                        _lr_mult = self.args.vit_lr / lr
                    elif is_aligner and self.args.aligner_lr:
                        scale_lr = True
                        _lr_mult = self.args.aligner_lr / lr

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
            return 0, 0
        logger.info(f'checkpoint_dir: {ckpt_dir}')
        tracker_path = os.path.join(ckpt_dir, 'latest_checkpointed_iteration.txt')
        if not os.path.exists(tracker_path):
            return 0, 0
        with open(tracker_path, 'r') as f:
            iteration = int(f.read())

        common_path = os.path.join(ckpt_dir, f'iter_{iteration:07d}', 'common.pt')
        if not os.path.exists(common_path):
            return iteration, 0

        state_dict = torch.load(common_path)
        set_checkpoint_version(state_dict.get('checkpoint_version', 0))
        if 'args' in state_dict and not args.finetune:
            checkpoint_args = state_dict['args']
            check_checkpoint_args(checkpoint_args)
            args.consumed_train_samples = getattr(checkpoint_args, 'consumed_train_samples', 0)
            args.skipped_train_samples = getattr(checkpoint_args, 'skipped_train_samples', 0)
            update_num_microbatches(consumed_samples=args.consumed_train_samples, verbose=True)
            # TODO: ignore
            # args.consumed_valid_samples = getattr(checkpoint_args, 'consumed_valid_samples', 0)
        else:
            print_rank_0('could not find arguments in the checkpoint ...')

        return iteration

    def setup_model_and_optimizer(self, model_provider_func, model_type, *_args, **kwargs):

        # read iteration
        args = self.args
        if not args.finetune:
            args.iteration = self._load_iteration()

        if args.apply_wd_to_qk_layernorm or self.args.vit_lr is not None or self.args.aligner_lr is not None:
            param_groups_context = self._patch_get_param_groups()
        else:
            param_groups_context = nullcontext()
        with self._patch_load_state_dict(self._load_base_checkpoint), param_groups_context:
            model, optimizer, opt_param_scheduler = self._origin_setup_model_and_optimizer(
                new_model_provider_func, model_type, *_args, **kwargs)
        self.wrapped_models = model
        return model, optimizer, opt_param_scheduler

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

    # Code borrowed from NVIDIA/Megatron-LM
    def _evaluate(
        self,
        forward_step_func,
        data_iterator,
        model,
        process_non_loss_data_func,
        config,
        verbose=False,
        non_loss_data_func=None,
        eval_iters=None,
    ):
        """Evaluation."""
        args = self.args
        timers = get_timers()

        timers('evaluate', log_level=0).start(barrier=True)
        if args.vision_pretraining and args.vision_pretraining_type == 'dino':
            from megatron.legacy.model.vision.knn_monitor import compute_feature_bank
            compute_feature_bank(model)

        # Turn on evaluation mode which disables dropout.
        for model_module in model:
            model_module.eval()

        # Disable result validation during evaluation
        rerun_state_machine = get_rerun_state_machine()
        rerun_mode = rerun_state_machine.get_mode()
        rerun_state_machine.set_mode(RerunMode.DISABLED)

        total_loss_dict = {}

        # make validation batch size independent from training batch size
        eval_batch_size = args.global_batch_size
        eval_num_microbatches = eval_batch_size // (args.micro_batch_size * args.data_parallel_size)
        forward_backward_func = get_forward_backward_func()
        if args.enable_cuda_graph and args.cuda_graph_scope == 'full_iteration':
            from megatron.core.full_cuda_graph import FullCudaGraphWrapper
            forward_backward_func = FullCudaGraphWrapper(
                forward_backward_func, cuda_graph_warmup_steps=args.cuda_graph_warmup_steps)

        if eval_iters is None:
            eval_iters = args.eval_iters

        with torch.no_grad(), tqdm(
                total=eval_iters, dynamic_ncols=True, disable=not is_last_rank(), desc='Evaluate: ') as prog_bar:
            iteration = 0
            if verbose:
                print_rank_0(f'Evaluating on {eval_iters * eval_batch_size} samples')
            while iteration < eval_iters:
                iteration += 1
                prog_bar.update()
                if verbose:
                    print_rank_0(f'Evaluating iter {iteration}/{eval_iters}')

                # Don't care about timing during evaluation
                config.timers = None
                ft_integration.on_eval_step_start()
                new_data_iterator = self._replace_data_iterator(data_iterator, model)
                loss_dicts = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=new_data_iterator,
                    model=model,
                    num_microbatches=eval_num_microbatches,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    forward_only=True,
                )
                ft_integration.on_eval_step_end()
                config.timers = get_timers()

                # Empty unused memory
                if args.empty_unused_memory_level >= 1:
                    torch.cuda.empty_cache()

                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    if self.mcore_013:
                        for key in loss_dicts[0].keys():
                            if key not in total_loss_dict:
                                total_loss_dict[key] = torch.tensor([0.0, 0.0], dtype=torch.float).cuda()
                            val = [x[key].view(-1) for x in loss_dicts]
                            if val[0].numel() == 2:
                                val = torch.vstack(val).sum(dim=0)
                                torch.distributed.all_reduce(
                                    val, group=mpu.get_data_parallel_group(with_context_parallel=True))
                                total_loss_dict[key] += val
                            elif val[0].numel() == 1:
                                val = torch.cat(val).sum()
                                total_loss_dict[key][0] += val
                                total_loss_dict[key][1] += len(loss_dicts)
                            else:
                                raise ValueError(f'Invalid value shape: {val[0].shape} for key {key}')
                    else:
                        # Reduce across processes.
                        for loss_dict in loss_dicts:
                            for key in loss_dict:
                                if key not in total_loss_dict:
                                    total_loss_dict[key] = torch.tensor([0.0, 0.0], dtype=torch.float).cuda()
                                val = loss_dict[key]
                                if isinstance(val, tuple) or isinstance(val, list):
                                    total_loss_dict[key][0] += val[0]
                                    total_loss_dict[key][1] += val[1]
                                else:
                                    total_loss_dict[key][0] += val
                                    total_loss_dict[key][1] += 1
                # args.consumed_valid_samples += eval_batch_size

                if args.exit_duration_in_mins:
                    train_time = (time.time() - training._TRAIN_START_TIME) / 60.0
                    done_cuda = torch.tensor([train_time > args.exit_duration_in_mins], dtype=torch.int, device='cuda')
                    torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
                    done = done_cuda.item()
                    if done:
                        rerun_state_machine.set_mode(rerun_mode)
                        print_rank_0('Exiting during evaluation, timelimit reached')
                        return None, None, True

            collected_non_loss_data = None
            if non_loss_data_func is not None:
                collected_non_loss_data = non_loss_data_func(model)
            elif process_non_loss_data_func is not None and is_last_rank():
                collected_non_loss_data = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=get_num_microbatches(),
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    forward_only=True,
                    collect_non_loss_data=True,
                )

        # Move model back to the train mode.
        for model_module in model:
            model_module.train()

        for key in total_loss_dict:
            numerator, denominator = total_loss_dict[key]
            total_loss_dict[key] = numerator / denominator
        if self.eval_metrics is not None:
            metric = self.eval_metrics.compute()
            for k, v in metric.items():
                total_loss_dict[k] = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            self.eval_metrics.reset()
        timers('evaluate').stop()
        timers.log(['evaluate'])
        self.custom_log(total_loss_dict, 'eval')
        rerun_state_machine.set_mode(rerun_mode)
        if is_last_rank():
            logs = {}
            for key, val in total_loss_dict.items():
                logs[f'eval_{key}'] = round(val.item(), 8)
            self.jsonl_writer.append(logs)
        return total_loss_dict, collected_non_loss_data, False

    def _evaluate_and_print_results(
        self,
        prefix,
        forward_step_func,
        data_iterator,
        model,
        iteration,
        process_non_loss_data_func,
        config,
        verbose=False,
        write_to_tensorboard=True,
        non_loss_data_func=None,
    ):
        """Helper function to evaluate and dump results on screen."""

        args = self.args
        if write_to_tensorboard:
            writer = get_tensorboard_writer()
        else:
            writer = None

        wandb_writer = get_wandb_writer()

        data_iterators = data_iterator if args.multiple_validation_sets else [data_iterator]

        if not args.multiple_validation_sets:
            eval_iters = [args.eval_iters]
        else:
            eval_iters = args.eval_iters

        if args.full_validation:
            assert len(eval_iters) == len(data_iterators)

            # with full validation we need to distribute eval_iters to all ranks
            if mpu.get_tensor_model_parallel_rank() == 0:
                eval_iters = torch.tensor(args.eval_iters, dtype=torch.long, device='cuda')
            else:
                eval_iters = torch.tensor([0] * len(eval_iters), dtype=torch.long, device='cuda')
            torch.distributed.broadcast(eval_iters, 0)
            eval_iters = eval_iters.tolist()
            args.eval_iters = eval_iters[0] if not args.multiple_validation_sets else eval_iters
        elif not args.multiple_validation_sets:
            eval_iters = [args.eval_iters]
        else:
            eval_iters = args.eval_iters

        for index, (iterator, iterations) in enumerate(zip(data_iterators, eval_iters)):
            suffix = ''
            if args.multiple_validation_sets:
                suffix = f'-{index}'
            total_loss_dict, collected_non_loss_data, timelimit = self.evaluate(
                forward_step_func,
                iterator,
                model,
                process_non_loss_data_func,
                config,
                verbose,
                non_loss_data_func,
                eval_iters=iterations,
            )
            # Timelimit hit during evaluation
            if timelimit:
                return
            string = f' validation{suffix} loss at {prefix} | '
            for key in total_loss_dict:
                string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
                ppl = None
                if key == 'loss':
                    ppl = math.exp(min(20, total_loss_dict[key].item()))
                    string += '{} PPL: {:.6E} | '.format(key, ppl)
                if writer:
                    writer.add_scalar('{} validation{}'.format(key, suffix), total_loss_dict[key].item(), iteration)
                    writer.add_scalar(
                        '{} validation{} vs samples'.format(key, suffix),
                        total_loss_dict[key].item(),
                        args.consumed_train_samples,
                    )
                    if args.log_validation_ppl_to_tensorboard and ppl is not None:
                        writer.add_scalar('{} validation{} ppl'.format(key, suffix), ppl, iteration)
                        writer.add_scalar('{} validation{} ppl vs samples'.format(key, suffix), ppl,
                                          args.consumed_train_samples)
                    if wandb_writer and is_last_rank():
                        wandb_writer.log({'{} validation{}'.format(key, suffix): total_loss_dict[key].item()},
                                         iteration)

            if process_non_loss_data_func is not None and writer and is_last_rank():
                process_non_loss_data_func(collected_non_loss_data, iteration, writer)

            length = len(string) + 1
            print_rank_last('-' * length)
            print_rank_last(string)
            print_rank_last('-' * length)

    def _get_metrics(self, total_loss_dict, mode):
        advanced_iters = total_loss_dict['advanced iterations'] if mode == 'train' else 1
        return {
            k: torch.tensor([v * advanced_iters], device='cuda')
            for k, v in SwiftMixin.compute_custom_metrics(self.custom_metrics[mode]).items()
        }

    def _remove_log(self, total_loss_dict):
        pass

    def custom_log(self, total_loss_dict, mode: Literal['train', 'eval'], iteration=None) -> None:
        writer = get_tensorboard_writer()
        wandb_writer = get_wandb_writer()
        metrics = self._get_metrics(total_loss_dict, mode)
        total_loss_dict.update(metrics)
        self._remove_log(total_loss_dict)
        if iteration is None:
            args = self.args
            iteration = args.curr_iteration + 1
        if writer:
            for k, v in metrics.items():
                writer.add_scalar(k, v, iteration)
        if wandb_writer:
            wandb_writer.log(metrics, iteration)

    # Code borrowed from NVIDIA/Megatron-LM
    def _training_log(self, loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration, loss_scale,
                      report_memory_flag, skipped_iter, grad_norm, params_norm, num_zeros_in_grad):
        """Log training information such as losses, timing, ...."""
        args = self.args
        timers = get_timers()
        writer = get_tensorboard_writer()
        wandb_writer = get_wandb_writer()

        # Advanced, skipped, and Nan iterations.
        advanced_iters_key = 'advanced iterations'
        skipped_iters_key = 'skipped iterations'
        nan_iters_key = 'nan iterations'
        # Advanced iterations.
        if not skipped_iter:
            total_loss_dict[advanced_iters_key] = total_loss_dict.get(advanced_iters_key, 0) + 1
        else:
            if advanced_iters_key not in total_loss_dict:
                total_loss_dict[advanced_iters_key] = 0
        # Skipped iterations.
        total_loss_dict[skipped_iters_key] = total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
        # Update losses and set nan iterations
        got_nan = False
        for key in loss_dict:
            if not skipped_iter:
                total_loss_dict[key] = total_loss_dict.get(key, torch.tensor([0.0], dtype=torch.float,
                                                                             device='cuda')) + loss_dict[key]
            else:
                value = loss_dict[key].float().sum().item()
                is_nan = value == float('inf') or value == -float('inf') or value != value
                got_nan = got_nan or is_nan
        total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(got_nan)

        # Logging.
        timers_to_log = [
            'forward-backward', 'forward-compute', 'backward-compute', 'batch-generator', 'forward-recv',
            'forward-send', 'backward-recv', 'backward-send', 'forward-send-forward-recv', 'forward-send-backward-recv',
            'backward-send-forward-recv', 'backward-send-backward-recv', 'forward-backward-send-forward-backward-recv',
            'layernorm-grads-all-reduce', 'embedding-grads-all-reduce', 'all-grads-sync', 'params-all-gather',
            'optimizer-copy-to-main-grad', 'optimizer-unscale-and-check-inf', 'optimizer-clip-main-grad',
            'optimizer-count-zeros', 'optimizer-inner-step', 'optimizer-copy-main-to-model-params', 'optimizer'
        ]

        # Calculate batch size.
        batch_size = args.micro_batch_size * args.data_parallel_size * get_num_microbatches()

        # Track app tag & app tag ID
        one_logger_utils.track_app_tag(batch_size, args.world_size, args.seq_length)

        total_iterations = total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]

        # learning rate will be None on ranks without trainable params, so we must gather across mp ranks
        learning_rate = reduce_max_stat_across_model_parallel_group(learning_rate)
        # Tensorboard values.
        # Timer requires all the ranks to call.
        if args.log_timers_to_tensorboard and (iteration % args.tensorboard_log_interval == 0):
            timers.write(timers_to_log, writer, iteration, normalizer=total_iterations)
        if writer and (iteration % args.tensorboard_log_interval == 0):
            if wandb_writer:
                wandb_writer.log({'samples vs steps': args.consumed_train_samples}, iteration)
            writer.add_scalar('learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'learning-rate': learning_rate}, iteration)
            if args.decoupled_lr is not None:
                writer.add_scalar('decoupled-learning-rate', decoupled_learning_rate, iteration)
            if args.skipped_train_samples > 0:
                writer.add_scalar('skipped-train-samples', args.skipped_train_samples, iteration)
                if wandb_writer:
                    wandb_writer.log({'skipped-train-samples': args.skipped_train_samples}, iteration)
            writer.add_scalar('batch-size', batch_size, iteration)
            writer.add_scalar('batch-size vs samples', batch_size, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'batch-size': batch_size}, iteration)
            log_loss_dict = loss_dict.copy()
            self._remove_log(log_loss_dict)
            for key in log_loss_dict:
                writer.add_scalar(key, loss_dict[key], iteration)
                writer.add_scalar(key + ' vs samples', loss_dict[key], args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({key: loss_dict[key]}, iteration)
            if args.log_loss_scale_to_tensorboard:
                writer.add_scalar('loss-scale', loss_scale, iteration)
                writer.add_scalar('loss-scale vs samples', loss_scale, args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'loss-scale': loss_scale}, iteration)
            if args.log_world_size_to_tensorboard:
                writer.add_scalar('world-size', args.world_size, iteration)
                writer.add_scalar('world-size vs samples', args.world_size, args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'world-size': args.world_size}, iteration)
            if grad_norm is not None:
                writer.add_scalar('grad-norm', grad_norm, iteration)
                writer.add_scalar('grad-norm vs samples', grad_norm, args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'grad-norm': grad_norm}, iteration)
            if num_zeros_in_grad is not None:
                writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
                writer.add_scalar('num-zeros vs samples', num_zeros_in_grad, args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'num-zeros': num_zeros_in_grad}, iteration)
            if params_norm is not None:
                writer.add_scalar('params-norm', params_norm, iteration)
                writer.add_scalar('params-norm vs samples', params_norm, args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'params-norm': params_norm}, iteration)
            if args.log_memory_to_tensorboard:
                mem_stats = torch.cuda.memory_stats()
                writer.add_scalar(
                    'mem-reserved-bytes',
                    mem_stats['reserved_bytes.all.current'],
                    iteration,
                )
                writer.add_scalar(
                    'mem-allocated-bytes',
                    mem_stats['allocated_bytes.all.current'],
                    iteration,
                )
                writer.add_scalar(
                    'mem-max-allocated-bytes',
                    mem_stats['allocated_bytes.all.peak'],
                    iteration,
                )
                writer.add_scalar(
                    'mem-allocated-count',
                    mem_stats['allocation.all.current'],
                    iteration,
                )
        if args.num_experts is not None:
            moe_loss_scale = 1 / get_num_microbatches()
            track_names = []
            if args.moe_router_load_balancing_type in ['aux_loss', 'seq_aux_loss']:
                track_names.append('load_balancing_loss')
            if args.moe_z_loss_coeff is not None:
                track_names.append('z_loss')
            track_moe_kwargs = {'mtp_num_layers': args.mtp_num_layers} if self.mcore_013 else {}
            track_moe_metrics(
                loss_scale=moe_loss_scale,
                iteration=iteration,
                writer=writer,
                wandb_writer=wandb_writer,
                total_loss_dict=total_loss_dict,
                per_layer_logging=args.moe_per_layer_logging,
                force_initialize=True,
                track_names=track_names,
                num_layers=args.num_layers,
                moe_layer_freq=args.moe_layer_freq,
                **track_moe_kwargs)
        if args.mtp_num_layers is not None:
            mtp_loss_scale = 1 / get_num_microbatches()
            MTPLossLoggingHelper.track_mtp_metrics(mtp_loss_scale, iteration, writer, wandb_writer, total_loss_dict)
        if iteration % args.log_interval == 0 or iteration == 1:
            self.custom_log(total_loss_dict, 'train')
            origin_total_loss_dict = total_loss_dict.copy()

            if args.record_memory_history and is_last_rank():
                snapshot = torch.cuda.memory._snapshot()
                from pickle import dump
                with open(args.memory_snapshot_path, 'wb') as f:
                    dump(snapshot, f)

            elapsed_time = timers('interval-time').elapsed(barrier=True)
            elapsed_time_per_iteration = elapsed_time / total_iterations
            train_percentage = iteration / args.train_iters
            total_elapsed_time = timers('interval-time').active_time()
            memory_GiB = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
            remaining_time = total_elapsed_time / train_percentage - total_elapsed_time
            total_elapsed_time = format_time(total_elapsed_time)
            remaining_time = format_time(remaining_time)

            throughput = num_floating_point_operations(args, batch_size) / (
                elapsed_time_per_iteration * 10**12 * args.world_size)

            one_logger_utils.track_e2e_metrics(args.log_throughput, throughput)

            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar('iteration-time', elapsed_time_per_iteration, iteration)
                if wandb_writer:
                    wandb_writer.log({'iteration-time': elapsed_time_per_iteration}, iteration)
            log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            log_string += ' iteration {:8d}/{:8d} |'.format(iteration, args.train_iters)
            log_string += ' consumed samples: {:12d} |'.format(args.consumed_train_samples)
            if args.skipped_train_samples > 0:
                log_string += ' skipped samples: {:12d} |'.format(args.skipped_train_samples)
            log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time_per_iteration * 1000.0)
            log_string += (f' memory(GiB): {memory_GiB} |'
                           f' elapsed time: {total_elapsed_time} | remaining time: {remaining_time} |')
            if args.log_throughput:
                log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
                if args.log_timers_to_tensorboard:
                    if writer:
                        writer.add_scalar('throughput', throughput, iteration)
                    if wandb_writer:
                        wandb_writer.log({'throughput': throughput}, iteration)
            # Decoupled_learning_rate should be not None only on first and last pipeline stage.
            log_string += f' learning rate: {learning_rate:.6E} |'
            if args.decoupled_lr is not None and (mpu.is_pipeline_first_stage(ignore_virtual=True)
                                                  or mpu.is_pipeline_last_stage(ignore_virtual=True)):
                assert decoupled_learning_rate is not None
                log_string += f' decoupled learning rate: {decoupled_learning_rate:.6E} |'
            else:
                assert decoupled_learning_rate is None
            log_string += f' global batch size: {batch_size:5d} |'
            for key in total_loss_dict:
                if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                    avg = total_loss_dict[key].item() / float(max(1, total_loss_dict[advanced_iters_key]))
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                    total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
            log_string += f' loss scale: {loss_scale:.1f} |'
            if grad_norm is not None:
                log_string += f' grad norm: {grad_norm:.3f} |'
            if num_zeros_in_grad is not None:
                log_string += f' num zeros: {num_zeros_in_grad} |'
            if params_norm is not None:
                log_string += f' params norm: {params_norm:.3f} |'
            log_string += ' number of skipped iterations: {:3d} |'.format(total_loss_dict[skipped_iters_key])
            log_string += ' number of nan iterations: {:3d} |'.format(total_loss_dict[nan_iters_key])
            total_loss_dict[advanced_iters_key] = 0
            total_loss_dict[skipped_iters_key] = 0
            total_loss_dict[nan_iters_key] = 0
            print_rank_last(log_string)
            if report_memory_flag:
                # Report memory after optimizer state has been initialized.
                if torch.distributed.get_rank() == 0:
                    num_microbatches = get_num_microbatches()
                    report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
                report_memory(f'(after {iteration} iterations)')
                report_memory_flag = False
            timers.log(timers_to_log, normalizer=args.log_interval)

            if is_last_rank():
                logs = {}
                for key in origin_total_loss_dict:
                    if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                        avg = origin_total_loss_dict[key].item() / float(
                            max(1, origin_total_loss_dict[advanced_iters_key]))
                        logs[key] = round(avg, 8)
                if grad_norm is not None:
                    logs['grad_norm'] = round(grad_norm, 8)
                if params_norm is not None:
                    logs['params_norm'] = round(params_norm, 8)
                logs['learning_rate'] = round(learning_rate, 8)
                logs['elapsed_time_per_iteration'] = round(elapsed_time_per_iteration, 8)
                logs['memory(GiB)'] = memory_GiB
                logs['elapsed_time'] = total_elapsed_time
                logs['remaining_time'] = remaining_time
                if args.log_throughput:
                    logs['throughput'] = round(throughput, 8)
                logs['loss_scale'] = round(loss_scale, 8)
                logs['consumed_samples'] = args.consumed_train_samples
                logs['global_step/max_steps'] = f'{iteration}/{args.train_iters}'
                self.jsonl_writer.append(logs)

        return report_memory_flag

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

    def _save_checkpoint(self, iteration, model, *_args, **kwargs):
        args = self.args
        output_dir = os.path.join(args.save, f'checkpoint-{iteration}')
        os.makedirs(output_dir, exist_ok=True)
        origin_save = args.save
        args.save = output_dir
        args_path = os.path.join(os.path.dirname(output_dir), 'args.json')
        self.copy_path(args_path, os.path.join(output_dir, 'args.json'))
        save_peft_format = args.tuner_type == 'lora' and not args.merge_lora
        if args.save_safetensors and args.no_save_optim:
            model = []
        with adapter_state_dict_context(is_peft_format=args.tuner_type == 'lora'):
            self._origin_save_checkpoint(iteration, model, *_args, **kwargs)
        args.save = origin_save
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
        train_metrics = {}
        while args.iteration < args.train_iters:
            metrics, grad_norm = self.train_step(train_dataloader)
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                self.aggregated_metrics(metrics, train_metrics)
            self.training_log(train_metrics, grad_norm)

            if args.eval_interval and args.iteration % args.eval_interval == 0:
                self.evaluate(val_dataloader)

            if args.save and args.save_interval and args.iteration % args.save_interval == 0:
                self.save_checkpoint()

    def save_checkpoint(self):
        print

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

        with torch.no_grad(), tqdm(
                total=val_dataloader, dynamic_ncols=True, disable=not is_last_rank(), desc='Evaluate: ') as prog_bar:
            iteration = 0
            while iteration < args.eval_iters:
                prog_bar.update()
                iteration += 1

                metrics = forward_backward_func(
                    forward_step_func=self.forward_step,
                    data_iterator=val_dataloader,
                    model=self.wrapped_models,
                    num_microbatches=get_num_microbatches(),
                    seq_length=args.max_length,
                    micro_batch_size=args.micro_batch_size,
                    forward_only=True,
                )
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    self.aggregated_metrics(metrics, eval_metrics)
        # TODO: log metrics
        logger.info(f'eval_metrics: {eval_metrics}')
        for m in self.wrapped_models:
            m.train()

    def train_step(self, train_dataloader):
        args = self.args
        forward_backward_func = get_forward_backward_func()
        for m in self.wrapped_models:
            m.zero_grad_buffer()
        self.optimizer.zero_grad()
        metrics = forward_backward_func(
            forward_step_func=self.forward_step,
            data_iterator=train_dataloader,
            model=self.wrapped_models,
            num_microbatches=get_num_microbatches(),
            seq_length=args.max_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=False,
        )

        update_successful, grad_norm, _ = self.optimizer.step()
        update_successful = logical_and_across_model_parallel_group(update_successful)
        grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)

        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
            args.iteration += 1
            args.consumed_train_samples += increment
            self.scheduler.step(increment=increment)

        return metrics, grad_norm

    def aggregated_metrics(self, metrics, total_metrics):
        for key in metrics[0].keys():
            if key not in total_metrics:
                total_metrics[key] = torch.tensor([0.0], dtype=torch.float32, device=torch.cuda.current_device())
            val = [x[key].view(-1) for x in metrics]
            val = torch.concat(val, dim=0)
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
            consumed_samples=args.consumed_train_samples,
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
            persistent_workers=args.dataloader_persistent_workers if args.num_workers > 0 else False,
            prefetch_factor=args.dataloader_prefetch_factor if args.num_workers > 0 else None,
            collate_fn=self.data_collator,
        )
        return dataloader

    @abstractmethod
    def forward_step(self, data_iterator, model):
        pass

    def _prepare_batch(self, data, vp_stage=None, num_samples=None):
        batch = get_batch_on_this_tp_rank(data, vp_stage=vp_stage)
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
        batch = get_batch_on_this_cp_rank(batch)
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
