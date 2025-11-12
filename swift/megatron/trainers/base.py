# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import os
import shutil
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Dict, List, Literal, Optional

import megatron.core
import torch
import torch.nn
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.optimizer import _update_min_and_max_lr_in_param_groups, param_group_identifier_keys
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.utils import StragglerDetector
from megatron.training import (checkpointing, ft_integration, get_args, get_model, get_tensorboard_writer, get_timers,
                               get_wandb_writer, is_last_rank, one_logger_utils, pretrain, print_rank_0,
                               print_rank_last, training)
from megatron.training.checkpointing import load_checkpoint
from megatron.training.theoretical_memory_usage import report_theoretical_memory
from megatron.training.training import num_floating_point_operations
from megatron.training.utils import reduce_max_stat_across_model_parallel_group, report_memory, unwrap_model
from packaging import version
from tqdm.auto import tqdm

from swift.llm import dynamic_gradient_checkpointing
from swift.plugin import MeanMetric
from swift.trainers import SwiftMixin
from swift.utils import JsonlWriter, deep_getattr, format_time, get_logger
from ..tuners import LoraParallelLinear
from ..utils import adapter_state_dict_context, copy_original_module_weight, patch_merge_fn, prepare_mcore_model
from .utils import (get_batch_on_this_cp_rank, get_batch_on_this_tp_rank, get_packed_seq_params,
                    get_swift_datasets_provider)

logger = get_logger()


class BaseMegatronTrainer(ABC):

    def __init__(self, args, template):
        self.args = args
        self.template = template
        self.stimer = StragglerDetector()
        self.unwrapped_models = []
        self.peft_models = []
        self._bridge = None
        logging_path = os.path.join(args.save, 'logging.jsonl')
        logger.info(f'logging_path: {logging_path}')
        self.jsonl_writer = JsonlWriter(logging_path, enable_async=True, write_on_rank='last')  # for evaluate
        self._patch_megatron()

        def _get_mean_metric():
            return MeanMetric(nan_value=None, group=mpu.get_data_parallel_group(with_context_parallel=True))

        self.custom_metrics = {
            'train': collections.defaultdict(_get_mean_metric),
            'eval': collections.defaultdict(_get_mean_metric)
        }
        self.megatron_core_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')

    @property
    def bridge(self):
        if self._bridge is None:
            self._bridge = self.args.megatron_model_meta.bridge_cls()
        return self._bridge

    @contextmanager
    def _get_iters(self, train_dataset, val_dataset):
        origin_initialize_megatron = training.initialize_megatron

        def initialize_megatron(*_args, **kwargs):
            res = origin_initialize_megatron(*_args, **kwargs)
            args = get_args()
            data_parallel_size = mpu.get_data_parallel_world_size()
            step_batch_size = args.micro_batch_size * data_parallel_size
            if args.train_iters is None and args.max_epochs is not None:
                if hasattr(train_dataset, '__len__'):
                    dataset_sample = len(train_dataset) // step_batch_size * step_batch_size
                    args.train_iters = dataset_sample * args.max_epochs // args.global_batch_size
                else:
                    raise ValueError(
                        'You are using a streaming training dataset. Please explicitly specify `--train_iters`.')
            if args.eval_iters < 0:
                if val_dataset is None:
                    args.eval_iters = 0
                elif hasattr(val_dataset, '__len__'):
                    dataset_sample = len(val_dataset) // step_batch_size * step_batch_size
                    args.eval_iters = max(dataset_sample // args.global_batch_size, 1)
                else:
                    raise ValueError(
                        'You are using a streaming validation dataset. Please explicitly specify `--eval_iters`.')
                logger.info(f'Setting args.eval_iters: {args.eval_iters}')
            return res

        training.initialize_megatron = initialize_megatron
        try:
            yield
        finally:
            training.initialize_megatron = origin_initialize_megatron

    def new_cyclic_iter(self, iterable):
        args = get_args()
        i = 0
        n_batch = 0
        while True:
            training = self.unwrapped_models[0].training
            if training:
                logger.info(f'The training of Epoch {i} starts...')
            if training and args.max_epochs and i >= args.max_epochs - 1:
                it = iter(iterable)
                num_microbatches = args.global_batch_size // (args.micro_batch_size * args.data_parallel_size)
                x = [next(it) for _ in range(num_microbatches - n_batch % num_microbatches)]
                while True:
                    try:
                        next_x = [next(it) for _ in range(num_microbatches)]
                    except StopIteration:
                        break
                    yield from x
                    x = next_x
                logger.info(f'Training of {i + 1} epochs has been completed, the training has finished.')
                x[0]['is_finished'] = True
                yield from x
            else:
                for x in iterable:
                    n_batch += 1
                    yield x
            i += 1

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
        if self.args.train_type == 'full':
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

        args = get_args()
        origin_load_state_dict = torch.nn.Module.load_state_dict
        origin_no_load_optim = args.no_load_optim
        origin_no_load_rng = args.no_load_rng
        origin_finetune = args.finetune

        def load_state_dict(self, state_dict, strict: bool = True, *args, **kwargs):
            strict = False
            return origin_load_state_dict(self, state_dict, strict, *args, **kwargs)

        if args.train_type != 'full':
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

        use_decoupled_learning_rate = decoupled_lr is not None

        # Map (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr) to params.
        params_map = {}
        for model_chunk in model_chunks:
            visual = model_chunk.module.module.visual
            for name, param in model_chunk.named_parameters():
                if not param.requires_grad:
                    continue

                is_expert_parallel = not getattr(param, 'allreduce', True)

                if no_weight_decay_cond is not None:
                    no_wd: bool = no_weight_decay_cond(name, param)
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
        if not self.args.megatron_model_meta.is_multimodal or (self.args.vit_lr is None
                                                               and self.args.aligner_lr is None):
            yield
            return
        from megatron.core import optimizer

        _get_param_groups = optimizer._get_param_groups
        optimizer._get_param_groups = self._get_param_groups
        try:
            yield
        finally:
            optimizer._get_param_groups = _get_param_groups

    def setup_model_and_optimizer(self, model_provider_func, model_type, *_args, **kwargs):

        args = get_args()

        def new_model_provider_func(*_args, **kwargs):
            model = model_provider_func(*_args, **kwargs)
            if args.load_safetensors:
                self.bridge.load_weights(model, args.model_dir)
            self.unwrapped_models.append(model)
            peft_model = prepare_mcore_model(model)
            if args.load_safetensors and args.train_type == 'lora':
                for adapters, name in [(args.adapters, 'default'), (args.ref_adapters, 'ref_adapter')]:
                    if adapters:
                        assert len(adapters) == 1, 'Currently only support one adapter.'
                        self.bridge.load_weights(model, adapters[0], is_peft_format=True, adapter_name=name)
            self.peft_models.append(peft_model)
            return model

        self._init_multimodal_full()
        with self._patch_load_state_dict(self._load_base_checkpoint), self._patch_get_param_groups():
            model, optimizer, opt_param_scheduler = self._origin_setup_model_and_optimizer(
                new_model_provider_func, model_type, *_args, **kwargs)
        if args.initialize_embedding:
            for m in self.unwrapped_models:
                self._initialize_embedding(m)
        if args.train_type != 'full' and args.modules_to_save:
            for m in self.unwrapped_models:
                copy_original_module_weight(m)
        if args.ref_adapter_load is not None:
            with self._patch_load_state_dict(self._load_adapter_base_checkpoint):
                args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                    model, optimizer, opt_param_scheduler, load_arg='ref_adapter_load', strict=False)
        if args.adapter_load is not None:
            with adapter_state_dict_context():
                args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                    model, optimizer, opt_param_scheduler, load_arg='adapter_load', strict=False)
        if args.is_multimodal:
            for m in self.unwrapped_models:
                self._prepare_vit_gradient_checkpointing(m)
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
        values = list(metric.values())
        reporting_metric = values[0].new_tensor(values)
        torch.distributed.all_reduce(reporting_metric, reduction, group=mpu.get_data_parallel_group())
        return {k: reporting_metric[i] for i, k in enumerate(metric.keys())}

    def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, *args,
                   **kwargs):
        new_data_iterator = self._replace_data_iterator(data_iterator, model)
        return self._origin_train_step(forward_step_func, new_data_iterator, model, optimizer, opt_param_scheduler,
                                       config, *args, **kwargs)

    # Code borrowed from NVIDIA/Megatron-LM
    def evaluate(
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
        args = get_args()
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
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True,
                )
                ft_integration.on_eval_step_end()
                config.timers = get_timers()

                # Empty unused memory
                if args.empty_unused_memory_level >= 1:
                    torch.cuda.empty_cache()

                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    if self.megatron_core_013:
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
                args.consumed_valid_samples += eval_batch_size

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
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True,
                    collect_non_loss_data=True,
                )

        # Move model back to the train mode.
        for model_module in model:
            model_module.train()

        for key in total_loss_dict:
            numerator, denominator = total_loss_dict[key]
            total_loss_dict[key] = numerator / denominator

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

    def custom_log(self, total_loss_dict, mode: Literal['train', 'eval']) -> None:
        advanced_iters = total_loss_dict['advanced iterations'] if mode == 'train' else 1
        total_loss_dict.update({
            k: torch.tensor([v * advanced_iters], device='cuda')
            for k, v in SwiftMixin.compute_custom_metrics(self.custom_metrics[mode]).items()
        })

    # Code borrowed from NVIDIA/Megatron-LM
    def training_log(self, loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration, loss_scale,
                     report_memory_flag, skipped_iter, grad_norm, params_norm, num_zeros_in_grad):
        """Log training information such as losses, timing, ...."""
        args = get_args()
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
            for key in loss_dict:
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
                moe_layer_freq=args.moe_layer_freq)
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
        for model in self.unwrapped_models:
            for module in model.modules():
                if isinstance(module, LoraParallelLinear):
                    # Merge all active adapters
                    module.merge(adapter_names=[adapter_name])

    def unmerge_lora_adapters(self):
        """Unmerge LoRA adapters to restore training state."""
        for model in self.unwrapped_models:
            for module in model.modules():
                if isinstance(module, LoraParallelLinear):
                    # Unmerge to restore separate LoRA weights for training
                    module.unmerge()

    def save_checkpoint(self, iteration, *_args, **kwargs):
        args = get_args()
        if args.train_type == 'lora' and args.merge_lora:
            self.merge_lora_adapters()
        save_peft_format = args.train_type == 'lora' and not args.merge_lora
        if args.save_safetensors:
            output_dir = os.path.join(args.save, f'checkpoint-{iteration}')
            self.bridge.save_weights(self.unwrapped_models, output_dir, is_peft_format=save_peft_format)
            if is_last_rank():
                args_path = os.path.join(os.path.dirname(output_dir), 'args.json')
                if os.path.exists(args_path):
                    shutil.copy(args_path, os.path.join(output_dir, 'args.json'))
        else:
            with adapter_state_dict_context(is_peft_format=save_peft_format):
                return self._origin_save_checkpoint(iteration, *_args, **kwargs)
        if args.train_type == 'lora' and args.merge_lora:
            self.unmerge_lora_adapters()

    def _patch_megatron(self):
        # support max_epochs
        self._origin_train_step = training.train_step
        training.train_step = self.train_step
        training.cyclic_iter = self.new_cyclic_iter
        # patch training_log
        self._origin_training_log = training.training_log
        training.training_log = self.training_log
        # patch evaluate
        self._origin_evaluate = training.evaluate
        training.evaluate = self.evaluate
        # patch model and optimizer
        self._origin_setup_model_and_optimizer = training.setup_model_and_optimizer
        training.setup_model_and_optimizer = self.setup_model_and_optimizer
        # patch save_checkpoint
        self._origin_save_checkpoint = training.save_checkpoint
        training.save_checkpoint = self.save_checkpoint

    def _init_multimodal_full(self):
        args = get_args()
        visual_cls = self.args.megatron_model_meta.visual_cls
        if args.train_type == 'full' and args.is_multimodal and visual_cls is not None:
            vision_tower = [f'visual.{vit}' for vit in visual_cls._vision_tower]
            aligner = [f'visual.{aligner}' for aligner in visual_cls._aligner]
            if args.freeze_llm:
                args.freeze_parameters.append('language_model')
            if args.freeze_vit:
                args.freeze_parameters += vision_tower
            if args.freeze_aligner:
                args.freeze_parameters += aligner
            else:
                args.trainable_parameters += aligner
            if args.freeze_parameters:
                logger.info(f'freeze_parameters: {args.freeze_parameters}')
            if args.trainable_parameters:
                logger.info(f'additional trainable_parameters: {args.trainable_parameters}')

    def train(self, train_dataset, val_dataset, data_collator):
        args = self.args
        datasets_provider = get_swift_datasets_provider(train_dataset, val_dataset)
        datasets_provider.is_distributed = True
        with self.patch_megatron_data_collator(data_collator), self._get_iters(train_dataset, val_dataset):
            extra_args_provider = args.megatron_model_meta.extra_args_provider
            pretrain(
                datasets_provider,
                args.megatron_model_meta.model_provider,
                ModelType.encoder_or_decoder,
                self.forward_step,
                extra_args_provider=extra_args_provider,
                args_defaults=args.extra_args)

    @contextmanager
    def patch_megatron_data_collator(self, data_collator):
        origin_build_pretraining_data_loader = training.build_pretraining_data_loader

        def build_pretraining_data_loader(*_args, **kwargs):
            args = get_args()
            res = origin_build_pretraining_data_loader(*_args, **kwargs)
            if res is not None and args.dataloader_type != 'external':
                res.collate_fn = data_collator
            return res

        training.build_pretraining_data_loader = build_pretraining_data_loader
        try:
            yield
        finally:
            training.build_pretraining_data_loader = origin_build_pretraining_data_loader

    @abstractmethod
    def forward_step(self, data_iterator, model):
        pass

    def _prepare_batch(self, data, vp_stage, num_samples=None):
        batch = get_batch_on_this_tp_rank(data, vp_stage=vp_stage)
        if num_samples is None:
            num_samples = batch.pop('num_samples')
        args = get_args()
        text_position_ids = batch.pop('text_position_ids', None)
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
        args = get_args()
        data = next(data_iterator)
        is_finished = data.pop('is_finished', False)
        if is_finished:
            args.train_iters = args.curr_iteration + 1
        return self._prepare_batch(data, vp_stage)
