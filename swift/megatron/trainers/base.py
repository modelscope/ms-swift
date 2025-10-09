# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from typing import Dict

import megatron.core
import torch
import torch.nn
from megatron.core import mpu
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from megatron.core.enums import ModelType
from megatron.core.inference.communication_utils import recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.utils import StragglerDetector
from megatron.training import (ft_integration, get_args, get_model, get_tensorboard_writer, get_timers,
                               get_wandb_writer, is_last_rank, one_logger_utils, pretrain, print_rank_0,
                               print_rank_last, training)
from megatron.training.checkpointing import load_checkpoint
from megatron.training.theoretical_memory_usage import report_theoretical_memory
from megatron.training.training import num_floating_point_operations
from megatron.training.utils import reduce_max_stat_across_model_parallel_group, report_memory, unwrap_model
from packaging import version
from transformers.utils import ContextManagers

from swift.llm import dynamic_gradient_checkpointing
from swift.plugin import MeanMetric
from swift.trainers import SwiftMixin
from swift.utils import JsonlWriter, deep_getattr, format_time, get_logger
from ..utils import adapter_state_dict_context, copy_original_module_weight, prepare_mcore_model
from .utils import get_swift_datasets_provider

logger = get_logger()


class BaseMegatronTrainer(ABC):

    def __init__(self, args, template):
        self.args = args
        self.template = template
        self.stimer = StragglerDetector()
        self.unwrapped_models = []
        self.peft_models = []
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

    def _replace_data_iterator(self, data_iterator):
        return data_iterator

    @staticmethod
    def _patch_merge_fn(state_dict_model):
        # https://github.com/NVIDIA/Megatron-LM/issues/1380

        def sh_ten_merge_fn(sub_state_dict):
            with torch.no_grad():
                shared_storage = sub_state_dict[0].untyped_storage()
                if all(shared_storage.data_ptr() == tensor.untyped_storage().data_ptr() for tensor in sub_state_dict):
                    element_size = sub_state_dict[0].element_size()
                    total_numel = sum(tensor.numel() for tensor in sub_state_dict)
                    if shared_storage.nbytes() == total_numel * element_size:
                        dim_0 = sum(tensor.shape[0] for tensor in sub_state_dict)
                        shape = (dim_0, ) + sub_state_dict[0].shape[1:]
                        combined_tensor = torch.empty(
                            shape, dtype=sub_state_dict[0].dtype,
                            device=sub_state_dict[0].device).set_(shared_storage, 0, shape)
                        return combined_tensor
                return torch.cat(sub_state_dict)

        for v in state_dict_model.values():
            if isinstance(v, ShardedTensorFactory) and 'apply_swiglu_sharded_factory' in v.merge_fn.__qualname__:
                v.merge_fn = sh_ten_merge_fn

    def _load_adapter_base_checkpoint(self, *_args, **kwargs):
        adapter_name = kwargs.pop('adapter_name', None) or 'ref_adapter'
        from megatron.training import checkpointing
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
            self._patch_merge_fn(state_dict_model)
        res = checkpointing.origin__load_base_checkpoint(*_args, **kwargs)
        for model_k in model_keys:
            state_dict = res[0][model_k]
            for k, origin_k in mapping[model_k].items():
                v = state_dict.pop(k)
                state_dict[origin_k] = v
        return res

    def _load_base_checkpoint(self, *_args, **kwargs):
        from megatron.training import checkpointing
        sharded_state_dict = kwargs.get('sharded_state_dict')
        if sharded_state_dict is None:
            return checkpointing.origin__load_base_checkpoint(*_args, **kwargs)
        model_keys = [k for k in sharded_state_dict.keys() if k.startswith('model')]
        if self.args.train_type == 'full':
            for k in model_keys:
                self._patch_merge_fn(sharded_state_dict[k])
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
            self._patch_merge_fn(state_dict_model)
        res = checkpointing.origin__load_base_checkpoint(*_args, **kwargs)
        for model_k in model_keys:
            state_dict = res[0][model_k]
            for k, origin_k in mapping[model_k].items():
                v = state_dict.pop(k)
                state_dict[origin_k] = v
        return res

    @contextmanager
    def _patch_load_state_dict(self, load_base_checkpoint):
        from megatron.training import checkpointing
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

    def setup_model_and_optimizer(self, model_provider_func, model_type, *_args, **kwargs):

        def new_model_provider_func(*args, **kwargs):
            model = model_provider_func(*args, **kwargs)
            self.unwrapped_models.append(model)
            self.peft_models.append(prepare_mcore_model(model))
            return model

        args = get_args()
        self._init_multimodal_full(args)
        with self._patch_load_state_dict(self._load_base_checkpoint):
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
        if args.model_meta.is_multimodal:
            for m in self.unwrapped_models:
                self._prepare_vit_gradient_checkpointing(m)
        return model, optimizer, opt_param_scheduler

    def _prepare_vit_gradient_checkpointing(self, model):
        visual = model.visual
        if visual is None:
            return
        args = get_args()
        for vision_tower in visual._vision_tower:
            module = deep_getattr(visual, vision_tower)
            if args.vit_gradient_checkpointing:
                dynamic_gradient_checkpointing(module, False)
                try:
                    module.gradient_checkpointing_enable(**(args.gradient_checkpointing_kwargs or {}))
                    module.enable_input_require_grads()
                except AttributeError:
                    pass

    @staticmethod
    def _initialize_embedding(model):
        # compat new_special_tokens
        init_method = model.config.init_method
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

    def _all_reduce_metric(self, metric: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        values = list(metric.values())
        reporting_metric = values[0].new_tensor(values)
        torch.distributed.all_reduce(
            reporting_metric, torch.distributed.ReduceOp.AVG, group=mpu.get_data_parallel_group())
        return {k: reporting_metric[i] for i, k in enumerate(metric.keys())}

    def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
        new_data_iterator = self._replace_data_iterator(data_iterator)
        return self._origin_train_step(forward_step_func, new_data_iterator, model, optimizer, opt_param_scheduler,
                                       config)

    # Code borrowed from NVIDIA/Megatron-LM
    def evaluate(self,
                 forward_step_func,
                 data_iterator,
                 model,
                 process_non_loss_data_func,
                 config,
                 verbose=False,
                 non_loss_data_func=None):
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
        with torch.no_grad():
            iteration = 0
            if verbose:
                print_rank_0(f'Evaluating on {args.eval_iters * eval_batch_size} samples')
            while iteration < args.eval_iters:
                iteration += 1
                if verbose:
                    print_rank_0(f'Evaluating iter {iteration}/{args.eval_iters}')

                forward_backward_func = get_forward_backward_func()
                # Don't care about timing during evaluation
                config.timers = None
                ft_integration.on_eval_step_start()
                new_data_iterator = self._replace_data_iterator(data_iterator)
                loss_dicts = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=new_data_iterator,
                    model=model,
                    num_microbatches=eval_num_microbatches,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True)
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
                    collect_non_loss_data=True)

        # Move model back to the train mode.
        for model_module in model:
            model_module.train()

        for key in total_loss_dict:
            numerator, denominator = total_loss_dict[key]
            total_loss_dict[key] = numerator / denominator

        timers('evaluate').stop()
        timers.log(['evaluate'])

        total_loss_dict.update({
            k: torch.tensor([v], device='cuda')
            for k, v in SwiftMixin.compute_custom_metrics(self.custom_metrics['eval'], 'eval_').items()
        })
        rerun_state_machine.set_mode(rerun_mode)
        if is_last_rank():
            logs = {}
            for key, val in total_loss_dict.items():
                logs[f'eval_{key}'] = round(val.item(), 8)
            self.jsonl_writer.append(logs)
        return total_loss_dict, collected_non_loss_data, False

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
            total_loss_dict.update({
                k: torch.tensor([v * total_loss_dict[advanced_iters_key]], device='cuda')
                for k, v in SwiftMixin.compute_custom_metrics(self.custom_metrics['train']).items()
            })
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

    def save_checkpoint(self, *args, **kwargs):
        with adapter_state_dict_context():
            return self._origin_save_checkpoint(*args, **kwargs)

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

    @staticmethod
    def _init_multimodal_full(args):
        visual_cls = args.megatron_model_meta.visual_cls
        if args.train_type == 'full' and args.model_meta.is_multimodal and visual_cls is not None:
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


class MegatronRLHFTrainer(BaseMegatronTrainer):

    def setup_model_and_optimizer(self, model_provider_func, model_type, *_args, **kwargs):
        args = get_args()
        if args.train_type == 'full':
            ref_models = get_model(model_provider_func, model_type, wrap_with_ddp=False)
            for m in ref_models:
                m = unwrap_model(m)
                m.requires_grad_(False).eval()
            if args.ref_load is None:
                args.ref_load = args.load
            args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                ref_models, None, None, load_arg='ref_load')
            self.ref_models = ref_models
        return super().setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs)

    @contextmanager
    def null_ref_context(self):
        args = get_args()
        contexts = []
        if args.train_type == 'full':
            ref_models = self.ref_models
        else:
            if args.ref_adapter_load is None:
                for m in self.peft_models:
                    contexts.append(m.disable_adapter())
            ref_models = self.unwrapped_models
        with ContextManagers(contexts):
            if args.ref_adapter_load:
                for m in self.peft_models:
                    m.set_adapter('ref_adapter')
            yield ref_models
            if args.ref_adapter_load:
                for m in self.peft_models:
                    m.set_adapter('default')

    @staticmethod
    def _forward_step_helper(model, inputs):
        args = get_args()
        if mpu.is_pipeline_first_stage():
            micro_batch_size = 1  # use qkv_format 'thd'
            seq_length = inputs['input_ids'].shape[1]
            if args.sequence_parallel:
                seq_length //= mpu.get_tensor_model_parallel_world_size()
            recv_shape_buffer = torch.tensor([seq_length, micro_batch_size, args.hidden_size],
                                             device=torch.cuda.current_device(),
                                             dtype=torch.int64)
        else:
            recv_shape_buffer = torch.empty((3, ), device=torch.cuda.current_device(), dtype=torch.int64)
            recv_from_prev_pipeline_rank_(recv_shape_buffer)
        if not mpu.is_pipeline_last_stage():
            send_to_next_pipeline_rank(recv_shape_buffer)
        shape = recv_shape_buffer.tolist()

        if not mpu.is_pipeline_first_stage():
            recv_buffer = torch.empty(shape, device=torch.cuda.current_device(), dtype=args.params_dtype)
            recv_from_prev_pipeline_rank_(recv_buffer)
            model.set_input_tensor(recv_buffer)
        output_tensor = model(**inputs)
        if not mpu.is_pipeline_last_stage():
            send_to_next_pipeline_rank(output_tensor)
            output_tensor = None

        return output_tensor
