# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager

import megatron.core
import torch
import torch.distributed as dist
import torch.nn
from megatron.core import mpu
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.core.utils import StragglerDetector
from megatron.training import ft_integration, get_args, get_timers, is_last_rank, pretrain, print_rank_0, training
from megatron.training.checkpointing import load_checkpoint
from packaging import version

from swift.utils import JsonlWriter, deep_getattr, get_logger
from ..utils import adapter_state_dict_context, copy_original_module_weight, prepare_mcore_model
from .utils import get_swift_datasets_provider

logger = get_logger()


class BaseMegatronTrainer(ABC):

    def __init__(self, args):
        self.args = args
        self.stimer = StragglerDetector()
        logging_path = os.path.join(args.save, 'logging.jsonl')
        logger.info(f'logging_path: {logging_path}')
        self.jsonl_writer = JsonlWriter(logging_path, enable_async=True, write_on_rank='last')  # for evaluate
        self._patch_megatron()

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

    @staticmethod
    def new_cyclic_iter(iterable):
        args = get_args()
        i = 0
        n_batch = 0
        while True:
            is_training = getattr(args, 'is_training', False)
            if is_training:
                logger.info(f'The training of Epoch {i} starts...')
            if is_training and args.max_epochs and i >= args.max_epochs - 1:
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

    @staticmethod
    @contextmanager
    def _training_context():
        args = get_args()
        args.is_training = True
        try:
            yield
        finally:
            args.is_training = False

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

    @contextmanager
    def _patch_load_state_dict(self):
        from megatron.training import checkpointing
        origin__load_base_checkpoint = checkpointing._load_base_checkpoint

        def _load_base_checkpoint(*_args, **kwargs):
            sharded_state_dict = kwargs.get('sharded_state_dict')
            if sharded_state_dict is None:
                return origin__load_base_checkpoint(*_args, **kwargs)
            if self.args.train_type == 'full':
                self._patch_merge_fn(sharded_state_dict['model'])
                return origin__load_base_checkpoint(*_args, **kwargs)
            state_dict_model = {}
            mapping = {}
            for k, v in sharded_state_dict['model'].items():
                if 'lora_A' in k or 'lora_B' in k or 'original_module' in k:
                    continue
                # lora
                if '.base_layer' in k:
                    origin_k = k
                    k = k.replace('.base_layer', '')
                    mapping[k] = origin_k
                    v.key = v.key.replace('.base_layer', '')
                elif '.modules_to_save' in k:
                    # modules to save
                    origin_k = k
                    k = k.replace('.modules_to_save.default', '')
                    mapping[k] = origin_k
                    v.key = v.key.replace('.modules_to_save.default', '')
                state_dict_model[k] = v
            sharded_state_dict['model'] = state_dict_model
            self._patch_merge_fn(state_dict_model)
            res = origin__load_base_checkpoint(*_args, **kwargs)
            state_dict = res[0]['model']
            for k, origin_k in mapping.items():
                v = state_dict.pop(k)
                state_dict[origin_k] = v
            return res

        origin_load_state_dict = torch.nn.Module.load_state_dict

        def load_state_dict(self, state_dict, strict: bool = True, *args, **kwargs):
            strict = False
            return origin_load_state_dict(self, state_dict, strict, *args, **kwargs)

        checkpointing._load_base_checkpoint = _load_base_checkpoint
        torch.nn.Module.load_state_dict = load_state_dict

        args = get_args()
        origin_no_load_optim = args.no_load_optim
        origin_no_load_rng = args.no_load_rng
        args.no_load_optim = True
        args.no_load_rng = True

        try:
            yield
        finally:
            checkpointing._load_base_checkpoint = origin__load_base_checkpoint
            torch.nn.Module.load_state_dict = origin_load_state_dict
            args.no_load_optim = origin_no_load_optim
            args.no_load_rng = origin_no_load_rng

    def setup_model_and_optimizer(self, model_provider_func, model_type, *_args, **kwargs):

        def new_model_provider_func(*args, **kwargs):
            self.unwrapped_model = model_provider_func(*args, **kwargs)
            self.peft_model = prepare_mcore_model(self.unwrapped_model)
            return self.unwrapped_model

        with self._patch_load_state_dict():
            model, optimizer, opt_param_scheduler = self._origin_setup_model_and_optimizer(
                new_model_provider_func, model_type, *_args, **kwargs)
        args = get_args()
        if args.adapter_load is not None:
            with adapter_state_dict_context():
                args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                    model, optimizer, opt_param_scheduler, load_arg='adapter_load', strict=False)
        if args.train_type != 'full' and args.modules_to_save:
            copy_original_module_weight(self.unwrapped_model)
        if args.initialize_embedding:
            self._initialize_embedding(self.unwrapped_model)
        return model, optimizer, opt_param_scheduler

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
            tensor = module.weight.new_empty(num_to_initialize, module.weight.shape[1])
            module.weight.data[initialize_mask] = init_method(tensor)

    def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
        with self._training_context():
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
        megatron_core_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
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
                    if megatron_core_013:
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

        rerun_state_machine.set_mode(rerun_mode)
        if is_last_rank():
            logs = {}
            for key, val in total_loss_dict.items():
                logs[f'eval_{key}'] = round(val.item(), 8)
            self.jsonl_writer.append(logs)
        return total_loss_dict, collected_non_loss_data, False

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
        # patch evaluate
        self._origin_evaluate = training.evaluate
        training.evaluate = self.evaluate
        # patch model and optimizer
        self._origin_setup_model_and_optimizer = training.setup_model_and_optimizer
        training.setup_model_and_optimizer = self.setup_model_and_optimizer
        # patch save_checkpoint
        self._origin_save_checkpoint = training.save_checkpoint
        training.save_checkpoint = self.save_checkpoint

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
