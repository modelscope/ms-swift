# Copyright (c) Alibaba, Inc. and its affiliates.
import base64
import gc
import inspect
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import pandas as pd
import torch
import torch.nn as nn
from accelerate.utils import broadcast_object_list
from dacite import from_dict
from megatron.core import mpu
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.training import get_args, get_wandb_writer, training
from vllm.distributed import parallel_state as vllm_ps

from swift.llm import RequestConfig, RolloutInferRequest, RowPreprocessor, Template, get_packed_seq_params, to_device
from swift.llm.infer.protocol import RolloutOutput
from swift.llm.template.template_inputs import TemplateInputs
from swift.plugin import MultiTurnScheduler, multi_turns, orms
from swift.trainers.rlhf_trainer.grpo_trainer import DataType
from swift.trainers.rlhf_trainer.utils import (FlattenedTensorBucket, aggressive_empty_cache, check_vllm_version_ge,
                                               nanstd, pad_logps_back_to_batch, replace_assistant_response_with_ids,
                                               set_expandable_segments)
from swift.utils import (get_current_device, get_logger, is_last_rank, is_vllm_available, is_wandb_available,
                         remove_response)
from ..argument import MegatronArguments, MegatronRLHFArguments
from ..utils import forward_step_helper, get_padding_to
from .rlhf_mixin import MegatronRLHFTrainer
from .utils import (gather, gather_object, get_swift_datasets_provider, load_megatron_model_to_gpu,
                    load_megatron_optimizer, offload_megatron_model_to_cpu, offload_megatron_optimizer,
                    profiling_context, profiling_decorator)

if is_wandb_available():
    import wandb

logger = get_logger()


class MegatronGRPOTrainer(MegatronRLHFTrainer):

    def __init__(self, args: MegatronRLHFArguments, template: Template, **kwargs):
        self.vllm_client = kwargs.pop('vllm_client')
        super().__init__(args, template)
        self.args = args
        self.hf_model_dir = args.model_info.model_dir
        self.processing_class = self.template.processor
        self._prepare_metrics()
        self._init_grpo_params()
        self._prepare_rewards()
        self._prepare_scheduler()  # TODO
        self._prepare_rollout_engine()

    def train(self, train_dataset, val_dataset, data_collator):
        # Store dataset provider for lazy resample iterator initialization
        # Used by both dynamic_sample and truncation_strategy='raise'(delete)
        if self.dynamic_sample or self.truncation_strategy == 'raise':
            self._train_valid_test_dataset_provider = get_swift_datasets_provider(train_dataset, val_dataset)
            self._train_valid_test_dataset_provider.is_distributed = True
        super().train(train_dataset, val_dataset, data_collator)

    def _init_grpo_params(self):
        args: MegatronArguments = self.args
        # distributed params
        self.world_size = torch.distributed.get_world_size()
        self.process_index = torch.distributed.get_rank()
        self.is_main_process = is_last_rank()
        self.device = get_current_device()
        # algorithm params
        self.num_generations = args.num_generations  # G in the GRPO paper
        self.num_generations_eval = args.num_generations_eval or self.num_generations
        self.beta = args.beta
        self.temperature = args.temperature
        self.loss_type = args.loss_type
        self.max_completion_length = args.max_completion_length
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.top_entropy_quantile = args.top_entropy_quantile
        self.importance_sampling_level = args.importance_sampling_level

        # SAPO, https://arxiv.org/abs/2511.20347
        self.tau_pos = args.tau_pos
        self.tau_neg = args.tau_neg

        # DAPO, https://arxiv.org/abs/2503.14476
        self.dynamic_sample = args.dynamic_sample
        self.max_resample_times = args.max_resample_times
        self.overlong_filter = args.overlong_filter

        # Dr. GRPO / RLOO / REINFORCE++
        self.scale_rewards = args.scale_rewards
        self.advantage_estimator = args.advantage_estimator
        self.kl_in_reward = args.kl_in_reward

        # Entropy mask settings, TODO
        self.log_entropy = args.log_entropy
        self.compute_entropy = self.log_entropy or self.top_entropy_quantile < 1.0

        # Rollout Importance Sampling Correction
        self.rollout_importance_sampling_mode = args.rollout_importance_sampling_mode
        self.rollout_importance_sampling_threshold = args.rollout_importance_sampling_threshold
        self.log_rollout_offpolicy_metrics = args.log_rollout_offpolicy_metrics

        # Off-Policy Sequence Masking
        self.off_policy_sequence_mask_delta = args.off_policy_sequence_mask_delta

        # batch size (completion-level)
        self.generation_batch_size = args.generation_batch_size
        self.steps_per_generation = args.steps_per_generation
        self.global_batch_size = args.global_batch_size
        self.micro_batch_size = args.micro_batch_size
        self.per_device_generation_batch_size = args.per_device_generation_batch_size

        self.enable_offload = False

        # sampling params
        self.request_config = RequestConfig(
            n=1,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop_words,
            return_details=True,
            logprobs=True)  # Enable logprobs for rollout importance sampling

        self._step = 0
        self._last_loaded_step = -1
        self._rollout_group = None  # Will be lazily initialized

        # truncation_strategy support
        self.truncation_strategy = self.template.truncation_strategy

    def _prepare_rollout_engine(self):
        args = self.args
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.use_vllm = args.use_vllm
        self.async_generate = args.async_generate  # TODO
        self.vllm_use_async_engine = False
        self.enable_offload = False
        self.use_gym_env = False
        self.enable_server_multi_turn = False  # TODO
        self.vllm_version_ge_0_10_2 = check_vllm_version_ge('0.10.2')

        self.disable_rollout_importance_sampling = not self.vllm_version_ge_0_10_2
        if not self.vllm_version_ge_0_10_2 and getattr(self.args, 'rollout_importance_sampling_mode', None) is not None:
            raise ValueError('rollout_importance_sampling_mode is not supported in vLLM version < 0.10.2, '
                             'please update vLLM to 0.10.2 or later.')
        # for multi-turn server, maybe the num of rollout outputs is not equal to the num of rollout inputs
        assert self.use_vllm
        if not is_vllm_available():
            raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                              'Please install vLLM with `pip install vllm -U` to use it.')
        if self.vllm_mode == 'server':
            pass
        elif self.vllm_mode == 'colocate':
            if not self.world_size % self.vllm_tensor_parallel_size == 0:
                raise ValueError(f'vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size '
                                 f'({self.world_size}) evenly.')

            self.enable_offload = self.args.offload_model or self.args.offload_optimizer
            context = self.offload_context if self.enable_offload else nullcontext

            with context():
                set_expandable_segments(False)
                self.engine = self.prepare_vllm()
                if self.args.sleep_level > 0:
                    self.engine.engine.sleep(self.args.sleep_level)
                set_expandable_segments(True)
        else:
            raise ValueError(f'Invalid vllm_mode: {self.vllm_mode}')

    def prepare_vllm(self):
        from swift.llm.infer.infer_engine import GRPOVllmEngine
        args = self.args
        max_num_seqs = args.vllm_max_num_seqs or self.per_device_generation_batch_size * self.vllm_tensor_parallel_size
        vllm_template = copy(self.template)
        vllm_template.padding_free = False
        vllm_template.sequence_parallel_size = 1
        logprobs_mode = 'processed_logprobs' if self.vllm_version_ge_0_10_2 else None

        # Use load_format from vllm_engine_kwargs if provided, otherwise default to 'dummy'
        vllm_engine_kwargs = self.args.vllm_engine_kwargs or {}
        load_format = vllm_engine_kwargs.pop('load_format', 'dummy')
        engine = GRPOVllmEngine(
            self.hf_model_dir,
            args.torch_dtype,
            model_type=args.model_type,
            use_async_engine=False,
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            enable_prefix_caching=self.args.vllm_enable_prefix_caching,
            max_num_seqs=max_num_seqs,
            enforce_eager=self.args.vllm_enforce_eager,
            limit_mm_per_prompt=self.args.vllm_limit_mm_per_prompt,
            enable_sleep_mode=self.args.sleep_level > 0,
            max_model_len=self.args.vllm_max_model_len,
            seed=self.process_index // self.vllm_tensor_parallel_size,
            disable_cascade_attn=self.args.vllm_disable_cascade_attn,
            load_format=load_format,
            mm_processor_cache_gb=args.vllm_mm_processor_cache_gb,
            template=vllm_template,
            distributed_executor_backend='external_launcher',
            engine_kwargs=vllm_engine_kwargs,
            logprobs_mode=logprobs_mode)
        if self.vllm_tensor_parallel_size > 1:
            self.vllm_tp_group = vllm_ps.get_tp_group().device_group
        self._buffered_inputs = None
        return engine

    @profiling_decorator
    def _move_model_to_vllm(self):
        # Handle LoRA: merge adapters before exporting weights
        is_lora_training = self.args.train_type == 'lora'

        try:
            if is_lora_training:
                self.merge_lora_adapters()

            # Export and load weights incrementally to avoid memory spikes
            self._export_and_load_weights()

        finally:
            # Unmerge adapters to restore training state
            if is_lora_training:
                self.unmerge_lora_adapters()

        # Reset prefix cache
        if self.vllm_mode == 'server' and self.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == 'colocate':
            self.engine.engine.reset_prefix_cache()

    @property
    def bridge(self):
        if self._bridge is None:
            self._bridge = self.args.megatron_model_meta.bridge_cls(disable_tqmd=True)
        return self._bridge

    def _export_and_load_weights(self):
        """
        Export weights from Megatron models and load to vLLM incrementally.

        For colocate mode: llm_model.load_weights accepts an iterator, so pass it directly.
        For server mode: Process weights in buckets to avoid memory spikes.
        """
        # Export weights returns an iterator
        target_device = None
        if self.args.offload_bridge:
            target_device = 'cpu'
        with profiling_context(self, 'export_weights'):
            weight_iterator = self.bridge.export_weights(self.unwrapped_models, target_device=target_device)

        if self.vllm_mode == 'colocate':
            # Colocate mode: load_weights supports iterator, pass directly
            llm_model = self.engine.inner_model
            llm_model.load_weights(weight_iterator)
        elif self.vllm_mode == 'server':
            # Server mode: process in buckets and sync with flattened tensors
            self._load_weights_to_server_in_buckets(weight_iterator)

    def _load_weights_to_server_in_buckets(self, weight_iterator):
        """
        Load weights to vLLM server in buckets using FlattenedTensorBucket.

        Args:
            weight_iterator: Iterator of (name, tensor) tuples from export_weights
        """
        # Get bucket size from environment or use default
        bucket_size_mb = int(os.environ.get('SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE', 512))
        bucket_size_bytes = bucket_size_mb * 1024 * 1024

        current_bucket = []
        current_size = 0

        for name, param in weight_iterator:
            param_size = param.numel() * param.element_size()
            current_bucket.append((name, param))
            current_size += param_size

            # If adding this param would exceed bucket size, process current bucket first
            if current_size > bucket_size_bytes and current_bucket:
                self._sync_bucket_to_server(current_bucket)
                current_bucket = []
                current_size = 0

        # Process remaining parameters in the last bucket
        if current_bucket:
            self._sync_bucket_to_server(current_bucket)

    def _sync_bucket_to_server(self, bucket_params: List[Tuple[str, torch.Tensor]]):
        """
        Synchronize a bucket of parameters to vLLM server using flattened tensors.

        Args:
            bucket_params: List of (name, tensor) tuples to sync
        """
        if not bucket_params or not self.is_main_process:
            return

        # Create FlattenedTensorBucket for efficient transfer
        bucket = FlattenedTensorBucket(named_tensors=bucket_params)
        metadatas = bucket.get_metadata()
        flattened_tensor = bucket.get_flattened_tensor()

        # Directly call vllm_client to update weights
        self.vllm_client.update_flattened_params(metadatas, flattened_tensor)

        # Clean up to free memory immediately
        del bucket, metadatas, flattened_tensor

    def _prepare_rewards(self):
        # TODO: reward model
        args = self.args
        reward_funcs = args.reward_funcs
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        # initilize reward functions
        if reward_funcs:
            for i, reward_func in enumerate(reward_funcs):
                if reward_func in orms:
                    reward_func_class = orms[reward_func]
                    reward_func_args = list(inspect.signature(reward_func_class.__init__).parameters)
                    reward_func_kwargs = {
                        key: getattr(args, key)
                        for key in reward_func_args if key not in ['self', 'args', 'kwargs'] and hasattr(args, key)
                    }
                    if 'tokenizer' in reward_func_args:
                        reward_func_kwargs['tokenizer'] = self.processing_class
                    reward_funcs[i] = reward_func_class(**reward_func_kwargs)
                elif not callable(reward_func):
                    raise ValueError(f'reward_function {reward_func} is not implemented in swift.plugin')

        # get reward name for logging
        self.reward_funcs = reward_funcs
        self.reward_func_names = []
        for reward_func in reward_funcs:
            if inspect.isfunction(reward_func):
                reward_func_name = reward_func.__name__
            else:
                reward_func_name = reward_func.__class__.__name__
            self.reward_func_names.append(reward_func_name)

        # set reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(f'Number of reward weights ({len(args.reward_weights)}) must match number of reward '
                                 f'functions ({len(reward_funcs)})')
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32).to(self.device)
        else:
            self.reward_weights = torch.ones(len(self.reward_func_names), dtype=torch.float32).to(self.device)

        # TODO: reward models
        self.reward_model_plugins = [None] * len(self.reward_funcs)

        assert self.reward_funcs, 'reward_funcs is not set'

    def _prepare_scheduler(self):
        """Prepare multi-turn scheduler"""
        args = self.args

        self.multi_turn_scheduler = None
        if not hasattr(args, 'multi_turn_scheduler'):
            return

        if args.multi_turn_scheduler:
            if isinstance(args.multi_turn_scheduler, str):
                assert args.multi_turn_scheduler in multi_turns
                multi_turn_scheduler = multi_turns[args.multi_turn_scheduler](max_turns=args.max_turns)
                self.multi_turn_scheduler: MultiTurnScheduler = multi_turn_scheduler
            else:
                assert isinstance(args.multi_turn_scheduler, MultiTurnScheduler)
                self.multi_turn_scheduler: MultiTurnScheduler = args.multi_turn_scheduler

    def _get_rollout_group(self):
        """
        Get or create the rollout process group (TP×PP×CP).

        The rollout group is used for:
        1. Data slicing: distributing rollout data across ranks with same data samples
        2. Gather operations: collecting results from ranks with same data samples

        Note: Groups are created per data parallel index, containing TP×PP×CP ranks each.
        This follows Megatron's data_iterator logic where same data_parallel_rank processes
        identical data samples.

        Key insight: ranks with the SAME data parallel index process the SAME data samples
        and must coordinate for rollout data distribution.
        Megatron rank order: TP → CP → EP → DP → PP
        """
        if self._rollout_group is not None:
            return self._rollout_group

        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            # No CP, use the standard MODEL_PARALLEL_GROUP
            self._rollout_group = mpu.get_model_parallel_group()
            return self._rollout_group

        # Use RankGenerator to create rollout groups following Megatron-LM logic
        global_rank = torch.distributed.get_rank()

        # Get parallel dimensions
        tp_size = mpu.get_tensor_model_parallel_world_size()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        dp_size = mpu.get_data_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()

        # Create RankGenerator following Megatron-LM pattern
        # Order: tp-cp-ep-dp-pp (default in Megatron-LM)
        decoder_rank_generator = mpu.RankGenerator(
            tp=tp_size,
            ep=1,
            dp=dp_size,
            pp=pp_size,
            cp=cp_size,
            order='tp-cp-ep-dp-pp',
            rank_offset=0,
        )

        # Create rollout groups based on data consistency from data_iterator
        # Same data_parallel_rank processes same data - group ranks with same DP index
        if not hasattr(self, '_rollout_groups_created'):
            # Use 'tp-cp-ep-pp' to get groups with same DP index (DP is excluded from variation)
            dp_groups = decoder_rank_generator.get_ranks('tp-cp-ep-pp')
            for dp_group_ranks in dp_groups:
                # Sort for consistency
                dp_group_ranks = sorted(dp_group_ranks)
                group = torch.distributed.new_group(ranks=dp_group_ranks, group_desc='ROLLOUT_GROUP')

                if global_rank in dp_group_ranks:
                    self._rollout_group = group
            self._rollout_groups_created = True

        return self._rollout_group

    def _init_resample_data_iterator(self):
        """
        Initialize an independent data iterator for dynamic resampling (lazy initialization).

        This method is called lazily during the first dynamic resampling, ensuring that
        pretrain() has already called initialize_megatron() to properly set up all args.
        Uses a different seed (args.seed + 1) to avoid overlapping with training samples.

        Note: pretrain() will automatically reset the random seed back to args.seed
        after this method completes, so we don't need manual state restoration.

        Args:
            train_valid_test_dataset_provider: Dataset provider function

        Returns:
            train_data_iterator: Independent data iterator with different random seed
        """
        from megatron.training.training import build_train_valid_test_data_iterators
        from megatron.training.initialize import _set_random_seed
        from megatron.training import training
        training.cyclic_iter = self._origin_cyclic_iter
        args = get_args()

        train_valid_test_dataset_provider = self._train_valid_test_dataset_provider
        # Use different seed for resample iterator (offset by 1 to avoid overlap)
        resample_seed = getattr(args, 'seed', 42) + 1
        try:
            # Set new seed for resample iterator creation
            _set_random_seed(
                resample_seed,
                args.data_parallel_random_init,
                args.te_rng_tracker,
                args.inference_rng_tracker,
                use_cudagraphable_rng=args.enable_cuda_graph,
            )

            # Build data iterators with new seed
            # TODO: VPP (Virtual Pipeline Parallelism)
            resample_data_iterator, _, _ = (build_train_valid_test_data_iterators(train_valid_test_dataset_provider))
        finally:
            # Restore original random states to avoid affecting training
            _set_random_seed(
                args.seed,
                args.data_parallel_random_init,
                args.te_rng_tracker,
                args.inference_rng_tracker,
                use_cudagraphable_rng=args.enable_cuda_graph,
            )
        return resample_data_iterator

    def _replace_data_iterator(self, data_iterator, model):
        if self._step % self.steps_per_generation == 0:
            num_iters_per_step = self.get_num_iters_per_step()
            rollout_batch = []
            for _ in range(num_iters_per_step):
                rollout_batch.extend(next(data_iterator))
            micro_batch_data = self._generate_and_score_completions(rollout_batch)
            num_mini_batch = self.global_batch_size // (self.micro_batch_size * mpu.get_data_parallel_world_size())
            mini_batch_data = [
                micro_batch_data[i:i + num_mini_batch] for i in range(0, len(micro_batch_data), num_mini_batch)
            ]
            assert len(mini_batch_data) == self.steps_per_generation
            self._buffered_inputs = mini_batch_data
        inputs = self._buffered_inputs[self._step % self.steps_per_generation]
        self._step += 1
        return RerunDataIterator(iter(inputs))

    def _generate_and_score_completions(self, batch):
        # Get or create the rollout group (TP×PP×CP)
        args = get_args()

        rollout_group = self._get_rollout_group()

        # Resample for encoding failed data when truncation_strategy is 'raise'(delete)
        # This handles: (1) prompt length exceeds max_length, (2) multimodal encoding failures
        # Do this before get_local_rollout_batch to process prompt-level data
        if self.truncation_strategy == 'raise':
            batch = self.resample_encode_failed_inputs(batch)

        rollout_batch = self.get_local_rollout_batch(batch)

        rollout_batch = self._generate_completions(rollout_batch)

        rewards_per_func = self._score_completions(rollout_batch)

        # Dynamic sampling for std=0 groups (DAPO)
        if self.dynamic_sample:
            rollout_batch, rewards_per_func = self._dynamic_sampling(rollout_batch, rewards_per_func)

        def _get_encoded_batch(rollout_batch):
            template = self.template
            with self._template_context(template):
                encoded_list = [template.encode(data, return_length=True) for data in rollout_batch]
                encoded_batch = to_device(
                    template.data_collator(encoded_list, padding_to=get_padding_to(args)), self.device)
                if 'cu_seq_lens_q' in encoded_batch:
                    cu_seq_lens_q = encoded_batch['cu_seq_lens_q']
                else:
                    cu_seq_lens_q = get_packed_seq_params(encoded_batch['position_ids'])['cu_seq_lens_q']
                seq_lengths = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]

            labels = encoded_batch['labels']
            batch_size = len(rollout_batch)
            max_seq_len = seq_lengths.max().item()
            assert self.template.padding_free

            truncated_mask = torch.tensor([b['is_truncated'] for b in rollout_batch],
                                          dtype=torch.bool,
                                          device=self.device)

            # completion_mask in rmpad format [1, total_tokens]
            completion_mask_rmpad = (labels != -100).float()
            completion_mask, _ = pad_logps_back_to_batch(
                logps_rmpad=completion_mask_rmpad,
                logits_to_keep=max_seq_len,
                batch_size=batch_size,
                seq_lengths=seq_lengths,
                pad_value=0.0)
            completion_mask = completion_mask.bool()

            encoded_batch.update({
                'completion_mask': completion_mask,  # [batch_size, max_seq_len]
                'truncated_mask': truncated_mask,  # [batch_size]
                'num_samples': batch_size,
                'seq_lengths': seq_lengths,  # [batch_size]
            })

            # Process rollout_logprobs for importance sampling correction
            rollout_per_token_logps = None
            rollout_logprobs_list = [data.get('rollout_logprobs') for data in rollout_batch]
            if all(lp is not None and lp for lp in rollout_logprobs_list):
                # Validate that logprobs count matches completion tokens count
                valid_logprobs = True
                for i, nested_lp in enumerate(rollout_logprobs_list):
                    total_logprobs = sum(len(turn_lps) for turn_lps in nested_lp)
                    completion_count = int(completion_mask[i].sum().item())
                    if total_logprobs != completion_count:
                        logger.warning(f'Rollout logprobs count ({total_logprobs}) does not match '
                                       f'completion tokens count ({completion_count}). '
                                       f'Skipping rollout importance sampling for this batch.')
                        valid_logprobs = False
                        break

                if valid_logprobs:
                    batch_size = completion_mask.shape[0]
                    seq_len = completion_mask.shape[1]
                    rollout_per_token_logps = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=self.device)
                    for i, nested_lp in enumerate(rollout_logprobs_list):
                        # Flatten logprobs for this sample
                        flat_lps = [lp for turn_lps in nested_lp for lp in turn_lps]
                        if flat_lps:
                            # Check for None values in flat_lps
                            if any(lp is None for lp in flat_lps):
                                logger.warning('Found None values in rollout_logprobs. '
                                               'Skipping rollout importance sampling for this batch.')
                                rollout_per_token_logps = None
                                break
                            # Get indices where completion_mask is True
                            completion_indices = completion_mask[i].nonzero(as_tuple=True)[0]
                            # Scatter logprobs to completion positions
                            rollout_per_token_logps[i, completion_indices] = torch.tensor(
                                flat_lps, dtype=torch.float32, device=self.device)

            encoded_batch['rollout_per_token_logps'] = rollout_per_token_logps

            return encoded_batch

        # Gather rollout data across rollout group
        total_batch = gather_object(rollout_batch, group=rollout_group)
        mini_batch_data = []

        # Step 1: Encode batches and compute logps first (unified flow like GRPOTrainer)
        for idx in range(0, len(total_batch), self.micro_batch_size):
            micro_batch_data = total_batch[idx:idx + self.micro_batch_size]
            micro_batch_data = self._maybe_replace_response_token(micro_batch_data)
            micro_batch_encoded = _get_encoded_batch(micro_batch_data)
            with profiling_context(self, 'compute_ref_old_logps'):
                micro_batch_encoded = self._maybe_compute_logps(micro_batch_encoded)
            mini_batch_data.append(micro_batch_encoded)

        # Step 2: Compute KL from logps if kl_in_reward is enabled
        kl_values = None
        if self.kl_in_reward and self.beta != 0.0:
            kl_values = self._compute_kl_from_batches(mini_batch_data)

        # Step 3: Compute advantages (with KL penalty if kl_in_reward is enabled)
        advantages = self._compute_advantages(rollout_batch, rewards_per_func, kl_values=kl_values)
        total_advantages = gather(advantages, group=rollout_group)

        # Step 4: Add advantages to encoded batches
        for idx, micro_batch_encoded in enumerate(mini_batch_data):
            start_idx = idx * self.micro_batch_size
            end_idx = start_idx + micro_batch_encoded['num_samples']
            micro_batch_advantages = total_advantages[start_idx:end_idx]
            micro_batch_encoded['advantages'] = micro_batch_advantages

        if self.loss_type in ['cispo', 'dapo']:
            # Calculate num_items_in_batch
            # Count tokens from all mini_batch_data (this includes gathered data from rollout_group)
            total_token_count = sum(batch_data['seq_lengths'].sum().item() if self.template.
                                    padding_free else batch_data['completion_mask'].sum().item()
                                    for batch_data in mini_batch_data)

            # All-reduce across all ranks
            total_token_count_tensor = torch.tensor(total_token_count, dtype=torch.int, device=self.device)
            torch.distributed.all_reduce(total_token_count_tensor)

            # Divide by rollout_group_size to account for duplicate counting within each rollout_group
            # Each rollout_group (TP×PP×CP ranks) has the same gathered data, so we need to normalize
            rollout_group_size = (
                mpu.get_tensor_model_parallel_world_size() * mpu.get_pipeline_model_parallel_world_size()
                * mpu.get_context_parallel_world_size())
            num_items_in_batch = total_token_count_tensor / rollout_group_size
            # Store num_items_in_batch in each mini_batch_data for CISPO/DAPO loss normalization
            for batch_data in mini_batch_data:
                batch_data['num_items_in_batch'] = num_items_in_batch

        return mini_batch_data

    @profiling_decorator
    def _generate_completions(self, batch):
        """
        Generate completions for a batch of rollout data using vLLM engine.

        This method processes rollout data for the current process, generates completions
        using the vLLM engine, and merges the results back into the original batch.

        Args:
            batch: Rollout data assigned to the current process.

        Returns:
            batch: The input batch with rollout completion results merged in.
        """
        # add prompt ids and system prompts
        batch = self._preprocess_inputs(batch)
        # Step 1: Wake up the engine if it's sleeping (vLLM colocate mode)
        if self.vllm_mode == 'colocate' and self.engine.inner_model_executor.is_sleeping:
            wake_up_params = inspect.signature(self.engine.engine.wake_up).parameters
            # Load weights only (faster and reduces memory peak)
            kwargs = {'tags': ['weights']} if 'tags' in wake_up_params else {}
            self.engine.engine.wake_up(**kwargs)

        # Step 2: Load model weights
        if self._step != self._last_loaded_step or self.args.sleep_level == 2:
            self._move_model_to_vllm()
            self._last_loaded_step = self._step

        context = self.offload_context if self.enable_offload else nullcontext
        with context():
            if (self.vllm_mode == 'colocate' and self.engine.inner_model_executor.is_sleeping
                    and 'tags' in inspect.signature(self.engine.engine.wake_up).parameters):
                aggressive_empty_cache()
                set_expandable_segments(False)
                self.engine.engine.wake_up(tags=['kv_cache'])

            # Step3: Rollout
            outputs: List[RolloutOutput] = self._rollout(batch)

            # Step4: Sleep to release memory
            if self.vllm_mode == 'colocate' and self.args.sleep_level > 0:
                self.engine.engine.reset_prefix_cache()
                self.engine.engine.sleep(level=self.args.sleep_level)
                aggressive_empty_cache()
                set_expandable_segments(True)
            batch = self.postprocess_rollout_data(batch, outputs)

        return batch

    def _rollout(self, batch) -> List[RolloutOutput]:
        batch = self._set_inputs_system(batch)
        request_config = self._get_request_config()
        if self.vllm_mode == 'server':
            rollout_outputs = self._server_rollout(batch, request_config)
        elif self.vllm_mode == 'colocate':
            rollout_outputs = self._colocate_rollout(batch, request_config)
        # log prompt and completions
        messages = gather_object([data['messages'] for data in batch])
        completions = gather_object([data.response.choices[0].message.content for data in rollout_outputs])
        self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(messages))
        self._logs['completion'].extend(completions)

        return rollout_outputs

    def postprocess_rollout_data(self, batch, outputs):
        """
        Post-process the raw vLLM generation outputs and merge them back into the
        original input batch.

        Args:
            batch (List[Dict[str, Any]]):
                Original rollout samples.
            outputs (List[RolloutOutput]):
                outputs from vLLM from vLLM TP group

        Returns:
            List[Dict[str, Any]]:
                Updated samples with rollout results merged in.
        """

        def merge_output_input_data(input_data: Dict[str, Union[torch.Tensor, Any]], output: RolloutOutput):
            response = output.response
            choice = response.choices[0]

            # Step 1: Update or append assistant message
            if output.messages:
                input_data['messages'] = output.messages  # Override full message history
            else:
                # not provided, append
                messages = input_data['messages']
                remove_response(messages)
                messages.append({'role': 'assistant', 'content': choice.message.content})
            # Step 2: Add token IDs and loss mask
            if output.response_token_ids:
                input_data['response_token_ids'] = output.response_token_ids
                if output.response_loss_mask:
                    input_data['response_loss_mask'] = output.response_loss_mask
            else:
                # for single turn, skip tokenizer response
                input_data['response_token_ids'] = output.response.choices[0].token_ids

            # Step 3: Attach rollout extra info
            if output.rollout_infos:
                input_data['rollout_infos'] = output.rollout_infos

            # Step 4: Store finish reason (used for truncation filters etc.)
            input_data['finish_reason'] = choice.finish_reason
            input_data['is_truncated'] = choice.finish_reason == 'length'
            input_data['add_eos'] = False

            # Step 5: Store rollout logprobs for importance sampling correction
            if output.rollout_logprobs:
                input_data['rollout_logprobs'] = output.rollout_logprobs
            elif choice.logprobs is not None:
                if 'content' in choice.logprobs:
                    rollout_logprobs = [item['logprob'] for item in choice.logprobs['content']]
                    input_data['rollout_logprobs'] = [rollout_logprobs]
            return input_data

        assert len(batch) == len(outputs)
        return [merge_output_input_data(input_data, output) for input_data, output in zip(batch, outputs)]

    def _get_request_config(self) -> RequestConfig:
        request_config = copy(self.request_config)
        if self.args.vllm_mode == 'colocate' and self.vllm_tensor_parallel_size > 1:
            # Set request_config.seed
            # 1. Ensure that the seed for vLLM Engines within each TP (Tensor Parallelism) group is the same;
            #   otherwise, the program may hang.
            # 2. Ensure that the seed for vLLM Engines across different TP groups is different;
            #   otherwise, identical completions will be generated.
            batch_size = self.per_device_generation_batch_size
            batch_size *= self.vllm_tensor_parallel_size
            # Since the TP (Tensor Parallelism) group gathers the inputs,
            # multiply the batch size by the TP parallel size.
            request_config.seed = batch_size * (self.process_index // self.vllm_tensor_parallel_size)

        return request_config

    def _server_rollout(self,
                        inputs: DataType,
                        request_config: RequestConfig,
                        is_global_inputs: bool = False) -> List[RolloutOutput]:
        # TODO: async generate
        infer_requests = self.inputs2requests(inputs)

        if is_global_inputs:
            per_device_size = len(infer_requests) // self.world_size
            all_requests = infer_requests
            all_requests_lengths = [per_device_size] + [0] * (self.world_size - 1)
        else:
            all_requests = gather_object(infer_requests)
            all_requests_lengths = gather_object([len(infer_requests)])

        if not any(requests for requests in all_requests):
            return []

        if self.is_main_process:
            all_outputs: List[RolloutOutput] = self.vllm_client.infer(
                infer_requests=all_requests, request_config=request_config)
            assert len(all_outputs) == len(all_requests)  # TODO: dynamic num of samples
        else:
            all_outputs = [None] * len(all_requests)

        if not is_global_inputs:
            all_outputs = broadcast_object_list(all_outputs, from_process=self.world_size - 1)
            start_idx = sum(all_requests_lengths[:self.process_index])
            end_idx = start_idx + all_requests_lengths[self.process_index]
            outputs = all_outputs[start_idx:end_idx]
        else:
            outputs = all_outputs if self.is_main_process else []
        return outputs

    def _colocate_rollout(self, batch, request_config: RequestConfig):
        if self.vllm_tensor_parallel_size > 1:
            local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
            local_input_length = len(batch)
            all_input_lengths = [None] * self.vllm_tensor_parallel_size
            torch.distributed.all_gather_object(all_input_lengths, local_input_length, group=self.vllm_tp_group)

            start_idx = sum(all_input_lengths[:local_rank_in_group])
            end_idx = start_idx + all_input_lengths[local_rank_in_group]

            gathered_batch = [None for _ in range(self.vllm_tensor_parallel_size)]
            torch.distributed.all_gather_object(gathered_batch, batch, group=self.vllm_tp_group)
            batch = [p for sublist in gathered_batch for p in sublist]

        outputs: List[RolloutOutput] = self.engine.infer(infer_requests=batch, request_config=request_config)

        if self.vllm_tensor_parallel_size > 1:
            outputs = outputs[start_idx:end_idx]

        return outputs

    @profiling_decorator
    def _score_completions(self, inputs: DataType) -> torch.Tensor:
        """Score completions using all reward functions.

        Args:
            inputs: List of input examples, each containing a 'messages' list with conversation history

        Returns:
            rewards_per_func: Tensor of shape (num_examples, num_reward_funcs) with local reward values
        """
        # Compute rewards using reward functions
        local_rewards_per_func = self._compute_rewards_per_func(inputs)

        return local_rewards_per_func

    def _compute_rewards_per_func(self, batch: DataType) -> torch.Tensor:
        """Compute rewards using all reward functions"""
        device = self.device
        rewards_per_func = torch.zeros((len(batch), len(self.reward_funcs)), device=device)
        completions = [inp['messages'][-1]['content'] for inp in batch]
        reward_kwargs = {}  # TODO: training step info
        for i, (reward_func, reward_model_plugin, reward_func_name) in enumerate(
                zip(self.reward_funcs, self.reward_model_plugins, self.reward_func_names)):
            with profiling_context(self, reward_func_name):
                # reward model
                if isinstance(reward_func, nn.Module):
                    output_reward_func = reward_model_plugin(inputs=batch, **reward_kwargs)
                # reward function
                else:
                    # Repeat all input columns (but "messages" and "completion") to match the number of generations
                    reward_kwargs.update(RowPreprocessor.rows_to_batched(batch))
                    output_reward_func = reward_func(completions, **reward_kwargs)
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs['completion'] = completions[nan_row_idx]
            logger.warning(f'All reward functions returned None for the following kwargs: {row_reward_kwargs}. '
                           'Please ensure that at least one reward function returns a valid reward.')

        return rewards_per_func

    def _compute_advantages(self,
                            batch: DataType,
                            rewards_per_func: torch.Tensor,
                            kl_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute advantages for RL training.

        Supports different advantage estimators:
        - 'grpo': Group mean baseline
        - 'rloo': Leave-One-Out baseline
        - 'reinforce_plus_plus': Similar to grpo but normalizes advantages std

        Args:
            batch: Local batch data samples
            rewards_per_func: Reward per function for local data samples
            kl_values: Optional KL values for kl_in_reward mode, shape [total_samples]

        Returns:
            advantages: Computed advantages for local batch, shape [local_batch_size]
        """

        def normalize_advantages(advantages: torch.Tensor, std_values: torch.Tensor) -> torch.Tensor:
            """Normalize advantages if configured; otherwise, return as-is."""
            if self.scale_rewards != 'none':
                return advantages / (std_values + 1e-4)
            return advantages

        mode = 'train' if self.unwrapped_models[0].training else 'eval'
        assert len(batch) == rewards_per_func.shape[0]
        total_rewards_per_func = gather(rewards_per_func)
        rewards = (total_rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)

        # Apply KL penalty to rewards if kl_in_reward is enabled
        if self.kl_in_reward and self.beta != 0.0 and kl_values is not None:
            self._metrics[mode]['kl'].append(kl_values.nanmean().item())
            rewards = rewards - self.beta * kl_values

        # Use num_generations_eval in eval mode
        num_generations = self.num_generations if mode == 'train' else self.num_generations_eval
        grouped_rewards = rewards.view(-1, num_generations)
        K = num_generations

        # Compute group statistics
        group_rewards_mean = grouped_rewards.mean(dim=1)

        # Broadcast stats back to the original shape
        group_rewards_mean = group_rewards_mean.repeat_interleave(K)

        # Compute advantages based on estimation type
        if self.advantage_estimator == 'rloo':
            # RLOO: Leave-One-Out baseline
            # A_i = r_i - mean(r_j for j != i)
            # = r_i * K/(K-1) - mean_all * K/(K-1)
            # Edge case: when K=1 (e.g., num_generations_eval=1), fall back to simple advantage
            if K > 1:
                advantages = rewards * K / (K - 1) - group_rewards_mean * K / (K - 1)
            else:
                advantages = rewards - group_rewards_mean
        else:  # 'grpo' or 'reinforce_plus_plus'
            # Both use group mean as baseline
            advantages = rewards - group_rewards_mean

        # Normalize advantages based on estimator and scale_rewards
        if self.advantage_estimator == 'reinforce_plus_plus':
            # REINFORCE++: Use std of advantages (not rewards)
            if self.scale_rewards == 'batch':
                # Global whitening: std computed on advantages
                if advantages.numel() > 1:
                    advantages_std = advantages.std().expand_as(advantages)
                else:  # edge case: num_generations_eval=batch_size=1
                    advantages_std = torch.zeros_like(advantages)
            elif self.scale_rewards == 'group':
                # Group-level whitening on advantages
                advantages_grouped = advantages.view(-1, K)
                if K > 1:
                    advantages_std = advantages_grouped.std(dim=1).repeat_interleave(K)
                else:  # edge case: num_generations_eval=1
                    advantages_std = torch.zeros_like(advantages)
            else:  # 'none'
                advantages_std = None
            if advantages_std is not None:
                advantages = normalize_advantages(advantages, advantages_std)
        else:  # 'grpo' or 'rloo'
            # GRPO/RLOO: Use std of original rewards
            if self.scale_rewards == 'batch':
                # Global batch-level normalization
                if rewards.numel() > 1:
                    rewards_std = rewards.std().expand_as(rewards)
                else:  # edge case: num_generations_eval=batch_size=1
                    rewards_std = torch.zeros_like(rewards)
            elif self.scale_rewards == 'group':
                # Group-level normalization (default)
                if K > 1:
                    rewards_std = grouped_rewards.std(dim=1).repeat_interleave(K)
                else:  # edge case: num_generations_eval=1
                    rewards_std = torch.zeros_like(rewards)
            else:  # 'none'
                rewards_std = None
            if rewards_std is not None:
                advantages = normalize_advantages(advantages, rewards_std)

        def log_rewards_metrics(rewards: torch.Tensor, rewards_per_func_for_metrics: torch.Tensor):
            """Log reward statistics for monitoring. Only log once per unique request_id."""
            # rewards: [prompt_batch_size, num_generations]
            # rewards_per_func_for_metrics: [prompt_batch_size*num_generations, self.num_reward_funcs]
            group_rewards = rewards.view(-1, num_generations)
            rewards_mean = group_rewards.mean(-1).mean().item()
            # Compute std based on scale_rewards setting for logging
            if self.scale_rewards in ['group', 'none']:
                # Handle edge case when num_generations_eval=1
                if num_generations > 1:
                    rewards_std = group_rewards.std(-1).mean().item()
                else:
                    rewards_std = 0.0
            elif self.scale_rewards == 'batch':
                rewards_std = rewards.std().item() if rewards.numel() > 1 else 0.0
            if num_generations > 1:
                is_std_zero = torch.isclose(group_rewards.std(dim=1), torch.zeros_like(group_rewards.std(dim=1)))
            else:
                is_std_zero = torch.ones(group_rewards.size(0), dtype=torch.bool, device=group_rewards.device)

            self._metrics[mode]['reward'].append(rewards_mean)
            self._metrics[mode]['reward_std'].append(rewards_std)
            self._metrics[mode]['frac_reward_zero_std'].append(is_std_zero.float().mean().item())

            # Log per-reward-function statistics using deduplicated rewards_per_func
            for i, name in enumerate(self.reward_func_names):
                col = rewards_per_func_for_metrics[:, i]
                self._metrics[mode][f'rewards/{name}/mean'].append(torch.nanmean(col).item())
                self._metrics[mode][f'rewards/{name}/std'].append(nanstd(col).item())

        log_rewards_metrics(rewards=grouped_rewards, rewards_per_func_for_metrics=total_rewards_per_func)
        self._logs['advantages'].extend(advantages.tolist())
        for i, name in enumerate(self.reward_func_names):
            self._logs['rewards'][name].extend(total_rewards_per_func[:, i].tolist())

        slice_start = self.process_index * len(batch)
        slice_end = slice_start + len(batch)
        advantages = advantages[slice_start:slice_end]

        return advantages

    def _dynamic_sampling(self, rollout_batch: DataType,
                          rewards_per_func: torch.Tensor) -> Tuple[DataType, torch.Tensor]:
        """
        Perform dynamic sampling to replace samples with zero-reward-variance groups.

        This method implements DAPO (https://arxiv.org/abs/2503.14476) by replacing
        samples from groups with zero reward variance (std=0) through resampling.

        Args:
            rollout_batch: local rollout data samples
            rewards_per_func: reward per function for local data samples
            rollout_group: rollout communication group

        Returns:
            tuple: (rollout_batch, rewards_per_func) with zero-variance groups replaced by resampled data
        """
        resample_count = 0
        valid_samples = []
        valid_rewards_per_func = []
        origin_data = (rollout_batch, rewards_per_func)

        while resample_count < self.max_resample_times:
            # Gather all samples and rewards across rollout group first
            global_rollout_batch = gather_object(rollout_batch)
            global_rewards_per_func = gather(rewards_per_func)

            # Compute reward std for the entire global batch
            # We need to compute std on the gathered data to get a global mask
            global_rewards = (global_rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)
            mode = 'train' if self.unwrapped_models[0].training else 'eval'
            num_generations = self.num_generations if mode == 'train' else self.num_generations_eval
            grouped_rewards = global_rewards.view(-1, num_generations)
            # Handle edge case when num_generations=1
            if num_generations > 1:
                group_rewards_std = grouped_rewards.std(dim=1).repeat_interleave(num_generations)
            else:
                group_rewards_std = torch.zeros_like(global_rewards)
            global_valid_mask = (group_rewards_std > 0)

            # Filter valid samples based on std > 0
            valid_samples.extend([sample for sample, mask in zip(global_rollout_batch, global_valid_mask) if mask])
            valid_rewards_per_func.append(global_rewards_per_func[global_valid_mask])

            if len(valid_samples) >= self.generation_batch_size:
                break

            # Lazy initialization of resample_data_iterator
            # Only initialize when needed, after pretrain() has set up args
            if not hasattr(self, 'resample_data_iterator') or self.resample_data_iterator is None:
                self.resample_data_iterator = self._init_resample_data_iterator()
            num_iters_per_step = self.get_num_iters_per_step()
            next_rollout_prompt_batch = []
            for _ in range(num_iters_per_step):
                next_rollout_prompt_batch.extend(next(self.resample_data_iterator))

            # Resample for encoding failed data when truncation_strategy is 'raise'(delete)
            if self.truncation_strategy == 'raise':
                next_rollout_prompt_batch = self.resample_encode_failed_inputs(next_rollout_prompt_batch)

            # Repeat num_generations times and get local slice
            rollout_batch = self.get_local_rollout_batch(next_rollout_prompt_batch)

            # Generate and score new completions
            rollout_batch = self._generate_completions(rollout_batch)
            rewards_per_func = self._score_completions(rollout_batch)
            resample_count += 1

        if len(valid_samples) >= self.generation_batch_size:
            # Get local slice of valid samples
            rank = self.process_index
            per_device_batch_size = self.per_device_generation_batch_size
            data_slice = slice(rank * per_device_batch_size, (rank + 1) * per_device_batch_size)
            rollout_batch = valid_samples[:self.generation_batch_size][data_slice]
            rewards_per_func = torch.cat(valid_rewards_per_func)[:self.generation_batch_size][data_slice]
        else:
            logger.warning(f'There are still std=0 groups present after {self.max_resample_times} retries.')
            rollout_batch, rewards_per_func = origin_data

        return rollout_batch, rewards_per_func

    def _maybe_compute_logps(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: entropy
        seq_lengths = batch['seq_lengths']
        batch_size = batch['num_samples']
        max_seq_len = batch['completion_mask'].shape[1]

        inputs = self._prepare_model_inputs(batch)
        if self.beta != 0.0:
            with torch.no_grad(), self.null_ref_context() as ref_models:
                assert len(ref_models) == 1, 'GRPO currently does not support VPP.'
                ref_model = ref_models[0]
                ref_per_token_logps_rmpad = self.model_forward(
                    ref_model, iter([deepcopy(inputs)]), no_grad=True, per_token=True)['logps']
                ref_per_token_logps, _ = pad_logps_back_to_batch(
                    logps_rmpad=ref_per_token_logps_rmpad,
                    logits_to_keep=max_seq_len,
                    batch_size=batch_size,
                    seq_lengths=seq_lengths)
                batch['ref_per_token_logps'] = ref_per_token_logps

        old_per_token_logps_rmpad = self.model_forward(
            self.unwrapped_models[0], iter([deepcopy(inputs)]), no_grad=True, per_token=True)['logps']
        old_per_token_logps, _ = pad_logps_back_to_batch(
            logps_rmpad=old_per_token_logps_rmpad,
            logits_to_keep=max_seq_len,
            batch_size=batch_size,
            seq_lengths=seq_lengths)
        batch['old_per_token_logps'] = old_per_token_logps

        return batch

    def _compute_kl_from_batches(self, mini_batch_data: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute per-sample KL divergence from encoded batches for kl_in_reward.

        The KL is computed as: sum over tokens of (old_logp - ref_logp) for each sample.

        Args:
            mini_batch_data: List of encoded batch dictionaries containing:
                - old_per_token_logps: [batch_size, max_seq_len] in batch format
                - ref_per_token_logps: [batch_size, max_seq_len] in batch format
                - completion_mask: [batch_size, max_seq_len] mask for completion tokens

        Returns:
            kl_values: Per-sample KL values, shape [total_samples]
        """
        kl_list = []
        assert self.beta != 0.0

        for batch in mini_batch_data:
            old_per_token_logps = batch['old_per_token_logps']  # [batch_size, max_seq_len]
            ref_per_token_logps = batch['ref_per_token_logps']  # [batch_size, max_seq_len]
            completion_mask = batch['completion_mask']  # [batch_size, max_seq_len]

            # Compute per-token KL: old_logp - ref_logp
            per_token_kl = old_per_token_logps - ref_per_token_logps  # [batch_size, max_seq_len]

            # Compute per-sample KL by summing over completion tokens
            sample_kl = (per_token_kl * completion_mask).sum(-1)  # [batch_size]
            kl_list.append(sample_kl)

        # Concatenate all KL values and gather across ranks
        kl_values = torch.cat(kl_list, dim=0)
        kl_values = gather(kl_values)

        return kl_values

    @contextmanager
    def _disable_maxlength_template_context(self, template: Template):
        # The max_length for prompt and completion has already been restricted, so there is no need for max_length here.
        max_length = template.max_length
        template.max_length = None
        try:
            yield
        finally:
            template.max_length = max_length

    def _maybe_replace_response_token(self, batch):
        # maybe replace the response token with the response token ids to avoid repetitive tokenize

        for data in batch:
            if 'response_token_ids' in data and data['response_token_ids']:
                loss_mask = None
                if 'response_loss_mask' in data and data['response_loss_mask']:
                    loss_mask = data['response_loss_mask']
                # token in token out
                data['messages'] = replace_assistant_response_with_ids(data['messages'], data['response_token_ids'],
                                                                       loss_mask)
        return batch

    @property
    def on_policy(self):
        return self.steps_per_generation == 1

    @contextmanager
    def patch_megatron_data_collator(self, data_collator):
        """
        Context manager that temporarily patches Megatron's data-loader factory so each
        prompt-level micro-batch size equals (original micro-batch size // num_generations),
        required by GRPO.  Restores the original size and loader on exit.
        """
        origin_build_pretraining_data_loader = training.build_pretraining_data_loader

        def build_pretraining_data_loader(*_args, **kwargs):
            args = get_args()
            org_micro_batch_size = args.micro_batch_size
            # args.micro_batch_size = org_micro_batch_size // self.num_generations
            res = origin_build_pretraining_data_loader(*_args, **kwargs)
            args.micro_batch_size = org_micro_batch_size
            if res is not None and args.dataloader_type != 'external':
                res.collate_fn = data_collator
            return res

        training.build_pretraining_data_loader = build_pretraining_data_loader
        try:
            yield
        finally:
            training.build_pretraining_data_loader = origin_build_pretraining_data_loader

    @profiling_decorator
    def forward_step(self, data_iterator, model):
        # train_batch_size
        # return: output_tensor, loss_func
        data = self.get_batch(data_iterator)
        data.pop('loss_scale', None)
        inputs = self._prepare_model_inputs(data)

        with self.stimer:
            output_tensor = model(**inputs)
        return output_tensor, partial(self.loss_func, data=data)

    @profiling_decorator
    def loss_func(self, output_tensor: torch.Tensor, data: Dict[str, Any]):
        # Get pre-padded data in batch format [batch_size, max_seq_len]
        advantages = data['advantages']  # [batch_size]
        labels = data['labels']
        completion_mask = data['completion_mask']  # [batch_size, max_seq_len]
        packed_seq_params = data['packed_seq_params']
        truncated_mask = data['truncated_mask']  # [batch_size]
        seq_lengths = data['seq_lengths']  # [batch_size]
        max_seq_len = completion_mask.shape[1]
        micro_batch_size = self.micro_batch_size

        # Use full sequence lengths directly (get_logps returns full sequences in CP mode)
        lengths = packed_seq_params.cu_seqlens_q[1:micro_batch_size
                                                 + 1] - packed_seq_params.cu_seqlens_q[:micro_batch_size]

        # get_logps with per_token=True returns rmpad format [1, total_tokens]
        # Pad to batch format [batch_size, max_seq_len]
        per_token_logps_rmpad = self.get_logps(
            output_tensor, labels, packed_seq_params, packed_seq_params.num_samples, per_token=True)
        per_token_logps, _ = pad_logps_back_to_batch(
            logps_rmpad=per_token_logps_rmpad,
            logits_to_keep=max_seq_len,
            batch_size=micro_batch_size,
            seq_lengths=seq_lengths)

        # Get pre-padded ref/old/rollout logps from data
        ref_per_token_logps = data.get('ref_per_token_logps')  # [batch_size, max_seq_len] or None
        old_per_token_logps = data.get('old_per_token_logps')  # [batch_size, max_seq_len]
        rollout_per_token_logps = data.get('rollout_per_token_logps')  # [batch_size, max_seq_len] or None

        # Rollout importance sampling correction
        rollout_correction_metrics = {}
        should_compute_rollout_metrics = (
            self.rollout_importance_sampling_mode is not None or self.log_rollout_offpolicy_metrics)
        local_has_rollout_per_token_logps = rollout_per_token_logps is not None
        dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        all_has_rollout_per_token_logps = gather_object([local_has_rollout_per_token_logps], group=dp_group)
        should_compute_rollout_metrics = should_compute_rollout_metrics and all(all_has_rollout_per_token_logps)
        if (not self.disable_rollout_importance_sampling and should_compute_rollout_metrics):
            # Compute off-policy diagnostic metrics
            rollout_correction_metrics = self._compute_rollout_offpolicy_metrics(old_per_token_logps,
                                                                                 rollout_per_token_logps,
                                                                                 completion_mask)

            # Apply importance sampling correction if mode is enabled
            if self.rollout_importance_sampling_mode is not None:
                rollout_log_ratio = old_per_token_logps - rollout_per_token_logps
                rollout_is_weights = self._apply_rollout_importance_sampling(rollout_log_ratio, completion_mask)

                # Compute IS-specific metrics
                is_metrics = self._compute_is_correction_metrics(rollout_log_ratio, rollout_is_weights, completion_mask)
                rollout_correction_metrics.update(is_metrics)

        # Apply truncated_mask filter (now in batch format)
        if self.args.overlong_filter and truncated_mask.any():
            if truncated_mask.all():
                logger.warning('All completions are truncated in this batch. Loss and grad_norm will be 0. '
                               'Consider increasing max_completion_length')
            # Expand truncated_mask from [batch_size] to [batch_size, max_seq_len]
            truncated_mask_expanded = truncated_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & (~truncated_mask_expanded)

        # Compute KL divergence if needed
        # Only compute KL for loss if kl_in_reward=False (GRPO style)
        # When kl_in_reward=True, KL penalty is already applied to rewards in _compute_advantages
        if self.beta != 0.0 and ref_per_token_logps is not None and not self.kl_in_reward:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)
        else:
            per_token_kl = None

        # Compute log ratio for importance sampling
        log_ratio = per_token_logps - old_per_token_logps

        # Compute importance weights based on level
        if self.importance_sampling_level == 'token':
            log_importance_weights = log_ratio
        elif self.importance_sampling_level in ['sequence', 'sequence_token']:
            # Sequence-level: compute mean log ratio per sequence
            seq_level_log_weights = ((log_ratio * completion_mask).sum(-1)
                                     / completion_mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
            if self.importance_sampling_level == 'sequence':
                log_importance_weights = seq_level_log_weights
            else:
                seq_level_log_weight = seq_level_log_weights.detach()
                log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                ",'sequence' and 'sequence_token'.")

        coef_1 = torch.exp(log_importance_weights)

        if self.loss_type == 'cispo':
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages.unsqueeze(1) * per_token_logps
        elif self.loss_type == 'sapo':
            gate_pos = torch.sigmoid(self.tau_pos * (coef_1 - 1))
            gate_neg = torch.sigmoid(self.tau_neg * (coef_1 - 1))
            is_positive = advantages.unsqueeze(1) > 0
            soft_gate = torch.where(is_positive, gate_pos, gate_neg)
            per_token_loss = -soft_gate * advantages.unsqueeze(1)
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo']:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        if self.rollout_importance_sampling_mode is not None:
            # Apply IS weights to loss
            per_token_loss = per_token_loss * rollout_is_weights
        # Add KL penalty if needed
        if self.beta != 0.0 and per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Apply off-policy sequence masking if enabled
        # Mask out sequences where delta > threshold AND advantage < 0
        if self.off_policy_sequence_mask_delta is not None:
            old_policy_per_token_logps = rollout_per_token_logps if rollout_per_token_logps is not None \
                else old_per_token_logps
            off_policy_seq_mask = self._compute_off_policy_sequence_mask(per_token_logps, old_policy_per_token_logps,
                                                                         completion_mask, advantages)
            # Expand sequence mask to token level and apply to completion_mask
            off_policy_seq_mask_expanded = off_policy_seq_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & off_policy_seq_mask_expanded

        if self.loss_type in ['grpo', 'sapo']:
            # Per-sample mean, then batch mean
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == 'bnpo':
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == 'dr_grpo':
            loss = (per_token_loss * completion_mask).sum() / (micro_batch_size * self.max_completion_length)
        elif self.loss_type in ['cispo', 'dapo']:
            # CISPO and DAPO: Normalize by total completion tokens across all processes
            num_items_in_batch = data['num_items_in_batch']
            dp_size = mpu.get_data_parallel_world_size()
            normalizer = num_items_in_batch / dp_size
            loss = (per_token_loss * completion_mask).sum() / normalizer.clamp(min=1.0)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        avg_metric = {
            'loss': loss.clone().detach(),
        }
        custom_metrics = {}
        total_lengths = gather(lengths, group=mpu.get_data_parallel_group(with_context_parallel=True))
        custom_metrics = {
            'completions/mean_length': total_lengths.float().mean(),
            'completions/max_length': total_lengths.float().max(),
            'completions/min_length': total_lengths.float().min(),
        }

        if self.beta != 0.0 and per_token_kl is not None:
            kl_value = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            avg_metric['kl'] = kl_value.clone().detach()

        mode = 'train' if self.unwrapped_models[0].training else 'eval'

        # Compute clipping metrics
        completion_token_count = completion_mask.sum().clamp(min=1.0)
        if self.loss_type == 'cispo':
            # CISPO: Only track upper bound clipping
            # coef_1 is [batch_size, max_seq_len] or [batch_size, 1] depending on importance_sampling_level
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            cispo_clip_ratio = (is_cispo_clipped.float() * completion_mask).sum() / completion_token_count
            # Store local clip ratio, _all_reduce_metric will handle averaging across ranks
            self._metrics[mode]['cispo_clip_ratio'].append(cispo_clip_ratio)
        elif self.loss_type == 'sapo':
            # SAPO: No hard clipping, skip clipping metrics
            pass
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo']:
            # coef_1 is [batch_size, max_seq_len] or [batch_size, 1] depending on importance_sampling_level
            # Use exp(log_importance_weights) to get the original ratios before clamping
            coef_1_for_metrics = torch.exp(log_importance_weights)
            is_low_clipped = (coef_1_for_metrics < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
            is_high_clipped = (coef_1_for_metrics > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            low_clip = (is_low_clipped.float() * completion_mask).sum() / completion_token_count
            high_clip = (is_high_clipped.float() * completion_mask).sum() / completion_token_count
            is_region_clipped = is_low_clipped | is_high_clipped
            clip_ratio = (is_region_clipped.float() * completion_mask).sum() / completion_token_count

            # For min/max, we need to gather values from all ranks to compute global min/max
            # For mean, let _all_reduce_metric handle averaging
            gathered_low_clip = gather(
                low_clip.unsqueeze(0), group=mpu.get_data_parallel_group(with_context_parallel=True))
            gathered_high_clip = gather(
                high_clip.unsqueeze(0), group=mpu.get_data_parallel_group(with_context_parallel=True))

            # Store local values for mean (will be averaged by _all_reduce_metric)
            self._metrics[mode]['clip_ratio/low_mean'].append(low_clip)
            self._metrics[mode]['clip_ratio/high_mean'].append(high_clip)
            self._metrics[mode]['clip_ratio/region_mean'].append(clip_ratio)
            # Store global min/max in custom_metrics (not through _all_reduce_metric to avoid incorrect averaging)
            custom_metrics['clip_ratio/low_min'] = gathered_low_clip.min()
            custom_metrics['clip_ratio/high_max'] = gathered_high_clip.max()

        # Add rollout correction metrics
        if rollout_correction_metrics:
            for key, value in rollout_correction_metrics.items():
                if isinstance(value, torch.Tensor):
                    custom_metrics[f'rollout_correction/{key}'] = value.clone().detach()
                else:
                    custom_metrics[f'rollout_correction/{key}'] = torch.tensor(value, device=loss.device)

        if self._metrics[mode]:
            addition_metrics = {
                key: torch.tensor(sum(val) / len(val), device=loss.device)
                for key, val in self._metrics[mode].items()
            }
            avg_metric.update(addition_metrics)

        avg_metric = self._all_reduce_metric(avg_metric)

        reporting_metric = {**avg_metric, **custom_metrics}

        # log_completions
        if (self.log_completions and self.is_main_process and (self._step - 1) % self.steps_per_generation == 0
                and self._step != self._last_logged_step):
            table = {
                'gen_step': [self._step - 1] * len(self._logs['prompt']),
                'prompt': list(self._logs['prompt']),
                'completion': list(self._logs['completion']),
                **{k: list(v)
                   for k, v in self._logs['rewards'].items()},
                'advantages': list(self._logs['advantages']),
            }
            self.jsonl_writer.append(table)
            wandb_writer = get_wandb_writer()
            if wandb_writer:
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                # if not self.init_custom_metric:
                #     wandb_writer.define_metric('completions', step_metric='gen_step')
                #     self.init_custom_metric = True
                wandb_writer.log({'completions': wandb.Table(dataframe=df)})
            self._last_logged_step = self._step

        return loss, reporting_metric

    def model_forward(self, model, data_iterator, no_grad=True, per_token=False):
        # used to calculate model forward (logps) in GRPO
        with self.stimer(bdata=True):
            data = self.get_batch(data_iterator)
        data.pop('loss_scale', None)
        labels = data.get('labels')
        context = torch.no_grad() if no_grad else nullcontext()
        with context:
            output_tensor = forward_step_helper(model, data)
        packed_seq_params = data['packed_seq_params']
        data['logps'] = None if labels is None else self.get_logps(
            output_tensor, labels, data['packed_seq_params'], packed_seq_params.num_samples, per_token=per_token)
        return data

    @contextmanager
    def offload_context(self):
        if self.args.offload_model:
            offload_megatron_model_to_cpu(self.wrapped_models)
            if hasattr(self, 'ref_models') and self.ref_models:
                offload_megatron_model_to_cpu(self.ref_models)
        if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
            offload_megatron_optimizer(self.optimizer)

        try:
            yield
        finally:
            # reload (load back) model when exiting context
            if self.args.offload_model:
                load_megatron_model_to_gpu(self.wrapped_models)
                if hasattr(self, 'ref_models') and self.ref_models:
                    load_megatron_model_to_gpu(self.ref_models)
            if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
                load_megatron_optimizer(self.optimizer)

    def inputs2requests(self, inputs: Union[DataType, List[RolloutInferRequest]]) -> List[RolloutInferRequest]:
        """Convert raw input data into RolloutInferRequest objects"""

        def _process_image_data(image_data: Union[dict, str]) -> str:
            if isinstance(image_data, dict):
                if image_data.get('bytes'):
                    return base64.b64encode(image_data['bytes']).decode('utf-8')
                if image_data.get('path'):
                    return image_data['path']
            return image_data

        if not inputs:
            return []

        args = self.args

        REQUEST_METADATA_FIELDS = ['messages', 'images', 'audios', 'videos', 'tools', 'objects', 'uuid']
        requests_list = []

        for data in inputs:
            if isinstance(data, RolloutInferRequest):
                request_obj = data
            else:
                request_data = {
                    key: data[key]
                    for key in REQUEST_METADATA_FIELDS if key in data and data[key] is not None
                }
                if 'uuid' not in request_data:
                    request_data['uuid'] = data['request_id']
                if hasattr(args, 'vllm_server_pass_dataset') and args.vllm_server_pass_dataset:
                    extra_fields = {
                        k: v
                        for k, v in data.items() if k not in REQUEST_METADATA_FIELDS and data[k] is not None
                    }
                    if extra_fields:
                        request_data['data_dict'] = extra_fields
                elif self.multi_turn_scheduler:
                    base_data_dict = {}
                    if 'data_dict' in data:
                        if isinstance(data['data_dict'], dict):
                            base_data_dict = data['data_dict']
                        else:
                            raise ValueError('data_dict exists but is not a dictionary')
                    extra_data = {
                        k: v
                        for k, v in data.items()
                        if k not in REQUEST_METADATA_FIELDS and k != 'data_dict' and data[k] is not None
                    }
                    final_data_dict = {**extra_data, **base_data_dict}
                    request_data['data_dict'] = final_data_dict if final_data_dict else {}

                if 'images' in request_data and request_data['images']:
                    imgs = request_data['images']
                    if not isinstance(imgs, list):
                        imgs = [imgs]
                    request_data['images'] = [_process_image_data(img) for img in imgs]

                if 'tools' in request_data and isinstance(request_data['tools'], str):
                    try:
                        request_data['tools'] = json.loads(request_data['tools'])
                    except json.JSONDecodeError:
                        pass

                request_obj = from_dict(RolloutInferRequest, request_data)
            requests_list.append(request_obj)

        return requests_list

    def _preprocess_inputs(self, inputs: DataType) -> DataType:
        """Preprocess inputs before inference"""
        processed_inputs = self._add_prompt_id_to_inputs(inputs)
        for input_item in processed_inputs:
            remove_response(input_item['messages'])
        return processed_inputs

    @profiling_decorator
    def resample_encode_failed_inputs(self, inputs: DataType, max_resample_rounds: int = 10) -> DataType:
        """
        Attempt to encode each input using the template. If encoding fails,
        resample from a backup iterator until we have enough valid samples.

        This method handles two cases:
        1. Prompt length exceeds max_length
        2. Encoding failures (e.g., multimodal data processing errors)

        Unlike GRPOTrainer which fetches one sample at a time, this method accumulates
        successfully encoded samples from each resample batch (micro_batch_size samples per fetch)
        to avoid wasting data.

        Args:
            inputs (DataType): A list of input data samples, each containing a `messages` field.
            max_resample_rounds (int, optional): Maximum number of resample rounds.
                Each round processes samples from pending_samples buffer. Defaults to 10.

        Returns:
            DataType: A list of successfully encoded input samples with the same length as inputs.

        Raises:
            RuntimeError: If we cannot collect enough valid samples after max_resample_rounds.
        """
        template = self.template
        required_count = len(inputs)
        valid_samples = []

        # Buffer for samples waiting to be validated
        pending_samples = list(inputs)
        # Lazy initialization of resample_data_iterator
        if not hasattr(self, 'resample_data_iterator') or self.resample_data_iterator is None:
            self.resample_data_iterator = self._init_resample_data_iterator()
        for _ in range(max_resample_rounds + 1):
            # Calculate how many more samples we need
            still_needed = required_count - len(valid_samples)
            if still_needed <= 0:
                break

            # Ensure pending_samples has enough samples to try
            while len(pending_samples) < still_needed:
                # Fetch a new batch of samples (micro_batch_size samples)
                pending_samples.extend(next(self.resample_data_iterator))

            # Try to encode samples from pending_samples until we have enough valid ones
            while pending_samples and len(valid_samples) < required_count:
                data = pending_samples.pop(0)
                try:
                    remove_response(data['messages'])
                    template.encode(data)
                    # Encoding succeeded, add to valid samples
                    valid_samples.append(data)
                except Exception as e:
                    # Encoding failed, skip this sample
                    logger.info(f'Encoding failed for one sample; will resample. {e}')

        if len(valid_samples) < required_count:
            raise RuntimeError(
                f'Failed to collect {required_count} valid samples after {max_resample_rounds} resample rounds. '
                f'Only collected {len(valid_samples)} valid samples. '
                'Consider increasing `max_length` or adjusting the `truncation_strategy`.')

        return valid_samples[:required_count]

    def _add_prompt_id_to_inputs(self, inputs: DataType) -> DataType:
        """Add unique prompt_id and request_id to each input"""
        if not inputs:
            return inputs

        all_messages = gather_object([inp['messages'] for inp in inputs])
        messages_to_prompt_id = {}
        prompt_id_counter = 0

        for messages in all_messages:
            key = json.dumps(messages)
            if key not in messages_to_prompt_id:
                messages_to_prompt_id[key] = f'prompt_{prompt_id_counter}'
                prompt_id_counter += 1

        for input_item in inputs:
            messages = input_item.get('messages')
            input_item['prompt_id'] = messages_to_prompt_id[json.dumps(messages)]
            input_item['request_id'] = f'chatcmpl-{str(uuid.uuid4().hex)}'

        return inputs

    def get_num_iters_per_step(self):
        mode = 'train' if self.unwrapped_models[0].training else 'eval'
        cache_key = f'_num_iters_per_step_{mode}'
        if hasattr(self, cache_key):
            return getattr(self, cache_key)
        # each rollout DP group will generate generation_batch_size / dp_size completions
        dp_size = mpu.get_data_parallel_world_size()
        completions_to_rollout = self.generation_batch_size // dp_size
        # completions will be repeated num_generations times after
        # so we need to divide num_iters_per_step by num_generations to get prompt batch size
        num_generations = self.num_generations if mode == 'train' else self.num_generations_eval
        prompts_to_rollout = completions_to_rollout // num_generations
        # every iter will generate micro_batch_size prompts
        num_iters_per_step = prompts_to_rollout // self.micro_batch_size
        assert num_iters_per_step > 0, (
            f'num_iters_per_step={num_iters_per_step} <= 0. '
            f'This means no prompts will be generated'
            f'generation_batch_size={self.generation_batch_size}, '
            f'data_parallel_world_size={mpu.get_data_parallel_world_size()}, '
            f'num_generations={num_generations}, '
            f'micro_batch_size={self.micro_batch_size}. '
            'Please adjust these parameters so that '
            'generation_batch_size // data_parallel_world_size // num_generations // micro_batch_size >= 1.')
        setattr(self, cache_key, num_iters_per_step)
        return num_iters_per_step

    def get_local_rollout_batch(self, batch):
        mode = 'train' if self.unwrapped_models[0].training else 'eval'
        num_generations = self.num_generations if mode == 'train' else self.num_generations_eval
        # repeat num_generations times
        rollout_group = self._get_rollout_group()
        global_rollout_batch = [deepcopy(item) for item in batch for _ in range(num_generations)]
        # get local rollout data
        rollout_rank = torch.distributed.get_rank(group=rollout_group)
        rollout_group_size = torch.distributed.get_world_size(group=rollout_group)

        per_device_batch_size = self.per_device_generation_batch_size
        assert rollout_group_size * per_device_batch_size == len(global_rollout_batch)
        data_slice = slice(rollout_rank * per_device_batch_size, (rollout_rank + 1) * per_device_batch_size)
        rollout_batch = global_rollout_batch[data_slice]
        return rollout_batch

    @contextmanager
    def _template_context(self, template: Template):
        # The max_length for prompt and completion has already been restricted, so there is no need for max_length here.
        max_length = template.max_length
        template.max_length = None
        try:
            yield
        finally:
            template.max_length = max_length

    def _prepare_metrics(self):
        args = self.args
        from swift.utils import JsonlWriter
        from collections import deque
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.jsonl_writer = JsonlWriter(os.path.join(args.save, 'completions.jsonl'), write_on_rank='last')
        self.init_custom_metric = False
        self._last_logged_step = -1
        self._logs = {
            'prompt': deque(maxlen=args.generation_batch_size),
            'completion': deque(maxlen=args.generation_batch_size),
            'rewards': defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            'advantages': deque(maxlen=args.generation_batch_size),
        }

        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}

    def _apply_chat_template_to_messages_list(self, messages_list: DataType):
        prompts_text = []
        for messages in messages_list:
            remove_response(messages)
            template_inputs = TemplateInputs.from_dict({'messages': messages})
            res = self.template.encode(template_inputs)
            prompts_text.append(self.template.safe_decode(res['input_ids']))
        return prompts_text

    def _set_inputs_system(self, batch: DataType) -> DataType:
        """
        Ensures the system message is consistently set for all conversations in the batch.

        The template handles the user-defined system message. However, in server mode,
        tokenization occurs on the rollout side. To prevent a mismatch where the system
        message is set only during training but missing during rollout, this method
        injects the default system message into each conversation if not already present.

        Args:
            batch: A list of data items, each containing a 'messages' list.

        Returns:
            The updated batch with the default system message inserted at the beginning
            of each conversation that lacks one.
        """

        if self.vllm_mode != 'server':
            return batch

        # Return early if no default system message is defined
        if not self.template.template_meta.default_system:
            return batch

        # Return early if all conversations already start with a system message
        if all(data['messages'][0]['role'] == 'system' for data in batch):
            return batch

        # Insert the default system message at the beginning of each conversation
        # that doesn't already have one
        for data in batch:
            messages = data['messages']
            if messages[0]['role'] != 'system':
                messages.insert(0, {'role': 'system', 'content': self.template.template_meta.default_system})

        return batch

    def _compute_sequence_level_ratios(self, is_ratio: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
        """
        Helper function to compute sequence-level importance sampling ratios.

        Args:
            is_ratio: Token-level IS ratios, shape [batch_size, seq_len]
            completion_mask: Boolean mask for completion tokens, shape [batch_size, seq_len]

        Returns:
            Sequence-level ratios as geometric mean of token-level ratios, shape [batch_size]
        """
        log_ratio = torch.log(is_ratio.clamp(min=1e-10))
        # Compute per-sequence mean of log ratios
        seq_log_ratios = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        seq_ratios = torch.exp(seq_log_ratios)
        return seq_ratios

    def _apply_rollout_importance_sampling(self, rollout_log_ratio: torch.Tensor,
                                           completion_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply rollout importance sampling correction using one of four modes.

        Args:
            rollout_log_ratio: log(π_θ / π_rollout) per token, shape [batch_size, seq_len]
            completion_mask: Boolean mask for completion tokens, shape [batch_size, seq_len]

        Returns:
            IS weights to multiply with loss, same shape as rollout_log_ratio
        """
        mode = self.rollout_importance_sampling_mode
        threshold = self.rollout_importance_sampling_threshold

        # Clamp log_ratio to prevent numerical overflow from padding values
        SAFETY_BOUND = 20.0
        rollout_log_ratio_safe = torch.clamp(rollout_log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)

        # Compute importance sampling ratios: exp(log_ratio)
        is_ratio = torch.exp(rollout_log_ratio_safe)

        if mode == 'token_truncate':
            # Token-level truncated IS: clip ratios from above at threshold
            is_weights = torch.clamp(is_ratio, max=threshold)

        elif mode == 'token_mask':
            # Token-level masked IS: mask out tokens with ratio > threshold
            is_weights = torch.where(is_ratio <= threshold, is_ratio, torch.zeros_like(is_ratio))

        elif mode == 'sequence_truncate':
            # Sequence-level truncated IS: compute sequence-level ratio and clip
            seq_ratios = self._compute_sequence_level_ratios(is_ratio, completion_mask)
            clipped_seq_ratios = torch.clamp(seq_ratios, max=threshold)

            # Expand back to token-level [batch_size] -> [batch_size, seq_len]
            is_weights = clipped_seq_ratios.unsqueeze(-1).expand_as(is_ratio)

        elif mode == 'sequence_mask':
            # Sequence-level masked IS: mask entire sequences with ratio > threshold
            seq_ratios = self._compute_sequence_level_ratios(is_ratio, completion_mask)
            seq_mask = (seq_ratios <= threshold).float()

            # Expand mask to token-level and apply to original token-level ratios
            seq_mask_expanded = seq_mask.unsqueeze(-1).expand_as(is_ratio)
            is_weights = is_ratio * seq_mask_expanded
        else:
            return is_ratio

        return is_weights

    def _compute_off_policy_sequence_mask(
        self,
        per_token_logps: torch.Tensor,
        old_policy_per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute off-policy sequence mask to filter out sequences that deviate too much
        from the old/rollout policy AND have negative advantage.

        This implements the Off-Policy Sequence Masking technique from DeepSeek-V3.2
        (https://arxiv.org/abs/2512.02556). The mask filters sequences where:
        1. mean(old_policy_logps - policy_logps) > off_policy_sequence_mask_delta
        2. AND advantage < 0

        Args:
            per_token_logps: Log probs from current policy, shape [batch_size, seq_len]
            old_policy_per_token_logps: Log probs from old/rollout policy, shape [batch_size, seq_len].
                Uses rollout_per_token_logps if available, otherwise old_per_token_logps.
            completion_mask: Boolean mask for completion tokens, shape [batch_size, seq_len]
            advantages: Advantage values per sample, shape [batch_size]

        Returns:
            Sequence mask, shape [batch_size], True = keep sequence, False = mask out
        """
        # Compute per-token log ratio: log(π_old / π_current)
        # Following DeepSeek-V3.2: positive delta means old policy assigns higher prob
        log_ratio = old_policy_per_token_logps - per_token_logps

        # Compute sequence-level mean of log ratio
        seq_mean_log_ratio = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)

        # Mask condition: delta > threshold AND advantage < 0
        # Keep sequences that do NOT meet this condition
        exceeds_threshold = seq_mean_log_ratio > self.off_policy_sequence_mask_delta
        negative_advantage = advantages < 0
        should_mask = exceeds_threshold & negative_advantage

        # Return mask: True = keep, False = mask out
        return ~should_mask

    def _compute_rollout_offpolicy_metrics(
        self,
        per_token_logps: torch.Tensor,
        rollout_per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute off-policy diagnostic metrics (always computed for monitoring).

        These metrics help diagnose the off-policy gap between rollout and training policies.

        Args:
            per_token_logps: Log probs from training policy model, shape [batch_size, seq_len]
            rollout_per_token_logps: Log probs from rollout policy, shape [batch_size, seq_len]
            completion_mask: Boolean mask for completion tokens, shape [batch_size, seq_len]

        Returns:
            Dictionary with off-policy diagnostic metrics
        """
        SAFETY_BOUND = 20.0
        metrics = {}
        dp_group = mpu.get_data_parallel_group(with_context_parallel=True)

        # Helper function for masked mean
        def masked_mean(x, mask, axis=None):
            if axis is None:
                return (x * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                return (x * mask).sum(axis) / mask.sum(axis).clamp(min=1.0)

        # 1. Training policy perplexity
        # Compute per-sequence mean log prob
        mean_log_prob_training = masked_mean(per_token_logps, completion_mask, axis=-1)  # [batch_size]
        training_ppl = torch.exp(-mean_log_prob_training).mean()
        gathered_training_ppl = gather(training_ppl.unsqueeze(0), group=dp_group)
        metrics['training_ppl'] = gathered_training_ppl.nanmean()
        metrics['training_log_ppl'] = gather((-mean_log_prob_training).mean().unsqueeze(0), group=dp_group).nanmean()

        # 2. Compute rollout off-policy metrics
        log_ratio = per_token_logps - rollout_per_token_logps
        log_ratio = log_ratio * completion_mask

        # 2a. kl: Direct estimator for KL(π_training || π_rollout)
        kl = masked_mean(log_ratio, completion_mask)
        metrics['kl'] = gather(kl.unsqueeze(0), group=dp_group).nanmean()

        # 2b. k3_kl: K3 estimator for KL
        k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
        k3_kl = masked_mean(k3_kl_matrix, completion_mask)
        metrics['k3_kl'] = gather(k3_kl.unsqueeze(0), group=dp_group).nanmean()

        # 2c. Rollout policy perplexity
        mean_log_prob_rollout = masked_mean(rollout_per_token_logps, completion_mask, axis=-1)  # [batch_size]
        rollout_ppl = torch.exp(-mean_log_prob_rollout).mean()
        metrics['rollout_ppl'] = gather(rollout_ppl.unsqueeze(0), group=dp_group).nanmean()
        metrics['rollout_log_ppl'] = gather((-mean_log_prob_rollout).mean().unsqueeze(0), group=dp_group).nanmean()

        # 2d. Log PPL difference
        log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        metrics['log_ppl_diff'] = gather(log_ppl_diff.mean().unsqueeze(0), group=dp_group).nanmean()
        metrics['log_ppl_abs_diff'] = gather(log_ppl_diff.abs().mean().unsqueeze(0), group=dp_group).nanmean()
        metrics['log_ppl_diff_max'] = gather(log_ppl_diff.max().unsqueeze(0), group=dp_group).max()
        metrics['log_ppl_diff_min'] = gather(log_ppl_diff.min().unsqueeze(0), group=dp_group).min()

        # 2e. PPL ratio
        ppl_ratio = torch.exp(log_ppl_diff).mean()
        metrics['ppl_ratio'] = gather(ppl_ratio.unsqueeze(0), group=dp_group).nanmean()

        # 2f. Chi-squared divergence
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_token = torch.exp(log_ratio_safe)
        rho_squared_token = rho_token.square()
        chi2_token = masked_mean(rho_squared_token, completion_mask) - 1.0
        metrics['chi2_token'] = gather(chi2_token.unsqueeze(0), group=dp_group).nanmean()

        # Sequence-level chi2
        log_ratio_mean = masked_mean(log_ratio, completion_mask, axis=-1)  # [batch_size]
        log_ratio_mean_safe = torch.clamp(log_ratio_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_geo = torch.exp(log_ratio_mean_safe)
        chi2_seq = (rho_geo.square().mean() - 1.0)
        metrics['chi2_seq'] = gather(chi2_seq.unsqueeze(0), group=dp_group).nanmean()

        return metrics

    def _compute_is_correction_metrics(
        self,
        rollout_log_ratio: torch.Tensor,
        is_weights: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute importance sampling correction metrics (ess, clipped_frac, is_weight_mean).
        Only called when rollout_importance_sampling_mode is enabled.

        Args:
            rollout_log_ratio: Log ratio log(π_policy / π_rollout), shape [batch_size, seq_len]
            is_weights: Importance sampling weights after correction, shape [batch_size, seq_len]
            completion_mask: Boolean mask for completion tokens, shape [batch_size, seq_len]

        Returns:
            Dictionary with IS-specific metrics
        """
        metrics = {}
        SAFETY_BOUND = 20.0
        threshold = self.rollout_importance_sampling_threshold
        threshold_lower = 1.0 / threshold
        dp_group = mpu.get_data_parallel_group(with_context_parallel=True)

        # Helper function for masked mean
        def masked_mean(x, mask):
            return (x * mask).sum() / mask.sum().clamp(min=1.0)

        # Compute IS ratio with safety bounds
        log_ratio_safe = torch.clamp(rollout_log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        is_ratio = torch.exp(log_ratio_safe)

        # 1. IS weight statistics
        mean_is_weight = masked_mean(is_weights, completion_mask)
        metrics['is_weight_mean'] = gather(mean_is_weight.unsqueeze(0), group=dp_group).nanmean()

        # 2. Compute Effective Sample Size (ESS) for IS weights
        weights_for_ess = is_weights.clamp(min=threshold_lower, max=threshold)
        mean_for_ess = masked_mean(weights_for_ess, completion_mask)
        is_weights_normalized = weights_for_ess / (mean_for_ess + 1e-8)
        ess = 1.0 / masked_mean(is_weights_normalized.square(), completion_mask).clamp(min=1e-10)
        metrics['ess'] = gather(ess.unsqueeze(0), group=dp_group).nanmean()

        # 3. Fraction of clipped/masked samples
        if self.rollout_importance_sampling_mode in ['token_truncate', 'token_mask']:
            # Token-level
            if self.rollout_importance_sampling_mode == 'token_truncate':
                clipped_frac = masked_mean((is_ratio > threshold).float(), completion_mask)
            else:  # token_mask
                clipped_frac = masked_mean((is_weights == 0).float(), completion_mask)
            metrics['clipped_frac'] = gather(clipped_frac.unsqueeze(0), group=dp_group).nanmean()
        else:
            # Sequence-level (both truncate and mask)
            seq_ratios = self._compute_sequence_level_ratios(is_ratio, completion_mask)
            clipped_frac = (seq_ratios > threshold).float().mean()
            metrics['clipped_frac'] = gather(clipped_frac.unsqueeze(0), group=dp_group).nanmean()

        return metrics

    def _prepare_model_inputs(self, inputs: 'DataType') -> Dict[str, Any]:
        """Filters inputs to create model_inputs, removing GRPO-specific keys."""
        return {
            k: v
            for k, v in inputs.items() if k not in [
                'logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps',
                'truncated_mask', 'seq_lengths', 'num_items_in_batch', 'rollout_per_token_logps'
            ]
        }

    def _collect_config_info(self) -> Dict[str, str]:
        config = {
            'dynamic_sample': str(self.args.dynamic_sample),
            'importance_sampling_level': str(self.args.importance_sampling_level),
            'advantage_estimator': str(self.args.advantage_estimator),
            'offpolicy_sequence_mask': 'enable' if self.args.off_policy_sequence_mask_delta is not None else 'disable',
            'rollout_importance_sampling':
            'enable' if self.args.rollout_importance_sampling_mode is not None else 'disable',
            'loss_type': str(self.args.loss_type)
        }
        return config
