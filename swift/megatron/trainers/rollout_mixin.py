# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Megatron Rollout Mixin - Provides vLLM integration for on-policy generation.

This mixin extracts common vLLM rollout functionality from MegatronGRPOTrainer
to be reused by GKD and other trainers that need online generation.
"""
import base64
import inspect
import json
import os
import re
import time
import torch
import uuid
from accelerate.utils import broadcast_object_list
from collections import OrderedDict, deque
from contextlib import contextmanager, nullcontext
from copy import copy
from dacite import from_dict
from dataclasses import asdict
from megatron.core import mpu
from typing import Any, Dict, List, Tuple, Union

from swift.infer_engine.protocol import RequestConfig, RolloutInferRequest, RolloutOutput
from swift.rl_core.data import OnPolicySample
from swift.rlhf_trainers.base_rollout_mixin import BaseRolloutTrainerMixin
from swift.rlhf_trainers.utils import (VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH, FlattenedTensorBucket,
                                       TensorLoRARequest, add_base_layer_suffix_by_param_names, aggressive_empty_cache,
                                       check_vllm_version_ge, expand_vllm_param_name_aliases, finish_vllm_weight_reload,
                                       patch_vllm_load_adapter, patch_vllm_moe_model_weight_loader, profiling_context,
                                       profiling_decorator, set_expandable_segments, vllm_supports_lora_load_inplace)
from swift.rollout import invoke_async_hook, run_multi_turn
from swift.utils import (JsonlWriter, get_current_device, get_logger, is_last_rank, is_vllm_available, remove_response,
                         synchronize, to_device)
from .utils import (gather_object, load_megatron_model_to_gpu, load_megatron_optimizer, offload_megatron_model_to_cpu,
                    offload_megatron_optimizer)

DataType = List[Dict[str, Union[torch.Tensor, Any]]]
logger = get_logger()


def create_rollout_group(trainer) -> torch.distributed.ProcessGroup:
    """
    Get or create the rollout process group (TP×PP×CP).

    This is a shared function used by both MegatronRolloutMixin and MegatronGRPOTrainer.

    The rollout group is used for:
    1. Data slicing: distributing rollout data across ranks with same data samples
    2. Gather operations: collecting results from ranks with same data samples

    Note: Groups are created per data parallel index, containing TP×PP×CP ranks each.
    This follows Megatron's data_iterator logic where same data_parallel_rank processes
    identical data samples.

    Key insight: ranks with the SAME data parallel index process the SAME data samples
    and must coordinate for rollout data distribution.
    Megatron rank order: TP → CP → EP → DP → PP

    Args:
        trainer: Trainer instance with _rollout_group and _rollout_groups_created attributes

    Returns:
        The rollout process group for this rank
    """
    if trainer._rollout_group is not None:
        return trainer._rollout_group

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        # No CP, use the standard MODEL_PARALLEL_GROUP
        trainer._rollout_group = mpu.get_model_parallel_group()
        return trainer._rollout_group

    # Use RankGenerator to create rollout groups following Megatron-LM logic
    global_rank = torch.distributed.get_rank()

    # Get parallel dimensions
    tp_size = mpu.get_tensor_model_parallel_world_size()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    dp_size = mpu.get_data_parallel_world_size()

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
    if not trainer._rollout_groups_created:
        # Use 'tp-cp-ep-pp' to get groups with same DP index (DP is excluded from variation)
        dp_groups = decoder_rank_generator.get_ranks('tp-cp-ep-pp')
        for dp_group_ranks in dp_groups:
            # Sort for consistency
            dp_group_ranks = sorted(dp_group_ranks)
            group = torch.distributed.new_group(ranks=dp_group_ranks, group_desc='ROLLOUT_GROUP')

            if global_rank in dp_group_ranks:
                trainer._rollout_group = group
        trainer._rollout_groups_created = True

    return trainer._rollout_group


class MegatronRolloutMixin(BaseRolloutTrainerMixin):

    # Per-sample container class; subclasses override (GRPOSample / GKDSample).
    sample_cls = OnPolicySample

    def _init_rollout_params(self):
        """Initialize rollout generation parameters."""
        args = self.args
        # distributed params
        self.world_size = torch.distributed.get_world_size()
        self.process_index = torch.distributed.get_rank()
        self.is_main_process = is_last_rank()
        self.device = get_current_device()

        # sampling params
        self.temperature = getattr(args, 'temperature', 1.0)
        self.max_completion_length = args.max_completion_length
        structured_outputs_regex = getattr(args, 'structured_outputs_regex', None)

        self.request_config = RequestConfig(
            n=1,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=getattr(args, 'top_p', 1.0),
            top_k=getattr(args, 'top_k', -1),
            repetition_penalty=getattr(args, 'repetition_penalty', 1.0),
            stop=getattr(args, 'stop_words', None),
            return_details=True,
            logprobs=True,
            structured_outputs_regex=structured_outputs_regex)

        self._last_loaded_step = -1
        self._step = 0
        self._rollout_group = None  # Lazily initialized rollout group (TP×PP×CP)
        self._rollout_groups_created = False  # Flag for group creation (all ranks must create together)
        self._bridge = None

    def _get_rollout_group(self):
        """Get or create the rollout process group (TP×PP×CP)."""
        return create_rollout_group(self)

    def _get_local_rollout_batch(self, samples: List[OnPolicySample]) -> List[OnPolicySample]:
        """Split batch within rollout group for distributed vLLM generation.

        The batch is evenly split across the rollout group (TP×PP×CP ranks with
        the same DP index). This is the base implementation that simply splits
        the batch without repetition.

        Subclasses (e.g., GRPO) may override this to implement custom logic like
        repeating each prompt num_generations times.

        Note: In Megatron, batch size should always be divisible by rollout group size.
        This is ensured by global_batch_size = micro_batch_size * num_microbatches * dp_size,
        where rollout_group_size = tp_size * pp_size * cp_size, and world_size = dp_size * rollout_group_size.

        Args:
            samples: Full batch of data samples

        Returns:
            Local slice of samples for this rank to process
        """
        rollout_group = self._get_rollout_group()
        rollout_rank = torch.distributed.get_rank(group=rollout_group)
        rollout_group_size = torch.distributed.get_world_size(group=rollout_group)

        total_batch_size = len(samples)
        assert total_batch_size % rollout_group_size == 0, \
            f'Batch size ({total_batch_size}) must be divisible by rollout group size ({rollout_group_size})'

        per_device_batch_size = total_batch_size // rollout_group_size
        start_idx = rollout_rank * per_device_batch_size
        end_idx = start_idx + per_device_batch_size

        return samples[start_idx:end_idx]

    def _gather_rollout_results(self, samples: List[OnPolicySample]) -> List[OnPolicySample]:
        """Gather rollout results from all ranks in the rollout group.

        Args:
            samples: Local rollout results from this rank

        Returns:
            Gathered results from all ranks in the rollout group
        """
        rollout_group = self._get_rollout_group()
        return gather_object(samples, group=rollout_group)

    def _init_rollout_engine(self):
        """Initialize vLLM engine for rollout generation."""
        args = self.args
        self._init_rollout_params()

        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
        self.use_vllm = args.use_vllm
        self.vllm_use_async_engine = False
        self.enable_offload = False
        self.vllm_version_ge_0_10_2 = check_vllm_version_ge('0.10.2')
        self.rollout_enable_lora = False
        self.enable_server_multi_turn = False
        self.base_sync_done = False
        self._cached_vllm_param_names = None

        if not args.use_vllm:
            return

        if args.rlhf_type == 'gkd' and args.lmbda == 0:
            return

        if not is_vllm_available():
            raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                              'Please install vLLM with `pip install vllm -U` to use it.')

        if self.vllm_mode == 'server':
            # Server mode uses external vLLM server
            if self.is_main_process:
                self.vllm_client.get_engine_type()
                self.vllm_client.reset_mm_cache()
                enable_lora = [self.vllm_client.enable_lora]
                enable_multi_turn = [self.vllm_client.enable_multi_turn]
            else:
                enable_lora = [False]
                enable_multi_turn = [False]
            self.rollout_enable_lora = broadcast_object_list(enable_lora, from_process=self.world_size - 1)[0]
            self.enable_server_multi_turn = broadcast_object_list(
                enable_multi_turn, from_process=self.world_size - 1)[0]
        elif self.vllm_mode == 'colocate':
            if self.world_size % self.vllm_tensor_parallel_size != 0:
                raise ValueError(f'vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size '
                                 f'({self.world_size}) evenly.')

            self.enable_offload = args.offload_model or args.offload_optimizer
            context = self.offload_context if self.enable_offload else nullcontext

            with context():
                set_expandable_segments(False)
                self.engine = self._prepare_vllm_engine()
                self.engine.engine.reset_mm_cache()
                if args.sleep_level > 0:
                    self.engine.engine.sleep(args.sleep_level)
                set_expandable_segments(True)
        else:
            raise ValueError(f'Invalid vllm_mode: {self.vllm_mode}')

    def _prepare_vllm_engine(self):
        """Create and configure vLLM engine for colocate mode."""
        from vllm.distributed import parallel_state as vllm_ps

        from swift.infer_engine import GRPOVllmEngine

        args = self.args
        per_device_batch_size = getattr(args, 'per_device_generation_batch_size', args.micro_batch_size)
        max_num_seqs = args.vllm_max_num_seqs or per_device_batch_size * self.vllm_tensor_parallel_size

        vllm_template = copy(self.template)
        vllm_template.padding_free = False
        vllm_template.sequence_parallel_size = 1

        logprobs_mode = 'processed_logprobs' if self.vllm_version_ge_0_10_2 else None

        vllm_engine_kwargs = args.vllm_engine_kwargs or {}
        load_format = vllm_engine_kwargs.pop('load_format', 'auto')

        if self.args.router_replay_mode == 'R3':
            assert check_vllm_version_ge('0.14.0'), \
                'The enable_return_routed_experts attribute is not supported. Please upgrade vllm to 0.14.0 or higher'
            vllm_engine_kwargs['enable_return_routed_experts'] = True
            # https://github.com/vllm-project/vllm/pull/39917
            import vllm
            from packaging import version
            vllm_version = vllm.__version__
            if vllm_version is not None and version.parse('0.21.0rc1') <= version.parse(vllm_version) <= version.parse(
                    '0.21.0'):
                vllm_engine_kwargs.setdefault('async_scheduling', False)

        enable_lora = False
        max_loras = 1
        max_lora_rank = args.lora_rank
        if args.tuner_type == 'lora' and args.vllm_enable_lora:
            enable_lora = True
            self.rollout_enable_lora = True
            patch_vllm_load_adapter()
            logger.info(f'Enabled vLLM LoRA adapter sync with max_lora_rank={args.lora_rank}')

        engine = GRPOVllmEngine(
            args.model_info.model_dir,
            torch_dtype=args.torch_dtype,
            model_type=args.model_type,
            use_async_engine=False,
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            enable_prefix_caching=args.vllm_enable_prefix_caching,
            max_num_seqs=max_num_seqs,
            enforce_eager=args.vllm_enforce_eager,
            limit_mm_per_prompt=args.vllm_limit_mm_per_prompt,
            enable_sleep_mode=args.sleep_level > 0,
            max_model_len=args.vllm_max_model_len,
            seed=self.process_index // self.vllm_tensor_parallel_size,
            disable_cascade_attn=args.vllm_disable_cascade_attn,
            load_format=load_format,
            mm_processor_cache_gb=args.vllm_mm_processor_cache_gb,
            template=vllm_template,
            distributed_executor_backend='external_launcher',
            enable_lora=enable_lora,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            engine_kwargs=vllm_engine_kwargs,
            logprobs_mode=logprobs_mode)

        if self.vllm_tensor_parallel_size > 1:
            self.vllm_tp_group = vllm_ps.get_tp_group().device_group

        return engine

    @profiling_decorator
    def _move_model_to_vllm(self):
        """Synchronize model weights to vLLM engine.

        - Full sync: when tuner_type != 'lora' (e.g. full, lora_llm), or first sync
          (base_sync_done=False), or sleep_level==2, or rollout_enable_lora is disabled.
        - Adapter-only sync: when tuner_type == 'lora' with rollout_enable_lora=True and
          base weights have already been synced.
        """
        args = self.args
        tuner_type = args.tuner_type

        if (tuner_type != 'lora' or (not self.base_sync_done or args.sleep_level == 2) or not self.rollout_enable_lora):
            self._move_full_model_to_vllm()
        else:
            self._move_adapter_to_vllm()

        self._reset_vllm_cache()

    def _reset_vllm_cache(self):
        # Reset prefix cache and encoder cache
        vllm_ge_16 = check_vllm_version_ge('0.16')
        if self.vllm_mode == 'server' and self.is_main_process:
            self.vllm_client.reset_prefix_cache()
            if vllm_ge_16:
                self.vllm_client.reset_encoder_cache()
        elif self.vllm_mode == 'colocate':
            self.engine.engine.reset_prefix_cache()
            if vllm_ge_16:
                self.engine.engine.reset_encoder_cache()

    def _move_full_model_to_vllm(self):
        """Transfer full model weights to vLLM engine.

        For LoRA training (tuner_type == 'lora'):
        - When rollout_enable_lora=False: merge LoRA into base, export merged weights, then unmerge.
        - When rollout_enable_lora=True: export base weights only (no merge needed),
          then follow up with adapter-only sync via _move_adapter_to_vllm.
        For lora_llm: always merge LLM LoRA into exported dense weights (vLLM has no separate
        adapter pass for this tuner_type; see _move_model_to_vllm).
        """
        is_lora_training = self.args.tuner_type in ('lora', 'lora_llm')
        is_pure_lora = self.args.tuner_type == 'lora'
        should_merge = not self.rollout_enable_lora
        if self.args.tuner_type == 'lora_llm' and self.rollout_enable_lora:
            logger.warning('lora_llm is not supported with vllm_enable_lora=True. plz set vllm_enable_lora to False')

        try:
            if should_merge:
                self.merge_lora_adapters()

            self._export_and_load_weights()

        finally:
            if should_merge:
                self.unmerge_lora_adapters()

        if is_lora_training:
            self.base_sync_done = True
            if self.rollout_enable_lora and is_pure_lora:
                self._move_adapter_to_vllm()

    def _move_adapter_to_vllm(self):
        """Transfer only LoRA adapter weights to vLLM engine.

        Uses bridge.export_weights(peft_format=True) to export LoRA delta weights.
        Yielded names follow PEFT convention: 'base_model.model.<hf_path>.lora_A.weight'.
        """
        target_device = 'cpu' if self.args.offload_bridge else None

        with profiling_context(self, 'export_adapter_weights'):
            adapter_iterator = self.bridge.export_weights(
                self.unwrapped_models, target_device=target_device, peft_format=True)
            lora_params = OrderedDict()
            for name, tensor in adapter_iterator:
                lora_params[name] = tensor.detach()

        peft_config = self.unwrapped_models[0].peft_config.get('default', None)

        if self.vllm_mode == 'colocate':
            req_kw = dict(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=asdict(peft_config),
                lora_tensors=lora_params,
            )
            if vllm_supports_lora_load_inplace():
                req_kw['load_inplace'] = True
            lora_request = TensorLoRARequest(**req_kw)
            self.engine.engine.add_lora(lora_request)
        elif self.vllm_mode == 'server' and self.is_main_process:
            bucket = FlattenedTensorBucket(named_tensors=list(lora_params.items()))
            metadatas = bucket.get_metadata()
            flattened_tensor = bucket.get_flattened_tensor()
            self.vllm_client.update_adapter_flattened_param(peft_config, metadatas, flattened_tensor)
            del bucket, metadatas, flattened_tensor

        del lora_params

    def _export_and_load_weights(self):
        """Export weights from Megatron and load to vLLM."""
        target_device = 'cpu' if self.args.offload_bridge else None

        with profiling_context(self, 'export_weights'):
            weight_iterator = self.bridge.export_weights(self.unwrapped_models, target_device=target_device)

        if self.rollout_enable_lora:
            vllm_param_names = self._get_vllm_param_names_for_mapping()
            if vllm_param_names:
                weight_iterator = add_base_layer_suffix_by_param_names(weight_iterator, vllm_param_names)

        if self.vllm_mode == 'colocate':
            llm_model = self.engine.inner_model
            patch_vllm_moe_model_weight_loader(llm_model)
            llm_model.load_weights(weight_iterator)
            _model_config = self.engine.engine.model_config
            finish_vllm_weight_reload(llm_model, model_config=_model_config, target_device=self.device)
        elif self.vllm_mode == 'server':
            self._load_weights_to_server_in_buckets(weight_iterator)
            if self.is_main_process:
                self.vllm_client.process_weights_after_loading()

    def _get_vllm_param_names_for_mapping(self):
        """Get vLLM runtime parameter names for base_layer mapping.

        Returns an alias-expanded set so bridge/HF names can match vLLM packed names.
        """
        if self.vllm_mode == 'colocate':
            llm_model = self.engine.inner_model
            raw_names = set(dict[Any, Any](llm_model.named_parameters()).keys())
            return expand_vllm_param_name_aliases(raw_names)

        if self.vllm_mode != 'server' or not self.is_main_process:
            return None

        if self._cached_vllm_param_names is None:
            keys = self.vllm_client.get_model_state_keys()
            self._cached_vllm_param_names = expand_vllm_param_name_aliases(set(keys))
        return self._cached_vllm_param_names

    def _load_weights_to_server_in_buckets(self, weight_iterator):
        """Load weights to vLLM server in buckets."""
        bucket_size_mb = int(os.environ.get('SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE', 512))
        bucket_size_bytes = bucket_size_mb * 1024 * 1024

        current_bucket = []
        current_size = 0

        for name, param in weight_iterator:
            param_size = param.numel() * param.element_size()
            current_bucket.append((name, param))
            current_size += param_size

            if current_size > bucket_size_bytes and current_bucket:
                self._sync_bucket_to_server(current_bucket)
                current_bucket = []
                current_size = 0

        if current_bucket:
            self._sync_bucket_to_server(current_bucket)

    def _sync_bucket_to_server(self, bucket_params: List[Tuple[str, torch.Tensor]]):
        """Synchronize a bucket of parameters to vLLM server."""
        if not bucket_params or not self.is_main_process:
            return

        # Ensure all async GPU ops (e.g. TP all-gather on NCCL stream from bridge.export_weights)
        # are complete before .copy_() reads param data on the default stream.
        synchronize()

        bucket = FlattenedTensorBucket(named_tensors=bucket_params)
        metadatas = bucket.get_metadata()
        flattened_tensor = bucket.get_flattened_tensor()

        self.vllm_client.update_flattened_params(metadatas, flattened_tensor)

        del bucket, metadatas, flattened_tensor

    @profiling_decorator
    def _generate_completions(self, samples: List[OnPolicySample]) -> List[OnPolicySample]:
        """Generate completions for a batch using vLLM engine.

        Args:
            samples: List of OnPolicySample carrying messages + rollout fields

        Returns:
            Batch with rollout completion results merged in
        """
        samples = self._preprocess_inputs(samples)

        # Wake up engine if sleeping (colocate mode)
        if self.vllm_mode == 'colocate' and self.engine.inner_model_executor.is_sleeping:
            wake_up_params = inspect.signature(self.engine.engine.wake_up).parameters
            kwargs = {'tags': ['weights']} if 'tags' in wake_up_params else {}
            aggressive_empty_cache()
            self.engine.engine.wake_up(**kwargs)

        # Load model weights if needed
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

            multi_turn_scheduler = getattr(self, 'multi_turn_scheduler', None)
            colocate_multi_turn = (
                multi_turn_scheduler is not None and not getattr(self, 'enable_server_multi_turn', False))

            if colocate_multi_turn:
                requests = self._inputs_to_requests(samples)
                invoke_async_hook(multi_turn_scheduler.on_trajectory_start(requests))
                request_config = self._get_request_config()
                outputs: List[RolloutOutput] = self._rollout_requests(requests, request_config)
                outputs = run_multi_turn(
                    requests=requests,
                    first_turn_outputs=outputs,
                    scheduler=multi_turn_scheduler,
                    rollout_fn=lambda reqs, cfg: self._rollout_requests(reqs, cfg),
                    request_config=request_config,
                    max_turns=self.args.max_turns,
                    gather_fn=lambda x: gather_object(x, group=self._get_rollout_group()),
                )
                self._log_rollout(samples, outputs)
            else:
                # Single-turn rollout (or server multi-turn handled by the engine).
                outputs: List[RolloutOutput] = self._rollout(samples)
                self._log_rollout(samples, outputs)

            # Sleep to release memory
            if self.vllm_mode == 'colocate' and self.args.sleep_level > 0:
                self.engine.engine.reset_prefix_cache()
                self.engine.engine.sleep(level=self.args.sleep_level)
                aggressive_empty_cache()
                set_expandable_segments(True)

            samples = self._postprocess_rollout_outputs(samples, outputs)

        return samples

    def _rollout_requests(self, requests: List[RolloutInferRequest],
                          request_config: RequestConfig) -> List[RolloutOutput]:
        """Continuation rollout taking already-prepared ``RolloutInferRequest`` objects.

        Used by the multi-turn driver (:func:`swift.rollout.run_multi_turn`) on
        every turn after the first. ``_set_inputs_system`` is skipped because the
        system prompt is already encoded into ``requests[i].messages``.
        """
        if self.vllm_mode == 'server':
            return self._server_rollout(requests, request_config)
        elif self.vllm_mode == 'colocate':
            return self._colocate_rollout(requests, request_config)
        raise ValueError(f'Invalid vllm_mode: {self.vllm_mode}')

    def _rollout(self, samples: List[OnPolicySample]) -> List[RolloutOutput]:
        """Execute rollout using vLLM engine (system already injected by _preprocess_inputs)."""
        request_config = self._get_request_config()

        if self.vllm_mode == 'server':
            return self._server_rollout(samples, request_config)
        elif self.vllm_mode == 'colocate':
            return self._colocate_rollout(samples, request_config)

    def _get_request_config(self) -> RequestConfig:
        """Get request config with proper seed for distributed TP groups."""
        request_config = copy(self.request_config)

        if self.vllm_mode == 'colocate' and self.vllm_tensor_parallel_size > 1:
            batch_size = getattr(self.args, 'per_device_generation_batch_size', self.args.micro_batch_size)
            batch_size *= self.vllm_tensor_parallel_size
            request_config.seed = batch_size * (self.process_index // self.vllm_tensor_parallel_size)

        return request_config

    def _server_rollout(self, samples: Union[List[OnPolicySample], List[RolloutInferRequest]],
                        request_config: RequestConfig) -> List[RolloutOutput]:
        """Perform rollout using vLLM server mode."""
        infer_requests = self._inputs_to_requests(samples)

        all_requests = gather_object(infer_requests)
        all_requests_lengths = gather_object([len(infer_requests)])

        if not any(requests for requests in all_requests):
            return []

        if self.is_main_process:
            all_outputs: List[RolloutOutput] = self.vllm_client.infer(
                infer_requests=all_requests, request_config=request_config)
            if len(all_outputs) != len(all_requests):
                # Per-turn-split multi-turn (`dynamic_num_samples`) is HF-only
                raise NotImplementedError(
                    'Per-turn-split multi-turn (dynamic_num_samples) is not supported on Megatron — '
                    f'server returned {len(all_outputs)} outputs for {len(all_requests)} requests. '
                    'Return one RolloutOutput per request from MultiTurnScheduler.run (combine turns '
                    'inside response_token_ids: List[List[int]]), or use the HF trainer.')
        else:
            all_outputs = [None] * len(all_requests)

        all_outputs = broadcast_object_list(all_outputs, from_process=self.world_size - 1)
        start_idx = sum(all_requests_lengths[:self.process_index])
        end_idx = start_idx + all_requests_lengths[self.process_index]
        outputs = all_outputs[start_idx:end_idx]

        return outputs

    def _colocate_rollout(self, samples: Union[List[OnPolicySample], List[RolloutInferRequest]],
                          request_config: RequestConfig) -> List[RolloutOutput]:
        """Perform co-located rollout with vLLM engine."""
        # Normalize samples (first turn) / RolloutInferRequest (continuation turns)
        # into engine-ready requests.
        samples = self._inputs_to_requests(samples)
        start_idx = 0
        end_idx = len(samples)

        # Handle vLLM tensor parallelism
        if self.vllm_tensor_parallel_size > 1:
            local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
            local_input_length = len(samples)
            all_input_lengths = [None] * self.vllm_tensor_parallel_size
            torch.distributed.all_gather_object(all_input_lengths, local_input_length, group=self.vllm_tp_group)

            start_idx = sum(all_input_lengths[:local_rank_in_group])
            end_idx = start_idx + all_input_lengths[local_rank_in_group]

            gathered = [None for _ in range(self.vllm_tensor_parallel_size)]
            torch.distributed.all_gather_object(gathered, samples, group=self.vllm_tp_group)
            samples = [p for sublist in gathered for p in sublist]

        outputs: List[RolloutOutput] = self.engine.infer(
            infer_requests=samples, request_config=request_config, use_tqdm=False)

        if self.vllm_tensor_parallel_size > 1:
            # R3 router replay: vLLM's routing capturer host cache only exists on
            # TP rank 0, so non-primary TP ranks have routed_experts=None in outputs.
            # Broadcast routed_experts from TP primary to all TP ranks.
            if getattr(self.args, 'router_replay_mode', None) == 'R3':
                routed_experts_list = [output.response.choices[0].routed_experts for output in outputs]
                tp_primary_global_rank = torch.distributed.get_global_rank(self.vllm_tp_group, 0)
                torch.distributed.broadcast_object_list(
                    routed_experts_list, src=tp_primary_global_rank, group=self.vllm_tp_group)
                if local_rank_in_group != 0:
                    for output, experts in zip(outputs, routed_experts_list):
                        output.response.choices[0].routed_experts = experts
            outputs = outputs[start_idx:end_idx]

        return outputs

    def _preprocess_inputs(self, samples: List[OnPolicySample]) -> List[OnPolicySample]:
        """Preprocess samples before rollout inference.

        Unified pre-processing (mirrors HF RolloutTrainerMixin._preprocess_inputs):
        1. Insert default system message if absent
        2. Assign unique request_id (for rollout tracking)
        3. Strip any prior assistant response (prompt-only for generation)
        """
        samples = self._set_inputs_system(samples)
        for s in samples:
            if not s.request_id:
                s.request_id = f'chatcmpl-{str(uuid.uuid4().hex)}'
            remove_response(s.messages)
        return samples

    def _inputs_to_requests(
            self, samples: Union[List[OnPolicySample], List[RolloutInferRequest]]) -> List[RolloutInferRequest]:
        """Convert samples into RolloutInferRequest objects.

        Already-built ``RolloutInferRequest`` (multi-turn continuation) pass
        through unchanged. The per-sample mapping lives in
        ``OnPolicySample.to_infer_request``.
        """
        if not samples:
            return []
        requests_list = []
        for data in samples:
            if isinstance(data, RolloutInferRequest):
                requests_list.append(data)
            else:
                requests_list.append(data.to_infer_request())
        return requests_list

    def _log_rollout(self, samples: List[OnPolicySample], outputs: List[RolloutOutput]) -> None:
        """Log rollout prompts/completions. Collects into ``_logs`` for periodic flush."""
        if not self.log_completions:
            return
        messages = gather_object([s.messages for s in samples])
        completions = gather_object([out.response.choices[0].message.content for out in outputs])
        self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(messages))
        self._logs['completion'].extend(completions)

    def _prepare_logging(self):
        """Initialize logging infrastructure (shared by GRPO and GKD)."""
        args = self.args
        self.log_completions = getattr(args, 'log_completions', False)
        self.wandb_log_unique_prompts = getattr(args, 'wandb_log_unique_prompts', False)
        self.jsonl_writer = JsonlWriter(os.path.join(args.output_dir, 'completions.jsonl'), write_on_rank='last')
        self._last_logged_step = -1
        self._logs = {
            'prompt': deque(),
            'completion': deque(),
        }

    def _flush_log_completions(self):
        """Flush accumulated completion logs to jsonl/wandb/swanlab."""
        if not (self.log_completions and self.is_main_process and len(self._logs['prompt']) > 0):
            return
        if self._step == self._last_logged_step:
            return
        self._last_logged_step = self._step

        table = self._build_log_table()
        self.jsonl_writer.append(table)
        self._logs['prompt'].clear()
        self._logs['completion'].clear()

        args = self.args
        if 'wandb' in args.report_to:
            import pandas as pd

            import wandb
            df = pd.DataFrame(table)
            if self.wandb_log_unique_prompts:
                df = df.drop_duplicates(subset=['prompt'])
            wandb.log({'completions': wandb.Table(dataframe=df)}, commit=False)
        if 'swanlab' in args.report_to:
            import swanlab
            headers = list(table.keys())
            rows = [[table[h][i] for h in headers] for i in range(len(table.get('gen_step', table['prompt'])))]
            swanlab.log({'completions': swanlab.echarts.Table().add(headers, rows)})

    def _build_log_table(self) -> Dict[str, list]:
        """Build the completion log table. Subclasses extend with extra columns (rewards/advantages)."""
        return {
            'gen_step': [self._step - 1] * len(self._logs['prompt']),
            'prompt': list(self._logs['prompt']),
            'completion': list(self._logs['completion']),
        }

    def _apply_chat_template_to_messages_list(self, messages_list):
        """Convert messages list to prompt text using template."""
        from swift.template import TemplateInputs
        prompts_text = []
        for messages in messages_list:
            remove_response(messages)
            template_inputs = TemplateInputs.from_dict({'messages': messages})
            res = self.template.encode(template_inputs)
            prompts_text.append(self.template.safe_decode(res['input_ids']))
        return prompts_text

    @contextmanager
    def offload_context(self):
        """Context manager for model/optimizer offloading during vLLM generation."""
        if self.args.offload_model:
            offload_megatron_model_to_cpu(self.wrapped_models)
            if hasattr(self, 'ref_models') and self.ref_models:
                offload_megatron_model_to_cpu(self.ref_models)
        if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
            offload_megatron_optimizer(self.optimizer)

        try:
            yield
        finally:
            if self.args.offload_model:
                load_megatron_model_to_gpu(self.wrapped_models)
                if hasattr(self, 'ref_models') and self.ref_models:
                    load_megatron_model_to_gpu(self.ref_models)
            if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
                load_megatron_optimizer(self.optimizer)
