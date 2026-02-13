# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Megatron Rollout Mixin - Provides vLLM integration for on-policy generation.

This mixin extracts common vLLM rollout functionality from MegatronGRPOTrainer
to be reused by GKD and other trainers that need online generation.
"""
import base64
import inspect
import os
import uuid
from contextlib import contextmanager, nullcontext
from copy import copy
from typing import Any, Dict, List, Tuple, Union

import json
import torch
from dacite import from_dict
from megatron.core import mpu

from swift.infer_engine.protocol import RequestConfig, RolloutInferRequest, RolloutOutput
from swift.rlhf_trainers.utils import (FlattenedTensorBucket, aggressive_empty_cache, check_vllm_version_ge,
                                       patch_vllm_moe_model_weight_loader, set_expandable_segments)
from swift.utils import get_current_device, get_logger, is_last_rank, is_vllm_available, remove_response, to_device
from .utils import (gather_object, load_megatron_model_to_gpu, load_megatron_optimizer, offload_megatron_model_to_cpu,
                    offload_megatron_optimizer, profiling_context, profiling_decorator)

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


class MegatronRolloutMixin:

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

    def _get_local_rollout_batch(self, batch: List[Dict]) -> List[Dict]:
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
            batch: Full batch of data samples

        Returns:
            Local slice of batch for this rank to process
        """
        rollout_group = self._get_rollout_group()
        rollout_rank = torch.distributed.get_rank(group=rollout_group)
        rollout_group_size = torch.distributed.get_world_size(group=rollout_group)

        total_batch_size = len(batch)
        assert total_batch_size % rollout_group_size == 0, \
            f'Batch size ({total_batch_size}) must be divisible by rollout group size ({rollout_group_size})'

        per_device_batch_size = total_batch_size // rollout_group_size
        start_idx = rollout_rank * per_device_batch_size
        end_idx = start_idx + per_device_batch_size

        return batch[start_idx:end_idx]

    def _gather_rollout_results(self, local_batch: List[Dict]) -> List[Dict]:
        """Gather rollout results from all ranks in the rollout group.

        Args:
            local_batch: Local rollout results from this rank

        Returns:
            Gathered results from all ranks in the rollout group
        """
        rollout_group = self._get_rollout_group()
        return gather_object(local_batch, group=rollout_group)

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

        if not args.use_vllm:
            return

        if args.rlhf_type == 'gkd' and args.lmbda == 0:
            return

        if not is_vllm_available():
            raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                              'Please install vLLM with `pip install vllm -U` to use it.')

        if self.vllm_mode == 'server':
            # Server mode uses external vLLM server
            pass
        elif self.vllm_mode == 'colocate':
            if self.world_size % self.vllm_tensor_parallel_size != 0:
                raise ValueError(f'vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size '
                                 f'({self.world_size}) evenly.')

            self.enable_offload = args.offload_model or args.offload_optimizer
            context = self.offload_context if self.enable_offload else nullcontext

            with context():
                set_expandable_segments(False)
                self.engine = self._prepare_vllm_engine()
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
        load_format = vllm_engine_kwargs.pop('load_format', 'dummy')

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
            engine_kwargs=vllm_engine_kwargs,
            logprobs_mode=logprobs_mode)

        if self.vllm_tensor_parallel_size > 1:
            self.vllm_tp_group = vllm_ps.get_tp_group().device_group

        return engine

    @property
    def bridge(self):
        """Lazy initialization of weight bridge for Megatron-to-vLLM weight transfer."""
        if self._bridge is None:
            self._bridge = self.args.megatron_model_meta.bridge_cls()
        return self._bridge

    @profiling_decorator
    def _move_model_to_vllm(self):
        """Synchronize model weights to vLLM engine."""
        is_lora_training = self.args.tuner_type == 'lora'

        try:
            if is_lora_training:
                self.merge_lora_adapters()

            self._export_and_load_weights()

        finally:
            if is_lora_training:
                self.unmerge_lora_adapters()

        # Reset prefix cache
        if self.vllm_mode == 'server' and self.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == 'colocate':
            self.engine.engine.reset_prefix_cache()

    def _export_and_load_weights(self):
        """Export weights from Megatron and load to vLLM."""
        target_device = 'cpu' if self.args.offload_bridge else None

        with profiling_context(self, 'export_weights'):
            weight_iterator = self.bridge.export_weights(self.unwrapped_models, target_device=target_device)

        if self.vllm_mode == 'colocate':
            llm_model = self.engine.inner_model
            # Patch MoE weight_loader if needed
            patch_vllm_moe_model_weight_loader(llm_model)
            llm_model.load_weights(weight_iterator)
        elif self.vllm_mode == 'server':
            self._load_weights_to_server_in_buckets(weight_iterator)

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

        bucket = FlattenedTensorBucket(named_tensors=bucket_params)
        metadatas = bucket.get_metadata()
        flattened_tensor = bucket.get_flattened_tensor()

        self.vllm_client.update_flattened_params(metadatas, flattened_tensor)

        del bucket, metadatas, flattened_tensor

    @profiling_decorator
    def _generate_completions(self, batch: DataType) -> DataType:
        """Generate completions for a batch using vLLM engine.

        Args:
            batch: List of input data with 'messages' field
            step: Current training step (for weight sync tracking)

        Returns:
            Batch with rollout completion results merged in
        """
        batch = self._preprocess_rollout_inputs(batch)

        # Wake up engine if sleeping (colocate mode)
        if self.vllm_mode == 'colocate' and self.engine.inner_model_executor.is_sleeping:
            wake_up_params = inspect.signature(self.engine.engine.wake_up).parameters
            kwargs = {'tags': ['weights']} if 'tags' in wake_up_params else {}
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

            # Rollout
            outputs: List[RolloutOutput] = self._rollout(batch)

            # Sleep to release memory
            if self.vllm_mode == 'colocate' and self.args.sleep_level > 0:
                self.engine.engine.reset_prefix_cache()
                self.engine.engine.sleep(level=self.args.sleep_level)
                aggressive_empty_cache()
                set_expandable_segments(True)

            batch = self._postprocess_rollout_outputs(batch, outputs)

        return batch

    def _rollout(self, batch: DataType) -> List[RolloutOutput]:
        """Execute rollout using vLLM engine."""
        batch = self._set_inputs_system(batch)
        request_config = self._get_request_config()

        if self.vllm_mode == 'server':
            return self._server_rollout(batch, request_config)
        elif self.vllm_mode == 'colocate':
            return self._colocate_rollout(batch, request_config)

    def _get_request_config(self) -> RequestConfig:
        """Get request config with proper seed for distributed TP groups."""
        request_config = copy(self.request_config)

        if self.vllm_mode == 'colocate' and self.vllm_tensor_parallel_size > 1:
            batch_size = getattr(self.args, 'per_device_generation_batch_size', self.args.micro_batch_size)
            batch_size *= self.vllm_tensor_parallel_size
            request_config.seed = batch_size * (self.process_index // self.vllm_tensor_parallel_size)

        return request_config

    def _server_rollout(self, inputs: DataType, request_config: RequestConfig) -> List[RolloutOutput]:
        """Perform rollout using vLLM server mode."""
        from accelerate.utils import broadcast_object_list

        infer_requests = self._inputs_to_requests(inputs)

        all_requests = gather_object(infer_requests)
        all_requests_lengths = gather_object([len(infer_requests)])

        if not any(requests for requests in all_requests):
            return []

        if self.is_main_process:
            all_outputs: List[RolloutOutput] = self.vllm_client.infer(
                infer_requests=all_requests, request_config=request_config)
            assert len(all_outputs) == len(all_requests)
        else:
            all_outputs = [None] * len(all_requests)

        all_outputs = broadcast_object_list(all_outputs, from_process=self.world_size - 1)
        start_idx = sum(all_requests_lengths[:self.process_index])
        end_idx = start_idx + all_requests_lengths[self.process_index]
        outputs = all_outputs[start_idx:end_idx]

        return outputs

    def _colocate_rollout(self, batch: DataType, request_config: RequestConfig) -> List[RolloutOutput]:
        """Perform co-located rollout with vLLM engine."""
        start_idx = 0
        end_idx = len(batch)

        # Handle vLLM tensor parallelism
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

        outputs: List[RolloutOutput] = self.engine.infer(
            infer_requests=batch, request_config=request_config, use_tqdm=False)

        if self.vllm_tensor_parallel_size > 1:
            outputs = outputs[start_idx:end_idx]

        return outputs

    def _preprocess_rollout_inputs(self, inputs: DataType) -> DataType:
        """Preprocess inputs before rollout inference."""
        # Add unique request_id
        for input_item in inputs:
            if 'request_id' not in input_item:
                input_item['request_id'] = f'chatcmpl-{str(uuid.uuid4().hex)}'
            remove_response(input_item['messages'])
        return inputs

    def _set_inputs_system(self, inputs: DataType) -> DataType:
        """Insert default system message if not present."""
        if not self.template.template_meta.default_system:
            return inputs
        if all(inp['messages'][0]['role'] == 'system' for inp in inputs):
            return inputs
        for inp in inputs:
            messages = inp['messages']
            if messages[0]['role'] != 'system':
                messages.insert(0, {'role': 'system', 'content': self.template.template_meta.default_system})
        return inputs

    def _inputs_to_requests(self, inputs: DataType) -> List[RolloutInferRequest]:
        """Convert raw input data into RolloutInferRequest objects."""

        def _process_image_data(image_data: Union[dict, str]) -> str:
            if isinstance(image_data, dict):
                if image_data.get('bytes'):
                    return base64.b64encode(image_data['bytes']).decode('utf-8')
                if image_data.get('path'):
                    return image_data['path']
            return image_data

        if not inputs:
            return []

        REQUEST_METADATA_FIELDS = ['messages', 'images', 'audios', 'videos', 'tools', 'objects', 'uuid']
        requests_list = []

        for data in inputs:
            if isinstance(data, RolloutInferRequest):
                requests_list.append(data)
                continue

            request_data = {key: data[key] for key in REQUEST_METADATA_FIELDS if key in data and data[key] is not None}
            if 'uuid' not in request_data:
                request_data['uuid'] = data.get('request_id', str(uuid.uuid4().hex))

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

    def _postprocess_rollout_outputs(self, inputs: DataType, outputs: List[RolloutOutput]) -> DataType:
        """Post-process rollout outputs and merge into inputs."""

        def merge_output(input_data: Dict, output: RolloutOutput) -> Dict:
            response = output.response
            choice = response.choices[0]

            if output.messages:
                input_data['messages'] = output.messages
            else:
                messages = input_data['messages']
                remove_response(messages)
                messages.append({'role': 'assistant', 'content': choice.message.content})

            if output.response_token_ids:
                input_data['response_token_ids'] = output.response_token_ids
                if output.response_loss_mask:
                    input_data['response_loss_mask'] = output.response_loss_mask
            else:
                input_data['response_token_ids'] = choice.token_ids

            if output.rollout_infos:
                input_data['rollout_infos'] = output.rollout_infos

            input_data['finish_reason'] = choice.finish_reason
            input_data['is_truncated'] = choice.finish_reason == 'length'
            input_data['add_eos'] = False

            if output.rollout_logprobs:
                input_data['rollout_logprobs'] = output.rollout_logprobs
            elif choice.logprobs is not None and 'content' in choice.logprobs:
                rollout_logprobs = [item['logprob'] for item in choice.logprobs['content']]
                input_data['rollout_logprobs'] = [rollout_logprobs]

            return input_data

        assert len(inputs) == len(outputs)
        return [merge_output(inp, out) for inp, out in zip(inputs, outputs)]

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
