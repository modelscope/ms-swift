# Copyright (c) Alibaba, Inc. and its affiliates.
import base64
import concurrent.futures
import inspect
import os
import re
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from math import ceil
from queue import Queue
from types import MethodType
from typing import Any, Dict, List, Optional, Union

import json
import torch
import torch.nn as nn
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model, set_seed
from dacite import from_dict
from peft.utils.save_and_load import get_peft_model_state_dict
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TrainerCallback

from swift.llm import MultiModelKeys, RequestConfig, RolloutInferRequest
from swift.llm.infer.protocol import ChatCompletionResponse, RolloutOutput
from swift.plugin import MultiTurnScheduler, multi_turns
from swift.trainers import RolloutTrainerArgumentsMixin
from swift.utils import get_logger, is_vllm_available, remove_response
from swift.utils.torch_utils import get_current_device
from .rlhf_mixin import RLHFTrainerMixin
from .utils import (FlattenedTensorBucket, TensorLoRARequest, _create_parameter_buckets,
                    _process_bucket_with_flattened_tensor, aggressive_empty_cache, get_even_process_data,
                    get_gather_if_zero3_context, patch_lora_merge, patch_lora_unmerge, patch_profiling_context,
                    patch_profiling_decorator, patch_vllm_load_adapter, set_expandable_segments)

DataType = List[Dict[str, Union[torch.Tensor, Any]]]
logger = get_logger()


@dataclass
class DataCache:
    """Cache container for rollout results"""
    results: DataType


class AsyncGenerateCallback(TrainerCallback):
    """Callback for async generation in training"""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.queue = self.trainer.train_queue
        train_dataloader = getattr(state, 'train_dataloader', None) or kwargs.get('train_dataloader')
        self.trainer._prefetch(train_dataloader)


class RolloutTrainerMixin(RLHFTrainerMixin):
    """
    Mixin for RLHF trainers that use rollout-based methods (e.g., GRPO, GKD).

    This mixin provides vLLM integration and rollout infrastructure.
    It should be used for trainers that require:
    - Policy rollout with vLLM engine (server or colocate mode)
    - Multi-turn dialogue support (GRPO only)
    - Async generation capabilities (GRPO only)
    """

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 *_args,
                 **kwargs):
        super().__init__(model, ref_model, *_args, **kwargs)

    def prepare_rollout(self):
        self._prepare_rollout_params()
        self._prepare_scheduler()
        self._prepare_vllm()
        self._prepare_async_generate()

    def _prepare_rollout_params(self):
        """Initialize rollout generation parameters"""
        args = self.args
        self.num_generations = args.num_generations if hasattr(args, 'num_generations') else 1
        self.temperature = args.temperature
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
        self.max_completion_length = args.max_completion_length
        self.completion_length_limit_scope = None
        if hasattr(args, 'completion_length_limit_scope'):  # GRPO colocate
            self.completion_length_limit_scope = args.completion_length_limit_scope
        self.async_generate = args.async_generate

        self.request_config = RequestConfig(
            n=1,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop_words,
            return_details=True)

    def _prepare_vllm(self):
        """Initialize vLLM engine (server or colocate mode)"""
        if not is_vllm_available():
            raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                              'Please install vLLM with `pip install vllm -U` to use it.')

        # Initialize default values
        args = self.args

        self.use_fast_infer = args.use_vllm
        self.enable_offload = False
        self.use_gym_env = False
        self.enable_server_multi_turn = False
        self.rollout_enable_lora = False
        self.vllm_use_async_engine = False

        if not args.use_vllm:
            return

        # split model parameters into batches for synchronized weight transfer
        self.parameter_groups, self.parameter_groups_no_lora = self.split_batches()

        if self.vllm_mode == 'server':
            if self.accelerator.is_main_process:
                self.vllm_client.get_engine_type()
                vllm_use_async_engine = [self.vllm_client.use_async_engine]
                use_gym_env = [self.vllm_client.use_gym_env]
                enable_multi_turn = [self.vllm_client.enable_multi_turn]
                enable_lora = [self.vllm_client.enable_lora]
            else:
                vllm_use_async_engine = [False]
                use_gym_env = [False]
                enable_multi_turn = [self.enable_server_multi_turn]
                enable_lora = [False]
            self.vllm_use_async_engine = broadcast_object_list(vllm_use_async_engine, from_process=0)[0]
            self.use_gym_env = broadcast_object_list(use_gym_env, from_process=0)[0]
            self.enable_server_multi_turn = broadcast_object_list(enable_multi_turn, from_process=0)[0]
            self.rollout_enable_lora = broadcast_object_list(enable_lora, from_process=0)[0]
            if self.use_gym_env:
                self.reward_func_names = ['gym_reward']

        elif self.vllm_mode == 'colocate':
            if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                raise ValueError(f'vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size '
                                 f'({self.accelerator.num_processes}) evenly.')

            if self.vllm_tensor_parallel_size > 1:
                self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration([
                    list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                    for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                ])
            self.enable_offload = args.offload_model or args.offload_optimizer
            context = self.offload_context if self.enable_offload else nullcontext

            with context():
                self.engine = self._prepare_vllm_engine()
                if args.sleep_level > 0:
                    self.engine.engine.sleep(args.sleep_level)
        self.dynamic_num_samples = False  # grpo multi-turn
        self.base_sync_done = False
        self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

    def _prepare_vllm_engine(self):
        """Create and configure vLLM engine for colocate mode"""
        from swift.tuners import Swift
        from swift.llm.infer.infer_engine import GRPOVllmEngine
        args = self.args
        model = self.model
        steps_per_generation = args.steps_per_generation if hasattr(args, 'steps_per_generation') else 1
        max_num_seqs = (args.per_device_train_batch_size * self.vllm_tensor_parallel_size * steps_per_generation)
        vllm_template = copy(self.template)
        vllm_template.padding_free = False
        lora_kwargs = {}
        is_moe = model.model_info.is_moe_model
        vllm_enable_lora = args.vllm_enable_lora

        if args.train_type == 'lora' and vllm_enable_lora:
            lora_kwargs = {
                'enable_lora': args.vllm_enable_lora,
                'max_loras': 1,
                'max_lora_rank': args.lora_rank,
            }
            self.rollout_enable_lora = True

            if is_moe:
                logger.warning(
                    'vLLM LoRA is enabled for an MoE model. This may cause errors when applying LoRA to expert layers, '
                    'as vLLM currently does not support LoRA in MoE configurations. If you encounter errors, '
                    'please set vllm_enable_lora to False.')

            if self.is_multimodal:
                logger.warning('vLLM LoRA is enabled for a multimodal model. This may lead to unexpected issues '
                               'when applying LoRA to the ViT component, as vLLM does not yet support this setup. '
                               'If errors occur, please disable LoRA by setting vllm_enable_lora to False.')

            patch_vllm_load_adapter()

        with Swift.grpo_context(model, self.template.processor):
            set_expandable_segments(False)
            engine = GRPOVllmEngine(
                model.model_dir,
                model.model_info.torch_dtype,
                model_type=model.model_meta.model_type,
                use_async_engine=False,
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                enable_prefix_caching=args.vllm_enable_prefix_caching,
                max_num_seqs=max_num_seqs,
                enforce_eager=args.vllm_enforce_eager,
                limit_mm_per_prompt=args.vllm_limit_mm_per_prompt,
                enable_sleep_mode=args.sleep_level > 0,
                max_model_len=args.vllm_max_model_len,
                seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                disable_cascade_attn=args.vllm_disable_cascade_attn,
                load_format='dummy',
                mm_processor_cache_gb=args.vllm_mm_processor_cache_gb,
                template=vllm_template,
                distributed_executor_backend='external_launcher',
                engine_kwargs=self.args.vllm_engine_kwargs,
                **lora_kwargs,
            )
            set_expandable_segments(True)

        return engine

    def split_batches(self):
        """Split model parameters into batches for synchronized weight transfer.

        This method divides model parameters into manageable batches to facilitate efficient
        weight synchronization between vLLM and DeepSpeed. The splitting strategy focuses on
        LLM layers while grouping other components separately.

        Batching Strategy:
            1. LLM layers: Split into N batches based on layer indices (configurable via args.move_model_batches)
            2. Embeddings and LM heads: Grouped into a single batch
            3. Multi-modal components (e.g., vision tower): Grouped into a single batch

        Returns:
            tuple[list[list[str]], list[list[str]]]: A tuple containing two lists:
                - First list: Batches of parameter names (with original prefixes and LoRA layers)
                - Second list: Batches of parameter names with LoRA layers removed and prefixes stripped
                  (for vLLM compatibility). Contains None if no batching is configured.

        Notes:
            - If args.move_model_batches is None, all parameters are returned in a single batch
            - Reference model parameters (containing 'ref_model') are excluded from batching
            - For multi-modal models, components are identified using model_arch metadata
            - LoRA layers are filtered based on the rollout_enable_lora flag

        Raises:
            AssertionError: If no ModuleList is found in the model to determine layer count
        """
        model = self.accelerator.unwrap_model(self.model)
        args = self.args

        if args.move_model_batches is None:
            return [[n for n, p in model.named_parameters() if 'ref_model' not in n]], [None]

        model_arch = model.model_meta.model_arch
        non_llm_parameters = []
        llm_embeds = []
        parameters = []
        pattern = r'\.(\d+)\.'

        layer_count = None
        for name, module in model.named_modules():
            if isinstance(module, ModuleList):
                if model_arch is not None and isinstance(model_arch, MultiModelKeys):
                    llm = model_arch.language_model
                    vision_tower = model_arch.vision_tower
                    if any(vt in name for vt in vision_tower):
                        continue
                    if isinstance(llm, list):
                        llm = llm[0]
                    if name.startswith('base_model'):
                        name = name.replace('base_model.', '')
                    if llm in name:
                        layer_count = len(module)
                else:
                    layer_count = len(module)
        assert layer_count is not None, 'Cannot find ModuleList to split modules.'

        n_layers = ceil(layer_count / args.move_model_batches)
        for _ in range(args.move_model_batches):
            parameters.append([])

        def replace_lora(name):
            if 'lora_' in name:
                return ''
            else:
                if not self.rollout_enable_lora:
                    return name.replace('.base_layer', '')
                else:
                    return name

        def remove_lora_and_prefix(names):
            names = set([re.sub(r'^_model\.', '', replace_lora(n)) for n in names])
            return [n for n in names if n]

        def split_llm(name):
            match = re.search(pattern, name)
            if match:
                number = match.group(1)
                group = int(number) // n_layers
                parameters[group].append(name)
            else:
                llm_embeds.append(name)

        for name, parameter in model.named_parameters():
            if 'ref_model' in name:
                continue
            if model_arch is not None and isinstance(model_arch, MultiModelKeys):
                llm = model_arch.language_model
                vision_tower = model_arch.vision_tower
                if any(vt in name for vt in vision_tower):
                    non_llm_parameters.append(name)
                elif isinstance(llm, list):
                    llm = llm[0]
                    if llm in name:
                        split_llm(name)
                    else:
                        non_llm_parameters.append(name)
            else:
                split_llm(name)

        if llm_embeds:
            parameters.append(llm_embeds)
        if non_llm_parameters:
            parameters.append(non_llm_parameters)
        parameters = [p for p in parameters if p]
        parameters_no_lora = [remove_lora_and_prefix(p_list) for p_list in parameters]
        return parameters, parameters_no_lora

    @patch_profiling_decorator
    def _move_model_to_vllm(self, skip_async_check=False):
        """Synchronize model weights to vLLM engine"""
        args = self.args

        if args.async_generate and not skip_async_check:
            self._wait_queue()

        train_type = args.train_type

        if train_type == 'full' or (train_type == 'lora' and not self.base_sync_done) or not self.rollout_enable_lora:
            self._move_full_model_to_vllm()
        else:
            self._move_adapter_to_vllm()

        # Reset prefix cache
        if self.vllm_mode == 'server' and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == 'colocate':
            self.engine.engine.reset_prefix_cache()

    def _move_adapter_to_vllm(self):
        """Transfer LoRA adapter weights to vLLM engine"""
        lora_params = OrderedDict()
        for i, parameter_group in enumerate(self.parameter_groups):
            parameters = [
                parameter for name, parameter in self.model.named_parameters()
                if not parameter_group or name in parameter_group
            ]
            gather_if_zero3 = get_gather_if_zero3_context(self)
            with gather_if_zero3(parameters), patch_lora_merge(self.model, parameter_group):
                assert len(parameters) == len(parameter_group)
                state_dict = {name: p for p, name in zip(parameters, parameter_group)}
                peft_config = self.model.peft_config.get('default', None)
                self.model.merge_adapter()
                cur_lora_params = get_peft_model_state_dict(self.model, state_dict)
                cur_lora_params = {
                    name: param.full_tensor().detach() if hasattr(param, 'full_tensor') else param.detach()
                    for name, param in cur_lora_params.items()
                }
                lora_params.update(cur_lora_params)
                with patch_lora_unmerge(self.model):
                    self.model.unmerge_adapter()
                del cur_lora_params

        if self.vllm_mode == 'server' and self.accelerator.is_main_process:
            bucket = FlattenedTensorBucket(named_tensors=list(lora_params.items()))
            metadatas = bucket.get_metadata()
            flattened_tensor = bucket.get_flattened_tensor()
            self.vllm_client.update_adapter_flattened_param(peft_config, metadatas, flattened_tensor)
        elif self.vllm_mode == 'colocate':
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_request = TensorLoRARequest(
                lora_name=f'{lora_int_id}',
                lora_int_id=lora_int_id,
                lora_path='dummy_lora_path',
                peft_config=asdict(peft_config),
                lora_tensors=lora_params,
            )
            self.engine.engine.add_lora(lora_request)
        del lora_params

    def _load_state_dict_to_vllm(self, state_dict):
        """Load state_dict to vLLM engine (server or colocate mode)"""
        if self.vllm_mode == 'server' and self.accelerator.is_main_process:
            bucket_size_mb = int(os.environ.get('SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE', 512))
            named_params = list(state_dict.items())
            parameter_buckets = _create_parameter_buckets(named_params, bucket_size_mb=bucket_size_mb)

            for bucket in parameter_buckets:
                _process_bucket_with_flattened_tensor(self, bucket)

            del named_params, parameter_buckets
        elif self.vllm_mode == 'colocate':
            llm_model = self.engine.inner_model
            llm_model.load_weights(state_dict.items())
        del state_dict

    def _move_full_model_to_vllm(self):
        """Transfer full model weights to vLLM engine"""
        gather_if_zero3 = get_gather_if_zero3_context(self)
        is_peft = is_peft_model(self.model)

        for i, parameter_group in enumerate(self.parameter_groups):
            parameter_group_no_lora = self.parameter_groups_no_lora[i]
            parameters = [
                parameter for name, parameter in self.model.named_parameters()
                if not parameter_group or name in parameter_group
            ]

            context_manager = patch_lora_merge(self.model, parameter_group) if is_peft else nullcontext()

            with gather_if_zero3(parameters), context_manager:
                if is_peft and self.should_merge_adapter:
                    self.model.merge_adapter()

                state_dict = self.model.state_dict()

                if is_peft:
                    prefix_removed = {k.removeprefix('base_model.model.'): v for k, v in state_dict.items()}
                    state_dict = prefix_removed if self.rollout_enable_lora else {
                        k.replace('.base_layer', ''): v
                        for k, v in prefix_removed.items()
                    }
                    state_dict = {k: v for k, v in state_dict.items() if self.model.prefix not in k}
                    state_dict = {
                        k.replace('modules_to_save.default.', ''): v
                        for k, v in state_dict.items() if 'original_module' not in k
                    }

                if parameter_group_no_lora:
                    if is_peft:
                        parameter_group_no_lora = [n.replace('base_model.model.', '') for n in parameter_group_no_lora]
                    state_dict = {k: v for k, v in state_dict.items() if k in parameter_group_no_lora}

                if is_peft:
                    assert len(state_dict) > 0 and all(
                        [state.shape != torch.Size([0]) for state in state_dict.values()])

                self._load_state_dict_to_vllm(state_dict)

                if is_peft and self.should_merge_adapter:
                    with patch_lora_unmerge(self.model):
                        self.model.unmerge_adapter()

        if is_peft:
            self.base_sync_done = True

    def _rollout(self,
                 inputs: Optional[DataType],
                 request_config: RequestConfig,
                 is_global_inputs: bool = False) -> List[RolloutOutput]:
        """Execute rollout using vLLM server or colocate mode"""
        request_config = self._get_request_config()
        if self.vllm_mode == 'server':
            rollout_outputs = self._server_rollout(inputs, request_config, is_global_inputs)
        else:
            rollout_outputs = self._colocate_rollout(inputs, request_config)
        return rollout_outputs

    def _get_request_config(self) -> RequestConfig:
        """Get request config with proper seed for distributed TP groups"""
        request_config = copy(self.request_config)
        args = self.args

        if args.vllm_mode == 'colocate' and self.vllm_tensor_parallel_size > 1:
            mode = 'train' if self.model.training else 'eval'
            batch_size = (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps if mode == 'train' else args.per_device_eval_batch_size)
            batch_size *= self.vllm_tensor_parallel_size
            request_config.seed = batch_size * (self.accelerator.process_index // self.vllm_tensor_parallel_size)

        return request_config

    def _set_inputs_system(self, inputs: DataType) -> DataType:
        """Insert default system message if not present"""
        if not self.template.template_meta.default_system:
            return inputs
        if all(_input['messages'][0]['role'] == 'system' for _input in inputs):
            return inputs
        for _input in inputs:
            messages = _input['messages']
            if messages[0]['role'] != 'system':
                messages.insert(0, {'role': 'system', 'content': self.template.template_meta.default_system})
        return inputs

    def _infer_single_or_multi_turn(self,
                                    inputs: DataType,
                                    request_config: RequestConfig,
                                    is_global_inputs: bool = False) -> List[DataType]:
        """Run inference for single-turn or multi-turn dialogue"""
        inputs = self._set_inputs_system(inputs)
        rollout_outputs: List[RolloutOutput] = self._rollout(inputs, request_config, is_global_inputs)

        if not self.multi_turn_scheduler or self.enable_server_multi_turn:
            return self._postprocess_rollout_outputs(inputs, rollout_outputs)

        return self._colocate_multi_turn_infer(inputs, rollout_outputs, request_config)

    def _colocate_multi_turn_infer(self, inputs: DataType, first_turn_rollout_outputs: List[RolloutOutput],
                                   request_config: RequestConfig) -> List[RolloutOutput]:
        """
        Handles multi-turn inference under colocate mode.

        This method iteratively rolls out turns until all dialogues are finished
        according to the multi_turn_scheduler.
        """
        args = self.args
        orig_size = len(inputs)
        # Preallocate to preserve order
        rollout_outputs: List[RolloutOutput] = [None] * orig_size
        rollout_infos = [{} for _ in range(orig_size)]
        response_token_ids = [[] for _ in range(orig_size)]
        response_loss_mask = [[] for _ in range(orig_size)]
        is_continuations = [False] * orig_size
        # Attach index to inputs for tracking
        requests = self.inputs2requests(inputs)
        index_to_infer = list(range(orig_size))
        current_turn = 1
        outputs = first_turn_rollout_outputs
        while True:
            has_local_data = bool(len(index_to_infer) > 0)
            has_global_data = gather_object([has_local_data])
            if not any(has_global_data):
                break
            assert len(index_to_infer) == len(outputs)
            for index, output in zip(index_to_infer, outputs):
                messages = requests[index].messages
                if messages[-1]['content'] is None:
                    # for continuation, we add dummy response, remove here
                    remove_response(messages)
                # Get model response
                response = output.response
                response_choice = response.choices[0]
                # Update conversation history
                completion = response_choice.message.content
                is_continuation = is_continuations[index] = False
                if messages[-1]['role'] == 'assistant':
                    messages[-1]['content'] += completion
                    is_continuation = is_continuations[index] = True
                else:
                    messages.append({'role': 'assistant', 'content': completion})

            current_requests = [requests[index] for index in index_to_infer]
            # Determine which dialogues are finished
            should_stops = [
                self.multi_turn_scheduler.check_finished(req, output.response.choices[0], current_turn)
                for req, output in zip(current_requests, outputs)
            ]

            # Prepare pending inputs for next turn
            next_turn_index_to_infer = []
            for stop, index, output in zip(should_stops, index_to_infer, outputs):
                if args.max_turns:
                    stop = stop or (current_turn >= args.max_turns)
                if stop:
                    rollout_outputs[index] = RolloutOutput(
                        response=output.response,
                        messages=requests[index].messages,
                        response_token_ids=response_token_ids[index],
                        response_loss_mask=response_loss_mask[index],
                        rollout_infos={
                            **rollout_infos[index], 'num_turns': current_turn
                        })
                    continue
                is_continuation = is_continuations[index]
                step_result = self.multi_turn_scheduler.step(requests[index], output.response.choices[0], current_turn)
                current_request: RolloutInferRequest = step_result['infer_request']
                # Track response tokens and masks
                return_token_id = False
                if 'response_token_ids' in step_result:
                    if is_continuation and response_token_ids[index]:
                        response_token_ids[index][-1].extend(step_result['response_token_ids'])
                    else:
                        response_token_ids[index].append(step_result['response_token_ids'])
                    return_token_id = True
                if 'response_loss_mask' in step_result:
                    assert return_token_id, 'You must return response_token_ids with response_loss_mask return'
                    assert len(step_result['response_loss_mask']) == len(step_result['response_token_ids']), \
                        'response_loss_mask must have the same length as response_token_ids'
                    if is_continuation and response_loss_mask[index]:
                        response_loss_mask[index][-1].extend(step_result['response_loss_mask'])
                    else:
                        response_loss_mask[index].append(step_result['response_loss_mask'])

                if 'rollout_infos' in step_result:
                    # Always overwrite the rollout info for this step.
                    # If you need to keep all step-wise details, switch to append or merge instead.
                    rollout_infos[index].update(step_result['rollout_infos'])
                if current_request.messages[-1]['role'] == 'assistant':
                    # for continuation, we add dummy response, add here
                    current_request.messages.append({'role': 'assistant', 'content': None})

                requests[index] = current_request
                next_turn_index_to_infer.append(index)
            current_turn += 1
            infer_requests = [requests[index] for index in next_turn_index_to_infer]
            # Rollout for the next turn
            outputs = self._rollout(infer_requests if has_local_data else [], request_config)
            index_to_infer = next_turn_index_to_infer

        assert all(o is not None for o in rollout_outputs)
        return self._postprocess_rollout_outputs(inputs, rollout_outputs)

    def _fast_infer(self, inputs: DataType) -> DataType:
        """Efficient inference with vLLM colocate mode support"""
        args = self.args
        assert isinstance(args, RolloutTrainerArgumentsMixin)

        if self.vllm_mode == 'colocate' and args.sleep_level > 0:
            if self.engine.inner_model_executor.is_sleeping:
                wake_up_params = inspect.signature(self.engine.engine.wake_up).parameters
                kwargs = {'tags': ['weights']} if 'tags' in wake_up_params else {}
                self.engine.engine.wake_up(**kwargs)

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        context = self.offload_context if self.enable_offload else nullcontext
        with context():

            if (self.vllm_mode == 'colocate' and self.engine.inner_model_executor.is_sleeping
                    and 'tags' in inspect.signature(self.engine.engine.wake_up).parameters):
                aggressive_empty_cache()
                set_expandable_segments(False)
                self.engine.engine.wake_up(tags=['kv_cache'])

            if hasattr(self, 'async_generate') and self.async_generate:
                all_inputs = gather_object(inputs)
                self.async_generate_rollout(all_inputs)

                data_cache = self._queue.get()
                all_outputs = gather_object(data_cache.results)

                per_device_datasize = len(all_outputs) // self.accelerator.num_processes
                process_slice = slice(
                    self.accelerator.process_index * per_device_datasize,
                    (self.accelerator.process_index + 1) * per_device_datasize,
                )
                outputs = all_outputs[process_slice]

            else:
                with self.multi_turn_completion_length_context():
                    outputs = self._infer_single_or_multi_turn(inputs, self.request_config)

            if self.vllm_mode == 'colocate' and args.sleep_level > 0:
                self.engine.engine.reset_prefix_cache()
                self.engine.engine.sleep(level=args.sleep_level)
                aggressive_empty_cache()
                set_expandable_segments(True)
        return outputs

    def _preprocess_inputs(self, inputs: DataType) -> DataType:
        """Preprocess inputs before inference"""
        processed_inputs = self._add_prompt_id_to_inputs(inputs)
        for input_item in processed_inputs:
            remove_response(input_item['messages'])
        return processed_inputs

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

    def _server_rollout(self, inputs: DataType, request_config: RequestConfig,
                        is_global_inputs: bool) -> List[RolloutOutput]:
        """Perform rollout inference using vLLM server mode"""
        infer_requests = self.inputs2requests(inputs)

        if is_global_inputs:
            per_device_size = len(infer_requests) // self.accelerator.num_processes
            all_requests = infer_requests
            all_requests_lengths = [per_device_size] + [0] * (self.accelerator.num_processes - 1)
        else:
            all_requests = gather_object(infer_requests)
            all_requests_lengths = gather_object([len(infer_requests)])

        if not any(requests for requests in all_requests):
            return []

        if self.accelerator.is_main_process:
            all_outputs: List[RolloutOutput] = self._engine_infer(
                infer_requests=all_requests, request_config=request_config)
            if len(all_outputs) != len(all_requests):
                all_outputs = self._sort_by_request_id(all_outputs)
        else:
            all_outputs = [None] * len(all_requests)

        if self.enable_server_multi_turn:
            self.dynamic_num_samples = False
            outputs_count = [len(all_outputs)] if self.accelerator.is_main_process else [0]
            outputs_count = gather_object(outputs_count)[0]
            if outputs_count != len(all_requests):
                self.dynamic_num_samples = True
                if self.dynamic_sample:
                    logger.warning('Mismatch between returned samples and requests detected.')
                if self.template.padding_free:
                    raise NotImplementedError('Padding free mode is not supported for dynamic sample')
            if not self.accelerator.is_main_process:
                all_outputs = [None] * outputs_count

        if not is_global_inputs:
            all_outputs = broadcast_object_list(all_outputs, from_process=0)

            if not self.enable_server_multi_turn or not self.dynamic_num_samples:
                start_idx = sum(all_requests_lengths[:self.accelerator.process_index])
                end_idx = start_idx + all_requests_lengths[self.accelerator.process_index]
                process_slice = slice(start_idx, end_idx)
                outputs = all_outputs[process_slice]
            else:
                outputs = get_even_process_data(self, all_outputs)
        else:
            outputs = all_outputs if self.accelerator.is_main_process else []

        return outputs

    def _colocate_rollout(self, inputs: DataType, request_config: RequestConfig) -> List[RolloutOutput]:
        """Perform co-located rollout inference with PTEngine or vLLMEngine"""
        if self.vllm_tensor_parallel_size > 1:
            local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
            local_input_length = len(inputs)
            all_input_lengths = [None] * self.vllm_tensor_parallel_size
            torch.distributed.all_gather_object(all_input_lengths, local_input_length, group=self.tp_group)

            start_idx = sum(all_input_lengths[:local_rank_in_group])
            end_idx = start_idx + all_input_lengths[local_rank_in_group]

            gathered_inputs = [None for _ in range(self.vllm_tensor_parallel_size)]
            torch.distributed.all_gather_object(gathered_inputs, inputs, group=self.tp_group)
            inputs = [p for sublist in gathered_inputs for p in sublist]

        outputs: List[RolloutOutput] = self._engine_infer(infer_requests=inputs, request_config=request_config)

        if self.vllm_tensor_parallel_size > 1:
            outputs = outputs[start_idx:end_idx]

        return outputs

    def _engine_infer(
        self,
        infer_requests: List[RolloutInferRequest],
        request_config: Optional[RequestConfig] = None,
        *,
        use_tqdm: Optional[bool] = False,
    ) -> List[RolloutOutput]:
        """Perform inference using configured engine"""
        with patch_profiling_context(self, 'generate'), self._disable_sp_context():
            if self.vllm_mode == 'server':
                res = self.vllm_client.infer([asdict(req) for req in infer_requests],
                                             asdict(request_config),
                                             use_tqdm=use_tqdm)
            else:
                res = self.engine.infer(infer_requests, request_config, use_tqdm=use_tqdm)
            if all(isinstance(r, RolloutOutput) for r in res):
                return res
            else:
                assert all(isinstance(r, ChatCompletionResponse) for r in res)
                return [RolloutOutput(response=r) for r in res]

    @property
    def should_merge_adapter(self):
        """Determine whether the LoRA adapter should be merged"""
        args = self.args

        assert args.train_type != 'full', 'Full-parameter training should not merge adapter'

        if not self.rollout_enable_lora:
            return True

        if args.resume_from_checkpoint:
            return True

        return False

    def _postprocess_rollout_outputs(self, inputs: DataType, outputs: List[RolloutOutput]) -> DataType:
        """Postprocess rollout outputs by merging them back into inputs"""

        def merge_output_input_data(input_data: Dict[str, Union[torch.Tensor, Any]], output: RolloutOutput):
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
                if not self.multi_turn_scheduler:
                    input_data['response_token_ids'] = output.response.choices[0].token_ids

            if output.rollout_infos:
                input_data['rollout_infos'] = output.rollout_infos

            input_data['finish_reason'] = choice.finish_reason
            input_data['is_truncated'] = choice.finish_reason == 'length'

            if output.rollout_infos:
                multi_modal_keys = ['images', 'videos', 'audios']
                for key in multi_modal_keys:
                    if key in output.rollout_infos:
                        input_data[key] = output.rollout_infos[key]
                        logger.info_once(f'Overriding multi-modal data from rollout_infos for key: {key}')

            return input_data

        if not self.dynamic_num_samples:
            if self.async_generate and not outputs:
                return outputs
            assert len(inputs) == len(outputs)
            return [
                merge_output_input_data(deepcopy(input_data), output) for input_data, output in zip(inputs, outputs)
            ]

        global_inputs = gather_object(inputs)
        results = []
        id2inputs = {}
        for input_data in global_inputs:
            request_id = input_data['request_id']
            if request_id not in id2inputs:
                id2inputs[request_id] = deepcopy(input_data)
        for output in outputs:
            request_id = output.response.id
            assert request_id in id2inputs, f'Request ID {request_id} not found in inputs'
            input_data = deepcopy(id2inputs[request_id])
            results.append(merge_output_input_data(input_data, output))

        return results

    @torch.no_grad()
    def offload_model(self, model):
        for param in model.parameters():
            param.data = param.data.to(torch.device('cpu'), non_blocking=True)

    @torch.no_grad()
    def load_model(self, model):
        device = get_current_device()
        for param in model.parameters():
            param.data = param.data.to(device, non_blocking=True)

    @torch.no_grad()
    def offload_optimizer(self):
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to('cpu', non_blocking=True)

    @torch.no_grad()
    def load_optimizer(self):
        device = get_current_device()
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device, non_blocking=True)

    @contextmanager
    def offload_context(self):
        """
        Context manager for model/optimizer offloading.

        This is a placeholder implementation. Subclasses (like GRPOTrainer)
        should override this method to provide actual offload/reload logic.
        """
        yield

    def _prepare_scheduler(self):
        """Prepare multi-turn scheduler"""
        args = self.args

        self.multi_turn_scheduler = None
        if not hasattr(args, 'multi_turn_scheduler'):
            # only GRPO support it now
            return

        if args.multi_turn_scheduler:
            if isinstance(args.multi_turn_scheduler, str):
                assert args.multi_turn_scheduler in multi_turns
                multi_turn_scheduler = multi_turns[args.multi_turn_scheduler](max_turns=args.max_turns)
                self.multi_turn_scheduler: MultiTurnScheduler = multi_turn_scheduler
            else:
                assert isinstance(args.multi_turn_scheduler, MultiTurnScheduler)
                self.multi_turn_scheduler: MultiTurnScheduler = args.multi_turn_scheduler

    @contextmanager
    def multi_turn_completion_length_context(self):
        """Context manager for multi-turn completion length limiting"""
        if not (self.multi_turn_scheduler and
                self.use_fast_infer) or self.vllm_mode == 'server' or self.completion_length_limit_scope == 'per_round':
            yield
            return

        original_fn = self.engine.set_default_max_tokens
        original_max_len = self.engine.max_model_len

        def set_default_max_tokens(_self, request_config: RequestConfig, inputs: Dict[str, Any]) -> None:
            original_max_len = _self.max_model_len or 8192
            assert isinstance(inputs, dict)
            prompt_tokens = _self._get_num_tokens(inputs)

            if not hasattr(_self, 'set_grpo_max_model_len'):
                max_len = min(original_max_len, prompt_tokens + request_config.max_tokens)
                _self.max_model_len = max_len
                _self.set_grpo_max_model_len = True
            else:
                if _self.max_model_len <= prompt_tokens:
                    num_tokens_avoid_crash = 10
                    _self.max_model_len = (prompt_tokens + num_tokens_avoid_crash)
                    request_config.max_tokens = num_tokens_avoid_crash

            original_fn(request_config, inputs)

        try:
            self.engine.set_default_max_tokens = MethodType(set_default_max_tokens, self.engine)
            yield
        finally:
            self.engine.set_default_max_tokens = original_fn
            self.engine.max_model_len = original_max_len
            del self.engine.set_grpo_max_model_len

    def inputs2requests(self, inputs: DataType) -> List[RolloutInferRequest]:
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
        requests_dicts = []

        for data in inputs:
            request_data = {key: data[key] for key in REQUEST_METADATA_FIELDS if key in data and data[key] is not None}
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

            requests_dicts.append(request_data)

        for request in requests_dicts:
            if 'images' in request and request['images']:
                request['images'] = ([_process_image_data(img) for img in request['images']] if isinstance(
                    request['images'], list) else _process_image_data(request['images']))

        # load tools json
        for request_data in requests_dicts:
            if 'tools' in request_data and isinstance(request_data['tools'], str):
                from json import JSONDecodeError
                try:
                    request_data['tools'] = json.loads(request_data['tools'])
                except JSONDecodeError:
                    pass

        return [from_dict(RolloutInferRequest, request_data) for request_data in requests_dicts]

    def async_generate_rollout(self, all_inputs):
        """Async generation task for rollout"""
        current_queue = self._queue

        def infer_task():
            try:
                with self.multi_turn_completion_length_context():
                    return self._infer_single_or_multi_turn(all_inputs, self.request_config, is_global_inputs=True)
            except Exception as e:
                logger.error('Inference task failed: %s', str(e))
                raise

        future: Future = self.executor.submit(infer_task)

        def done(future):
            try:
                result = future.result()
                current_queue.put(DataCache(result))
            except Exception as e:
                logger.error('Error in async_generate_rollout callback: %s', str(e))

        future.add_done_callback(done)

    def _prepare_async_generate(self):
        """Initialize async generation queues and callback"""
        self.train_queue = Queue()
        self.eval_queue = Queue()
        args = self.args

        if args.async_generate:
            self.add_callback(AsyncGenerateCallback(self))

    @property
    def _queue(self):
        """Get the appropriate queue based on training/eval mode"""
        if self.control.should_evaluate:
            return self.eval_queue
        else:
            return self.train_queue

    def _wait_queue(self):
        """Wait for queue to have items"""
        while self._queue.empty():
            time.sleep(0.01)

    def _sort_by_request_id(self, all_outputs: List[RolloutOutput]) -> List[RolloutOutput]:
        """Sort rollout outputs by request_id to group outputs together"""
        request_ids = [output.response.id for output in all_outputs]
        output_pairs = list(zip(request_ids, all_outputs))
        output_pairs.sort(key=lambda x: x[0])
        sorted_outputs = [output for _, output in output_pairs]
        return sorted_outputs

    def _prefetch(self, dataloader: DataLoader):
        inputs = next(iter(dataloader))
        if self.template.truncation_strategy == 'raise':
            inputs = self.resample_encode_failed_inputs(inputs)
        inputs = self._preprocess_inputs(inputs)
        all_inputs = gather_object(inputs)
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm(skip_async_check=True)
            self._last_loaded_step = self.state.global_step
        results = self._infer_single_or_multi_turn(all_inputs, self.request_config, is_global_inputs=True)
        self._queue.put(DataCache(results))

    @contextmanager
    def _disable_sp_context(self):
        from swift.trainers.sequence_parallel import sequence_parallel
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        # Save original SP state
        origin_size = sequence_parallel.world_size

        # Save and restore original attention functions
        flash_attn_backup = None
        sdpa_backup = None
        if 'flash_attention_2_origin' in ALL_ATTENTION_FUNCTIONS:
            flash_attn_backup = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
            ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin']
        if 'sdpa_origin' in ALL_ATTENTION_FUNCTIONS:
            sdpa_backup = ALL_ATTENTION_FUNCTIONS['sdpa']
            ALL_ATTENTION_FUNCTIONS['sdpa'] = ALL_ATTENTION_FUNCTIONS['sdpa_origin']

        # Disable SP
        sequence_parallel.world_size = 1

        try:
            yield
        finally:
            # Restore SP state
            sequence_parallel.world_size = origin_size

            # Restore patched attention functions
            if flash_attn_backup is not None:
                ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = flash_attn_backup
            if sdpa_backup is not None:
                ALL_ATTENTION_FUNCTIONS['sdpa'] = sdpa_backup
