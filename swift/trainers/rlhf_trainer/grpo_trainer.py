# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/trl.
import concurrent.futures
import inspect
import os
import re
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial
from math import ceil
from queue import Queue
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import torch
import torch.nn as nn
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from packaging import version
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TrainerCallback
from transformers.trainer import Trainer
from trl import GRPOTrainer as HFGRPOTrainer
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.models import prepare_deepspeed
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_trainer import nanmax, nanmin, nanstd

from swift.llm import (InferRequest, MultiModelKeys, RequestConfig, RolloutInferRequest, RowPreprocessor, Template,
                       get_model_arch, to_device)
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.llm.model.utils import get_llm_model
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.plugin import loss_scale_map, multi_turns, orms, rm_plugins
from swift.plugin.multi_turn import MultiTurnScheduler
from swift.utils import (JsonlWriter, gc_collect, get_current_device, get_device, get_logger, is_vllm_available,
                         is_wandb_available, seed_worker, unwrap_model_for_generation)
from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin
from .utils import _ForwardRedirection, patch_lora_merge, patch_lora_unmerge
from .vllm_client import VLLMClient

del HFGRPOTrainer.__init__
del HFGRPOTrainer.log

logger = get_logger()
if is_wandb_available():
    import wandb

InputsType = List[Dict[str, Union[torch.Tensor, Any]]]
# tuple: (messages, finish_reason)
OutputsType = List[Tuple[List[Dict], str]]


class GRPOCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer

    # offload original_modules to cpu, to save memory
    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.queue = self.trainer.train_queue
        train_dataloader = getattr(state, 'train_dataloader', None) or kwargs.get('train_dataloader')
        self.trainer._prefetch(train_dataloader)


@dataclass
class DataCache:
    inputs: List[Dict] = field(default_factory=list)
    outputs: List[Dict] = field(default_factory=list)


def identity_data_collator(features):
    return features


class GRPOTrainer(RLHFTrainerMixin, SwiftMixin, HFGRPOTrainer):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_model: Optional[List[Union[PreTrainedModel, nn.Module]]] = None,
                 reward_funcs: Optional[List[Union[str, Callable]]] = None,
                 *_args,
                 **kwargs):
        from swift.trainers.rlhf_arguments import GRPOConfig
        args: GRPOConfig = kwargs['args']
        self.args = args
        # for async generate
        self.train_queue = Queue()
        self.eval_queue = Queue()

        self.processing_class = kwargs.get('template').tokenizer

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

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
                    raise ValueError(f'reward_function {reward_func} is not implemented in swift.llm.plugin')

        self.reward_funcs = reward_funcs
        self.reward_func_names = []
        for reward_func in reward_funcs:
            if inspect.isfunction(reward_func):
                reward_func_name = reward_func.__name__
            else:
                reward_func_name = reward_func.__class__.__name__
            self.reward_func_names.append(reward_func_name)

        self.reward_model_plugins = [None] * len(self.reward_funcs)

        if reward_model is not None:
            reward_template = kwargs.pop('reward_template')
            reward_plugins = args.reward_model_plugin
            if reward_plugins is None:
                reward_plugins = ['default'] * len(reward_model)
            assert len(reward_plugins) == len(reward_model), (
                f"The number of 'reward_model_plugin' ({len(reward_plugins)}) does not match "
                f"the number of 'reward_model' ({len(reward_model)}). "
                "Please provide a corresponding 'reward_model_plugin' for each 'reward_model'.")
            for rm, rm_plugin, rm_template in zip(reward_model, reward_plugins, reward_template):
                # Set encoding mode train(see details in Template.encode).
                # Set max_length to None to disable truncation, as the input length has already been truncated earlier.
                rm_template.set_mode('train')
                rm_template.max_length = None
                if rm_plugin not in rm_plugins:
                    raise ValueError(f'rm_plugin {rm_plugin} is not implemented in swift.llm.plugin')
                self.reward_model_plugins.append(rm_plugins[rm_plugin](model=rm, template=rm_template))
                self.reward_funcs.append(rm)
                self.reward_func_names.append(rm.config._name_or_path.split('/')[-1])

        if not self.reward_funcs:
            raise ValueError('You must specify reward_funcs or reward_model')

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(f'Number of reward weights ({len(args.reward_weights)}) must match number of reward '
                                 f'functions ({len(reward_funcs)})')
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        self.multi_turn_scheduler = None
        if self.args.multi_turn_scheduler:
            if isinstance(self.args.multi_turn_scheduler, str):
                assert self.args.multi_turn_scheduler in multi_turns
                multi_turn_scheduler = multi_turns[self.args.multi_turn_scheduler](max_turns=self.args.max_turns)
                self.multi_turn_scheduler: MultiTurnScheduler = multi_turn_scheduler
            else:
                assert isinstance(multi_turn_scheduler, MultiTurnScheduler)
                self.multi_turn_scheduler: MultiTurnScheduler = self.args.multi_turn_scheduler

        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.loss_type = args.loss_type
        self.max_completion_length = args.max_completion_length
        self.completion_length_limit_scope = args.completion_length_limit_scope
        model.warnings_issued['estimate_tokens'] = True

        kwargs['data_collator'] = identity_data_collator  # No data collation is needed in GRPO
        self.shuffle_dataset = args.dataset_shuffle

        self.use_vllm = args.use_vllm
        self.async_generate = args.async_generate
        vllm_client = kwargs.pop('vllm_client')  # for external vllm

        super().__init__(model, ref_model, *_args, **kwargs)
        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper

        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        self.use_liger_loss = self.args.use_liger_kernel
        if self.use_liger_loss:
            from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )
            self._forward_redirection = _ForwardRedirection()

        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'completions.jsonl'))
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            'prompt': deque(maxlen=maxlen),
            'completion': deque(maxlen=maxlen),
            'rewards': defaultdict(lambda: deque(maxlen=maxlen)),
        }
        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        if is_peft_model(self.model):
            self.parameter_groups, self.parameter_groups_no_lora = self.split_batches()
        self.use_fast_infer = self.use_vllm  # whether to use the PT backend
        self.vllm_use_async_engine = False
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                                  'Please install vLLM with `pip install vllm -U` to use it.')
            if self.vllm_mode == 'server':
                self.vllm_client: VLLMClient = vllm_client
                if self.accelerator.is_main_process:
                    vllm_use_async_engine = [self.vllm_client.get_engine_type() == 'AsyncLLMEngine']
                else:
                    vllm_use_async_engine = [False]
                self.vllm_use_async_engine = broadcast_object_list(vllm_use_async_engine, from_process=0)[0]

            elif self.vllm_mode == 'colocate':
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f'vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size '
                        f'({self.accelerator.num_processes}) evenly.')

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration([
                        list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                        for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                    ])

                self.engine = self.prepare_vllm(model)
        else:
            from swift.llm import PtEngine
            self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit

        self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation
        self.request_config = RequestConfig(
            n=1,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop_words,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
        self.padding_free = self.template.padding_free
        self.template.padding_free = False
        self.template._packing = False
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True)

        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle. # noqa
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        if self.async_generate:
            self.add_callback(GRPOCallback(self))

        if self.args.dynamic_sample:
            self.resample_dataset = deepcopy(self.train_dataset)

            def cyclic_iter(iterable):
                while True:
                    for x in iterable:
                        yield x

            self.resample_iterator = cyclic_iter(self.get_resample_dataloader())
        # flag indicating whether the evaluation has started
        self.eval_flag = False

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, Union[torch.Tensor,
                                                                Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = 'train' if self.model.training else 'eval'
        if mode == 'train':
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(generation_batch)
                self._buffered_inputs = generation_batch  # < this is the change
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def split_batches(self):
        """Sync weights in batches
        Only split LLM layers for now:
        1. N batches for layers
        2. other, embeds, lm_heads in one batch
        3. multi-modal components in one batch
        """
        model = self.accelerator.unwrap_model(self.model)
        if self.args.move_model_batches is None:
            # All in one
            return [[n for n, p in model.named_parameters() if 'ref_model' not in n]], [None]

        model_arch = get_model_arch(model.model_meta.model_arch)
        non_llm_parameters = []
        llm_embeds = []
        parameters = []
        pattern = r'\.(\d+)\.'

        layer_count = None
        # Get the number of layers in LLM modules
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

        n_layers = ceil(layer_count / self.args.move_model_batches)
        for _ in range(self.args.move_model_batches):
            parameters.append([])

        def replace_lora(name):
            if 'lora_' in name:
                return ''
            else:
                return name.replace('base_layer.', '')

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

    def prepare_vllm(self, model):
        from swift.tuners import Swift
        from swift.llm.infer.infer_engine import GRPOVllmEngine
        max_num_seqs = (
            self.args.per_device_train_batch_size * self.vllm_tensor_parallel_size * self.args.steps_per_generation)
        current_device = get_device()
        with Swift.grpo_context(model, self.template.processor):
            engine = GRPOVllmEngine(
                model.model_dir,
                model.model_info.torch_dtype,
                model_type=model.model_meta.model_type,
                use_async_engine=False,  # TODO: async engine for colocate
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                max_num_seqs=max_num_seqs,
                enforce_eager=self.args.vllm_enforce_eager,
                limit_mm_per_prompt=self.args.vllm_limit_mm_per_prompt,
                enable_sleep_mode=self.args.sleep_level > 0,
                device=current_device,
                max_model_len=self.args.vllm_max_model_len,
                seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                template=self.template,
                distributed_executor_backend='external_launcher',
            )
        return engine

    @contextmanager
    def _template_context(self, template: Template):
        # The max_length for prompt and completion has already been restricted, so there is no need for max_length here.
        max_length = template.max_length
        mode = template.mode
        if mode in {'vllm', 'pt', 'lmdeploy'}:
            template.set_mode('train')
        template.max_length = None
        try:
            yield
        finally:
            template.set_mode(mode)
            template.max_length = max_length

    @profiling_decorator
    def _move_model_to_vllm(self, skip_async_check=False):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed
            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.args.async_generate and not skip_async_check:
            # before sync weight, we should wait async generate finish
            self._wait_queue()

        if is_peft_model(self.model):
            for i, parameter_group in enumerate(self.parameter_groups):  # < this is the change
                parameter_group_no_lora = self.parameter_groups_no_lora[i]
                parameters = [
                    parameter for name, parameter in self.model.named_parameters()
                    if not parameter_group or name in parameter_group
                ]
                with gather_if_zero3(parameters), patch_lora_merge(self.model, parameter_group):
                    self.model.merge_adapter()
                    state_dict = self.model.state_dict()
                    state_dict = {
                        k.removeprefix('base_model.model.').replace('.base_layer', ''): v
                        for k, v in state_dict.items()
                    }
                    state_dict = {k: v for k, v in state_dict.items() if self.model.prefix not in k}
                    # When module to save, remove its prefix and discard the original module
                    state_dict = {
                        k.replace('modules_to_save.default.', ''): v
                        for k, v in state_dict.items() if 'original_module' not in k
                    }
                    if parameter_group_no_lora:
                        parameter_group_no_lora = [n.replace('base_model.model.', '') for n in parameter_group_no_lora]
                        state_dict = {k: v for k, v in state_dict.items() if k in parameter_group_no_lora}
                    assert len(state_dict) > 0 and all(
                        [state.shape != torch.Size([0]) for state in state_dict.values()])

                    if self.vllm_mode == 'server' and self.accelerator.is_main_process:
                        for name, param in state_dict.items():
                            self.vllm_client.update_named_param(name, param)
                    elif self.vllm_mode == 'colocate':
                        llm_model = self.engine.inner_model
                        llm_model.load_weights(state_dict.items())
                    with patch_lora_unmerge(self.model):
                        self.model.unmerge_adapter()
        else:
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.vllm_mode == 'server' and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
                    elif self.vllm_mode == 'colocate':
                        llm_model = self.engine.inner_model
                        llm_model.load_weights([(name, param.data)])

        if self.vllm_mode == 'server' and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == 'colocate':
            # since vLLM model weights has been updated, we should reset the prefix cache
            self.engine.engine.reset_prefix_cache()

    def _wait_queue(self):
        while self._queue.empty():
            time.sleep(0.01)

    def _infer(self,
               inputs: Optional[InputsType],
               request_config: RequestConfig,
               is_global_inputs: bool = False) -> List[ChatCompletionResponse]:
        request_config = self._get_request_config()
        # keys from InferRequest
        per_device_size = len(inputs)
        if is_global_inputs:
            per_device_size //= self.accelerator.num_processes
        if self.vllm_mode == 'server':
            # for server mode, we gather all the inputs and send to remote vllm server in main process
            if is_global_inputs:
                # async generate, pre-gather to avoid potential communicate operator
                all_inputs = inputs
                all_input_lengths = [per_device_size] + [0] * (self.accelerator.num_processes - 1)
            else:
                all_inputs = gather_object(inputs)
                all_input_lengths = gather_object([len(inputs)])

            if not any(inputs for inputs in all_inputs):
                return []

            if self.accelerator.is_main_process:
                results: List[ChatCompletionResponse] = self._engine_infer(
                    infer_requests=all_inputs, request_config=request_config)
            else:
                results = [None] * len(all_inputs)
            # Broadcast the results from the main process to all processes,
            # ensuring each process receives its corresponding slice.
            if not is_global_inputs:
                results = broadcast_object_list(results, from_process=0)
                start_idx = sum(all_input_lengths[:self.accelerator.process_index])
                end_idx = start_idx + all_input_lengths[self.accelerator.process_index]
                results = results[start_idx:end_idx]
            else:
                results = results if self.accelerator.is_main_process else []
        else:
            # pt / vllm colocate
            if self.vllm_tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                # Note: The input sizes may differ across ranks (e.g., in multi-turn scenarios,
                # the amount of data each rank continues to process may vary).
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                local_input_length = len(inputs)
                all_input_lengths = [None] * self.vllm_tensor_parallel_size
                torch.distributed.all_gather_object(all_input_lengths, local_input_length, group=self.tp_group)
                start_idx = sum(all_input_lengths[:local_rank_in_group])
                end_idx = start_idx + all_input_lengths[local_rank_in_group]

                # orig_size = len(inputs)/
                gathered_inputs = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_inputs, inputs, group=self.tp_group)
                inputs = [p for sublist in gathered_inputs for p in sublist]
            # Set request_config.seed
            # 1. Ensure that the seed for vLLM Engines within each TP (Tensor Parallelism) group is the same;
            #   otherwise, the program may hang.
            # 2. Ensure that the seed for vLLM Engines across different TP groups is different;
            #   otherwise, identical completions will be generated.
            results: List[ChatCompletionResponse] = self._engine_infer(
                infer_requests=inputs, request_config=request_config)

            if self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                results = results[start_idx:end_idx]
        return results

    def _get_request_config(self) -> RequestConfig:
        request_config = copy(self.request_config)
        if self.args.vllm_mode == 'colocate' and self.vllm_tensor_parallel_size > 1:
            # Set request_config.seed
            # 1. Ensure that the seed for vLLM Engines within each TP (Tensor Parallelism) group is the same;
            #   otherwise, the program may hang.
            # 2. Ensure that the seed for vLLM Engines across different TP groups is different;
            #   otherwise, identical completions will be generated.
            mode = 'train' if self.model.training else 'eval'
            batch_size = (
                self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps if mode == 'train' else self.args.per_device_eval_batch_size)
            batch_size *= self.vllm_tensor_parallel_size
            # Since the TP (Tensor Parallelism) group gathers the inputs,
            # multiply the batch size by the TP parallel size.
            request_config.seed = batch_size * (self.accelerator.process_index // self.vllm_tensor_parallel_size)

        return request_config

    def _set_inputs_system(self, inputs: InputsType) -> InputsType:
        if not self.template.template_meta.default_system:
            return
        if all(_input['messages'][0]['role'] == 'system' for _input in inputs):
            return
        for _input in inputs:
            messages = _input['messages']
            if messages[0]['role'] != 'system':
                messages.insert(0, {'role': 'system', 'content': self.template.template_meta.default_system})

    def _infer_single_or_multi_turn(self,
                                    inputs: InputsType,
                                    request_config: RequestConfig,
                                    is_global_inputs: bool = False) -> OutputsType:
        """Perform multi-turn or single-turn inference

        Args:
            inputs: list of input requests
            request_config: Inference configuration parameters
            is_global_inputs:
                A boolean indicating whether the inputs are global. When set to True,
                the returned results in the main process will be a complete list of
                global_outputs, while other processes will return an empty list [].
        Returns:
            List of outputs where each entry contains:
            - List of responses per prompt
            - Each response is a tuple of (message_history, finish_reason)
        """
        self._set_inputs_system(inputs)
        # infer first turn
        results: List[ChatCompletionResponse] = self._infer(inputs, request_config, is_global_inputs)

        outputs = []
        if not self.multi_turn_scheduler and not self.vllm_use_async_engine:
            # message concatenation
            for i, output in enumerate(results):
                _choices = []
                for choice in output.choices:
                    _input: Dict = deepcopy(inputs[i])
                    InferRequest.remove_response(_input['messages'])
                    _input['messages'].append({'role': 'assistant', 'content': choice.message.content})
                    _choices.append((_input['messages'], choice.finish_reason))
                outputs.append(_choices)
            outputs = [item for sublist in outputs for item in sublist]
        else:
            # vLLMAsyncLLMEngine, only server mode is supported right now.
            # NOTE: The message concatenation has already been done in the engine.
            if self.vllm_use_async_engine:
                for i, output in enumerate(results):
                    _choices = []
                    for choice in output.choices:
                        # concated in Engine
                        _choices.append((choice.messages, choice.finish_reason))
                    outputs.append(_choices)
                outputs = [item for sublist in outputs for item in sublist]
            else:
                # PTEngine or vLLMLLMEngine
                orig_size = len(inputs)
                outputs = [None] * orig_size
                # we remove origin response in first turn
                current_turn = 1
                while True:
                    has_local_data = len(inputs) > 0
                    has_global_data = gather_object([has_local_data])
                    if not any(has_global_data):
                        break
                    # inputs for current turn
                    current_inputs = []
                    cnt = 0
                    # combine completions from results with messages
                    for i, output in enumerate(results):
                        for choice in output.choices:
                            current_input = deepcopy(inputs[i])
                            messages = current_input['messages']

                            if current_turn == 1 or not messages[-1]['content'] or messages[-1]['content'] == '<None>':
                                # first turn or the last message content is empty(dummy), remove the response
                                InferRequest.remove_response(messages)
                            if messages[-1]['role'] == 'assistant':
                                # If the last message was assistant, concatenate the new content to it
                                messages[-1]['content'] += choice.message.content
                            else:
                                # append a new message from the assistant
                                messages.append({'role': 'assistant', 'content': choice.message.content})

                            if 'index' not in current_input:
                                current_input['index'] = cnt
                            current_input['finish_reason'] = choice.finish_reason
                            cnt += 1
                            current_inputs.append(current_input)

                    # Process messages in the multi-turn function
                    should_stops = [
                        self.multi_turn_scheduler.check_finished(request, result.choices[0], current_turn)
                        for request, result in zip(self.inputs_to_rolloutrequest(current_inputs), results)
                    ]

                    # Retain messages that are not yet finished for the next round of rollout
                    pending_inputs = []
                    for stop, _input, result in zip(should_stops, current_inputs, results):
                        index = _input['index']
                        if stop:
                            outputs[index] = (_input['messages'], _input['finish_reason'])
                        else:
                            current_request = self.inputs_to_rolloutrequest([_input])[0]
                            infer_request = self.multi_turn_scheduler.step(current_request, result.choices[0],
                                                                           current_turn)
                            pending_input = asdict(infer_request)
                            pending_input['index'] = index
                            pending_inputs.append(pending_input)

                    current_infer_inputs = pending_inputs if has_local_data else []
                    results = self._infer(current_infer_inputs, request_config)

                    inputs = pending_inputs
                    current_turn += 1
                assert not any([o is None for o in outputs])

        # flatten 2D list to 1D list
        return outputs

    def async_infer(self, all_inputs):
        current_queue = self._queue

        def infer_task():
            try:
                with self.multi_turn_completion_length_context():
                    return self._infer_single_or_multi_turn(all_inputs, self.request_config, is_global_inputs=True)
            except Exception as e:
                logger.error('Inference task failed: %s', str(e))
                raise

        future: Future = self.executor.submit(infer_task)

        # pre-fetch the queue to avoid switching back to eval_queue at the end of training sample sampling

        def done(future):
            try:
                result = future.result()
                current_queue.put(DataCache(all_inputs, result))
            except Exception as e:
                logger.error('Error in async_infer callback: %s', str(e))

        future.add_done_callback(done)

    def _prefetch(self, dataloader: DataLoader):
        inputs = next(iter(dataloader))
        all_inputs = gather_object(inputs)
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm(skip_async_check=True)
            self._last_loaded_step = self.state.global_step
        outputs = self._infer_single_or_multi_turn(all_inputs, self.request_config, is_global_inputs=True)
        self._queue.put(DataCache(all_inputs, outputs))

    def _fast_infer(self, inputs: InputsType) -> Tuple[InputsType, OutputsType]:
        if self.vllm_mode == 'colocate' and self.args.sleep_level > 0:
            if self.args.offload_model:
                self.offload_model(self.accelerator.unwrap_model(self.model))
                if self.ref_model:
                    self.offload_model(self.ref_model)
            if self.args.offload_optimizer:
                self.offload_optimizer()
            if self.args.gc_collect_after_offload:
                gc_collect()
            # Skip the first wake_up to avoid the warning "Executor is not sleeping"
            if self.engine.inner_model_executor.is_sleeping:
                self.engine.engine.wake_up()
        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        if self.async_generate:
            # send this step data to server
            # we gather inputs outside the thread for prevent potential gather deadlock
            all_inputs = gather_object(inputs)
            self.async_infer(all_inputs)
            # cached data from last step
            data_cache = self._queue.get()
            all_inputs = data_cache.inputs
            all_outputs = gather_object(data_cache.outputs)
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )
            inputs = all_inputs[process_slice]
            outputs = all_outputs[process_slice]

        else:
            with self.multi_turn_completion_length_context():
                outputs = self._infer_single_or_multi_turn(inputs, self.request_config)

        if self.vllm_mode == 'colocate' and self.args.sleep_level > 0:
            self.engine.engine.sleep(level=self.args.sleep_level)
            if self.args.gc_collect_after_offload:
                gc_collect()
            if self.args.offload_model:
                self.load_model(self.accelerator.unwrap_model(self.model))
                if self.ref_model:
                    self.load_model(self.ref_model)
            if self.args.offload_optimizer:
                self.load_optimizer()
        return inputs, outputs

    def _generate_completions(self, inputs: InputsType) -> InputsType:
        """Generate completions for given inputs using either fast inference or standard PyTorch inference.

        Args:
            inputs: List of input examples containing conversation messages.

        Returns:
            Modified inputs with generated completions added to the last message
            and truncation flag set in 'is_truncated' field.
        """
        mode = 'train' if self.model.training else 'eval'
        if self.use_fast_infer:
            inputs, outputs = self._fast_infer(inputs)
        else:
            with unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ), self.template.generate_context(), self.multi_turn_completion_length_context():
                outputs = self._infer_single_or_multi_turn(inputs, self.request_config)
                if mode == 'train':
                    # In training mode, ensure the model is returned to train() mode after inference
                    # This is necessary as pt engines set the model to eval mode during generation
                    self.model.train()

        for i, output in enumerate(outputs):
            inputs[i]['messages'] = output[0]
            inputs[i]['is_truncated'] = output[1] == 'length'

        return inputs

    def _generate_and_score_completions(self, inputs: InputsType) -> InputsType:

        inputs = self._generate_completions(inputs)
        total_rewards_per_func, total_rewards, completions = self._score_completions(inputs)
        mode = 'train' if self.model.training else 'eval'

        if self.args.dynamic_sample and mode == 'train':
            # dynamic sampling for std=0 groups
            inputs, total_rewards, total_rewards_per_func, completions = \
                self._dynamic_sampling(inputs, total_rewards, total_rewards_per_func, completions)

        # Prepare final outputs with advantages and other required fields
        batch_encoded_inputs = self._prepare_batch_inputs(inputs, total_rewards)
        # Log metrics
        messages = [inputs[i]['messages'][:-1] for i in range(len(inputs))]

        self._log_metrics(batch_encoded_inputs, messages, completions, total_rewards, total_rewards_per_func)

        return batch_encoded_inputs

    def _score_completions(self, inputs: InputsType) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Score completions using all reward functions

        Args:
            inputs: List of input examples, each containing a 'messages' list with conversation history

        Returns:
            Tuple containing:
            - rewards_per_func: Tensor of shape (num_examples, num_reward_funcs) with individual rewards
            - total_rewards: Tensor of shape (num_examples,) with weighted sum of rewards
            - completions: List of generated completion strings
        """
        device = self.accelerator.device
        completions = [example['messages'][-1]['content'] for example in inputs]
        rewards_per_func = torch.zeros((len(inputs), len(self.reward_funcs)), device=device)

        for i, (reward_func, reward_model_plugin, reward_func_name) in enumerate(
                zip(self.reward_funcs, self.reward_model_plugins, self.reward_func_names)):
            with profiling_context(self, reward_func_name):
                # reward model
                if isinstance(reward_func, nn.Module):
                    output_reward_func = reward_model_plugin(inputs=inputs)
                # reward function
                else:
                    # Repeat all input columns (but "messages" and "completion") to match the number of generations
                    reward_kwargs = RowPreprocessor.rows_to_batched(inputs)
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

        total_rewards_per_func = gather(rewards_per_func)
        total_rewards = (total_rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        return total_rewards_per_func, total_rewards, completions

    def _dynamic_sampling(self, inputs, rewards, rewards_per_func, completions):
        # DAPO https://arxiv.org/abs/2503.14476
        # Replaces samples with zero-reward-variance groups (std=0)
        resample_count = 0
        valid_samples = []
        valid_rewards = []
        valid_rewards_per_func = []
        valid_completions = []

        origin_data = (inputs, rewards, rewards_per_func, completions)

        while resample_count < self.args.max_resample_times:
            grouped_rewards = rewards.view(-1, self.num_generations)
            group_std = grouped_rewards.std(dim=1)

            valid_mask = (group_std > 0).repeat_interleave(self.num_generations)
            all_inputs = gather_object(inputs)
            valid_samples.extend([inp for inp, mask in zip(all_inputs, valid_mask) if mask])
            valid_rewards.append(rewards[valid_mask])
            valid_rewards_per_func.append(rewards_per_func[valid_mask])
            valid_completions.extend(
                [inp['messages'][-1]['content'] for inp, mask in zip(all_inputs, valid_mask) if mask])

            if len(valid_samples) >= self.args.generation_batch_size:
                break

            inputs = next(self.resample_iterator)
            inputs = Trainer._prepare_inputs(self, inputs)
            inputs = self._generate_completions(inputs)
            rewards_per_func, rewards, completions = self._score_completions(inputs)
            resample_count += 1

        if len(valid_samples) >= self.args.generation_batch_size:
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )
            inputs = valid_samples[:self.args.generation_batch_size][process_slice]
            rewards = torch.cat(valid_rewards)[:self.args.generation_batch_size]
            rewards_per_func = torch.cat(valid_rewards_per_func)[:self.args.generation_batch_size]
            completions = valid_completions[:self.args.generation_batch_size][process_slice]
        else:
            logger.warning(f'There are still std=0 groups present after {self.args.max_resample_times} retries.')
            inputs, rewards, rewards_per_func, completions = origin_data

        return inputs, rewards, rewards_per_func, completions

    def split_by_mini_batches(self, inputs, advantages):
        # Slice to keep only the local part of the data
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(inputs),
            (self.accelerator.process_index + 1) * len(inputs),
        )
        advantages = advantages[process_slice]

        mode = 'train' if self.model.training else 'eval'
        bs = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size
        spg = self.args.steps_per_generation if mode == 'train' else 1

        assert len(inputs) == bs * spg, f'Expected {bs * spg} inputs, got {len(inputs)}'
        spg_chunks = [inputs[i * bs:(i + 1) * bs] for i in range(spg)]
        # Split advantages by spg chunks
        advantage_chunks = torch.chunk(advantages, spg)
        return spg_chunks, advantage_chunks

    def _prepare_batch_inputs(self, inputs: InputsType, rewards: torch.Tensor) -> List[InputsType]:
        """
        Prepare the final batch inputs with advantages, ref/old_policy logps and other fields for RL training.

        Args:
            inputs (InputsType): List of input samples. Original shape is [spg*bs] where:
                - spg: steps_per_generation
                - bs: per-device batch size
            rewards (torch.Tensor): Tensor of rewards corresponding to the inputs.
                Shape should match the total number of samples (spg*bs*num_generations)

        Returns:
            List[InputsType]: A list of prepared batch inputs, organized as [spg][bs]
        """
        # Compute advantages
        grouped_rewards = rewards.view(-1, self.num_generations)
        mean_grouped_rewards = grouped_rewards.mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = grouped_rewards.std(dim=1).repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards)
        if self.args.scale_rewards:
            advantages /= (std_grouped_rewards + 1e-4)
        template = self.template

        gas_chunks, advantage_chunks = self.split_by_mini_batches(inputs, advantages)
        ga_batch_encoded_inputs = []
        for i, (batch, batch_advantages) in enumerate(zip(gas_chunks, advantage_chunks)):
            # Encode and process each batch (size=bs)
            with self._template_context(template):
                batch_encoded_inputs = [template.encode(infer_request) for infer_request in batch]
                batch_encoded_inputs = to_device(template.data_collator(batch_encoded_inputs), self.model.device)

            # Process labels and masks
            labels = batch_encoded_inputs.pop('labels')
            logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
            batch_encoded_inputs.update({
                'completion_mask':
                labels[:, -logits_to_keep:] != -100,
                'truncated_mask':
                torch.tensor([b['is_truncated'] for b in batch], dtype=torch.bool),
                'logits_to_keep':
                logits_to_keep,
                'advantages':
                batch_advantages
            })

            with torch.no_grad():
                batch_encoded_inputs['old_per_token_logps'] = (
                    self._get_per_token_logps(self.model, batch_encoded_inputs) if self.old_policy() else None)

            ga_batch_encoded_inputs.append(batch_encoded_inputs)

        return ga_batch_encoded_inputs

    def _log_metrics(self, inputs, messages, completions, rewards, rewards_per_func):
        """Log training/evaluation metrics"""
        mode = 'train' if self.model.training else 'eval'
        device = self.accelerator.device

        # Calculate completion length metrics
        agg_completion_mask = gather(torch.cat([inp['completion_mask'].sum(1) for inp in inputs]))

        self._metrics[mode]['completions/mean_length'].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]['completions/min_length'].append(agg_completion_mask.float().min().item())
        self._metrics[mode]['completions/max_length'].append(agg_completion_mask.float().max().item())
        # Calculate clip ratio
        agg_truncated_mask = gather(torch.cat([inp['truncated_mask'] for inp in inputs]).to(device))

        term_completion_mask = agg_completion_mask[agg_truncated_mask]
        clipped_completions_ratio = len(term_completion_mask) / len(agg_completion_mask)

        self._metrics[mode]['completions/clipped_ratio'].append(clipped_completions_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f'rewards/{reward_func_name}/mean'].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f'rewards/{reward_func_name}/std'].append(std_rewards)

        # Log overall reward stats
        grouped_rewards = rewards.view(-1, self.num_generations)
        self._metrics[mode]['reward'].append(grouped_rewards.mean().item())
        self._metrics[mode]['reward_std'].append(grouped_rewards.std(dim=1).mean().item())

        # Log prompt and completion texts
        self._textual_logs['prompt'].extend(self._apply_chat_template_to_messages_list(gather_object(messages)))
        self._textual_logs['completion'].extend(gather_object(completions))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs['rewards'][name].extend(rewards_per_func[:, i].tolist())

    def _apply_chat_template_to_messages_list(self, messages_list: InputsType):
        prompts_text = []
        for messages in messages_list:
            InferRequest.remove_response(messages)
            template_inputs, _ = StdTemplateInputs.from_dict({'messages': messages})
            res_context_list, _, _ = self.template._swift_encode(template_inputs)
            prompts_text.append(''.join(res_context_list))
        return prompts_text

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Compute the per-token log probabilities for the model, return_outputs=True in mini-batch training
        if isinstance(inputs, list):
            assert len(inputs) == 1
            inputs = inputs[0]
        if self.use_liger_loss:
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        completion_mask = inputs['completion_mask']
        truncated_mask = inputs['truncated_mask']
        # apply the completion_mask to exclude loss and metrics for overlong completions
        if self.args.overlong_filter and any(truncated_mask):
            if all(truncated_mask):
                logger.info('All completions are overlong and truncated, '
                            'resulting in NaN some values for some metrics (e.g., KL)')
            truncated_mask = truncated_mask.unsqueeze(-1).expand_as(completion_mask).to(completion_mask.device)
            completion_mask = completion_mask * (~truncated_mask)

        per_token_logps = self._get_per_token_logps(model, inputs)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, inputs)
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(self.model, inputs)

            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)

        advantages = inputs['advantages']
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs['old_per_token_logps'] is None else inputs['old_per_token_logps'])

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == 'grpo':
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == 'bnpo':
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == 'dr_grpo':
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        # Log the metrics
        mode = 'train' if self.model.training else 'eval'

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]['kl'].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]['clip_ratio/low_mean'].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]['clip_ratio/low_min'].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]['clip_ratio/high_mean'].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]['clip_ratio/high_max'].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]['clip_ratio/region_mean'].append(gathered_clip_ratio.nanmean().item())

        return loss

    @contextmanager
    def padding_free_context(self, model: torch.nn.Module):
        ctx = {}

        def _padding_free_input_hook(module, args, kwargs):
            attention_mask = kwargs['attention_mask']
            # used in _padding_free_output_hook
            ctx['padding_left'] = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if 'input_ids' in kwargs and kwargs.get('input_ids') is not None:
                # llm models
                kwargs['position_ids'] = torch.arange(kwargs['input_ids'].shape[1]).unsqueeze(0).repeat(
                    kwargs['input_ids'].shape[0], 1).to(kwargs['input_ids'].dtype).to(kwargs['input_ids'].device)
                kwargs['input_ids'] = kwargs['input_ids'][attention_mask.bool()].unsqueeze(0)
            else:
                # mllm models
                kwargs['position_ids'] = torch.arange(kwargs['inputs_embeds'].shape[1]).unsqueeze(0).repeat(
                    kwargs['inputs_embeds'].shape[0], 1).to(torch.int64).to(kwargs['inputs_embeds'].device)
                kwargs['inputs_embeds'] = kwargs['inputs_embeds'][attention_mask.bool()].unsqueeze(0)
            kwargs['position_ids'] = kwargs['position_ids'][attention_mask.bool()].unsqueeze(0)
            kwargs.pop('attention_mask', None)
            return args, kwargs

        def _padding_free_output_hook(module, args, kwargs, result):
            position_ids = kwargs['position_ids']
            seq_lengths = []
            pos = position_ids[0]
            resets = torch.where(pos[1:] < pos[:-1])[0] + 1

            if len(resets) == 0:
                # Only one sequence in this batch item
                seq_lengths = [pos.max().item() + 1]
            else:
                # Multiple sequences
                start = 0
                for end in resets:
                    seq_lengths.append(end - start)
                    start = end
                seq_lengths.append(pos.shape[0] - start)

            max_length = max(seq_lengths)
            last_hidden_state = result.last_hidden_state.squeeze(0)
            unpacked_logits = []

            start = 0
            for length in seq_lengths:
                seq_state = last_hidden_state[start:start + length]
                padding = torch.zeros(
                    (max_length - length,
                     last_hidden_state.shape[-1])).to(last_hidden_state.dtype).to(last_hidden_state.device)
                # re-padding
                if ctx['padding_left']:
                    seq_state = torch.cat((padding, seq_state), dim=0)
                else:
                    seq_state = torch.cat((seq_state, padding), dim=0)
                unpacked_logits.append(seq_state)
                start += length
            result.last_hidden_state = torch.stack(unpacked_logits, dim=0)
            return result

        if self.padding_free:
            llm_model = get_llm_model(model)
            if hasattr(llm_model, 'thinker'):
                base_model = llm_model.thinker.model
            else:
                base_model = llm_model.model
            remove_handle1 = base_model.register_forward_pre_hook(
                _padding_free_input_hook, with_kwargs=True, prepend=True)
            remove_handle2 = base_model.register_forward_hook(_padding_free_output_hook, with_kwargs=True, prepend=True)
        yield
        if self.padding_free:
            remove_handle1.remove()
            remove_handle2.remove()

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, inputs):
        from trl.trainer.utils import selective_log_softmax
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        unwrapped_model = self.accelerator.unwrap_model(model)
        if is_peft_model(unwrapped_model):
            parameters = inspect.signature(unwrapped_model.base_model.model.forward).parameters
        else:
            parameters = inspect.signature(unwrapped_model.forward).parameters
        if not unwrapped_model.model_meta.is_multimodal and 'logits_to_keep' in parameters:
            # save memory
            with self.padding_free_context(model):
                return super()._get_per_token_logps(model, input_ids, inputs['attention_mask'], logits_to_keep)
        inputs = {
            k: v
            for k, v in inputs.items() if k not in [
                'logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps',
                'truncated_mask'
            ]
        }

        with self._template_context(self.template), self.padding_free_context(model):
            logits = model(**inputs).logits
        # exclude the last logit: it corresponds to the next token pred
        logits = logits[:, -(logits_to_keep + 1):-1, :]
        logits = logits / self.temperature
        input_ids = input_ids[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    @profiling_decorator
    def _get_last_hidden_state(self, unwrapped_model, inputs, logits_to_keep):
        # unwrap the model to access the model.model
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        if not unwrapped_model.model_meta.is_multimodal:
            last_hidden_state = unwrapped_model.model(
                input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state
        else:
            inputs = {
                k: v
                for k, v in inputs.items() if k not in [
                    'logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps',
                    'truncated_mask'
                ]
            }
            with self._template_context(self.template):
                outputs = unwrapped_model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]

        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        input_ids = inputs['input_ids']
        logits_to_keep = inputs['logits_to_keep']
        completion_ids = input_ids[:, -logits_to_keep:]
        completion_mask = inputs['completion_mask']

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = None
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, inputs)
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(self.model, inputs)

        # get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(unwrapped_model, inputs, logits_to_keep)
        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs['advantages'],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs['old_per_token_logps'],
            ref_per_token_logps=ref_per_token_logps,
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = 'eval' if self.control.should_evaluate else 'train'
        if self.beta != 0.0:
            self._metrics[mode]['kl'].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics[mode]['clip_ratio'].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    def evaluation_loop(self, dataloader, *args, **kwargs):
        # Wait for the training rollout to complete
        if self.args.async_generate:
            while not self.is_async_generate_train_rollout_done():
                time.sleep(0.1)
        if self._queue.empty() and self.args.async_generate:
            self._prefetch(dataloader)
        metric_key_prefix = kwargs['metric_key_prefix']
        output = super().evaluation_loop(dataloader, *args, **kwargs)
        metrics = {f'{metric_key_prefix}_{key}': sum(val) / len(val) for key, val in self._metrics['eval'].items()}
        output.metrics.update(metrics)
        self.eval_flag = True
        return output

    def training_step(self, model: nn.Module, inputs: InputsType, num_items_in_batch=None) -> torch.Tensor:
        if self.args.async_generate:
            # Wait for the eval rollout to complete
            while not self.is_async_generate_eval_rollout_done():
                time.sleep(0.1)
        return super().training_step(model, inputs, num_items_in_batch)

    def _engine_infer(
        self,
        infer_requests: InputsType,
        request_config: Optional[RequestConfig] = None,
        *,
        use_tqdm: Optional[bool] = False,
    ) -> List[ChatCompletionResponse]:
        with profiling_context(self, 'generate'):
            if self.vllm_mode == 'server':
                request_keys = ['messages', 'images', 'audios', 'videos', 'tools', 'objects']

                infer_requests = [{
                    **{k: request[k]
                       for k in request_keys if k in request},
                    **({
                        'data_dict': {k: request[k]
                                      for k in request if k not in request_keys}
                    } if self.multi_turn_scheduler and self.vllm_use_async_engine else {})
                } for request in infer_requests]

                self._process_infer_requests_images(infer_requests)
                return self.vllm_client.infer(infer_requests, asdict(request_config), use_tqdm=use_tqdm)
            else:
                return self.engine.infer(infer_requests, request_config, use_tqdm=use_tqdm)

    def _process_infer_requests_images(self, infer_requests: InputsType):
        # Process image format into a format that session.post can accept
        import base64
        if not any('images' in request for request in infer_requests):
            return
        for request in infer_requests:
            if 'images' not in request:
                continue
            for i, img in enumerate(request['images']):
                if 'bytes' in img and img['bytes']:
                    request['images'][i] = base64.b64encode(img['bytes']).decode('utf-8')
                elif 'path' in img and img['path']:
                    request['images'][i] = img['path']
        return

    def old_policy(self):
        return self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps

    @property
    def _queue(self):
        if self.control.should_evaluate:
            return self.eval_queue
        else:
            return self.train_queue

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
    def multi_turn_completion_length_context(self):
        """
        Context manager that temporarily adjusts the engine's max length handling
        for multi-turn generation scenarios.

        Ensures the total sequence length (prompt + completion) never exceeds:
            min(original_max_len, prompt_tokens + max_completion_length)
        """
        if not (self.multi_turn_scheduler and
                self.use_fast_infer) or self.vllm_mode == 'server' or self.completion_length_limit_scope == 'per_round':
            yield
            return

        original_fn = self.engine.set_default_max_tokens
        original_max_len = self.engine.max_model_len

        def set_default_max_tokens(_self, request_config: RequestConfig, inputs: InputsType) -> None:
            # Calculate required context window
            original_max_len = _self.max_model_len or 8192
            if isinstance(inputs, dict):
                inputs = [inputs]
            prompt_tokens = max(_self._get_num_tokens(inp) for inp in inputs)

            if not hasattr(_self, 'set_grpo_max_model_len'):
                # set max model len in first round
                max_len = min(original_max_len, prompt_tokens + request_config.max_tokens)
                _self.max_model_len = max_len
                _self.set_grpo_max_model_len = True
            else:
                if _self.max_model_len <= prompt_tokens:
                    # modify max_model_len > prompt_tokens to avoid crash
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

    def get_resample_dataloader(self) -> DataLoader:
        resample_dataset = self.resample_dataset
        data_collator = self.data_collator
        if isinstance(resample_dataset, datasets.Dataset):
            resample_dataset = self._remove_unused_columns(resample_dataset, description='training')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='training')

        dataloader_params = {
            'batch_size': self._train_batch_size * self.args.steps_per_generation,
            'collate_fn': data_collator,
            'num_workers': self.args.dataloader_num_workers,
            'pin_memory': self.args.dataloader_pin_memory,
            'persistent_workers': self.args.dataloader_persistent_workers,
        }

        @contextmanager
        def seed_context(self):
            seed = self.args.seed
            self.args.seed = seed + 1
            yield
            self.args.seed = seed

        if not isinstance(resample_dataset, torch.utils.data.IterableDataset):
            with seed_context(self):  # Set a different seed for resampling than the train_dataset.
                dataloader_params['sampler'] = self._get_train_sampler()
            dataloader_params['drop_last'] = self.args.dataloader_drop_last
            dataloader_params['worker_init_fn'] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index)
            dataloader_params['prefetch_factor'] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(resample_dataset, **dataloader_params))

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = 'train' if self.model.training else 'eval'
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == 'eval':
            metrics = {f'eval_{key}': val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse('4.47.0.dev0'):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            table = {
                'step': [str(self.state.global_step)] * len(self._textual_logs['prompt']),
                'prompt': self._textual_logs['prompt'],
                'completion': self._textual_logs['completion'],
                **self._textual_logs['rewards'],
            }
            self.jsonl_writer.append(table)
            if self.args.report_to and 'wandb' in self.args.report_to and wandb.run is not None:
                import pandas as pd
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                wandb.log({'completions': wandb.Table(dataframe=df)})

    def is_async_generate_eval_rollout_done(self):
        return not self.eval_flag or not self.eval_queue.empty()

    def is_async_generate_train_rollout_done(self):
        return not self.train_queue.empty()

    def inputs_to_rolloutrequest(self, inputs: InputsType) -> RolloutInferRequest:

        request_keys = ['messages', 'images', 'audios', 'videos', 'tools', 'objects']
        infer_requests = [
            RolloutInferRequest(
                **{
                    **{k: request[k]
                       for k in request_keys if k in request}, 'data_dict':
                    {k: request[k]
                     for k in request if k not in request_keys}
                }) for request in inputs
        ]

        return infer_requests
