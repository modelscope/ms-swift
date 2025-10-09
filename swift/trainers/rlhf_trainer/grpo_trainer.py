# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/trl.
import base64
import concurrent.futures
import inspect
import os
import re
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from math import ceil
from queue import Queue
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import json
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from dacite import from_dict
from packaging import version
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TrainerCallback
from transformers.trainer import Trainer
from trl import GRPOTrainer as HFGRPOTrainer
from trl.models import prepare_deepspeed
from trl.trainer import grpo_trainer
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_trainer import RepeatSampler, nanmax, nanmin, nanstd
from trl.trainer.utils import selective_log_softmax

from swift.llm import (InferRequest, MultiModelKeys, RequestConfig, RolloutInferRequest, RowPreprocessor, Template,
                       to_device)
from swift.llm.infer.protocol import ChatCompletionResponse, RolloutOutput
from swift.llm.model.utils import get_llm_model
from swift.llm.template.template_inputs import TemplateInputs
from swift.plugin import multi_turns, orms, rm_plugins
from swift.plugin.multi_turn import MultiTurnScheduler
from swift.utils import (JsonlWriter, empty_cache, get_current_device, get_logger, is_swanlab_available,
                         is_vllm_available, is_wandb_available, remove_response, seed_worker,
                         unwrap_model_for_generation)
from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin
from .utils import (_ForwardRedirection, compute_chord_loss, identity_data_collator, load_pil_img,
                    make_chord_sft_dataset, patch_lora_merge, patch_lora_unmerge, patch_profiling_context,
                    patch_profiling_decorator, patch_save_last_checkpoint, replace_assistant_response_with_ids,
                    set_expandable_segments)
from .vllm_client import VLLMClient

try:
    from trl.trainer.utils import entropy_from_logits
except ImportError:
    from .utils import entropy_from_logits

del HFGRPOTrainer.__init__
del HFGRPOTrainer.log
grpo_trainer.seed_worker = seed_worker  # fix transformers 4.51.3

logger = get_logger()
if is_wandb_available():
    import wandb
if is_swanlab_available():
    import swanlab

DataType = List[Dict[str, Union[torch.Tensor, Any]]]
T = TypeVar('T')


class AsyncGenerateCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer

    # offload original_modules to cpu, to save memory
    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.queue = self.trainer.train_queue
        train_dataloader = getattr(state, 'train_dataloader', None) or kwargs.get('train_dataloader')
        self.trainer._prefetch(train_dataloader)


@dataclass
class DataCache:
    results: DataType


class GRPOTrainer(RLHFTrainerMixin, SwiftMixin, HFGRPOTrainer):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_model: Optional[List[Union[PreTrainedModel, nn.Module]]] = None,
                 reward_funcs: Optional[List[Union[str, Callable]]] = None,
                 *_args,
                 **kwargs):
        patch_save_last_checkpoint()
        from swift.trainers.rlhf_arguments import GRPOConfig
        args: GRPOConfig = kwargs['args']
        self.args = args
        self.ref_adapter_name = getattr(args, 'ref_adapter_name', None)
        self.model_adapter_name = None
        # for async generate
        self.train_queue = Queue()
        self.eval_queue = Queue()
        self.is_multimodal = model.model_meta.is_multimodal
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
                    raise ValueError(f'reward_function {reward_func} is not implemented in swift.plugin')

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

        self.multi_turn_scheduler = None
        if self.args.multi_turn_scheduler:
            if isinstance(self.args.multi_turn_scheduler, str):
                assert self.args.multi_turn_scheduler in multi_turns
                multi_turn_scheduler = multi_turns[self.args.multi_turn_scheduler](max_turns=self.args.max_turns)
                self.multi_turn_scheduler: MultiTurnScheduler = multi_turn_scheduler
            else:
                assert isinstance(self.args.multi_turn_scheduler, MultiTurnScheduler)
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

        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys() if not hasattr(model, 'get_base_model') else
            inspect.signature(model.get_base_model().forward).parameters.keys())
        chord_sft_dataset = kwargs.pop('chord_sft_dataset', None)
        super().__init__(model, ref_model, *_args, **kwargs)
        self.chord_sft_iterator = None
        if chord_sft_dataset:
            self.chord_sft_iterator = make_chord_sft_dataset(self, chord_sft_dataset)
        if self.args.eval_strategy != 'no':
            total_eval_batch_size = self.args.per_device_eval_batch_size * \
                self.accelerator.num_processes // self.args.num_generations
            assert len(self.eval_dataset) >= total_eval_batch_size, (
                f'eval_dataset size {len(self.eval_dataset)} is smaller than '
                f'total_eval_batch_size {total_eval_batch_size}. '
                f'Please increase the size of eval_dataset or set a larger value for split_dataset_ratio.')
        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper

        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        self.top_entropy_quantile = args.top_entropy_quantile
        self.importance_sampling_level = args.importance_sampling_level

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
        self._logs = {
            'prompt': deque(maxlen=args.generation_batch_size),
            'completion': deque(maxlen=args.generation_batch_size),
            'rewards': defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            'advantages': deque(maxlen=args.generation_batch_size),
        }
        self.compute_entropy = self.args.log_entropy or self.top_entropy_quantile < 1.0
        if self.args.log_entropy:
            self._logs.update({'entropy': deque(maxlen=args.generation_batch_size)})
        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        if is_peft_model(self.model):
            self.parameter_groups, self.parameter_groups_no_lora = self.split_batches()
        self.use_fast_infer = self.use_vllm  # whether to use the PT backend
        self.vllm_use_async_engine = False
        self.enable_offload = False
        self.use_gym_env = False
        self.enable_server_multi_turn = False
        # for multi-turn server, maybe the num of rollout outputs is not equal to the num of rollout inputs
        self.dynamic_num_samples = False
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                                  'Please install vLLM with `pip install vllm -U` to use it.')
            if self.vllm_mode == 'server':
                self.vllm_client: VLLMClient = vllm_client
                if self.accelerator.is_main_process:
                    self.vllm_client.get_engine_type()
                    vllm_use_async_engine = [self.vllm_client.use_async_engine]
                    use_gym_env = [self.vllm_client.use_gym_env]
                    enable_multi_turn = [self.vllm_client.enable_multi_turn]
                else:
                    vllm_use_async_engine = [False]
                    use_gym_env = [False]
                    enable_multi_turn = [self.enable_server_multi_turn]
                self.vllm_use_async_engine = broadcast_object_list(vllm_use_async_engine, from_process=0)[0]
                self.use_gym_env = broadcast_object_list(use_gym_env, from_process=0)[0]
                self.enable_server_multi_turn = broadcast_object_list(enable_multi_turn, from_process=0)[0]
                if self.use_gym_env:
                    self.reward_func_names = ['gym_reward']

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
                self.enable_offload = self.args.offload_model or self.args.offload_optimizer
                context = self.offload_context if self.enable_offload else nullcontext

                with context():
                    self.engine = self.prepare_vllm(model)
                    if self.args.sleep_level > 0:
                        self.engine.engine.sleep(self.args.sleep_level)

        else:
            from swift.llm import PtEngine
            infer_template = copy(self.template)
            infer_template.padding_free = False
            self.engine = PtEngine.from_model_template(self.model, infer_template, max_batch_size=0)  # 0: no limit

        if not self.reward_funcs and not self.use_gym_env:
            raise ValueError('You must specify reward_funcs or reward_model')

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(f'Number of reward weights ({len(args.reward_weights)}) must match number of reward '
                                 f'functions ({len(reward_funcs)})')
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32).to(self.accelerator.device)
        else:
            self.reward_weights = torch.ones(
                len(self.reward_func_names), dtype=torch.float32).to(self.accelerator.device)
        self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation
        self.request_config = RequestConfig(
            n=1,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop_words,
            return_details=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
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
            self.add_callback(AsyncGenerateCallback(self))

        if self.args.dynamic_sample or self.template.truncation_strategy == 'raise':

            def cyclic_iter(iterable):
                while True:
                    for x in iterable:
                        yield x

            @contextmanager
            def seed_context():
                # Use a different seed to ensure the resample dataset does not overlap with train_dataset
                seed = self.args.seed
                self.args.seed = seed + 1
                yield
                self.args.seed = seed

            with seed_context():
                if self.args.dynamic_sample:
                    self.dynamic_resample_iterator = cyclic_iter(self.get_train_dataloader())

                if self.template.truncation_strategy == 'raise':

                    @contextmanager
                    def single_sample_context():
                        # Patch generation-related parameters to ensure that only one sample is processed per iteration
                        # when resampling truncated data.
                        origin_ng = self.num_generations
                        origin_gbs = self.args.generation_batch_size
                        origin_spg = self.args.steps_per_generation
                        try:
                            self.num_generations = 1
                            self.args.generation_batch_size = 1
                            self.args.steps_per_generation = 1
                            yield
                        finally:
                            self.num_generations = origin_ng
                            self.args.generation_batch_size = origin_gbs
                            self.args.steps_per_generation = origin_spg

                    with single_sample_context():
                        self.truncated_resample_iterator = cyclic_iter(self.get_train_dataloader())
        # flag indicating whether the evaluation has started
        self.eval_flag = False
        # Record the number of samples that need to be padded for even distribution across processes
        self.rollout_pad_count = 0

        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            self.args.gradient_accumulation_steps = self.args.gradient_accumulation_steps * sequence_parallel.world_size

    def _get_train_sampler(self, train_dataset=None):
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            return RepeatSampler(
                data_source=train_dataset or self.train_dataset,
                mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size // self.num_generations,
                repeat_count=self.num_iterations * self.args.steps_per_generation * sequence_parallel.world_size,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
            )
        else:
            return super()._get_train_sampler(train_dataset)

    @patch_profiling_decorator
    def _prepare_inputs(self, generation_batch: Dict[str, Union[torch.Tensor,
                                                                Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
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
            num_rollout_samples = self.args.steps_per_generation * self.template.sequence_parallel_size
            generate_every = num_rollout_samples * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(generation_batch)
                self._buffered_inputs = generation_batch  # < this is the change
            inputs = self._buffered_inputs[self._step % num_rollout_samples]
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

        model_arch = model.model_meta.model_arch
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
        vllm_template = copy(self.template)
        vllm_template.padding_free = False
        with Swift.grpo_context(model, self.template.processor):
            set_expandable_segments(False)
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
                max_model_len=self.args.vllm_max_model_len,
                seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                disable_cascade_attn=self.args.vllm_disable_cascade_attn,
                load_format='dummy',
                template=vllm_template,
                distributed_executor_backend='external_launcher',
            )
            set_expandable_segments(True)
        return engine

    @contextmanager
    def _template_context(self, template: Template):
        # The max_length for prompt and completion has already been restricted, so there is no need for max_length here.
        max_length = template.max_length
        template.max_length = None
        try:
            yield
        finally:
            template.max_length = max_length

    @patch_profiling_decorator
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
                    del state_dict
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

    def _rollout(self,
                 inputs: Optional[DataType],
                 request_config: RequestConfig,
                 is_global_inputs: bool = False) -> List[RolloutOutput]:
        request_config = self._get_request_config()
        if self.vllm_mode == 'server':
            rollout_outputs = self._server_rollout(inputs, request_config, is_global_inputs)
        else:
            rollout_outputs = self._colocate_rollout(inputs, request_config)
        return rollout_outputs

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

    def _set_inputs_system(self, inputs: DataType) -> DataType:
        """
        Inserts a default system message at the beginning of each input if specified.

        If a default system message is defined in the template and the first message in
        an input is not already a system message, this method inserts the default system
        message at the beginning of the messages list for each input. If no default system
        message is provided, no modification is made.

        Args:
            inputs (DataType): A list of input data entries, each containing a 'messages' field.

        Returns:
            DataType: The input list, with the default system message prepended if applicable.
        """
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
        """
        Runs inference for either single-turn or multi-turn dialogue.

        Args:
            inputs: Input data for inference.
            request_config: Configuration for the inference request.
            is_global_inputs: Whether the inputs are from the global process.

        Returns:
            List of processed outputs.
        """
        # for external server, pass the system args which may define in trainer

        # Step 1: Prepare inputs with system prompts (if any)
        inputs = self._set_inputs_system(inputs)

        # Step 2: First-turn rollout
        rollout_outputs: List[RolloutOutput] = self._rollout(inputs, request_config, is_global_inputs)

        # Step 3: Handle single-turn or server multi-turn
        if not self.multi_turn_scheduler or self.enable_server_multi_turn:
            return self._postprocess_rollout_outputs(inputs, rollout_outputs)

        # Step 4: Handle multi-turn colocate
        return self._colocate_multi_turn_infer(inputs, rollout_outputs, request_config)

    def async_generate_rollout(self, all_inputs):
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
                current_queue.put(DataCache(result))
            except Exception as e:
                logger.error('Error in async_generate_rollout callback: %s', str(e))

        future.add_done_callback(done)

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

    def _fast_infer(self, inputs: DataType) -> DataType:
        """
        Efficient inference logic with support for vLLM colocate mode, async generation,
        and model weight offloading.
        """

        # Step 1: Wake up the engine if it's sleeping (vLLM colocate mode)
        if self.vllm_mode == 'colocate' and self.args.sleep_level > 0:
            if self.engine.inner_model_executor.is_sleeping:
                wake_up_params = inspect.signature(self.engine.engine.wake_up).parameters
                # Load weights only (faster and reduces memory peak)
                kwargs = {'tags': ['weights']} if 'tags' in wake_up_params else {}
                self.engine.engine.wake_up(**kwargs)

        # Step 2: Load model weights if global_step has changed
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Step 3: Offload model/optimizer if enabled
        context = self.offload_context if self.enable_offload else nullcontext
        with context():
            set_expandable_segments(False)
            # Step 4: Wake up kv_cache after offloading (vLLM colocate only)
            if (self.vllm_mode == 'colocate' and self.engine.inner_model_executor.is_sleeping
                    and 'tags' in inspect.signature(self.engine.engine.wake_up).parameters):
                # Load the kv_cache only after updating and offload the weights.
                self.engine.engine.wake_up(tags=['kv_cache'])

            # Step 5: Handle rollout for async generate or sync
            if self.async_generate:
                # Pre-gather inputs to avoid potential gather deadlocks
                all_inputs = gather_object(inputs)
                self.async_generate_rollout(all_inputs)

                # Retrieve cached outputs from the last step
                data_cache: DataCache = self._queue.get()
                all_outputs = gather_object(data_cache.results)

                # Slice inputs/outputs for the current process
                per_device_datasize = len(all_outputs) // self.accelerator.num_processes
                process_slice = slice(
                    self.accelerator.process_index * per_device_datasize,
                    (self.accelerator.process_index + 1) * per_device_datasize,
                )
                outputs = all_outputs[process_slice]

            else:
                with self.multi_turn_completion_length_context():
                    outputs = self._infer_single_or_multi_turn(inputs, self.request_config)

            # Step 6: Reset prefix cache and sleep to release memory
            if self.vllm_mode == 'colocate' and self.args.sleep_level > 0:
                # Reset prefix cache before sleeping to prevent using stale cache upon waking up
                # https://github.com/modelscope/ms-swift/pull/5143
                self.engine.engine.reset_prefix_cache()
                self.engine.engine.sleep(level=self.args.sleep_level)
                empty_cache()
            set_expandable_segments(True)
        return outputs

    def _generate_completions(self, inputs: DataType) -> DataType:
        # add prompt ids and system prompts
        inputs = self._preprocess_inputs(inputs)

        mode = 'train' if self.model.training else 'eval'
        if self.use_fast_infer:
            results = self._fast_infer(inputs)
        else:
            with unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ), self.template.generate_context(), self.multi_turn_completion_length_context():
                results = self._infer_single_or_multi_turn(inputs, self.request_config)
                if mode == 'train':
                    # In training mode, ensure the model is returned to train() mode after inference
                    # This is necessary as pt engines set the model to eval mode during generation
                    self.model.train()

        return results

    @patch_profiling_decorator
    def _generate_and_score_completions(self, inputs: DataType) -> DataType:
        # resample for encoding failed data when set truncation_strategy 'delete'
        if self.template.truncation_strategy == 'raise':
            inputs = self.resample_encode_failed_inputs(inputs)

        inputs = self._generate_completions(inputs)
        total_rewards_per_func = self._score_completions(inputs)
        mode = 'train' if self.model.training else 'eval'

        if self.args.dynamic_sample and mode == 'train':
            # dynamic sampling for std=0 groups
            inputs, total_rewards_per_func = self._dynamic_sampling(inputs, total_rewards_per_func)  # noqa
        total_advantages = self._compute_advantages(inputs, total_rewards_per_func)

        local_advantages = self.get_even_process_data(total_advantages)
        assert len(local_advantages) == len(inputs)
        for i, advantage in enumerate(local_advantages):
            inputs[i]['advantages'] = advantage
        # log metrics in inputs
        self._logs['advantages'].extend(total_advantages.tolist())

        batch_encoded_inputs = self._prepare_batch_inputs(inputs)

        with patch_profiling_context(self, 'log_metrics'):
            # --- logs (prompts + completions) ---
            messages = [inp['messages'][:-1] for inp in inputs]
            completions = [deepcopy(inp['messages'][-1]['content']) for inp in inputs]
            for i, completion in enumerate(completions):
                if isinstance(completion, str):
                    continue
                if isinstance(completion, list):
                    token_ids = completion
                elif isinstance(completion, dict):
                    token_ids = completion['token_ids']
                completions[i] = self.processing_class.decode(token_ids)
            valid_messages = self._gather_and_flatten(messages, flatten_level=0)
            valid_completions = self._gather_and_flatten(completions, flatten_level=0)
            self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(valid_messages))
            self._logs['completion'].extend(valid_completions)

            # Example: if you want to log extra data in the wandb / swanlab table,
            #          add them to metrics_to_gather
            # NOTE: every key you register must appear in ALL rollout outputs
            #       to avoid potential communication / synchronization issues
            metrics_for_logs_to_gather = {}
            if all('images' in data and data['images'] is not None for data in inputs):
                metrics_for_logs_to_gather['image'] = [inp['images'] for inp in inputs]

            if all('solution' in inp for inp in inputs):
                metrics_for_logs_to_gather['solution'] = [inp['solution'] for inp in inputs]

            if all('rollout_infos' in inp and 'num_turns' in inp['rollout_infos'] for inp in inputs):
                metrics_for_logs_to_gather['num_turns'] = [inp['rollout_infos']['num_turns'] for inp in inputs]

            if metrics_for_logs_to_gather:
                for key, value in metrics_for_logs_to_gather.items():
                    if key not in self._logs:
                        self._logs[key] = deque(maxlen=self.args.generation_batch_size)
                    self._logs[key].extend(self._gather_and_flatten(value, flatten_level=0))

        return batch_encoded_inputs

    @patch_profiling_decorator
    def _score_completions(self, inputs: DataType) -> torch.Tensor:
        """Score completions using all reward functions.

        Args:
            inputs: List of input examples, each containing a 'messages' list with conversation history

        Returns:
            rewards_per_func: Tensor of shape (num_examples, num_reward_funcs) with all reward values
        """
        device = self.accelerator.device
        # If using gym environment, extract rewards directly from inputs
        if self.use_gym_env:
            reward_from_gym = [inp['rollout_infos']['total_reward'] for inp in inputs]
            # For gym environment, there's only one total reward, so rewards_per_func is just local_rewards reshaped
            local_rewards_per_func = torch.tensor(
                reward_from_gym, dtype=torch.float32, device=device).unsqueeze(1)  # shape: [num_examples, 1]
        else:
            # Compute rewards using reward functions
            local_rewards_per_func = self._compute_rewards_per_func(inputs)

        # gather rewards
        if not self.dynamic_num_samples:
            total_rewards_per_func = gather(local_rewards_per_func)
        else:
            # gather_object to avoid shape mismatch
            local_rewards_list = [row.tolist() for row in local_rewards_per_func]
            total_rewards_per_func = gather_object(local_rewards_list)
            total_rewards_per_func = torch.tensor(
                total_rewards_per_func, dtype=torch.float32, device=self.accelerator.device)

        return total_rewards_per_func

    def _compute_rewards_per_func(self, inputs: DataType) -> torch.Tensor:
        """Compute rewards using all reward functions"""
        device = self.accelerator.device
        rewards_per_func = torch.zeros((len(inputs), len(self.reward_funcs)), device=device)
        completions = [inp['messages'][-1]['content'] for inp in inputs]
        for i, (reward_func, reward_model_plugin, reward_func_name) in enumerate(
                zip(self.reward_funcs, self.reward_model_plugins, self.reward_func_names)):
            with patch_profiling_context(self, reward_func_name):
                # reward model
                reward_kwargs = {'trainer_state': self.state}
                if self.enable_server_multi_turn:
                    trajectory_inputs = self._get_trajectory_inputs(inputs)
                    reward_kwargs.update({'trajectory_inputs': trajectory_inputs})
                if isinstance(reward_func, nn.Module):
                    output_reward_func = reward_model_plugin(inputs=inputs, **reward_kwargs)
                # reward function
                else:
                    # Repeat all input columns (but "messages" and "completion") to match the number of generations
                    reward_kwargs.update(RowPreprocessor.rows_to_batched(inputs))
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

    def _compute_advantages(self, inputs: DataType, rewards_per_func: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages for RL training.

        Supports two modes:
        1. **Default grouped mode** (no prompt_ids / request_ids provided)
        - Assumes rewards are grouped by prompt and each prompt has exactly
            `self.num_generations` completions.
        - Computes advantages relative to group mean.
        2. **Request-aware mode** (multi-turn conversations, variable number of samples)
        - Groups rewards by unique `request_id` and computes statistics per prompt.
        - Handles dynamic sample sizes where multiple request_ids may share the same prompt.

        Args:
            inputs (DataType):
                Input data samples.
            rewards_per_func (torch.Tensor):
                Reward values for each reward function, shape `(N, num_reward_funcs)`.

        Returns:
            **advantages** (torch.Tensor):
                Computed advantages, shape `(N,)`.
        """

        def normalize_advantages(advantages: torch.Tensor, rewards_std: torch.Tensor) -> torch.Tensor:
            """Normalize advantages if configured; otherwise, return as-is."""
            if self.args.scale_rewards:
                return advantages / (rewards_std + 1e-4)
            return advantages

        def log_rewards_metrics(rewards: torch.Tensor, rewards_per_func_for_metrics: torch.Tensor):
            """Log reward statistics for monitoring. Only log once per unique request_id."""
            # rewards: [prompt_batch_size, self.num_generations]
            # rewards_per_func_for_metrics: [prompt_batch_size*self.num_generations, self.num_reward_funcs]
            mode = 'train' if self.model.training else 'eval'
            group_rewards = rewards.view(-1, self.num_generations)
            rewards_mean = group_rewards.mean(-1).mean().item()
            rewards_std = group_rewards.std(-1).mean().item()
            is_std_zero = torch.isclose(group_rewards.std(dim=1), torch.zeros_like(group_rewards.std(dim=1)))

            self._metrics[mode]['reward'].append(rewards_mean)
            self._metrics[mode]['reward_std'].append(rewards_std)
            self._metrics[mode]['frac_reward_zero_std'].append(is_std_zero.float().mean().item())

            # Log per-reward-function statistics using deduplicated rewards_per_func
            for i, name in enumerate(self.reward_func_names):
                col = rewards_per_func_for_metrics[:, i]
                self._metrics[mode][f'rewards/{name}/mean'].append(torch.nanmean(col).item())
                self._metrics[mode][f'rewards/{name}/std'].append(nanstd(col).item())

        def log_rewards_all(rewards_per_func: torch.Tensor):
            """Log all rewards for debugging."""
            for i, name in enumerate(self.reward_func_names):
                self._logs['rewards'][name].extend(rewards_per_func[:, i].tolist())

        # Step 0. Aggregate final reward using reward weights
        device = self.accelerator.device
        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)

        # --------------------------------------------------
        # Case 1: Default grouped mode
        # --------------------------------------------------
        if not self.dynamic_num_samples:
            grouped_rewards = rewards.view(-1, self.num_generations)
            group_rewards_mean = grouped_rewards.mean(dim=1)
            group_rewards_std = grouped_rewards.std(dim=1)

            # Broadcast stats back to the original shape
            group_rewards_mean = group_rewards_mean.repeat_interleave(self.num_generations)
            group_rewards_std = group_rewards_std.repeat_interleave(self.num_generations)

            # Compute advantages relative to group mean
            advantages = rewards - group_rewards_mean
            advantages = normalize_advantages(advantages, group_rewards_std)

            # Log metrics once per group
            log_rewards_metrics(rewards=grouped_rewards, rewards_per_func_for_metrics=rewards_per_func)

            # Log all rewards
            log_rewards_all(rewards_per_func)

            return advantages

        # --------------------------------------------------
        # Case 2: Request-aware mode
        # --------------------------------------------------
        else:
            prompt_ids = gather_object([inp['prompt_id'] for inp in inputs])
            request_ids = gather_object([inp['request_id'] for inp in inputs])
            assert rewards.shape[0] == len(prompt_ids) == len(request_ids)
            device = self.accelerator.device

            # Step 1. Deduplicate request_ids
            unique_indices = self._get_last_indices(request_ids)
            unique_request_ids = [request_ids[i] for i in unique_indices.cpu()]
            unique_prompt_ids = [prompt_ids[i] for i in unique_indices.cpu()]

            # Step 2. Validate rewards consistency within the same request_id
            for rid in set(request_ids):
                idxs = [i for i, r in enumerate(request_ids) if r == rid]
                if not torch.allclose(rewards[idxs], rewards[idxs[0]].expand(len(idxs)), atol=1e-6):
                    raise ValueError(f'Inconsistent rewards detected for request_id={rid}.')

            # Step 3. Group rewards by prompt_id and compute prompt-level mean/std
            unique_rewards = rewards[unique_indices]
            prompt_to_indices = {}
            for idx, pid in enumerate(unique_prompt_ids):
                prompt_to_indices.setdefault(pid, []).append(idx)

            prompt_means = torch.zeros(len(unique_rewards), device=device)
            prompt_stds = torch.zeros(len(unique_rewards), device=device)
            for pid, idxs in prompt_to_indices.items():
                idx_tensor = torch.tensor(idxs, device=device)
                r_group = unique_rewards[idx_tensor]
                prompt_means[idx_tensor] = r_group.mean()
                prompt_stds[idx_tensor] = r_group.std()

            # Step 4. Compute advantages
            request_advantages = unique_rewards - prompt_means
            request_advantages = normalize_advantages(request_advantages, prompt_stds)

            # Map advantages back to original order
            rid_to_idx = {rid: idx for idx, rid in enumerate(unique_request_ids)}
            indices_in_unique = torch.tensor([rid_to_idx[r] for r in request_ids], device=device)
            advantages = request_advantages[indices_in_unique]

            # Step 5. Log metrics for unique request_ids
            log_rewards_metrics(rewards=unique_rewards, rewards_per_func_for_metrics=rewards_per_func[unique_indices])

            # Step 6. Log all rewards
            log_rewards_all(rewards_per_func)

            return advantages

    @patch_profiling_decorator
    def _dynamic_sampling(self, inputs, rewards_per_func):
        """
        Perform dynamic sampling to replace samples with zero-reward-variance groups.

        This method implements DAPO (https://arxiv.org/abs/2503.14476) by replacing
        samples from groups with zero reward variance (std=0) through resampling.

        Args:
            inputs: local input data samples
            rewards_per_func: reward per function for global data samples

        Returns:
            tuple: (inputs, rewards_per_func) with zero-variance groups replaced by resampled data
        """
        # DAPO https://arxiv.org/abs/2503.14476
        # Replaces samples with zero-reward-variance groups (std=0)
        resample_count = 0
        valid_samples = []
        valid_rewards_per_func = []
        origin_data = (inputs, rewards_per_func)

        while resample_count < self.args.max_resample_times:
            rewards_std = self.compute_std(inputs, rewards_per_func)
            valid_mask = (rewards_std > 0)
            all_inputs = gather_object(inputs)
            valid_samples.extend([inp for inp, mask in zip(all_inputs, valid_mask) if mask])
            valid_rewards_per_func.append(rewards_per_func[valid_mask])
            if len(valid_samples) >= self.args.generation_batch_size:
                break

            inputs = next(self.dynamic_resample_iterator)
            inputs = Trainer._prepare_inputs(self, inputs)
            inputs = self._generate_completions(inputs)
            rewards_per_func = self._score_completions(inputs)
            resample_count += 1

        if len(valid_samples) >= self.args.generation_batch_size:
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )
            inputs = valid_samples[:self.args.generation_batch_size][process_slice]
            rewards_per_func = torch.cat(valid_rewards_per_func)[:self.args.generation_batch_size]
        else:
            logger.warning(f'There are still std=0 groups present after {self.args.max_resample_times} retries.')
            inputs, rewards_per_func = origin_data

        return inputs, rewards_per_func

    def compute_std(self, inputs: DataType, rewards_per_func: torch.Tensor) -> torch.Tensor:
        """Compute the standard deviation of the rewards per function."""
        device = self.accelerator.device
        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)

        if not self.dynamic_num_samples:
            grouped_rewards = rewards.view(-1, self.num_generations)
            group_rewards_std = grouped_rewards.std(dim=1).repeat_interleave(self.num_generations)
            return group_rewards_std
        else:
            prompt_ids = gather_object([inp['prompt_id'] for inp in inputs])
            request_ids = gather_object([inp['request_id'] for inp in inputs])
            device = self.accelerator.device
            unique_indices = self._get_last_indices(request_ids)
            unique_request_ids = [request_ids[i] for i in unique_indices.cpu()]
            unique_prompt_ids = [prompt_ids[i] for i in unique_indices.cpu()]

            unique_rewards = rewards[unique_indices]
            prompt_to_indices = {}
            for idx, pid in enumerate(unique_prompt_ids):
                prompt_to_indices.setdefault(pid, []).append(idx)

            prompt_stds = torch.zeros(len(unique_rewards), device=device)
            for pid, idxs in prompt_to_indices.items():
                idx_tensor = torch.tensor(idxs, device=device)
                r_group = unique_rewards[idx_tensor]
                prompt_stds[idx_tensor] = r_group.std()
            rid_to_idx = {rid: idx for idx, rid in enumerate(unique_request_ids)}
            indices_in_unique = torch.tensor([rid_to_idx[r] for r in request_ids], device=device)
            rewards_std = prompt_stds[indices_in_unique]

            return rewards_std

    def split_by_mini_batches(self, inputs: DataType) -> List[DataType]:
        """
        Split inputs into mini-batches, handling variable generation counts.

        When rollout count differs from expected (bs * spg * num_generations),
        we need to adjust the splitting logic to maintain proper batch sizes.

        This method divides the input data into chunks based on the steps per generation (spg).
        If the total number of inputs is not evenly divisible by spg, the remainder is
        distributed across the first few chunks to ensure all data is included.

        Args:
            inputs (DataType): List of input data samples to be split into mini-batches.

        Returns:
            List[DataType]: A list of data chunks, where each chunk represents one step
                           in the generation process. The number of chunks equals spg.
        """
        # Slice to keep only the local part of the data
        if self.template.sequence_parallel_size == 1:
            mode: str = 'train' if self.model.training else 'eval'
            spg: int = self.args.steps_per_generation if mode == 'train' else 1

            chunk_size: int = len(inputs) // spg
            remainder: int = len(inputs) % spg
            spg_chunks: List[DataType] = []

            start_idx: int = 0
            for i in range(spg):
                current_chunk_size: int = chunk_size + (1 if i < remainder else 0)
                end_idx: int = start_idx + current_chunk_size
                spg_chunks.append(inputs[start_idx:end_idx])
                start_idx = end_idx

            return spg_chunks
        else:
            from swift.trainers.sequence_parallel import sequence_parallel
            """Split by mini batches for GRPO sequence parallel training"""
            output = [None] * sequence_parallel.sp_world_size
            # gather inputs within a sp group
            dist.all_gather_object(output, inputs, group=sequence_parallel.sp_group)
            if sequence_parallel.rp_world_size > 1:
                output_rp = [None] * sequence_parallel.rp_world_size
                output = [p for sublist in output for p in sublist]
                dist.all_gather_object(output_rp, output, group=sequence_parallel.rp_group)
                output = output_rp
            output = [p for sublist in output for p in sublist]
            inputs = output

            mode = 'train' if self.model.training else 'eval'
            spg = self.args.steps_per_generation * sequence_parallel.world_size if mode == 'train' else 1

            if mode == 'eval':
                # TODO only take the first bs rows, because eval does not support loop
                bs = self.args.per_device_eval_batch_size
                inputs = inputs[:bs]
                spg = 1

            # Use the new dynamic splitting logic
            chunk_size: int = len(inputs) // spg
            remainder: int = len(inputs) % spg
            spg_chunks: List[DataType] = []

            start_idx: int = 0
            for i in range(spg):
                current_chunk_size: int = chunk_size + (1 if i < remainder else 0)
                end_idx: int = start_idx + current_chunk_size
                spg_chunks.append(inputs[start_idx:end_idx])
                start_idx = end_idx

            spg_chunks = to_device(spg_chunks, device=self.accelerator.device)
            return spg_chunks

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(self.model).disable_adapter() if is_peft_model(
                self.model) and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or 'default')

    @patch_profiling_decorator
    def _prepare_batch_inputs(self, inputs: DataType) -> List[DataType]:
        """
        Prepare the final batch inputs with ref/old_policy logps and other fields for RL training.

        Args:
            inputs (DataType): List of local input samples.

        Returns:
            List[DataType]: A list of prepared batch inputs, organized as [spg][bs]
        """
        template = self.template
        gas_chunks = self.split_by_mini_batches(inputs)
        ga_batch_encoded_inputs = []
        for batch in gas_chunks:
            # Encode and process each batch (size=bs)
            with self._template_context(template):
                for data in batch:
                    if 'response_token_ids' in data and data['response_token_ids']:
                        loss_mask = None
                        if 'response_loss_mask' in data and data['response_loss_mask']:
                            loss_mask = data['response_loss_mask']
                        data['messages'] = replace_assistant_response_with_ids(data['messages'],
                                                                               data['response_token_ids'], loss_mask)
                batch_encoded_inputs = [template.encode(data, return_length=True) for data in batch]
                batch_encoded_inputs = to_device(template.data_collator(batch_encoded_inputs), self.model.device)
                if self.dynamic_num_samples and self.is_multimodal:
                    batch_encoded_inputs['_origin_data'] = batch

            # Process labels and masks
            labels = batch_encoded_inputs.pop('labels')
            logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
            extra_kwargs = {
                'completion_mask':
                labels[:, -logits_to_keep:] != -100,
                'truncated_mask':
                torch.tensor([b['is_truncated'] for b in batch], dtype=torch.bool, device=self.accelerator.device),
                'logits_to_keep':
                logits_to_keep,
            }
            if self.template.padding_free:
                position_ids = batch_encoded_inputs.get('text_position_ids')
                if position_ids is None:
                    position_ids = batch_encoded_inputs.get('position_ids')
                position_ids = position_ids.squeeze()
                assert position_ids is not None
                lengths = torch.diff(
                    torch.cat([(position_ids == 0).nonzero(as_tuple=True)[0],
                               torch.tensor([len(position_ids)]).to(position_ids.device)]))
                total_lengths = lengths.sum()
                # The first sentence has its prompt portion removed due to logits_to_keep
                lengths[0] = lengths[0] - (total_lengths - logits_to_keep)
                extra_kwargs.update({'seq_lengths': lengths})
                advantages_stacked = torch.stack([data['advantages'] for data in batch])
                all_advantages = torch.repeat_interleave(advantages_stacked, lengths)
            else:
                all_advantages = torch.stack([data['advantages'] for data in batch])
            extra_kwargs.update({'advantages': all_advantages})
            batch_encoded_inputs.update(extra_kwargs)

            with torch.no_grad():
                batch_encoded_inputs['old_per_token_logps'] = (
                    self._get_per_token_logps_and_entropies(self.model, batch_encoded_inputs)[0]
                    if self.old_policy() else None)
                if self.beta == 0.0:
                    ref_per_token_logps = None
                elif self.ref_model is not None:
                    ref_per_token_logps = \
                        self._get_per_token_logps_and_entropies(self.ref_model, batch_encoded_inputs)[0]
                else:
                    with self.null_ref_context():
                        ref_per_token_logps = \
                            self._get_per_token_logps_and_entropies(self.model, batch_encoded_inputs)[0]
                batch_encoded_inputs['ref_per_token_logps'] = ref_per_token_logps

            ga_batch_encoded_inputs.append(batch_encoded_inputs)

        # --- log completion lengths ---
        mode = 'train' if self.model.training else 'eval'
        device = self.accelerator.device
        if self.template.padding_free:
            local_lengths = [inp['seq_lengths'].tolist() for inp in ga_batch_encoded_inputs]
        else:
            local_lengths = [inp['completion_mask'].sum(1).tolist() for inp in ga_batch_encoded_inputs]
        total_lengths = self._gather_and_flatten(local_lengths, dtype=torch.float32, device=device, flatten_level=1)

        self._metrics[mode]['completions/mean_length'].append(total_lengths.mean().item())
        self._metrics[mode]['completions/min_length'].append(total_lengths.min().item())
        self._metrics[mode]['completions/max_length'].append(total_lengths.max().item())

        # --- log completion clipped ratio ---
        local_trunc_masks = [inp['truncated_mask'].tolist() for inp in ga_batch_encoded_inputs]
        total_trunc_masks = self._gather_and_flatten(
            local_trunc_masks, dtype=torch.bool, device=device, flatten_level=1)

        if not self.dynamic_num_samples:
            clipped_ratio = total_trunc_masks.sum().item() / total_lengths.shape[0]
            self._metrics[mode]['completions/clipped_ratio'].append(clipped_ratio)

            if all('rollout_infos' in inp and 'num_turns' in inp['rollout_infos'] for inp in inputs):
                num_turns = torch.tensor(
                    gather_object([inp['rollout_infos']['num_turns'] for inp in inputs]), device=device)
                self._metrics[mode]['num_turns'].append(num_turns.float().mean().item())
        else:
            request_ids = gather_object([inp['request_id'] for inp in inputs])
            last_indices = self._get_last_indices(request_ids)

            final_trunc_masks = total_trunc_masks[last_indices]
            clipped_ratio = final_trunc_masks.sum().item() / final_trunc_masks.shape[0]
            self._metrics[mode]['completions/clipped_ratio'].append(clipped_ratio)

            if all('rollout_infos' in inp and 'num_turns' in inp['rollout_infos'] for inp in inputs):
                num_turns_all = torch.tensor(
                    gather_object([inp['rollout_infos']['num_turns'] for inp in inputs]), device=device)
                final_num_turns = num_turns_all[last_indices]
                self._metrics[mode]['num_turns'].append(final_num_turns.float().mean().item())

        return ga_batch_encoded_inputs

    def _apply_chat_template_to_messages_list(self, messages_list: DataType):
        prompts_text = []
        for messages in messages_list:
            InferRequest.remove_response(messages)
            template_inputs = TemplateInputs.from_dict({'messages': messages})
            res = self.template.encode(template_inputs)
            prompts_text.append(self.template.safe_decode(res['input_ids']))
        return prompts_text

    @patch_profiling_decorator
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
        mode = 'train' if self.model.training else 'eval'

        # Check batch size and decide processing strategy
        batch_size = inputs['seq_lengths'].shape[0] if self.template.padding_free else inputs['input_ids'].shape[0]
        expected_bs = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size

        should_chunk = self.dynamic_num_samples and any(gather_object([batch_size > expected_bs]))
        if not should_chunk:
            return self._compute_loss_single(model, inputs)
        else:
            # maybe dynamic rollout num for multi-turn training
            return self._compute_loss_chunked(model, inputs)

    def _compute_loss_single(self, model, inputs):
        """Original loss computation logic for single batch processing."""
        loss, metrics_data = self._compute_loss_and_metrics(model, inputs)
        self._update_metrics(metrics_data)
        return loss

    def _compute_loss_and_metrics(self, model, inputs):
        """Core loss computation without metrics recording."""
        mode = 'train' if self.model.training else 'eval'

        completion_mask = inputs['completion_mask']
        truncated_mask = inputs['truncated_mask']
        if self.template.padding_free:
            lengths = inputs['seq_lengths']
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, inputs, compute_entropy=self.compute_entropy)

        entropy_mask = None
        entropy_metrics = {}

        if self.compute_entropy:
            # fill the padded token with NaN
            entropies = entropies.masked_fill(completion_mask == 0, float('nan'))
            if self.args.log_entropy:
                if self.template.padding_free:
                    entropy_list = torch.split(entropies, lengths.tolist())
                    per_completion_entropies_mean = torch.stack([torch.nanmean(e) for e in entropy_list])
                else:
                    per_completion_entropies_mean = torch.nanmean(entropies, dim=1)
                global_per_completion_entropies_mean = gather(per_completion_entropies_mean)
                entropy_metrics = {
                    'entropy_logs': global_per_completion_entropies_mean.tolist(),
                    'entropy_mean': global_per_completion_entropies_mean.nanmean().item(),
                    'entropy_max': nanmax(global_per_completion_entropies_mean).item(),
                    'entropy_min': nanmin(global_per_completion_entropies_mean).item()
                }

            # compute the entropy threshold across all tokens in the batch
            if self.args.top_entropy_quantile < 1.0:
                entropy_threshold = torch.nanquantile(entropies.flatten().float(), 1 - self.top_entropy_quantile)
                entropy_metrics['entropy_threshold'] = entropy_threshold.item()
                entropy_mask = entropies >= entropy_threshold

        # apply the completion_mask to exclude loss and metrics for overlong completions
        if self.args.overlong_filter and any(truncated_mask):
            if all(truncated_mask):
                logger.info('All completions are overlong and truncated, '
                            'resulting in NaN some values for some metrics (e.g., KL)')
            if self.template.padding_free:
                truncated_mask = torch.repeat_interleave(truncated_mask, lengths).unsqueeze(0)
                assert truncated_mask.shape == completion_mask.shape
            else:
                truncated_mask = truncated_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & (~truncated_mask)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs['ref_per_token_logps']
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)

        advantages = inputs['advantages']
        # When under on-policy training
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs['old_per_token_logps'] is None else inputs['old_per_token_logps'])

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == 'token':
            log_importance_weights = log_ratio
        elif self.importance_sampling_level in ['sequence', 'sequence_token']:
            if self.template.padding_free:
                # split to batch, compute seq-level normalization
                log_ratio_list = torch.split(log_ratio.squeeze(0), lengths.tolist())
                mask_list = torch.split(completion_mask.squeeze(0), lengths.tolist())
                seq_weights = [(lr * m).sum() / m.sum().clamp(min=1.0) for lr, m in zip(log_ratio_list, mask_list)]
                seq_level_log_weights = torch.stack(seq_weights).to(log_ratio.dtype).unsqueeze(-1)
                if self.importance_sampling_level == 'sequence':
                    log_importance_weights = seq_level_log_weights
                else:
                    seq_level_log_weight = seq_level_log_weights.detach()
                    seq_level_log_weight = torch.repeat_interleave(seq_level_log_weight, lengths).unsqueeze(0)
                    log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight
            else:
                seq_level_log_weights = ((log_ratio * completion_mask).sum(-1)
                                         / completion_mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
                if self.importance_sampling_level == 'sequence':
                    log_importance_weights = seq_level_log_weights
                else:
                    # GSPO-token: sg[si(θ)] * πθ(yi,t)/sg[πθ(yi,t)]
                    seq_level_log_weight = seq_level_log_weights.detach()
                    log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight

        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'.")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        if self.template.padding_free:
            advantages = advantages[-coef_1.shape[1]:]
            per_token_loss1 = coef_1 * advantages.unsqueeze(0)
            per_token_loss2 = coef_2 * advantages.unsqueeze(0)
        else:
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == 'grpo':
            if self.template.padding_free:
                loss_list = torch.split(per_token_loss.squeeze(0), lengths.tolist())
                mask_list = torch.split(completion_mask.squeeze(0), lengths.tolist())
                sample_loss = [(loss * mask).sum() / mask.sum().clamp(min=1.0)
                               for loss, mask in zip(loss_list, mask_list)]
                loss = torch.stack(sample_loss).mean()
            else:
                loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == 'bnpo':
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == 'dr_grpo':
            batch_size = lengths.shape[0] if self.template.padding_free else inputs['input_ids'].shape[0]
            loss = (per_token_loss * completion_mask).sum() / (batch_size * self.max_completion_length)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            # compute for token-level average
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        # Prepare metrics data
        metrics_data = {
            'mode': mode,
            'entropy': entropy_metrics,
            'completion_mask': completion_mask,
            'completion_token_count': completion_token_count,
        }

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            metrics_data['kl'] = self.accelerator.gather_for_metrics(mean_kl).nanmean().item()

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)

        metrics_data['clipping'] = {
            'low_clip_mean': gathered_low_clip.nanmean().item(),
            'low_clip_min': nanmin(gathered_low_clip).item(),
            'high_clip_mean': gathered_high_clip.nanmean().item(),
            'high_clip_max': nanmax(gathered_high_clip).item(),
            'region_clip_mean': gathered_clip_ratio.nanmean().item()
        }
        if mode == 'train' and self.chord_sft_iterator is not None:
            loss = compute_chord_loss(self, grpo_loss=loss)

        return loss, metrics_data

    def _update_metrics(self, metrics_data):
        """Update metrics from metrics_data."""
        mode = metrics_data['mode']

        # Update entropy metrics
        if metrics_data['entropy']:
            entropy_metrics = metrics_data['entropy']
            if 'entropy_logs' in entropy_metrics:
                self._logs['entropy'].extend(entropy_metrics['entropy_logs'])
                self._metrics[mode]['entropy/mean'].append(entropy_metrics['entropy_mean'])
                self._metrics[mode]['entropy/max'].append(entropy_metrics['entropy_max'])
                self._metrics[mode]['entropy/min'].append(entropy_metrics['entropy_min'])
            if 'entropy_threshold' in entropy_metrics:
                self._metrics[mode]['entropy/threshold'].append(entropy_metrics['entropy_threshold'])

        # Update KL metrics
        if 'kl' in metrics_data:
            self._metrics[mode]['kl'].append(metrics_data['kl'])

        # Update clipping metrics
        if 'clipping' in metrics_data:
            clipping = metrics_data['clipping']
            self._metrics[mode]['clip_ratio/low_mean'].append(clipping['low_clip_mean'])
            self._metrics[mode]['clip_ratio/low_min'].append(clipping['low_clip_min'])
            self._metrics[mode]['clip_ratio/high_mean'].append(clipping['high_clip_mean'])
            self._metrics[mode]['clip_ratio/high_max'].append(clipping['high_clip_max'])
            self._metrics[mode]['clip_ratio/region_mean'].append(clipping['region_clip_mean'])

    def _compute_loss_chunked(self, model, inputs: DataType):
        """
        Compute loss in **fixed-size chunks** to reduce peak GPU memory.

        The function guarantees that **all ranks step through the same number of
        chunks**, so that collective communication remain synchronized
        even when local ``batch_size`` differs.
        """
        mode = 'train' if self.model.training else 'eval'
        chunk_size = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size
        batch_size = inputs['seq_lengths'].shape[0] if self.template.padding_free else inputs['input_ids'].shape[0]

        # Decide how many chunks every rank must run
        batch_sizes = gather_object([batch_size])
        chunks_per_device = [(bs + chunk_size - 1) // chunk_size for bs in batch_sizes]
        max_chunks = max(chunks_per_device)

        # Re-compute chunk size so that max_chunks * new_chunk_size covers entire batch
        new_chunk_size = (batch_size + max_chunks - 1) // max_chunks

        losses, weights = [], []
        all_metrics_data = []
        chunk_inputs = {}
        for chunk_idx in range(max_chunks):
            start_idx = chunk_idx * new_chunk_size
            end_idx = min(start_idx + new_chunk_size, batch_size)

            if start_idx < batch_size:
                chunk_inputs = self.get_chunked_inputs(inputs, start_idx, end_idx)

            # Compute loss and metrics for this chunk (without updating global metrics)
            chunk_loss, chunk_metrics_data = self._compute_loss_and_metrics(model, chunk_inputs)
            chunk_weight = end_idx - start_idx

            if start_idx < batch_size:
                losses.append(chunk_loss * chunk_weight)
                weights.append(chunk_weight)
                all_metrics_data.append((chunk_metrics_data, chunk_weight))

        # Compute weighted average loss
        total_weight = sum(weights)
        if total_weight > 0:
            final_loss = torch.stack(losses).sum() / total_weight
        else:
            final_loss = torch.tensor(0.0, device=model.device)

        # Aggregate metrics across all chunks
        self._aggregate_and_update_metrics(all_metrics_data, mode)

        return final_loss

    def _aggregate_and_update_metrics(self, all_metrics_data, mode):
        """Aggregate metrics from multiple chunks and update global metrics."""
        if not all_metrics_data:
            return

        # Separate metrics by type for aggregation
        entropy_logs, entropy_stats, kl_values = [], [], []
        clip_values = {'low': [], 'high': [], 'region': [], 'low_min': [], 'high_max': []}
        entropy_thresholds = []

        for chunk_metrics, chunk_weight in all_metrics_data:
            chunk_tokens = chunk_metrics['completion_token_count']

            # Collect entropy metrics
            if chunk_metrics['entropy']:
                entropy_metrics = chunk_metrics['entropy']
                if 'entropy_logs' in entropy_metrics:
                    entropy_logs.extend(entropy_metrics['entropy_logs'])
                    entropy_stats.append({
                        'mean': entropy_metrics['entropy_mean'],
                        'max': entropy_metrics['entropy_max'],
                        'min': entropy_metrics['entropy_min']
                    })
                if 'entropy_threshold' in entropy_metrics:
                    entropy_thresholds.append(entropy_metrics['entropy_threshold'])

            # Collect KL metrics
            if 'kl' in chunk_metrics:
                kl_values.append(chunk_metrics['kl'])

            # Collect clipping metrics (weighted by tokens)
            if 'clipping' in chunk_metrics:
                clipping = chunk_metrics['clipping']
                weight = chunk_tokens.item() if hasattr(chunk_tokens, 'item') else chunk_tokens
                clip_values['low'].append((clipping['low_clip_mean'], weight))
                clip_values['high'].append((clipping['high_clip_mean'], weight))
                clip_values['region'].append((clipping['region_clip_mean'], weight))
                clip_values['low_min'].append(clipping['low_clip_min'])
                clip_values['high_max'].append(clipping['high_clip_max'])

        # Build aggregated metrics
        aggregated_metrics = {'mode': mode, 'entropy': {}}

        # Aggregate entropy
        if entropy_logs:
            # Directly update entropy logs
            self._logs['entropy'].extend(entropy_logs)
            aggregated_metrics['entropy'] = {
                'entropy_mean': sum(s['mean'] for s in entropy_stats) / len(entropy_stats),
                'entropy_max': max(s['max'] for s in entropy_stats),
                'entropy_min': min(s['min'] for s in entropy_stats)
            }
        if entropy_thresholds:
            aggregated_metrics['entropy']['entropy_threshold'] = sum(entropy_thresholds) / len(entropy_thresholds)

        # Aggregate KL
        if kl_values:
            aggregated_metrics['kl'] = sum(kl_values) / len(kl_values)

        # Aggregate clipping (token-weighted averages)
        if clip_values['low']:

            def weighted_avg(values):
                return sum(v * w for v, w in values) / sum(w for _, w in values)

            aggregated_metrics['clipping'] = {
                'low_clip_mean': weighted_avg(clip_values['low']),
                'low_clip_min': min(clip_values['low_min']),
                'high_clip_mean': weighted_avg(clip_values['high']),
                'high_clip_max': max(clip_values['high_max']),
                'region_clip_mean': weighted_avg(clip_values['region'])
            }

        # Update metrics
        self._update_metrics(aggregated_metrics)

    def _get_per_token_logps_and_entropies_sp(
            self,
            model: torch.nn.Module,
            inputs: 'DataType',
            compute_entropy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get per token logps for GRPO sequence parallel training"""
        try:
            from trl.trainer.utils import selective_log_softmax
        except ImportError:
            raise ImportError('trl is required for GRPO training. Please install it with: pip install trl')

        from swift.trainers.sequence_parallel.utils import GatherLoss
        from swift.trainers.sequence_parallel import sequence_parallel

        # original logits to keep
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        inputs = {
            k: v
            for k, v in inputs.items() if k not in [
                'logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps',
                'truncated_mask'
            ]
        }
        sequence_parallel.prepare_inputs(inputs)
        with self._template_context(self.template):
            output = model(**inputs)
            logits = output.logits
        # split input_ids to labels
        position_ids = sequence_parallel.real_position_ids
        _, _, labels, _, _, _ = sequence_parallel.pad_and_split_inputs(
            None, None, input_ids.clone(), None, None, None, real_position_ids=position_ids)

        labels = torch.where(labels == -100, self.processing_class.pad_token_id, labels)
        logits = logits / self.temperature
        per_token_logps = selective_log_softmax(logits, labels)
        entropies = None
        per_token_logps, _ = GatherLoss.apply(per_token_logps, labels, 1, position_ids)
        if compute_entropy:
            entropies = entropy_from_logits(logits)
            entropies, _ = GatherLoss.apply(entropies, labels, 1, position_ids)

        per_token_logps = per_token_logps[:, -logits_to_keep - 1:-1]
        if compute_entropy:
            entropies = entropies[:, -logits_to_keep - 1:-1]
        # ignore the last token
        return per_token_logps, entropies

    @patch_profiling_decorator
    def _get_per_token_logps_and_entropies(self,
                                           model,
                                           inputs,
                                           compute_entropy=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute per-token log probabilities and entropies with memory-efficient batching.

        When rollout count is larger than expected, we process in smaller batches
        to control memory usage.
        """
        batch_size = inputs['seq_lengths'].shape[0] if self.template.padding_free else inputs['input_ids'].shape[0]
        mode = 'train' if self.model.training else 'eval'
        expected_bs = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size  # noqa
        should_chunk = self.dynamic_num_samples and any(gather_object([batch_size > expected_bs]))
        if not should_chunk:
            return self._get_per_token_logps_and_entropies_single(model, inputs, compute_entropy=compute_entropy)
        else:
            return self._get_per_token_logps_and_entropies_chunked(model, inputs, compute_entropy=compute_entropy)

    def _get_per_token_logps_and_entropies_single(self,
                                                  model,
                                                  inputs,
                                                  compute_entropy=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.template.sequence_parallel_size > 1:
            return self._get_per_token_logps_and_entropies_sp(model, inputs, compute_entropy=compute_entropy)
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        unwrapped_model = self.accelerator.unwrap_model(model)
        if is_peft_model(unwrapped_model):
            parameters = inspect.signature(unwrapped_model.base_model.model.forward).parameters
        else:
            parameters = inspect.signature(unwrapped_model.forward).parameters
        use_local_entropy = not hasattr(super(), '_get_per_token_logps_and_entropies') and compute_entropy

        can_use_super = (not self.is_multimodal and 'logits_to_keep' in parameters and not use_local_entropy)

        if can_use_super:
            # save memory
            if hasattr(super(), '_get_per_token_logps_and_entropies'):
                logps, entropies = super()._get_per_token_logps_and_entropies(
                    model, input_ids, inputs['attention_mask'], logits_to_keep, compute_entropy=compute_entropy)
            else:
                logps = super()._get_per_token_logps(model, input_ids, inputs['attention_mask'], logits_to_keep)
                entropies = None
        else:
            inputs = {
                k: v
                for k, v in inputs.items() if k not in [
                    'logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps',
                    'truncated_mask', 'seq_lengths'
                ]
            }
            if 'logits_to_keep' in self.model_kwarg_keys:
                inputs['logits_to_keep'] = logits_to_keep + 1
            logits = model(**inputs).logits
            # exclude the last logit: it corresponds to the next token pred
            logits = logits[:, -(logits_to_keep + 1):-1, :]
            logits = logits / self.temperature
            input_ids = input_ids[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens
            entropies = None
            if compute_entropy:
                entropies = entropy_from_logits(logits)

        return logps, entropies

    def _get_per_token_logps_and_entropies_chunked(self,
                                                   model,
                                                   inputs,
                                                   compute_entropy=False
                                                   ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute per-token log-probabilities (and optionally entropies) in **fixed-size
        chunks** to bound peak GPU memory.

        This routine **guarantees that every rank executes the same number of
        chunks**, even when the local batch sizes differ.

        Parameters
        ----------
        model : torch.nn.Module
            The model used to compute log-probs and entropies.
        inputs : DataType
            A list of dictionary of tensors that constitute the full rollout/training batch.
        compute_entropy : bool, optional
            Whether to compute per-token entropies as well (default: False).

        Returns
        -------
        final_logps : torch.Tensor
            Concatenated per-token log-probabilities for the **entire batch**.
        final_entropies : torch.Tensor or None
            Concatenated per-token entropies, or ``None`` if ``compute_entropy`` is
            ``False``.
        """
        batch_size = inputs['seq_lengths'].shape[0] if self.template.padding_free else inputs['input_ids'].shape[0]
        mode = 'train' if self.model.training else 'eval'
        chunk_size = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size

        batch_sizes = gather_object([batch_size])  # list[int]
        chunks_per_device = [(bs + chunk_size - 1) // chunk_size for bs in batch_sizes]
        max_chunks = max(chunks_per_device)

        new_chunk_size = (batch_size + max_chunks - 1) // max_chunks

        all_logps, all_entropies = [], [] if compute_entropy else None

        # Process in chunks
        chunk_inputs = {}
        for chunk_idx in range(max_chunks):
            start_idx = chunk_idx * new_chunk_size
            end_idx = min(start_idx + new_chunk_size, batch_size)

            if start_idx < end_idx:
                chunk_inputs = self.get_chunked_inputs(inputs, start_idx, end_idx)

            chunk_logps, chunk_entropies = self._get_per_token_logps_and_entropies_single(
                model, chunk_inputs, compute_entropy)

            if start_idx < end_idx:
                all_logps.append(chunk_logps)
                if compute_entropy and chunk_entropies is not None:
                    all_entropies.append(chunk_entropies)

        # Concatenate results
        final_logps = torch.cat(all_logps, dim=0)
        final_entropies = torch.cat(all_entropies, dim=0) if all_entropies else None

        return final_logps, final_entropies

    @patch_profiling_decorator
    def _get_last_hidden_state(self, unwrapped_model, inputs, logits_to_keep):
        # unwrap the model to access the model.model
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        if not self.is_multimodal:
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
            if 'logits_to_keep' in self.model_kwarg_keys:
                inputs['logits_to_keep'] = logits_to_keep + 1

            last_hidden_state = unwrapped_model.model(**inputs).last_hidden_state

        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        assert not self.template.padding_free
        input_ids = inputs['input_ids']
        logits_to_keep = inputs['logits_to_keep']
        completion_ids = input_ids[:, -logits_to_keep:]
        completion_mask = inputs['completion_mask']

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
            old_per_token_logps=inputs.get('old_per_token_logps'),
            ref_per_token_logps=inputs.get('ref_per_token_logps'),
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
        output = super().evaluation_loop(dataloader, *args, **kwargs)
        self.eval_flag = True
        return output

    def training_step(self, model: nn.Module, inputs: DataType, num_items_in_batch=None) -> torch.Tensor:
        if self.args.async_generate:
            # Wait for the eval rollout to complete
            while not self.is_async_generate_eval_rollout_done():
                time.sleep(0.1)
        return super().training_step(model, inputs, num_items_in_batch)

    def _engine_infer(
        self,
        infer_requests: List[RolloutInferRequest],
        request_config: Optional[RequestConfig] = None,
        *,
        use_tqdm: Optional[bool] = False,
    ) -> List[RolloutOutput]:
        """
        Perform inference using the configured engine (VLLM server or colocate engine).

        Args:
            infer_requests: List of rollout inference requests to process
            request_config: Optional configuration for the requests
            use_tqdm: Whether to show progress bar during inference

        Returns:
            List of RolloutOutput objects containing the inference results
        """
        with patch_profiling_context(self, 'generate'):
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

    def old_policy(self):
        if self.template.sequence_parallel_size == 1:
            return (self.num_iterations > 1
                    or self.args.gradient_accumulation_steps % self.args.steps_per_generation != 0)
        else:
            from swift.trainers.sequence_parallel import sequence_parallel
            return (self.num_iterations > 1 or self.args.gradient_accumulation_steps %
                    (self.args.steps_per_generation * sequence_parallel.world_size) != 0)

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

        def set_default_max_tokens(_self, request_config: RequestConfig, inputs: Dict[str, Any]) -> None:
            # Calculate required context window
            original_max_len = _self.max_model_len or 8192
            assert isinstance(inputs, dict)
            prompt_tokens = _self._get_num_tokens(inputs)

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

    @patch_profiling_decorator
    def resample_encode_failed_inputs(self, inputs: DataType, n_try_fetch: int = 10) -> DataType:
        """
        Attempt to encode each input using the template. If encoding fails,
        resample from a backup iterator until successful or until the maximum
        number of retries is reached.

        Args:
            inputs (DataType): A list of input data samples, each containing a `messages` field.
            n_try_fetch (int, optional): Maximum number of retries to fetch a new sample
                when encoding fails. Defaults to 10.

        Returns:
            DataType: A list of successfully encoded input samples.

        Raises:
            RuntimeError: If encoding fails after `n_try_fetch` resampling attempts.
        """
        template = self.template
        last_messages = None
        last_valid_data = None

        for i, data in enumerate(inputs):
            # Skip samples with the same `messages` as the previous one.
            # If the last sample was successfully encoded, reuse it.
            if last_messages is not None and data['messages'] == last_messages:
                if last_valid_data is not None:
                    inputs[i] = last_valid_data
                    continue

            current_data = data
            n_try = 0

            while True:
                try:
                    # Attempt to encode the current sample.
                    template.encode(current_data)
                    # If successful, store the result and update the last valid data.
                    inputs[i] = current_data
                    last_messages = current_data['messages']
                    last_valid_data = current_data
                    break

                except Exception as e:
                    # Encoding failed — attempt to resample a new input.
                    logger.warning(f'Encoding failed for one sample; resampling a new input. {e}')
                    n_try += 1

                    # Stop if the maximum retry limit is exceeded.
                    if n_try > n_try_fetch:
                        raise RuntimeError('Failed to obtain a valid sample after multiple attempts. '
                                           'Consider increasing `max_length` or adjusting the '
                                           '`truncation_strategy` to avoid excessive truncation.')

                    # Fetch a new sample from the resampling iterator.
                    current_data = next(self.truncated_resample_iterator)[0]

        return inputs

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        mode = 'train' if self.model.training else 'eval'
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == 'eval':
            metrics = {f'eval_{key}': val for key, val in metrics.items()}

        logs.update(metrics)
        if version.parse(transformers.__version__) >= version.parse('4.47.0.dev0'):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        # - entropy only includes samples that went through training (computed in _compute_loss)
        # - Other fields (e.g., prompt/completion/reward) are collected from rollout (in _prepare_inputs)
        # Therefore, if entropy exists, to ensure length consistency across fields,
        # we align all data based on the number of samples in entropy.
        seen_nums = len(self._logs['entropy']) \
            if 'entropy' in self._logs else len(self._logs['prompt'])
        if self.accelerator.is_main_process and self.log_completions:
            table = {
                'step': [str(self.state.global_step)] * seen_nums,
                'prompt': list(self._logs['prompt'])[:seen_nums],
                'completion': list(self._logs['completion'])[:seen_nums],
                **{k: list(v)[:seen_nums]
                   for k, v in self._logs['rewards'].items()},
                'advantages': list(self._logs['advantages'])[:seen_nums],
            }
            for key, value in self._logs.items():
                if key not in table and key not in ['image', 'rewards']:
                    table[key] = list(value)[:seen_nums]

            if self.args.log_entropy:
                table.update({'entropy': list(self._logs['entropy'])[:seen_nums]})

            report_to_wandb = self.args.report_to and 'wandb' in self.args.report_to and wandb.run is not None
            report_to_swanlab = self.args.report_to and 'swanlab' in self.args.report_to and swanlab.get_run(
            ) is not None

            self.jsonl_writer.append(table)

            if report_to_wandb:
                import pandas as pd
                # Create a copy to avoid modifying the original table used by other loggers.
                wandb_table = table.copy()
                if self._logs.get('image'):
                    wandb_table['image'] = [
                        wandb.Image(load_pil_img(img)) if img is not None else None for img in self._logs['image']
                    ]
                df = pd.DataFrame(wandb_table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                wandb.log({'completions': wandb.Table(dataframe=df)})

            if report_to_swanlab:
                headers = list(table.keys())
                rows = []
                for i in range(len(table['step'])):
                    row = []
                    for header in headers:
                        row.append(table[header][i])
                    rows.append(row)
                swanlab.log({'completions': swanlab.echarts.Table().add(headers, rows)})

    def is_async_generate_eval_rollout_done(self):
        return not self.eval_flag or not self.eval_queue.empty()

    def is_async_generate_train_rollout_done(self):
        return not self.train_queue.empty()

    @contextmanager
    def offload_context(self):
        if self.args.offload_model:
            self.offload_model(self.accelerator.unwrap_model(self.model))
            if self.ref_model:
                self.offload_model(self.ref_model)
        if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
            self.offload_optimizer()
        empty_cache()

        try:
            yield
        finally:
            # reload (load back) model when exiting context
            if self.args.offload_model:
                self.load_model(self.accelerator.unwrap_model(self.model))
                if self.ref_model:
                    self.load_model(self.ref_model)
            if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
                self.load_optimizer()
            empty_cache()

    def _add_prompt_id_to_inputs(self, inputs: DataType) -> DataType:
        """
        Adds a unique `prompt_id` to each input based on their `messages` content.

        In distributed environments, inputs with identical `messages` across different processes
        will share the same `prompt_id`. Each input also gets a unique `request_id` for vLLM request tracking.

        Args:
            inputs (DataType): A list of dictionaries, each containing a 'messages' key.

        Returns:
            DataType: The input list with each item containing new 'prompt_id' and 'request_id' fields.

        Example:
            >>> inputs = [
            ...     {"messages": [{"role": "user", "content": "hello"}], "data": 1},
            ...     {"messages": [{"role": "user", "content": "hello"}], "data": 2},
            ...     {"messages": [{"role": "assistant", "content": "hi"}], "data": 3},
            ... ]
            >>> self._add_prompt_id_to_inputs(inputs)
            [
                {"messages": [...], "data": 1, "prompt_id": "a1b2c3...", "request_id": "req1"},
                {"messages": [...], "data": 2, "prompt_id": "a1b2c3...", "request_id": "req2"},
                {"messages": [...], "data": 3, "prompt_id": "d4e5f6...", "request_id": "req3"},
            ]
        """
        if not inputs:
            return inputs

        # Gather all messages from all processes to ensure consistent prompt_id assignment
        all_messages = gather_object([inp['messages'] for inp in inputs])

        # Create a mapping from messages to prompt_id
        messages_to_prompt_id = {}
        prompt_id_counter = 0

        # Process all inputs from all processes to create consistent mapping
        for messages in all_messages:
            key = json.dumps(messages)
            if key not in messages_to_prompt_id:
                messages_to_prompt_id[key] = f'prompt_{prompt_id_counter}'
                prompt_id_counter += 1

        # Apply the mapping to current process inputs
        for input_item in inputs:
            messages = input_item.get('messages')
            input_item['prompt_id'] = messages_to_prompt_id[json.dumps(messages)]
            input_item['request_id'] = f'chatcmpl-{str(uuid.uuid4().hex)}'  # Each request gets a unique ID

        return inputs

    def _server_rollout(self, inputs: DataType, request_config: RequestConfig,
                        is_global_inputs: bool) -> List[RolloutOutput]:
        """
        Perform rollout inference using vLLM server mode.

        Args:
            inputs: List of input data to be processed
            request_config: Configuration dictionary for the inference request
            is_global_inputs: Flag indicating whether inputs are shared across all processes (async-generate)

        Returns:
            List of RolloutOutput objects containing inference results
                For non-global inputs(async-generate), returns only the portion assigned to this process.

        Notes:
            - async engine with multi-turn scenarios, the outputs count may exceed inputs count
            - For distributed inputs, outputs are scattered to processes
            - Main process coordinates inference and broadcasts outputs to other processes
        """
        # Convert inputs to inference requests
        infer_requests = self.inputs2requests(inputs)

        if is_global_inputs:
            per_device_size = len(infer_requests) // self.accelerator.num_processes
            # for async generate, data have been pre-gathered to avoid potential communicate operator
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
                # dynamic num of samples, sort by request_id to group outputs with the same trajectory together
                all_outputs = self._sort_by_request_id(all_outputs)
        else:
            all_outputs = [None] * len(all_requests)
        # Handle async engine the outputs count may exceed inputs count
        if self.enable_server_multi_turn:
            # reset to false before get the num of rollout outputs
            self.dynamic_num_samples = False
            outputs_count = [len(all_outputs)] if self.accelerator.is_main_process else [0]
            outputs_count = gather_object(outputs_count)[0]  # Broadcast count to all processes
            if outputs_count != len(all_requests):
                self.dynamic_num_samples = True
                if self.args.dynamic_sample:
                    logger.warning('Mismatch between returned samples and requests detected. '
                                   'With --dynamic_sample enabled, only the last valid sample of each '
                                   f'{self.args.generation_batch_size}-sized batch will be kept; '
                                   'some requests may therefore be dropped.')
                if self.template.padding_free:
                    raise NotImplementedError('Padding free mode is not supported for dynamic sample')
            # Initialize empty outputs for non-main processes
            if not self.accelerator.is_main_process:
                all_outputs = [None] * outputs_count

        # Distribute outputs to all processes for non-global inputs
        if not is_global_inputs:
            all_outputs = broadcast_object_list(all_outputs, from_process=0)

            # Calculate slice for this process's outputs
            if not self.enable_server_multi_turn or not self.dynamic_num_samples:
                # Special handling for colocated + multi-turn inference with varying request counts
                start_idx = sum(all_requests_lengths[:self.accelerator.process_index])
                end_idx = start_idx + all_requests_lengths[self.accelerator.process_index]
                process_slice = slice(start_idx, end_idx)
                outputs = all_outputs[process_slice]
            else:
                # Standard equal distribution case
                outputs = self.get_even_process_data(all_outputs)

        else:
            # For global inputs, only main process keeps outputs
            outputs = all_outputs if self.accelerator.is_main_process else []

        return outputs

    def _colocate_rollout(self, inputs: DataType, request_config: RequestConfig) -> List[RolloutOutput]:
        """
        Perform co-located rollout inference with PTEngine or vLLMEngine(TP supported).

        Args:
            inputs: Input data for the current process
            request_config: Configuration parameters for the inference request

        Returns:
            List[RolloutOutput]: Inference results for this process's portion of inputs

        Notes:
            - For tensor parallel groups (vllm_tensor_parallel_size > 1):
              * Gathers inputs from all ranks in the tensor parallel group
              * Each rank processes the full input set but keeps only its assigned portion
              * Ensures consistent seeds within TP groups for synchronization
            - In single-process mode, directly processes the inputs
        """
        # Handle tensor parallel group processing
        if self.vllm_tensor_parallel_size > 1:
            # Gather prompts from all ranks in the TP group and flatten.
            # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
            # Note: The input sizes may differ across ranks (e.g., in multi-turn scenarios,
            # the amount of data each rank continues to process may vary).

            # Step 1: Gather input lengths from all ranks in the TP group
            local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
            local_input_length = len(inputs)
            all_input_lengths = [None] * self.vllm_tensor_parallel_size
            torch.distributed.all_gather_object(all_input_lengths, local_input_length, group=self.tp_group)

            # Calculate slice indices for this rank's outputs
            start_idx = sum(all_input_lengths[:local_rank_in_group])
            end_idx = start_idx + all_input_lengths[local_rank_in_group]

            # Step 2: Gather actual inputs from all TP group ranks
            gathered_inputs = [None for _ in range(self.vllm_tensor_parallel_size)]
            torch.distributed.all_gather_object(gathered_inputs, inputs, group=self.tp_group)

            # Flatten the gathered inputs
            inputs = [p for sublist in gathered_inputs for p in sublist]

        # Critical seed configuration for TP groups:
        # 1. Same seed within TP group - ensures synchronization and prevents hangs
        # 2. Different seeds across TP groups - avoids duplicate generations
        outputs: List[RolloutOutput] = self._engine_infer(infer_requests=inputs, request_config=request_config)

        # For TP groups, each rank keeps only its assigned portion of outputs
        if self.vllm_tensor_parallel_size > 1:
            outputs = outputs[start_idx:end_idx]

        return outputs

    def inputs2requests(self, inputs: DataType) -> List[RolloutInferRequest]:
        """
        Convert raw input data into RolloutInferRequest objects with proper data processing.

        Args:
            inputs: List of raw input dictionaries containing messages and multimedia data

        Returns:
            List[RolloutInferRequest]: Processed inference request objects ready for engine

        Processing includes:
        - Image data conversion (bytes to base64, path handling)
        - Field filtering based on request metadata requirements
        - UUID assignment using unique request_id for vLLM request tracking
        - Optional preservation of additional fields for multi-turn async scenarios
        """

        def _process_image_data(image_data: Union[dict, str]) -> str:
            """Convert image data from various formats into standardized representation.

            Args:
                image_data: Either a dict with 'bytes' or 'path', or a direct string path

            Returns:
                str: Base64 encoded image data or original file path
            """
            if isinstance(image_data, dict):
                if image_data.get('bytes'):
                    return base64.b64encode(image_data['bytes']).decode('utf-8')
                if image_data.get('path'):
                    return image_data['path']
            return image_data

        if not inputs:
            return []

        # Define core metadata fields required for all requests
        REQUEST_METADATA_FIELDS = ['messages', 'images', 'audios', 'videos', 'objects', 'uuid']
        requests_dicts = []

        for data in inputs:
            # Extract required metadata fields
            request_data = {key: data[key] for key in REQUEST_METADATA_FIELDS if key in data}
            if 'uuid' not in request_data:
                request_data['uuid'] = data['request_id']  # Use unique request_id for vLLM
            # Preserve additional fields for multi-turn async scenarios
            if self.args.vllm_server_pass_dataset:
                # data_dict is already concatenated inside async engine
                extra_fields = {k: v for k, v in data.items() if k not in REQUEST_METADATA_FIELDS}
                if extra_fields:
                    request_data['data_dict'] = extra_fields
            elif self.multi_turn_scheduler:
                # Concatenate data_dict here
                base_data_dict = {}
                if 'data_dict' in data:
                    if isinstance(data['data_dict'], dict):
                        base_data_dict = data['data_dict']
                    else:
                        raise ValueError('data_dict exists but is not a dictionary')
                # Add fields that are not in metadata fields and not 'data_dict'
                extra_data = {k: v for k, v in data.items() if k not in REQUEST_METADATA_FIELDS and k != 'data_dict'}
                # Merge additional fields and existing data_dict
                final_data_dict = {**extra_data, **base_data_dict}
                request_data['data_dict'] = final_data_dict if final_data_dict else {}

            requests_dicts.append(request_data)

        # Process image data in each request
        for request in requests_dicts:
            if 'images' in request and request['images']:
                request['images'] = ([_process_image_data(img) for img in request['images']] if isinstance(
                    request['images'], list) else _process_image_data(request['images']))

        # Convert dictionaries to formal request objects
        return [from_dict(RolloutInferRequest, request_data) for request_data in requests_dicts]

    def _preprocess_inputs(self, inputs: DataType) -> DataType:
        """Preprocess input data before inference.

        Args:
            inputs: List of input dictionaries containing conversation messages

        Returns:
            Processed inputs with:
            - Added prompt IDs for grouping (same messages share same prompt_id)
            - Added unique request IDs for vLLM request tracking
            - Removed existing assistant responses from messages

        Processing Steps:
        1. Adds prompt IDs and unique request IDs to each input
        2. Cleans each message sequence by removing existing assistant responses
        """
        processed_inputs = self._add_prompt_id_to_inputs(inputs)

        for input_item in processed_inputs:
            remove_response(input_item['messages'])

        return processed_inputs

    def _postprocess_rollout_outputs(self, inputs: DataType, outputs: List[RolloutOutput]) -> DataType:
        """
        Postprocess rollout outputs by merging them back into the input data structures.

        Depending on the mode(if enable_server_multi_turn), it either matches inputs by request_id
        or assumes a one-to-one correspondence.
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
                if not self.multi_turn_scheduler:
                    # for single turn, skip tokenizer response
                    input_data['response_token_ids'] = output.response.choices[0].token_ids

            # Step 3: Attach rollout extra info
            if output.rollout_infos:
                input_data['rollout_infos'] = output.rollout_infos

            # Step 4: Store finish reason (used for truncation filters etc.)
            input_data['finish_reason'] = choice.finish_reason
            input_data['is_truncated'] = choice.finish_reason == 'length'

            # Step 5: override multi-modal data from rollout_infos
            if output.rollout_infos:
                multi_modal_keys = ['images', 'videos', 'audios']
                for key in multi_modal_keys:
                    if key in output.rollout_infos:
                        input_data[key] = output.rollout_infos[key]
                        logger.info_once(f'Overriding multi-modal data from rollout_infos for key: {key}')

            return input_data

        if not self.dynamic_num_samples:
            if self.async_generate and not outputs:
                # In async generation, only the main process receives outputs; non-main ranks get an empty list.
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

    def _colocate_multi_turn_infer(self, inputs: DataType, first_turn_rollout_outputs: List[RolloutOutput],
                                   request_config: RequestConfig) -> List[RolloutOutput]:
        """
        Handles multi-turn inference under colocate mode.

        This method iteratively rolls out turns until all dialogues are finished
        according to the multi_turn_scheduler.
        """
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
                if self.args.max_turns:
                    stop = stop or (current_turn >= self.args.max_turns)
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

    def get_even_process_data(self, global_data: List[T]) -> List[T]:
        """
        Evenly splits `global_data` among all processes.

        Each process receives a contiguous chunk of data. If `len(global_data)` is not
        perfectly divisible by the number of processes, the first `remainder` processes
        will receive one additional item.

        Args:
            global_data (List[T]): The full list of data to be distributed.

        Returns:
            List[T]: The subset of `global_data` assigned to this process.
        """
        num_procs = self.accelerator.num_processes
        proc_idx = self.accelerator.process_index
        total = len(global_data)

        base_size = total // num_procs
        remainder = total % num_procs

        # Calculate the number of samples that need to be padded
        # This ensures all processes have the same number of samples for gather operations
        self.rollout_pad_count = 0
        if remainder > 0 and proc_idx >= remainder:
            # Processes with extra samples need padding
            self.rollout_pad_count = 1

        if proc_idx < remainder:
            start = proc_idx * (base_size + 1)
            end = start + base_size + 1
        else:
            start = remainder * (base_size + 1) + (proc_idx - remainder) * base_size
            end = start + base_size

        return global_data[start:end]

    def _gather_and_flatten(self, local_list, dtype=None, device=None, flatten_level: int = 1):
        """
        Gather data from all ranks with `gather_object` and flatten as required.

        Args
        ----
        local_list : Sequence[Any]
            The per-rank data to be gathered. Can be any picklable structure.
        dtype : torch.dtype, optional
            If provided, the flattened result is converted to a tensor with this dtype.
        device : torch.device, optional
            Target device for the resulting tensor. Ignored if dtype is None.
        flatten_level : int
            0  keep the outer list of per-rank results: List[rank_data]
            1  flatten across ranks: List[element]
            2  flatten one more level (assumes rank_data is iterable): List[sub_element]

        Returns
        -------
        Any
            - List[Any] when dtype is None
            - torch.Tensor when dtype is given
        """
        gathered = gather_object(local_list)  # List[rank][...] returned by gather_object

        if flatten_level == 0:
            flat = gathered
        elif flatten_level == 1:  # flatten over ranks
            flat = [elem for rank_data in gathered for elem in rank_data]
        elif flatten_level == 2:  # flatten one additional level
            flat = [item for rank_data in gathered for sublist in rank_data for item in sublist]
        else:
            raise ValueError(f'Invalid flatten_level: {flatten_level}')

        if dtype is not None:
            try:
                return torch.tensor(flat, dtype=dtype, device=device)
            except (TypeError, ValueError) as e:
                raise RuntimeError(f'Cannot convert gathered+flattened data to tensor: {e}') from e
        return flat

    def _sort_by_request_id(self, all_outputs: List[RolloutOutput]) -> List[RolloutOutput]:
        """
        Sort rollout outputs by request_id to group outputs with the same request_id together.

        Args:
            all_outputs: List of RolloutOutput objects

        Returns:
            List[RolloutOutput]: Sorted list where outputs with the same request_id are adjacent
        """
        # Extract request_ids from outputs
        request_ids = [output.response.id for output in all_outputs]

        # Create pairs of (request_id, output) for sorting
        output_pairs = list(zip(request_ids, all_outputs))

        # Sort by request_id
        output_pairs.sort(key=lambda x: x[0])

        # Extract sorted outputs
        sorted_outputs = [output for _, output in output_pairs]

        return sorted_outputs

    def _group_inputs_by_request_id(self, inputs: DataType) -> Dict[str, List[Dict]]:
        """
        Group inputs by request_id for multi-turn reward computation.

        Args:
            inputs: List of input dictionaries, each containing a 'request_id' field

        Returns:
            Dict[str, List[Dict]]: A dictionary where keys are request_ids and values are
                                  lists of input dictionaries with the same request_id
        """
        inputs_by_request_id = {}

        for input_data in inputs:
            request_id = input_data.get('request_id')
            if request_id is None:
                # Skip inputs without request_id
                continue

            if request_id not in inputs_by_request_id:
                inputs_by_request_id[request_id] = []

            inputs_by_request_id[request_id].append(input_data)

        return inputs_by_request_id

    def _get_trajectory_inputs(self, inputs: DataType) -> Dict[str, List[Dict]]:
        """
        Retrieve trajectory data corresponding to the request_ids present in the current inputs.

        This method performs the following steps:
        1. Extract the set of request_ids from the current inputs
        2. Gather all inputs across processes
        3. Filter out entries whose request_id is not present in the local inputs
        4. Group the remaining inputs by request_id
        5. Keep only trajectory data for request_ids found in the current inputs

        Args:
            inputs: The current batch of input data. Each item is a dictionary
                containing at least the field 'request_id'.

        Returns:
            Dict[str, List[Dict]]: A mapping from request_id to the list of
            corresponding input records (trajectory data).
        """
        # Collect request_id set from the current inputs
        current_request_ids = {input_data['request_id'] for input_data in inputs}

        # Gather all inputs across processes
        total_inputs = gather_object(inputs)

        # Keep only entries whose request_id exists in the current inputs
        filtered_total_inputs = [
            input_data for input_data in total_inputs if input_data['request_id'] in current_request_ids
        ]

        # Group inputs by request_id
        inputs_by_request_id = self._group_inputs_by_request_id(filtered_total_inputs)

        return inputs_by_request_id

    def _get_last_indices(self, request_ids: List[str]) -> torch.Tensor:
        seen = {}
        for i, rid in enumerate(request_ids):
            seen[rid] = i
        return torch.tensor(list(seen.values()), dtype=torch.long, device=self.accelerator.device)

    def get_chunked_inputs(self, inputs, start_idx, end_idx):
        chunk_inputs = {}
        # for LLM, slice the inputs
        for key, val in inputs.items():
            if isinstance(val, torch.Tensor):
                chunk_inputs[key] = val[start_idx:end_idx]
            else:
                chunk_inputs[key] = val
        if self.is_multimodal:
            # for MLLM, re-encode to get mm-related inputs
            origin_data = inputs['_origin_data'][start_idx:end_idx]
            template = self.template
            with self._template_context(template):
                encoded_data = [template.encode(data) for data in origin_data]
                chunk_inputs.update(to_device(template.data_collator(encoded_data), self.model.device))
                chunk_inputs.pop('labels', None)
        return chunk_inputs
