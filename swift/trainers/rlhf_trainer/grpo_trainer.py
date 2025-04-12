# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/trl.
import concurrent.futures
import inspect
import os
import re
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, field
from math import ceil
from queue import Queue
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias, Union

import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from packaging import version
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TrainerCallback
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
from transformers.trainer_utils import seed_worker
from trl import GRPOTrainer as HFGRPOTrainer

from swift.llm import InferRequest, MultiModelKeys, RequestConfig, RowPreprocessor, get_model_arch, to_device
from swift.llm.infer.infer_engine import set_device_context
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.plugin import orms
from swift.plugin.multi_turn import multi_turns
from swift.utils import (JsonlWriter, gc_collect, get_device, get_device_count, get_dist_setting, get_logger,
                         get_node_setting, is_lmdeploy_available, is_vllm_available, is_wandb_available)
from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin
from .utils import _split_into_mini_batches, patch_lora_merge, patch_lora_unmerge, round_robin

try:
    from trl.extras.profiling import profiling_decorator
except ImportError:
    raise ImportError('Please install trl: `pip install -U trl`')
del HFGRPOTrainer.__init__
del HFGRPOTrainer.log

logger = get_logger()
if is_wandb_available():
    import wandb

InputsType: TypeAlias = List[Dict[str, Union[torch.Tensor, Any]]]
OutputsType: TypeAlias = List[List[Tuple[List[Dict], str]]]


@contextmanager
def unwrap_model_for_generation(
    model,
    accelerator,
    gather_deepspeed3_params=True,
    gather_parameters: List = None,
):
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            import deepspeed
            parameters = [
                parameter for name, parameter in model.named_parameters()
                if not gather_parameters or name in gather_parameters
            ]
            with deepspeed.zero.GatheredParameters(parameters):
                from trl.models.utils import remove_hooks
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                from trl.models.utils import add_hooks
                add_hooks(model)
    else:
        yield unwrapped_model


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
    distributed_idx: List[List] = field(default_factory=list)


class GRPOTrainer(RLHFTrainerMixin, SwiftMixin, HFGRPOTrainer):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_funcs: Optional[List[Union[str, Callable]]] = None,
                 *_args,
                 **kwargs):
        from swift.trainers.rlhf_arguments import GRPOConfig
        args: GRPOConfig = kwargs['args']
        self.args = args
        self.train_queue = Queue()
        self.eval_queue = Queue()
        self.processing_class = kwargs.get('template').tokenizer
        self.offload_modules = {}
        self.offload_states = {}
        _, _, _, local_world_size = get_dist_setting()

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

        self.multi_turn_func = None
        if self.args.multi_turn_func:
            if isinstance(self.args.multi_turn_func, str):
                assert self.args.multi_turn_func in multi_turns
                multi_turn_func = multi_turns[self.args.multi_turn_func]
                self.multi_turn_func = multi_turn_func
            else:
                self.multi_turn_func = self.args.multi_turn_func

        self.reward_templates = [None] * len(self.reward_funcs)
        if reward_model is not None:
            self.reward_templates.append(kwargs.pop('reward_template', None))
            self.reward_funcs.append(reward_model)
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

        self.num_generations = args.num_generations
        self.temperature = args.temperature
        model.warnings_issued['estimate_tokens'] = True
        kwargs['data_collator'] = lambda features: features

        use_vllm = args.use_vllm
        use_lmdeploy = args.use_lmdeploy

        if self.args.tensor_parallel_size > 1 and self.multi_turn_func:
            import torch.distributed as dist
            rank, _, _, _ = get_dist_setting()
            for tp_group in self.tp_group_ranks():
                group = dist.new_group(tp_group)
                if rank in tp_group:
                    self.group = group

        super().__init__(model, ref_model, *_args, **kwargs)

        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self.log_completions = args.log_completions
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'completions.jsonl'))
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            'prompt': deque(maxlen=maxlen),
            'completion': deque(maxlen=maxlen),
            'rewards': defaultdict(lambda: deque(maxlen=maxlen)),
        }

        num_processes = self.accelerator.num_processes
        self.global_train_batch_size = global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f'The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly '
                f'divisible by the number of generations per prompt ({self.num_generations}). Given the current train '
                f'batch size, the valid values for the number of generations are: {possible_values}.')
        if self.args.eval_strategy != 'no':
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f'The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly '
                    f'divisible by the number of generations per prompt ({self.num_generations}). Given the current '
                    f'eval batch size, the valid values for the number of generations are: {possible_values}.')

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        self.parameter_groups, self.parameter_groups_no_lora = self.split_batches()
        self.infer_device = None
        self.use_fast_infer = use_vllm or use_lmdeploy  # whether to use the PT backend
        if self.use_fast_infer:
            if self.infer_rank >= 0:
                fast_infer_device = self.args.vllm_device or self.args.lmdeploy_device
                if fast_infer_device[0] == 'auto':
                    if get_device_count() == 1:
                        fast_infer_device = [get_device()]  # particular case when training with only 1 GPU: share it
                    else:
                        fast_infer_device = []
                        for idx in range(get_device_count() - self.args.num_infer_workers, get_device_count()):
                            fast_infer_device.append(get_device(idx))

                for _device in fast_infer_device:
                    # Check that the requested device is available
                    if _device.split(':')[0] in {'cuda', 'npu'} and int(_device.split(':')[1]) >= get_device_count():
                        raise ValueError(f'The requested device for vllm ({_device}) is not available. '
                                         f'You are likely using vLLM '
                                         'without restricting the number of GPUs for training. '
                                         'Set the `--num_processes` argument to a '
                                         'value lower than the number of GPUs available on your machine—typically, '
                                         'reducing it by one is sufficient. '
                                         f'In your case: `--num_processes {get_device_count() - 1}`.')

                if use_vllm:
                    if not is_vllm_available():
                        raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                                          'Please install vLLM with `pip install vllm -U` to use it.')
                    self.prepare_vllm(model, fast_infer_device)
                    self.infer_device = fast_infer_device[self.local_infer_rank]
                elif use_lmdeploy:
                    if not is_lmdeploy_available():
                        raise ImportError('LMDeploy is not available and `use_lmdeploy` is set to True.'
                                          'Please install LMDeploy with `pip install lmdeploy -U` to use it.')
                    from swift.llm import LmdeployEngine
                    from swift.tuners import Swift
                    with Swift.grpo_context(model, self.template.processor):
                        fast_infer_device = int(fast_infer_device[self.local_infer_rank].split(':')[1])
                        self.engine = LmdeployEngine(
                            model.model_dir,
                            model.model_info.torch_dtype,
                            model_type=model.model_meta.model_type,
                            devices=[fast_infer_device],
                            session_len=args.lmdeploy_session_len,
                            cache_max_entry_count=args.lmdeploy_cache_max_entry_count,
                            reload_weights=True)
                        self.infer_device = fast_infer_device
                        from lmdeploy.turbomind.turbomind import TurboMind
                        lmdeploy_engine = self.engine.engine.engine
                        assert isinstance(lmdeploy_engine, TurboMind), (
                            "Currently only LMDeploy's TurboMind backend is supported. "
                            'The current model is incompatible - please use vLLM or PyTorch backend instead.')
                self.engine.default_template = copy(self.template)  # Avoid thread-unsafe modifications of the mode.
            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            from swift.llm import PtEngine
            self.engine = PtEngine.from_model_template(self.model, copy(self.template), max_batch_size=0)  # 0: no limit
        # Avoid thread-unsafe modifications of the mode.
        self.request_config = RequestConfig(
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop_words,
        )

        if local_world_size == self.args.num_infer_workers == get_device_count() and local_world_size > 1:
            self.request_config.n = self.args.tensor_parallel_size
            if self.infer_rank >= 0:
                self.request_config.seed = self.infer_rank // self.args.tensor_parallel_size

        self.model_accepts_loss_kwargs = False
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel) and is_deepspeed_zero3_enabled():
                from trl.models.utils import prepare_deepspeed
                prepare_deepspeed(reward_func, self.accelerator)  # Does not wrap DeepSpeedEngine

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle. # noqa
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps
        if self.args.async_generate:
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

    def prepare_vllm(self, model, fast_infer_device):
        from swift.tuners import Swift
        from swift.llm import VllmEngine
        from swift.llm.infer.infer_engine import GRPOVllmEngine
        _, _, _, local_world_size = get_dist_setting()
        if self.args.tensor_parallel_size > 1:
            vllm_kwargs = {'distributed_executor_backend': 'external_launcher'}
        else:
            vllm_kwargs = {}
        if local_world_size == self.args.num_infer_workers == get_device_count() and local_world_size > 1:
            # Compatibility with TP
            cls = GRPOVllmEngine
            engine_kwargs = {'seed': 0}
        else:
            cls = VllmEngine
            engine_kwargs = {}
        with Swift.grpo_context(model, self.template.processor):
            self.engine = cls(
                model.model_dir,
                model.model_info.torch_dtype,
                model_type=model.model_meta.model_type,
                device=fast_infer_device[self.local_infer_rank],
                tensor_parallel_size=self.args.tensor_parallel_size,
                gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                max_num_seqs=self.args.vllm_max_num_seqs,
                enforce_eager=self.args.vllm_enforce_eager,
                limit_mm_per_prompt=self.args.vllm_limit_mm_per_prompt,
                num_infer_workers=self.args.num_infer_workers,
                enable_sleep_mode=self.args.sleep_level > 0,
                use_async_engine=False,
                max_model_len=self.args.vllm_max_model_len,
                engine_kwargs=engine_kwargs,
                **vllm_kwargs)
            self.engine.default_template = self.template

    @property
    def infer_rank(self):
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        node_rank = get_node_setting()[0]
        for _vllm_rank in range(self.args.num_infer_workers):
            if local_rank == _vllm_rank:
                return node_rank * self.args.num_infer_workers + _vllm_rank
        if local_rank == -1:
            return 0
        return -1

    @property
    def infer_rank_tp_0(self):
        # whether is tp rank0, get data from this rank
        # vllm needs all tp ranks inputs and sampling params are the same
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        node_rank = get_node_setting()[0]
        for _vllm_rank in range(self.args.num_infer_workers):
            if local_rank == _vllm_rank and _vllm_rank % self.args.tensor_parallel_size == 0:
                return (node_rank * self.args.num_infer_workers + _vllm_rank // self.args.tensor_parallel_size)
        if local_rank == -1:
            return 0
        return -1

    @property
    def local_infer_rank(self):
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        for _vllm_rank in range(self.args.num_infer_workers):
            if local_rank == _vllm_rank:
                return _vllm_rank

        return -1

    def tp_group_ranks(self):
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        return [
            list(range(0, world_size))[i:i + self.args.tensor_parallel_size]
            for i in range(0, world_size, self.args.tensor_parallel_size)
        ]

    @contextmanager
    def _template_context(self, template):
        # The max_length for prompt and completion has already been restricted, so there is no need for max_length here.
        max_length = template.max_length
        mode = template.mode
        if mode in {'vllm', 'pt', 'lmdeploy'}:
            template.set_mode('train')
        template.max_length = None
        loss_scale = template.loss_scale
        if self.multi_turn_func:
            template.loss_scale = 'default'
        try:
            yield
        finally:
            template.loss_scale = loss_scale
            template.set_mode(mode)
            template.max_length = max_length

    @profiling_decorator
    def _move_model_to_vllm_lmdeploy(self):
        from accelerate.utils.other import is_compiled_module

        for i, parameter_group in enumerate(self.parameter_groups):
            parameter_group_no_lora = self.parameter_groups_no_lora[i]
            with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    gather_parameters=parameter_group) as unwrapped_model:

                if is_compiled_module(unwrapped_model):
                    unwrapped_model = unwrapped_model._orig_mod
                if is_peft_model(unwrapped_model):
                    with patch_lora_merge(unwrapped_model, parameter_group):
                        unwrapped_model.merge_adapter()
                    state_dict = unwrapped_model.state_dict()
                    # Remove base_model and base_layer prefixes
                    state_dict = {
                        k.removeprefix('base_model.model.').replace('.base_layer', ''): v
                        for k, v in state_dict.items()
                    }
                    # Remove values with adapter prefix (example: "_lora")
                    state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                    # When module to save, remove its prefix and discard the original module
                    state_dict = {
                        k.replace('modules_to_save.default.', ''): v
                        for k, v in state_dict.items() if 'original_module' not in k
                    }
                else:
                    state_dict = unwrapped_model.state_dict()
                if parameter_group_no_lora:
                    parameter_group_no_lora = [n.replace('base_model.model.', '') for n in parameter_group_no_lora]
                    state_dict = {k: v for k, v in state_dict.items() if k in parameter_group_no_lora}
                assert len(state_dict) > 0 and all([state.shape != torch.Size([0]) for state in state_dict.values()])
                if self.infer_rank >= 0:
                    if self.args.async_generate:
                        self._wait_queue()
                    if self.args.use_vllm:
                        llm_model = self.engine.inner_model
                    else:
                        llm_model = self.engine.engine.engine
                    llm_model.load_weights(state_dict.items())
                    del state_dict
                    gc_collect()
                # Unmerge the adapter to restore the model to its original state.
                # This must be done after loading weights to ensure they correspond to the merged state.
                if is_peft_model(unwrapped_model):
                    with patch_lora_unmerge(unwrapped_model):
                        unwrapped_model.unmerge_adapter()

        if self.infer_rank >= 0 and self.args.use_vllm and self.args.vllm_enable_prefix_caching:
            self.engine.engine.reset_prefix_cache()

    def _wait_queue(self):
        while self._queue.empty():
            time.sleep(0.01)

    @staticmethod
    def reorder_outputs(outputs, distributed_idx):
        index_to_output = {}
        current_position = 0
        for output_idx in distributed_idx:
            for idx in output_idx:
                index_to_output[idx] = outputs[current_position]
                current_position += 1

        return [index_to_output[idx] for idx in sorted(index_to_output.keys())]

    def _infer_multi_turn(self, inputs_slice: np.ndarray, request_config: RequestConfig) -> Union[OutputsType, List]:
        """Perform multi-turn or single-turn inference with support for tensor parallelism.

        Args:
            inputs_slice: Array of input requests
            request_config: Inference configuration parameters

        Returns:
            List of outputs where each entry contains:
            - List of responses per prompt (length = tensor_parallel_size)
            - Each response is a tuple of (message_history, finish_reason)
        """
        from swift.llm.infer.protocol import ChatCompletionResponse
        rank, _, _, _ = get_dist_setting()
        request_config = copy(request_config)
        results: List[ChatCompletionResponse] = self.engine.infer(
            infer_requests=inputs_slice, request_config=request_config, use_tqdm=False)
        prompt_lens = len(inputs_slice)
        messages_list = [None] * (len(inputs_slice) * self.args.tensor_parallel_size)
        if self.multi_turn_func:
            remove_response = True
            while len(inputs_slice) > 0:
                request_config.n = 1
                if self.infer_rank_tp_0 >= 0 or not self.use_fast_infer:
                    inputs = []
                    cnt = 0
                    for i, output in enumerate(results):
                        for choice in output.choices:
                            _input: Dict = deepcopy(inputs_slice[i])
                            if remove_response or _input['messages'][-1]['role'] != 'assistant' or not \
                                    _input['messages'][-1]['content']:
                                InferRequest.remove_response(_input['messages'])
                                _input['messages'].append({'role': 'assistant', 'content': choice.message.content})
                            else:
                                _input['messages'][-1]['content'] += choice.message.content
                            if 'index' not in _input:
                                _input['index'] = cnt
                            _input['finish_reason'] = choice.finish_reason
                            cnt += 1
                            inputs.append(_input)
                    results: List[Dict] = self.multi_turn_func(inputs)  # noqa
                else:
                    length = sum([len(results[i].choices) for i in range(len(results))])
                    results = [None] * length

                if self.args.tensor_parallel_size > 1:
                    # avoid duplicate calling in the same tensor parallel group
                    import torch.distributed as dist
                    if 'group_src' in inspect.signature(dist.broadcast_object_list).parameters:
                        dist.broadcast_object_list(results, group_src=0, group=self.group)
                    else:
                        global_src = dist.get_global_rank(self.group, 0)
                        dist.broadcast_object_list(results, src=global_src, group=self.group)
                inputs_slice = [r for r in results if not r['finished']]
                for idx, r in enumerate(results):
                    if r['finished'] or r['finish_reason'] == 'length':
                        messages_list[r['index']] = (r['messages'], r['finish_reason'])
                if len(inputs_slice) > 0:
                    _input_std = []
                    for _input in inputs_slice:
                        _input_std.append(StdTemplateInputs.from_dict(_input))
                        # StdTemplateInputs will not remove responses in infer
                    results = self.engine.infer(
                        infer_requests=_input_std, request_config=request_config, use_tqdm=False)
                # concat responses from the second loop
                remove_response = False

            outputs = []
            assert not any([m is None for m in messages_list])
            for i in range(0, len(messages_list), self.args.tensor_parallel_size):
                # reformat to [[x, x, x, x] [x, x, x, x]]
                # this is the same format of sampling_params.n > 1
                outputs.append(messages_list[i:i + self.args.tensor_parallel_size])
            assert len(outputs) == prompt_lens
            assert all([len(o) == self.args.tensor_parallel_size for o in outputs])
        else:
            # single turn
            outputs = []
            for i, output in enumerate(results):
                _choices = []
                for choice in output.choices:
                    _input: Dict = deepcopy(inputs_slice[i])
                    InferRequest.remove_response(_input['messages'])
                    _input['messages'].append({'role': 'assistant', 'content': choice.message.content})
                    _choices.append((_input['messages'], choice.finish_reason))
                outputs.append(_choices)
            assert len(outputs) == prompt_lens
            assert all([len(o) == self.args.tensor_parallel_size for o in outputs])

        if self.args.tensor_parallel_size > 1:
            if self.infer_rank_tp_0 < 0:
                outputs = []
            else:
                _outputs = []
                for tp_idx in range(self.args.tensor_parallel_size):
                    for prompt_idx in range(len(outputs)):
                        _outputs.append(outputs[prompt_idx][tp_idx])
                outputs = [_outputs]

        return outputs

    def async_infer(self, inputs, inputs_slice, distributed_idx):

        def infer_task():
            with set_device_context(self.infer_device), self.multi_turn_completion_length_context():
                return self._infer_multi_turn(inputs_slice, self.request_config)

        future: Future = self.executor.submit(infer_task)
        # pre-fetch the queue to avoid switching back to eval_queue at the end of training sample sampling
        current_queue = self._queue

        def done(_self):
            current_queue.put(DataCache(inputs, _self.result(), distributed_idx))

        future.add_done_callback(done)

    def _prefetch(self, dataloader: DataLoader):
        inputs = next(iter(dataloader))
        all_inputs = gather_object(inputs)
        nnodes = get_node_setting()[1]
        distributed_idx = round_robin(len(all_inputs), nnodes * self.args.num_infer_workers)
        if self.infer_rank >= 0:
            _input_slice = np.array(all_inputs)[distributed_idx[self.infer_rank]]
            with self.multi_turn_completion_length_context():
                outputs = self._infer_multi_turn(_input_slice, self.request_config)
            self._queue.put(DataCache(inputs, outputs, distributed_idx))
        else:
            self._queue.put(DataCache(inputs, [], distributed_idx))
        if self.accelerator.num_processes > 1:
            self.accelerator.wait_for_everyone()

    def _fast_infer(self, inputs: InputsType) -> Tuple[InputsType, OutputsType]:
        """
        This function performs fast inference by managing model and optimizer offloading,
        loading weights if necessary, distributing inputs among workers, and generating
        completions using the vLLM/LMDeploy framework. It supports both synchronous and asynchronous
        inference modes.
        inputs: local inputs
        """

        if self.args.sleep_level > 0 and self.infer_rank >= 0:
            if self.args.offload_model:
                self.offload_model()
            if self.args.offload_optimizer:
                self.offload_optimizer()
            if self.args.gc_collect_after_offload:
                gc_collect()
            # Skip the first wake_up to avoid the warning "Executor is not sleeping"
            if self.engine.inner_model_executor.is_sleeping:
                self.engine.engine.wake_up()
        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm_lmdeploy()
            self._last_loaded_step = self.state.global_step
        all_inputs = gather_object(inputs)
        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        # Distribute inputs to different workers
        # for example, 2 workers, 6 inputs, 0/2/4 dispatch to the first worker
        # 1/3/5 dispatch to the second worker
        # trying to shuffle and average the length
        distributed_idx = round_robin(len(all_inputs), get_node_setting()[1] * self.args.num_infer_workers)
        if self.infer_rank >= 0:
            _input_slice = np.array(all_inputs)[distributed_idx[self.infer_rank]]
            if self.args.async_generate:
                self.async_infer(inputs, _input_slice, distributed_idx)
                data_cache = self._queue.get()
                inputs = data_cache.inputs
                outputs = data_cache.outputs
                distributed_idx = data_cache.distributed_idx
            else:
                with set_device_context(self.infer_device):
                    request_config = copy(self.request_config)
                    if self.args.tensor_parallel_size > 1:
                        request_config.seed += self.state.global_step
                    with self.multi_turn_completion_length_context():
                        outputs = self._infer_multi_turn(_input_slice, self.request_config)
        else:
            if self.args.async_generate:
                # using old model to generate, which will ignore the `clip` of advantages.
                self._queue.put(DataCache(inputs, [], distributed_idx))
                data_cache = self._queue.get()
                inputs = data_cache.inputs
                distributed_idx = data_cache.distributed_idx
            outputs = []
        outputs = gather_object(outputs)
        if self.args.tensor_parallel_size > 1:
            outputs = [[item] for output in outputs for item in output]
        outputs = self.reorder_outputs(outputs, distributed_idx)
        if self.args.sleep_level > 0 and self.infer_rank >= 0:
            self.engine.engine.sleep(level=self.args.sleep_level)
            if self.args.gc_collect_after_offload:
                gc_collect()
            if self.args.offload_model:
                self.load_model()
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
        if self.use_fast_infer:
            inputs, outputs = self._fast_infer(inputs)
            # Slice to keep only the local part of the data
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )
            outputs = outputs[process_slice]
        else:
            # pt infer
            is_multimodal = self.model.model_meta.is_multimodal
            if is_multimodal:
                models = self.template.remove_post_encode_hook()
            with unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ), self.multi_turn_completion_length_context():
                outputs = self._infer_multi_turn(inputs, self.request_config)
                self.model.train()
            if is_multimodal:
                self.template.register_post_encode_hook(models)
            if isinstance(outputs[0][0], list):
                outputs = [output[0] for output in outputs]

        for i, output in enumerate(outputs):
            inputs[i]['messages'] = output[0][0]
            inputs[i]['is_truncated'] = output[0][1] == 'length'

        return inputs

    def _generate_and_score_completions(self, inputs: InputsType) -> InputsType:

        inputs = self._generate_completions(inputs)
        total_rewards_per_func, total_rewards, completions = self._score_completions(inputs)
        mode = 'eval' if self.control.should_evaluate else 'train'

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

        for i, (reward_func, reward_template) in enumerate(zip(self.reward_funcs, self.reward_templates)):
            # reward model
            if isinstance(reward_func, nn.Module):
                with self._template_context(reward_template):
                    batched_inputs = [reward_template.encode(infer_request) for infer_request in inputs]
                    reward_inputs = to_device(reward_template.data_collator(batched_inputs), reward_func.device)

                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            # reward function
            else:
                # Repeat all input columns (but "messages" and "completion") to match the number of generations
                reward_kwargs = RowPreprocessor.rows_to_batched(inputs)
                output_reward_func = reward_func(completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        total_rewards_per_func = gather(rewards_per_func)
        total_rewards = (total_rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

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

            if (group_std > 0).all():
                break

            valid_mask = (group_std > 0).repeat_interleave(self.num_generations)
            all_inputs = gather_object(inputs)
            valid_samples.extend([inp for inp, mask in zip(all_inputs, valid_mask) if mask])
            valid_rewards.append(rewards[valid_mask])
            valid_rewards_per_func.append(rewards_per_func[valid_mask])
            valid_completions.extend(
                [inp['messages'][-1]['content'] for inp, mask in zip(all_inputs, valid_mask) if mask])

            if len(valid_samples) >= self.global_train_batch_size:
                break

            inputs = next(self.resample_iterator)
            inputs = Trainer._prepare_inputs(self, inputs)
            inputs = self._generate_completions(inputs)
            rewards_per_func, rewards, completions = self._score_completions(inputs)
            resample_count += 1

        if len(valid_samples) >= self.global_train_batch_size:
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )
            inputs = valid_samples[:self.global_train_batch_size][process_slice]
            rewards = torch.cat(valid_rewards)[:self.global_train_batch_size]
            rewards_per_func = torch.cat(valid_rewards_per_func)[:self.global_train_batch_size]
            completions = valid_completions[:self.global_train_batch_size][process_slice]
        else:
            logger.warning(f'There are still std=0 groups present after {self.args.max_resample_times} retries.')
            inputs, rewards, rewards_per_func, completions = origin_data

        return inputs, rewards, rewards_per_func, completions

    def _prepare_batch_inputs(self, inputs: InputsType, rewards: torch.Tensor) -> InputsType:
        """Prepare the final batch inputs with advantages and other required fields"""
        # Compute advantages
        grouped_rewards = rewards.view(-1, self.num_generations)
        mean_grouped_rewards = grouped_rewards.mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = grouped_rewards.std(dim=1).repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards)
        if self.args.scale_rewards:
            advantages /= (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(inputs),
            (self.accelerator.process_index + 1) * len(inputs),
        )
        advantages = advantages[process_slice]

        # Prepare mini-batches
        mini_batch_inputs = _split_into_mini_batches(inputs, mini_batch_size=self.args.mini_batch_size)
        batch_encoded_inputs = []
        template = self.template

        for mini_batch in mini_batch_inputs:
            with self._template_context(template):
                mini_batch_encoded_inputs = [template.encode(infer_request) for infer_request in mini_batch]
                mini_batch_encoded_inputs = to_device(
                    template.data_collator(mini_batch_encoded_inputs), self.model.device)

            labels = mini_batch_encoded_inputs.pop('labels')
            logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
            mini_batch_encoded_inputs['logits_to_keep'] = logits_to_keep
            mini_batch_encoded_inputs['completion_mask'] = labels[:, -logits_to_keep:] != -100

            with torch.no_grad():
                if self.old_policy:
                    mini_batch_encoded_inputs['old_per_token_logps'] = self._get_per_token_logps(
                        self.model, mini_batch_encoded_inputs)
                else:
                    mini_batch_encoded_inputs['old_per_token_logps'] = None

                if self.beta == 0.0:
                    ref_per_token_logps = None
                elif self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, mini_batch_encoded_inputs)
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(self.model, mini_batch_encoded_inputs)

                mini_batch_encoded_inputs['ref_per_token_logps'] = ref_per_token_logps
                mini_batch_encoded_inputs['truncated_mask'] = \
                    torch.tensor([mb['is_truncated'] for mb in mini_batch], dtype=torch.bool)
                batch_encoded_inputs.append(mini_batch_encoded_inputs)
        # Split advantages into mini-batches
        mini_batch_advantages = _split_into_mini_batches(advantages, mini_batch_size=self.args.mini_batch_size)
        for i, mini_batch_advantage in enumerate(mini_batch_advantages):
            batch_encoded_inputs[i].update({'advantages': mini_batch_advantage})

        return batch_encoded_inputs

    def _log_metrics(self, inputs, messages, completions, rewards, rewards_per_func):
        """Log training/evaluation metrics"""
        mode = 'eval' if self.control.should_evaluate else 'train'
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

        # Get the names of the reward functions
        reward_func_names = []
        for reward_func in self.reward_funcs:
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split('/')[-1]
            else:
                if inspect.isfunction(reward_func):
                    reward_func_name = reward_func.__name__
                else:
                    reward_func_name = reward_func.__class__.__name__

            reward_func_names.append(reward_func_name)

        for i, reward_func_name in enumerate(reward_func_names):
            mean_rewards = rewards_per_func[:, i].mean().item()
            self._metrics[mode][f'rewards/{reward_func_name}/mean'].append(mean_rewards)
            std_rewards = rewards_per_func[:, i].std().item()
            self._metrics[mode][f'rewards/{reward_func_name}/std'].append(std_rewards)

        # Log overall reward stats
        grouped_rewards = rewards.view(-1, self.num_generations)
        self._metrics[mode]['reward'].append(grouped_rewards.mean().item())
        self._metrics[mode]['reward_std'].append(grouped_rewards.std(dim=1).mean().item())

        # Log prompt and completion texts
        self._textual_logs['prompt'].extend(gather_object(messages))
        self._textual_logs['completion'].extend(gather_object(completions))

        for i, name in enumerate(reward_func_names):
            self._textual_logs['rewards'][name].extend(rewards_per_func[:, i].tolist())

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Compute the per-token log probabilities for the model, return_outputs=True in mini-batch training
        if isinstance(inputs, list):
            assert len(inputs) == 1
            inputs = inputs[0]
        completion_mask = inputs['completion_mask']
        truncated_mask = inputs['truncated_mask']
        # apply the completion_mask to exclude loss and metrics for overlong completions
        if self.args.overlong_filter and any(truncated_mask):
            if all(truncated_mask):
                logger.info('All completions are overlong, loss and KL will be zero')
            truncated_mask = truncated_mask.unsqueeze(-1).expand_as(completion_mask).to(completion_mask.device)
            completion_mask = completion_mask * (~truncated_mask)

        per_token_logps = self._get_per_token_logps(model, inputs)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs['ref_per_token_logps']
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)

        advantages = inputs['advantages']
        old_per_token_logps = inputs['old_per_token_logps'] if self.old_policy else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        completions_length = completion_mask.sum()
        if completions_length == 0:
            # Prevent division by zero issues after all completions are filtered by the overlong filter
            completions_length += 1e-4
        loss = (per_token_loss * completion_mask).sum() / completions_length

        # Log the metrics
        metrics = {}
        mode = 'eval' if self.control.should_evaluate else 'train'

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completions_length
            metrics['kl'] = mean_kl

        is_clipped = ((coef_1 < 1 - self.epsilon_low) &
                      (advantages.unsqueeze(1) < 0)) | ((coef_1 > 1 + self.epsilon_high) &
                                                        (advantages.unsqueeze(1) > 0))

        clip_ratio = (is_clipped * completion_mask).sum() / completions_length
        metrics['clip_ratio'] = clip_ratio

        # Log metrics or return them
        if return_outputs:
            metrics['completions_length'] = completions_length
            return loss, metrics
        else:
            for key, value in metrics.items():
                self._metrics[mode][key].append(self.accelerator.gather_for_metrics(value).mean().item())
            return loss

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, inputs):
        from trl.trainer.utils import selective_log_softmax
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        unwrapped_model = self.accelerator.unwrap_model(model)
        parameters = inspect.signature(unwrapped_model.forward).parameters
        if not unwrapped_model.model_meta.is_multimodal and 'logits_to_keep' in parameters:
            # save memory
            return super()._get_per_token_logps(model, input_ids, inputs['attention_mask'], logits_to_keep)
        inputs = {
            k: v
            for k, v in inputs.items() if k not in [
                'logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps',
                'truncated_mask'
            ]
        }
        with self._template_context(self.template):
            logits = model(**inputs).logits
        # exclude the last logit: it corresponds to the next token pred
        logits = logits[:, -(logits_to_keep + 1):-1, :]
        logits = logits / self.temperature
        input_ids = input_ids[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    def evaluation_loop(self, dataloader, *args, **kwargs):
        # Wait for the training rollout to complete
        if self.args.async_generate:
            while not self.is_async_generate_eval_rollout_done():
                time.sleep(0.1)
        # set mini_batch_size None in evaluation
        mini_batch_size = self.args.mini_batch_size
        self.args.mini_batch_size = None
        if self._queue.empty() and self.args.async_generate:
            self._prefetch(dataloader)
        metric_key_prefix = kwargs['metric_key_prefix']
        output = super().evaluation_loop(dataloader, *args, **kwargs)
        metrics = {f'{metric_key_prefix}_{key}': sum(val) / len(val) for key, val in self._metrics['eval'].items()}
        output.metrics.update(metrics)
        self.args.mini_batch_size = mini_batch_size
        self.eval_flag = True
        return output

    def training_step(self, model: nn.Module, inputs: InputsType, num_items_in_batch=None) -> torch.Tensor:
        if self.args.async_generate:
            # Wait for the eval rollout to complete
            while not self.is_async_generate_eval_rollout_done():
                time.sleep(0.1)
        if self.args.mini_batch_size is None:
            return super().training_step(model, inputs, num_items_in_batch)
        model.train()
        if hasattr(self.optimizer, 'train') and callable(self.optimizer.train):
            self.optimizer.train()

        batch_inputs = self._prepare_inputs(inputs)

        total_loss = torch.tensor(0.0, device=batch_inputs[0]['input_ids'].device)
        # Initialize metrics accumulators
        total_kl = 0.0
        total_clip_ratio = 0.0
        total_completion_length = 0
        for mini_batch in batch_inputs:

            with self.compute_loss_context_manager():
                mini_batch_loss, mini_batch_metrics = self.compute_loss(model, mini_batch, return_outputs=True)
                mb_completion_length = mini_batch_metrics['completions_length']

            self.accelerator.backward(mini_batch_loss)
            # Token-level metrics are weighted by completion length to ensure a fair average over all tokens.
            if self.beta != 0.0:
                total_kl += mini_batch_metrics['kl'] * mb_completion_length
            total_clip_ratio += mini_batch_metrics['clip_ratio'] * mb_completion_length
            total_completion_length += mb_completion_length
            total_loss += mini_batch_loss * mb_completion_length

        mode = 'eval' if self.control.should_evaluate else 'train'
        if self.beta != 0.0:
            self._metrics[mode]['kl'].append(
                self.accelerator.gather_for_metrics(total_kl / total_completion_length).mean().item())
        self._metrics[mode]['clip_ratio'].append(
            self.accelerator.gather_for_metrics(total_clip_ratio / total_completion_length).mean().item())

        total_loss = total_loss / total_completion_length

        del inputs, batch_inputs
        if (self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0):
            gc_collect()

        return total_loss.detach()

    @property
    def old_policy(self):
        return self.num_iterations > 1

    @property
    def _queue(self):
        if self.control.should_evaluate:
            return self.eval_queue
        else:
            return self.train_queue

    @torch.no_grad()
    def offload_model(self):
        if len(self.offload_modules) > 0:
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for name, module in unwrapped_model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                self.offload_modules[name] = module.weight.device
                module.to('cpu')
            elif not hasattr(module, 'device'):
                pass
            elif module.device.type != 'cpu':
                self.offload_modules[name] = module.device
                module.to('cpu')

    @torch.no_grad()
    def load_model(self):
        if len(self.offload_modules) == 0:
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for name, device in self.offload_modules.items():
            module = unwrapped_model.get_submodule(name)
            if isinstance(module, torch.nn.Embedding):
                module.weight.to(device)
            else:
                module.to(device)
        self.offload_modules.clear()

    @torch.no_grad()
    def offload_optimizer(self):
        if len(self.offload_states) > 0:
            return
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        self.offload_states[key] = value.device
                        state[key] = value.to('cpu', non_blocking=True)

    @torch.no_grad()
    def load_optimizer(self):
        if len(self.offload_states) == 0:
            return
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.offload_states[key], non_blocking=True)
        self.offload_states.clear()

    @contextmanager
    def multi_turn_completion_length_context(self):
        """
        Context manager that temporarily adjusts the engine's max length handling
        for multi-turn generation scenarios.

        Ensures the total sequence length (prompt + completion) never exceeds:
            min(original_max_len, prompt_tokens + max_completion_length)
        """
        if not (self.multi_turn_func and self.infer_rank >= 0):
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
            'batch_size': self._train_batch_size,
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
            dataloader_params['worker_init_fn'] = seed_worker
            dataloader_params['prefetch_factor'] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(resample_dataset, **dataloader_params))

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # compatible with trl0.16 and trl0.17.0.dev
        # remove this function when next trl release(0.17.0)

        mode = 'eval' if self.control.should_evaluate else 'train'
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
                if self.args.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                wandb.log({'completions': wandb.Table(dataframe=df)})

    def is_async_generate_eval_rollout_done(self):
        return not self.eval_flag or not self.eval_queue.empty()

    def is_async_generate_train_rollout_done(self):
        return not self.train_queue.empty()
