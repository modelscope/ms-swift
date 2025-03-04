# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/trl.
import concurrent.futures
import inspect
import os
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Union, re

import numpy as np
import torch
import torch.nn as nn
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from transformers import PreTrainedModel, TrainerCallback
from trl import GRPOTrainer as HFGRPOTrainer
from trl.models import unwrap_model_for_generation

from swift.llm import InferRequest, RequestConfig, RowPreprocessor, to_device
from swift.plugin import orms
from swift.utils import (JsonlWriter, get_device, get_device_count, get_dist_setting, get_logger, get_node_setting,
                         is_lmdeploy_available, is_vllm_available, is_wandb_available)
from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

try:
    from trl.extras.profiling import profiling_decorator
except ImportError:
    raise ImportError('Please install trl from source using: `pip install git+https://github.com/huggingface/trl.git`')

del HFGRPOTrainer.__init__

logger = get_logger()
if is_wandb_available():
    import wandb


class GRPOCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer

    # offload original_modules to cpu, to save memory
    def on_train_begin(self, args, state, control, **kwargs):
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
        self.queue = Queue()
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
        model.warnings_issued['estimate_tokens'] = True
        kwargs['data_collator'] = lambda features: features
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}

        use_vllm = args.use_vllm
        use_lmdeploy = args.use_lmdeploy

        super().__init__(model, ref_model, *_args, **kwargs)

        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
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

        if use_vllm or use_lmdeploy:
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
                    # Check that the requested device is not also used for training
                    if _device in {get_device(idx) for idx in range(self.accelerator.num_processes)}:
                        logger.warning(f'The requested device {_device} is also used for training. '
                                       f'This may lead to unexpected behavior. '
                                       f'It is recommended to use a dedicated device for vLLM.')

                if use_vllm:
                    if not is_vllm_available():
                        raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                                          'Please install vLLM with `pip install vllm` to use it.')
                    from swift.llm import VllmEngine
                    from swift.tuners import Swift
                    with Swift.grpo_context(model, self.template.processor):
                        self.engine = VllmEngine(
                            model.model_dir,
                            model.model_info.torch_dtype,
                            model_type=model.model_meta.model_type,
                            device=fast_infer_device[self.local_infer_rank],
                            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                            enable_prefix_caching=args.vllm_enable_prefix_caching,
                            max_num_seqs=args.vllm_max_num_seqs,
                            enforce_eager=args.vllm_enforce_eager,
                            limit_mm_per_prompt=args.vllm_limit_mm_per_prompt,
                            max_model_len=args.vllm_max_model_len)
                    self.engine.default_template = self.template
                elif use_lmdeploy:
                    # https://github.com/tastelikefeet/lmdeploy.git@feat/reload_state_dict_064
                    # Compile:https://github.com/tastelikefeet/lmdeploy/blob/main/docs/en/get_started/installation.md
                    if not is_lmdeploy_available():
                        raise ImportError('Please install `pip install lmdeploy==0.7.1`'
                                          ' and replace three files with:\n'
                                          '1. https://github.com/tastelikefeet/lmdeploy/blob/feat/'
                                          'reload_state_dict_064/lmdeploy/messages.py\n'
                                          '2. https://github.com/tastelikefeet/lmdeploy/blob/feat/'
                                          'reload_state_dict_064/lmdeploy/turbomind/turbomind.py\n'
                                          '3. https://github.com/tastelikefeet/lmdeploy/blob/feat/'
                                          'reload_state_dict_064/lmdeploy/turbomind/deploy/loader.py\n')
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
                    self.engine.default_template = self.template
            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            from swift.llm import PtEngine
            self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop_words,
        )

        self.model_accepts_loss_kwargs = False
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
        self.log_completions = args.log_completions
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'completions.jsonl'))

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon = args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle. # noqa
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps
        if self.args.async_generate:
            self.add_callback(GRPOCallback(self))

    @property
    def infer_rank(self):
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        assert local_world_size % self.args.num_infer_workers == 0
        step = local_world_size // self.args.num_infer_workers
        for _vllm_rank in range(self.args.num_infer_workers):
            _assigned = _vllm_rank * step
            if local_rank == _assigned:
                return get_node_setting()[0] * self.args.num_infer_workers + _vllm_rank

        return -1

    @property
    def local_infer_rank(self):
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        assert local_world_size % self.args.num_infer_workers == 0
        step = local_world_size // self.args.num_infer_workers
        for _vllm_rank in range(self.args.num_infer_workers):
            _assigned = _vllm_rank * step
            if local_rank == _assigned:
                return _vllm_rank

        return -1

    @staticmethod
    def round_robin(num_reqs, nodes):
        distribution = [[] for _ in range(nodes)]
        for idx in range(num_reqs):
            node_id = idx % nodes
            distribution[node_id].append(idx)
        return distribution

    @staticmethod
    @contextmanager
    def _template_context(template):
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
    def _move_model_to_vllm_lmdeploy(self):
        from accelerate.utils.other import is_compiled_module
        with unwrap_model_for_generation(
                self.model, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
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
            if self.infer_rank >= 0:
                if self.args.async_generate:
                    self._wait_queue()
                if self.args.use_vllm:
                    llm_model = self.engine.engine.engine.model_executor.driver_worker.model_runner.model
                else:
                    llm_model = self.engine.engine.engine
                llm_model.load_weights(state_dict.items())
            # Unmerge the adapter to restore the model to its original state.
            # This must be done after loading weights to ensure they correspond to the merged state.
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()

    def _wait_queue(self):
        while self.queue.empty():
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

    def async_infer(self, inputs, inputs_slice, distributed_idx):
        future: Future = self.executor.submit(
            self.engine.infer, infer_requests=inputs_slice, request_config=self.request_config, use_tqdm=False)

        def done(_self):
            self.queue.put(DataCache(inputs, _self.result(), distributed_idx))

        future.add_done_callback(done)

    def _prefetch(self, train_dataloader):
        inputs = next(iter(train_dataloader))
        all_inputs = gather_object(inputs)
        distributed_idx = self.round_robin(len(all_inputs), get_node_setting()[1] * self.args.num_infer_workers)
        if self.infer_rank >= 0:
            _input_slice = np.array(all_inputs)[distributed_idx[self.infer_rank]]
            outputs = self.engine.infer(_input_slice, self.request_config, use_tqdm=False)
            self.queue.put(DataCache(inputs, outputs, distributed_idx))
        else:
            self.queue.put(DataCache(inputs, [], distributed_idx))
        if self.accelerator.num_processes > 1:
            self.accelerator.wait_for_everyone()

    def _fast_infer(self, inputs):
        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm_lmdeploy()
            self._last_loaded_step = self.state.global_step
        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        all_inputs = gather_object(inputs)
        # Distribute inputs to different workers
        # for example, 2 workers, 6 inputs, 0/2/4 dispatch to the first worker
        # 1/3/5 dispatch to the second worker
        # trying to shuffle and average the length
        distributed_idx = self.round_robin(len(all_inputs), get_node_setting()[1] * self.args.num_infer_workers)
        if self.infer_rank >= 0:
            _input_slice = np.array(all_inputs)[distributed_idx[self.infer_rank]]
            if self.args.async_generate:
                self.async_infer(inputs, _input_slice, distributed_idx)
                data_cache = self.queue.get()
                inputs = data_cache.inputs
                outputs = data_cache.outputs
                distributed_idx = data_cache.distributed_idx
            else:
                outputs = self.engine.infer(_input_slice, self.request_config, use_tqdm=False)
        else:
            if self.args.async_generate:
                self.queue.put(DataCache(inputs, [], distributed_idx))
                data_cache = self.queue.get()
                inputs = data_cache.inputs
                distributed_idx = data_cache.distributed_idx
            outputs = []
        outputs = gather_object(outputs)
        outputs = self.reorder_outputs(outputs, distributed_idx)
        return inputs, outputs

    @property
    def old_policy(self):
        return self.num_iterations > 1 or self.args.async_generate

    def _generate_and_score_completions(
            self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:

        device = self.accelerator.device
        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm or self.args.use_lmdeploy:
            inputs, outputs = self._fast_infer(inputs)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            # outputs = broadcast_object_list(outputs, from_process=0)
        else:
            # Regular generation path
            is_multimodal = self.model.model_meta.is_multimodal
            if is_multimodal:
                models = self.template.remove_post_encode_hook()
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator):
                # same reference
                outputs = self.engine.infer(inputs, self.request_config, use_tqdm=False)
                self.model.train()
            if is_multimodal:
                self.template.register_post_encode_hook(models)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(inputs),
            (self.accelerator.process_index + 1) * len(inputs),
        )
        if self.args.use_vllm or self.args.use_lmdeploy:
            outputs = outputs[process_slice]

        for i, output in enumerate(outputs):
            messages = inputs[i]['messages']
            InferRequest.remove_response(messages)
            messages.append({'role': 'assistant', 'content': output.choices[0].message.content})
        from copy import copy
        template = copy(self.template)
        with self._template_context(template):
            batched_inputs = [template.encode(infer_request) for infer_request in inputs]
            outputs = to_device(template.data_collator(batched_inputs), self.model.device)

        # we only need to compute the logits for the completion tokens
        labels = outputs.pop('labels')
        logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
        outputs['logits_to_keep'] = logits_to_keep
        outputs['completion_mask'] = labels[:, -logits_to_keep:] != -100

        with torch.no_grad():
            if self.old_policy:
                outputs['old_per_token_logps'] = self._get_per_token_logps(self.model, outputs)
            else:
                outputs['old_per_token_logps'] = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, outputs)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(self.model, outputs)

        rewards_per_func = torch.zeros((len(inputs), len(self.reward_funcs)), device=device)
        completions = [example['messages'][-1]['content'] for example in inputs]

        for i, (reward_func, reward_template) in enumerate(zip(self.reward_funcs, self.reward_templates)):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                with self._template_context(reward_template):
                    batched_inputs = [reward_template.encode(infer_request) for infer_request in inputs]
                    reward_inputs = to_device(reward_template.data_collator(batched_inputs), reward_func.device)

                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # Repeat all input columns (but "messages" and "completion") to match the number of generations
                reward_kwargs = RowPreprocessor.rows_to_batched(inputs)
                output_reward_func = reward_func(completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)
        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages = advantages[process_slice]

        # Log the metrics
        mode = 'eval' if self.control.should_evaluate else 'train'
        completion_length = self.accelerator.gather_for_metrics(outputs['completion_mask'].sum(1)).float().mean().item()
        self._metrics[mode]['completion_length'].append(completion_length)
        # clip ratio
        response_clip_ratio = torch.gt(
            self.accelerator.gather_for_metrics(outputs['completion_mask'].sum(1)),
            self.args.max_completion_length).float().mean().item()
        self._metrics[mode]['response_clip_ratio'].append(response_clip_ratio)
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split('/')[-1]
            else:
                if inspect.isfunction(reward_func):
                    reward_func_name = reward_func.__name__  # function
                else:
                    reward_func_name = reward_func.__class__.__name__  # method
            self._metrics[mode][f'rewards/{reward_func_name}'].append(reward_per_func[i].item())

        self._metrics[mode]['reward'].append(rewards.mean().item())
        self._metrics[mode]['reward_std'].append(std_grouped_rewards.mean().item())
        outputs.update({
            'ref_per_token_logps': ref_per_token_logps,
            'advantages': advantages,
        })
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            # For logging
            table = {
                'step': [str(self.state.global_step)] * len(rewards),
                'messages': [inputs['messages'][:-1] for inputs in gather_object(inputs)],
                'completion': gather_object(completions),
                'reward': rewards.tolist(),
            }
            self.jsonl_writer.append(table)
            if 'wandb' in self.args.report_to and wandb.run is not None and self.accelerator.is_main_process:
                import pandas as pd
                df = pd.DataFrame(table)
                wandb.log({'completions': wandb.Table(dataframe=df)})

        return outputs

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError('The GRPOTrainer does not support returning outputs')
        # Compute the per-token log probabilities for the model
        completion_mask = inputs['completion_mask']
        per_token_logps = self._get_per_token_logps(model, inputs)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs['ref_per_token_logps']
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)

        advantages = inputs['advantages']
        old_per_token_logps = inputs['old_per_token_logps'] if self.old_policy else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = 'eval' if self.control.should_evaluate else 'train'

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics[mode]['kl'].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]['clip_ratio'].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
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
            for k, v in inputs.items() if k not in
            ['logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps']
        }
        logits = model(**inputs).logits
        # exclude the last logit: it corresponds to the next token pred
        logits = logits[:, -(logits_to_keep + 1):-1, :]
        input_ids = input_ids[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    def evaluation_loop(self, *args, **kwargs):
        metric_key_prefix = kwargs['metric_key_prefix']
        output = super().evaluation_loop(*args, **kwargs)
        metrics = {f'{metric_key_prefix}_{key}': sum(val) / len(val) for key, val in self._metrics['eval'].items()}
        output.metrics.update(metrics)
        return output
