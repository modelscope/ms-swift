# Copyright (c) Alibaba, Inc. and its affiliates.
import gc
import inspect
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from functools import partial
from types import MethodType
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from megatron.core import mpu
from megatron.training import get_args, training
from trl.trainer.grpo_trainer import nanstd
from vllm.distributed import parallel_state as vllm_ps

from swift.llm import RequestConfig, RowPreprocessor, Template, to_device
from swift.llm.infer.protocol import RolloutOutput
from swift.plugin import orms
from swift.trainers.rlhf_trainer.grpo_trainer import DataType
from swift.trainers.rlhf_trainer.utils import replace_assistant_response_with_ids
from swift.utils import get_current_device, get_logger, is_vllm_available, remove_response
from ..argument import MegatronArguments, MegatronRLHFArguments
from .rlhf_mixin import MegatronRLHFTrainer
from .utils import (gather, gather_object, load_megatron_model_to_gpu, load_megatron_optimizer, log_gpu_memory,
                    offload_megatron_model_to_cpu, offload_megatron_optimizer, profiling_context)

try:
    from mbridge import AutoBridge
except ImportError:
    pass

logger = get_logger()


class MegatronGRPOTrainer(MegatronRLHFTrainer):

    def __init__(self, args: MegatronRLHFArguments, template: Template):
        super().__init__(args, template)
        self.args = args
        self.hf_model_dir = args.model_info.model_dir
        self.processing_class = self.template.processor
        # TODO: multi turn scheduler(colocate multi turn)
        self._prepare_template_data_collator()
        self._init_grpo_params()
        self._prepare_rewards()
        self._prepare_rollout_engine()
        # debug: use mbridge to convert mcore to hf
        self.bridge = None
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}

    def _prepare_template_data_collator(self):
        template = self.template
        args = self.args
        data_collator = template.data_collator
        padding_to = None
        if args.tensor_model_parallel_size > 1 and args.sequence_parallel:
            padding_to = args.tensor_model_parallel_size
        if args.context_parallel_size > 1:
            padding_to = (padding_to or 1) * args.context_parallel_size
        if args.fp8_format:
            padding_to = max((padding_to or 1) * 8, 16)
        logger.info(f'padding_to: {padding_to}')
        data_collator = partial(data_collator, padding_to=padding_to)
        template.data_collator = data_collator

    def _init_grpo_params(self):
        args: MegatronArguments = self.args
        # distributed params
        self.world_size = torch.distributed.get_world_size()
        self.process_index = torch.distributed.get_rank()
        self.is_main_process = self.process_index == 0
        self.device = get_current_device()
        # algorithm params
        self.num_generations = args.num_generations  # G in the GRPO paper
        self.beta = args.beta
        self.temperature = args.temperature
        self.loss_type = args.loss_type
        self.max_completion_length = args.max_completion_length
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.top_entropy_quantile = args.top_entropy_quantile
        self.importance_sampling_level = args.importance_sampling_level
        self.enable_offload = False
        # batch size (completion-level)
        self.generation_batch_size = args.generation_batch_size
        self.steps_per_generation = args.steps_per_generation
        self.global_batch_size = args.global_batch_size
        self.micro_batch_size = args.micro_batch_size
        self.per_device_generation_batch_size = args.per_device_generation_batch_size

        # sampling params
        self.request_config = RequestConfig(
            n=1,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop_words,
            return_details=True)

        self._step = 0

    def _prepare_rollout_engine(self):
        args = self.args
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.use_vllm = args.use_vllm
        self.async_generate = args.async_generate
        self.use_fast_infer = self.use_vllm  # whether to use the PT backend
        self.vllm_use_async_engine = False
        self.enable_offload = False
        self.use_gym_env = False
        self.enable_server_multi_turn = False  # TODO
        # for multi-turn server, maybe the num of rollout outputs is not equal to the num of rollout inputs
        self.dynamic_num_samples = False
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                                  'Please install vLLM with `pip install vllm -U` to use it.')
            assert self.vllm_mode == 'colocate'  # TODO: server mode

            if not self.world_size % self.vllm_tensor_parallel_size == 0:
                raise ValueError(f'vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size '
                                 f'({self.world_size}) evenly.')

            self.enable_offload = self.args.offload_model or self.args.offload_optimizer
            context = self.offload_context if self.enable_offload else nullcontext

            with context():
                self.engine = self.prepare_vllm()
                if self.args.sleep_level > 0:
                    self.engine.engine.sleep(self.args.sleep_level)
                    log_gpu_memory('after sleep vLLM engine')

    def prepare_vllm(self):
        from swift.llm.infer.infer_engine import GRPOVllmEngine
        args = self.args
        max_num_seqs = self.per_device_generation_batch_size * self.vllm_tensor_parallel_size
        vllm_template = copy(self.template)
        vllm_template.padding_free = False
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
            load_format='dummy',
            template=vllm_template,
            distributed_executor_backend='external_launcher',
        )
        if self.vllm_tensor_parallel_size > 1:
            self.vllm_tp_group = vllm_ps.get_tp_group().device_group
        self._buffered_inputs = None
        return engine

    def _move_model_to_vllm(self):
        # TODO: LoRA, server
        if self.bridge is None:
            self.bridge = AutoBridge.from_pretrained(self.hf_model_dir)
            self._patch_mbridge(self.bridge)
        per_tensor_params = self.bridge.export_weights(self.unwrapped_models)
        self.engine.inner_model.load_weights(per_tensor_params)  # TODO: check tensor_model_parallel

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

    def _patch_mbridge(self, bridge):
        original_method = bridge._weight_to_hf_format

        def _weight_to_hf_format_patched(mcore_weights_name, mcore_weights):
            # skip ViT weights
            if 'visual' in mcore_weights_name:
                if 'visual.visual' in mcore_weights_name:
                    mcore_weights_name = mcore_weights_name.replace('visual.visual', 'visual')
                return [mcore_weights_name], [mcore_weights]
            return original_method(mcore_weights_name, mcore_weights)

        bridge._weight_to_hf_format = _weight_to_hf_format_patched

    def _replace_data_iterator(self, data_iterator, model):

        if self._step % self.steps_per_generation == 0:
            # each rollout DP group will generate generation_batch_size / world_size completions
            completions_to_rollout = self.generation_batch_size // mpu.get_data_parallel_world_size()
            # completions will be repeated num_generations times after
            # so we need to divide num_iters_per_step by num_generations to get prompt batch size
            prompts_to_rollout = completions_to_rollout // self.num_generations
            # every iter will generate micro_batch_size prompts
            num_iters_per_step = prompts_to_rollout // self.micro_batch_size
            assert num_iters_per_step > 0, (
                f'num_iters_per_step={num_iters_per_step} <= 0. '
                f'This means no prompts will be generated'
                f'generation_batch_size={self.generation_batch_size}, '
                f'data_parallel_world_size={mpu.get_data_parallel_world_size()}, '
                f'num_generations={self.num_generations}, '
                f'micro_batch_size={self.micro_batch_size}. '
                'Please adjust these parameters so that '
                'generation_batch_size // data_parallel_world_size // num_generations // micro_batch_size >= 1.')
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
        self._step += 1
        inputs = self._buffered_inputs[self._step % self.steps_per_generation]
        return iter(inputs)

    def _generate_and_score_completions(self, batch):
        rollout_group = mpu.get_model_parallel_group()

        # batch : same across DP groups
        def get_local_rollout_batch(batch):
            # repeat num_generations times
            global_rollout_batch = [deepcopy(item) for item in batch for _ in range(self.num_generations)]
            # get local rollout data
            rollout_rank = torch.distributed.get_rank(group=rollout_group)
            rollout_group_size = torch.distributed.get_world_size(group=rollout_group)
            per_device_batch_size = self.per_device_generation_batch_size
            assert rollout_group_size * per_device_batch_size == len(global_rollout_batch)
            data_slice = slice(rollout_rank * per_device_batch_size, (rollout_rank + 1) * per_device_batch_size)
            rollout_batch = global_rollout_batch[data_slice]
            return rollout_batch

        # Step1: Rollout / Reward / Advantage

        rollout_batch = get_local_rollout_batch(batch)

        rollout_batch = self._generate_completions(rollout_batch)

        rewards_per_func = self._score_completions(rollout_batch)

        advantages = self._compute_advantages(rollout_batch, rewards_per_func)

        def _get_encoded_batch(rollout_batch, advantages):
            template = self.template
            encoded_batch = [template.encode(data, return_length=True) for data in rollout_batch]
            encoded_batch = to_device(template.data_collator(encoded_batch), self.device)
            labels = encoded_batch['labels']
            # TODO: logits_to_keep
            # logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
            if self.template.padding_free:
                position_ids = encoded_batch.get('text_position_ids')
                if position_ids is None:
                    position_ids = encoded_batch.get('position_ids')
                squeezed_position_ids = position_ids.squeeze()
                assert squeezed_position_ids is not None
                # Remove trailing padding zeros from position_ids to avoid interference
                # Find the last non-zero position
                last_nonzero_idx = (squeezed_position_ids != 0).nonzero(as_tuple=True)[0]
                if len(last_nonzero_idx) > 0:
                    # Keep only up to the last non-zero position + 1 to include the last valid position
                    squeezed_position_ids = squeezed_position_ids[:last_nonzero_idx[-1] + 1]

                # Calculate lengths based on sequence boundaries (position_ids == 0)
                lengths = torch.diff(
                    torch.cat([(squeezed_position_ids == 0).nonzero(as_tuple=True)[0],
                               torch.tensor([len(squeezed_position_ids)]).to(squeezed_position_ids.device)]))
                advantages = torch.repeat_interleave(advantages, lengths)

                # Pad advantages to match the original position_ids length
                original_length = position_ids.shape[1]
                if advantages.shape[0] < original_length:
                    padding_length = original_length - advantages.shape[0]
                    padding = torch.zeros(padding_length, device=advantages.device, dtype=advantages.dtype)
                    advantages = torch.cat([advantages, padding])

            encoded_batch.update({
                'completion_mask':
                labels != -100,
                'truncated_mask':
                torch.tensor([b['is_truncated'] for b in rollout_batch], dtype=torch.bool, device=self.device),
                'advantages':
                advantages,
            })

            return encoded_batch

        # Step2: ref/old logps
        rollout_group
        total_batch = gather_object(rollout_batch, group=rollout_group)
        total_advantages = gather(advantages, group=rollout_group)
        mini_batch_data = []
        for idx in range(0, len(total_batch), self.micro_batch_size):
            micro_batch_data = _get_encoded_batch(total_batch[idx:idx + self.micro_batch_size],
                                                  total_advantages[idx:idx + self.micro_batch_size])
            micro_batch_data = self._maybe_compute_logps(micro_batch_data)
            mini_batch_data.append(micro_batch_data)

        return mini_batch_data

    def _generate_completions(self, batch):
        """
        Generate completions for a batch of rollout data using vLLM engine.

        This method processes rollout data for the current process, generates completions
        using the vLLM engine, and merges the results back into the original batch.

        Args:
            batch: Rollout data assigned to the current process.

        Returns:
            batch: The input batch with rollout completion results merged in.

        Note:
            Currently only supports colocate mode. Server mode support is planned for future implementation.
        """
        # TODO: server mode
        assert self.vllm_mode == 'colocate'
        # Step 1: Wake up the engine if it's sleeping (vLLM colocate mode)
        context = self.offload_context if self.enable_offload else nullcontext
        with context():
            if self.engine.inner_model_executor.is_sleeping:
                wake_up_params = inspect.signature(self.engine.engine.wake_up).parameters
                # Load weights only (faster and reduces memory peak)
                kwargs = {'tags': ['weights']} if 'tags' in wake_up_params else {}
                self.engine.engine.wake_up(**kwargs)
                log_gpu_memory(f'after wake up vLLM engine with {kwargs}')

            # Step 2: Load model weights
            self._move_model_to_vllm()

            if (self.engine.inner_model_executor.is_sleeping
                    and 'tags' in inspect.signature(self.engine.engine.wake_up).parameters):
                self.engine.engine.wake_up(tags=['kv_cache'])
                log_gpu_memory('after wake up vLLM engine with kv_cache')

            # Step3: Rollout
            batch = self.preprocess_rollout_data(batch)
            outputs: List[RolloutOutput] = self._rollout(batch)

            # Step4: Sleep to release memory
            if self.args.sleep_level > 0:
                self.engine.engine.sleep(self.args.sleep_level)
                log_gpu_memory('after sleep vLLM engine')
            batch = self.postprocess_rollout_data(batch, outputs)

        return batch

    def preprocess_rollout_data(self, batch):
        """
        Gather rollout trajectories across the vLLM tensor-parallel (TP) group.

        This method collect the full batch on every rank, then flattens
        the nested lists into a single list of samples.

        Args:
            batch (list): List of rollout samples local to this TP rank.

        Returns:
            list: Flattened list containing all rollout samples from every
                rank in the TP group.
        """
        if self.vllm_tensor_parallel_size == 1:
            return batch

        gathered_batch = [None for _ in range(self.vllm_tensor_parallel_size)]
        torch.distributed.all_gather_object(gathered_batch, batch, group=self.vllm_tp_group)
        flattened_batch = [p for sublist in gathered_batch for p in sublist]
        return flattened_batch

    def _rollout(self, batch) -> List[RolloutOutput]:
        request_config = self._get_request_config()
        # TODO: server mode
        rollout_outputs = self._colocate_rollout(batch, request_config)
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

        if self.vllm_tensor_parallel_size > 1:
            local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
            orig_size = len(outputs) // self.vllm_tensor_parallel_size
            tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
            outputs = outputs[tp_slice]

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

    def _colocate_rollout(self, batch, request_config: RequestConfig):
        outputs: List[RolloutOutput] = self.engine.infer(infer_requests=batch, request_config=request_config)
        return outputs

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

    def _compute_advantages(self, batch: DataType, rewards_per_func: torch.Tensor) -> torch.Tensor:
        """Compute advantages for RL training."""

        def maybe_normalize_advantages(advantages: torch.Tensor, rewards_std: torch.Tensor) -> torch.Tensor:
            """Normalize advantages if configured; otherwise, return as-is."""
            if self.args.scale_rewards:
                return advantages / (rewards_std + 1e-4)
            return advantages

        total_rewards_per_func = gather(rewards_per_func)
        rewards = (total_rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)
        grouped_rewards = rewards.view(-1, self.num_generations)
        group_rewards_mean = grouped_rewards.mean(dim=1)
        group_rewards_std = grouped_rewards.std(dim=1)

        # Broadcast stats back to the original shape
        group_rewards_mean = group_rewards_mean.repeat_interleave(self.num_generations)
        group_rewards_std = group_rewards_std.repeat_interleave(self.num_generations)

        # Compute advantages relative to group mean
        advantages = rewards - group_rewards_mean
        advantages = maybe_normalize_advantages(advantages, group_rewards_std)

        def log_rewards_metrics(rewards: torch.Tensor, rewards_per_func_for_metrics: torch.Tensor):
            """Log reward statistics for monitoring. Only log once per unique request_id."""
            # rewards: [prompt_batch_size, self.num_generations]
            # rewards_per_func_for_metrics: [prompt_batch_size*self.num_generations, self.num_reward_funcs]
            mode = 'train' if self.unwrapped_models[0].training else 'eval'
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

        log_rewards_metrics(rewards=grouped_rewards, rewards_per_func_for_metrics=rewards_per_func)

        slice_start = self.process_index * len(batch)
        slice_end = slice_start + len(batch)
        advantages = advantages[slice_start:slice_end]

        return advantages

    def _maybe_compute_logps(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: entropy
        inputs = {k: v for k, v in batch.items() if k not in ['completion_mask', 'advantages', 'truncated_mask']}
        if self.beta != 0.0:
            with torch.no_grad(), self.null_ref_context() as ref_models:
                assert len(ref_models) == 1, 'GRPO currently does not support VPP.'
                ref_model = ref_models[0]
                batch['ref_per_token_logps'] = self.model_forward(
                    ref_model, iter([inputs]), no_grad=True, per_token=True)['logps']

        if not self.on_policy:
            batch['old_per_token_logps'] = self.model_forward(
                self.unwrapped_models[0], iter([inputs]), no_grad=True, per_token=True)['logps']
        return batch

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

    def forward_step(self, data_iterator, model):
        # train_batch_size
        # return: output_tensor, loss_func
        data = self.get_batch(data_iterator)
        data.pop('loss_scale', None)
        inputs = {
            k: v
            for k, v in data.items() if k not in
            ['completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps', 'truncated_mask']
        }

        with self.stimer:
            output_tensor = model(**inputs)
        return output_tensor, partial(self.loss_func, data=data)

    def loss_func(self, output_tensor: torch.Tensor, data: Dict[str, Any]):
        advantages = data['advantages']
        labels = data['labels']
        completion_mask = data['completion_mask']
        packed_seq_params = data['packed_seq_params']
        truncated_mask = data['truncated_mask']
        micro_batch_size = self.micro_batch_size
        lengths = packed_seq_params.cu_seqlens_q[1:micro_batch_size
                                                 + 1] - packed_seq_params.cu_seqlens_q[:micro_batch_size]
        lengths_with_padding = packed_seq_params.cu_seqlens_q[1:] - packed_seq_params.cu_seqlens_q[:-1]
        per_token_logps = self.get_logps(
            output_tensor, labels, packed_seq_params, packed_seq_params.num_samples, per_token=True)

        if self.args.overlong_filter and any(truncated_mask):
            # TODO: non-padding-free
            truncated_mask = torch.repeat_interleave(truncated_mask, lengths).unsqueeze(0)
            padding_length = completion_mask.shape[1] - truncated_mask.shape[1]
            if padding_length > 0:
                padding = torch.zeros((1, padding_length), device=truncated_mask.device, dtype=truncated_mask.dtype)
                truncated_mask = torch.cat([truncated_mask, padding], dim=1)
            completion_mask = completion_mask & (~truncated_mask)

        if self.beta != 0.0:
            ref_per_token_logps = data.get('ref_per_token_logps')
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)

        old_per_token_logps = (
            per_token_logps.detach() if data.get('old_per_token_logps') is None else data['old_per_token_logps'])
        log_ratio = per_token_logps - old_per_token_logps

        if self.importance_sampling_level == 'token':
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == 'sequence':
            log_ratio_list = torch.split(log_ratio.squeeze(0), lengths_with_padding.tolist())
            mask_list = torch.split(completion_mask.squeeze(0), lengths_with_padding.tolist())
            seq_weights = [(lr * m).sum() / m.sum().clamp(min=1.0) for lr, m in zip(log_ratio_list, mask_list)]
            seq_level_log_weights = torch.stack(seq_weights).to(log_ratio.dtype).unsqueeze(-1)
            if self.importance_sampling_level == 'sequence':
                log_importance_weights = seq_level_log_weights
            else:
                seq_level_log_weight = seq_level_log_weights.detach()
                seq_level_log_weight = torch.repeat_interleave(seq_level_log_weight, lengths).unsqueeze(0)
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
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == 'grpo':
            loss_list = torch.split(per_token_loss.squeeze(0), lengths_with_padding.tolist())
            mask_list = torch.split(completion_mask.squeeze(0), lengths_with_padding.tolist())
            sample_loss = [(loss * mask).sum() / mask.sum().clamp(min=1.0) for loss, mask in zip(loss_list, mask_list)]
            loss = torch.stack(sample_loss[:micro_batch_size]).mean()
        elif self.loss_type == 'bnpo':
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == 'dr_grpo':
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        loss = loss.mean()
        avg_metric = {
            'loss': loss.clone().detach(),
            'completions/mean_length': lengths.float().mean(),
        }
        max_metric = {
            'completions/max_length': lengths.float().max(),
        }
        min_metric = {
            'completions/min_length': lengths.float().min(),
        }
        if self.beta != 0.0:
            avg_metric['kl'] = per_token_kl.mean().item()
        avg_reporting_metric = loss.new_tensor(list(avg_metric.values()))
        max_reporting_metric = loss.new_tensor(list(max_metric.values()))
        min_reporting_metric = loss.new_tensor(list(min_metric.values()))
        torch.distributed.all_reduce(
            avg_reporting_metric, torch.distributed.ReduceOp.AVG, group=mpu.get_data_parallel_group())

        torch.distributed.all_reduce(
            max_reporting_metric, torch.distributed.ReduceOp.MAX, group=mpu.get_data_parallel_group())
        torch.distributed.all_reduce(
            min_reporting_metric, torch.distributed.ReduceOp.MIN, group=mpu.get_data_parallel_group())
        avg_reporting_metric = {k: avg_reporting_metric[i] for i, k in enumerate(avg_metric.keys())}
        max_reporting_metric = {k: max_reporting_metric[i] for i, k in enumerate(max_metric.keys())}
        min_reporting_metric = {k: min_reporting_metric[i] for i, k in enumerate(min_metric.keys())}
        addition_metrics = {
            key: torch.tensor(sum(val) / len(val), device=loss.device)
            for key, val in self._metrics['train'].items()
        }

        reporting_metric = {**avg_reporting_metric, **max_reporting_metric, **min_reporting_metric, **addition_metrics}
        # fix megatron-lm bug
        # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.12.0/megatron/core/pipeline_parallel/schedules.py#L291
        loss = loss / mpu.get_context_parallel_world_size()
        return loss, reporting_metric

    def model_forward(self, model, data_iterator, no_grad=True, per_token=False):
        # used to calculate model forward (logps) in GRPO
        with self.stimer(bdata=True):
            data = self.get_batch(data_iterator)
        data.pop('loss_scale', None)
        labels = data.get('labels')
        context = torch.no_grad() if no_grad else nullcontext()
        with context:
            output_tensor = self._forward_step_helper(model, data)
        packed_seq_params = data['packed_seq_params']
        data['logps'] = None if labels is None else self.get_logps(
            output_tensor, labels, data['packed_seq_params'], packed_seq_params.num_samples, per_token=per_token)
        return data

    @contextmanager
    def offload_context(self):
        if self.args.offload_model:
            offload_megatron_model_to_cpu(self.unwrapped_models)
            log_gpu_memory('after offload model to cpu')
        # if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
        #     self.offload_optimizer()

        try:
            yield
        finally:
            # reload (load back) model when exiting context
            if self.args.offload_model:
                load_megatron_model_to_gpu(self.unwrapped_models)
                log_gpu_memory('after load model to gpu')
            # if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
            #     self.load_optimizer()
