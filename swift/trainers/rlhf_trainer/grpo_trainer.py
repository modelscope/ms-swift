# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/trl.

# fmt: off
# apply patch before importing trl, which may internally reference GuidedDecodingParams
try:
    import vllm
    try:
        from vllm.sampling_params import GuidedDecodingParams
    except ImportError:
        import vllm.sampling_params
        # removed in https://github.com/vllm-project/vllm/pull/22772
        vllm.sampling_params.GuidedDecodingParams = vllm.sampling_params.StructuredOutputsParams
except ImportError:
    pass
# fmt: on

import concurrent.futures
import inspect
import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from packaging import version
from transformers import PreTrainedModel
from transformers.trainer import Trainer
from trl import GRPOTrainer as HFGRPOTrainer
from trl.models import prepare_deepspeed
from trl.trainer import grpo_trainer
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_trainer import RepeatSampler, nanmax, nanmin, nanstd
from trl.trainer.utils import selective_log_softmax

from swift.llm import RowPreprocessor, Template, to_device
from swift.llm.template.template_inputs import TemplateInputs
from swift.plugin import orms, rm_plugins
from swift.utils import (JsonlWriter, get_logger, is_swanlab_available, is_wandb_available, remove_response,
                         seed_worker, unwrap_model_for_generation)
from ..mixin import SwiftMixin
from .rollout_mixin import DataType, RolloutTrainerMixin
from .utils import (_ForwardRedirection, compute_chord_loss, get_even_process_data, identity_data_collator,
                    load_pil_img, make_chord_sft_dataset, pad_logps_back_to_batch, patch_profiling_context,
                    patch_profiling_decorator, patch_save_last_checkpoint, replace_assistant_response_with_ids)

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


class GRPOTrainer(RolloutTrainerMixin, SwiftMixin, HFGRPOTrainer):
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
        self.is_multimodal = model.model_meta.is_multimodal

        model.warnings_issued['estimate_tokens'] = True
        kwargs['data_collator'] = identity_data_collator  # No data collation is needed in GRPO

        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys() if not hasattr(model, 'get_base_model') else
            inspect.signature(model.get_base_model().forward).parameters.keys())

        self.vllm_client = kwargs.pop('vllm_client', None)
        self.chord_sft_dataset = kwargs.pop('chord_sft_dataset', None)
        reward_templates = kwargs.pop('reward_template', None)
        self._prepare_algorithm_params()
        super().__init__(model, ref_model, *_args, **kwargs)
        self._prepare_chord_dataset()
        self.prepare_rollout()
        self._prepare_rewards(reward_funcs, reward_model, reward_templates)

        if not self.reward_funcs and not self.use_gym_env:
            raise ValueError('You must specify reward_funcs or reward_model')

        if self.args.eval_strategy != 'no':
            total_eval_batch_size = self.args.per_device_eval_batch_size * \
                self.accelerator.num_processes // self.args.num_generations
            assert len(self.eval_dataset) >= total_eval_batch_size, (
                f'eval_dataset size {len(self.eval_dataset)} is smaller than '
                f'total_eval_batch_size {total_eval_batch_size}. '
                f'Please increase the size of eval_dataset or set a larger value for split_dataset_ratio.')

        self._prepare_liger_loss()
        self._prepare_metrics()

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if not self.args.use_vllm:
            from swift.llm import PtEngine
            infer_template = copy(self.template)
            infer_template.padding_free = False
            infer_template.sequence_parallel_size = 1
            self.engine = PtEngine.from_model_template(self.model, infer_template, max_batch_size=0)  # 0: no limit

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        if self.args.dynamic_sample or self.template.truncation_strategy == 'raise':
            self._prepare_resample_data_iterator()
        # flag indicating whether the evaluation has started
        self.eval_flag = False

        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            self.args.gradient_accumulation_steps = self.args.gradient_accumulation_steps * sequence_parallel.world_size

        # for multi-turn server, maybe the num of rollout outputs is not equal to the num of rollout inputs
        self.dynamic_num_samples = False
        # Record the number of samples that need to be padded for even distribution across processes
        self.rollout_pad_count = 0

        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle. # noqa
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

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

        if self.dynamic_sample and mode == 'train':
            # dynamic sampling for std=0 groups
            inputs, total_rewards_per_func = self._dynamic_sampling(inputs, total_rewards_per_func)  # noqa

        batch_encoded_inputs = self._prepare_batch_inputs(inputs)

        total_advantages = self._compute_advantages(inputs, total_rewards_per_func, batch_encoded_inputs)

        local_advantages = get_even_process_data(self, total_advantages)
        assert len(local_advantages) == len(inputs)
        for i, advantage in enumerate(local_advantages):
            inputs[i]['advantages'] = advantage
        # log metrics in inputs
        self._logs['advantages'].extend(total_advantages.tolist())

        # Add advantages to each batch in batch_encoded_inputs
        gas_chunks = self.split_by_mini_batches(inputs)
        assert len(gas_chunks) == len(batch_encoded_inputs), \
            f'Mismatch: {len(gas_chunks)} chunks vs {len(batch_encoded_inputs)} batches'

        for batch, batch_encoded in zip(gas_chunks, batch_encoded_inputs):
            # Advantages are always [batch_size], will be broadcast to [batch_size, seq_len] in loss computation
            all_advantages = torch.stack([data['advantages'] for data in batch])
            batch_encoded['advantages'] = all_advantages

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
            template = None if not hasattr(reward_model_plugin, 'template') else reward_model_plugin.template
            with patch_profiling_context(self, reward_func_name), self._disable_sp_context(template):
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

    def _compute_advantages(self, inputs: DataType, rewards_per_func: torch.Tensor,
                            batch_encoded_inputs: List[DataType]) -> torch.Tensor:
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
            if self.scale_rewards != 'none':
                return advantages / (rewards_std + 1e-4)
            return advantages

        def log_rewards_metrics(rewards: torch.Tensor, rewards_per_func_for_metrics: torch.Tensor):
            """Log reward statistics for monitoring. Only log once per unique request_id."""
            # rewards: [prompt_batch_size, self.num_generations]
            # rewards_per_func_for_metrics: [prompt_batch_size*self.num_generations, self.num_reward_funcs]
            mode = 'train' if self.model.training else 'eval'
            group_rewards = rewards.view(-1, self.num_generations)
            rewards_mean = group_rewards.mean(-1).mean().item()
            if self.scale_rewards in ['group', 'none']:
                rewards_std = group_rewards.std(-1).mean().item()
            elif self.scale_rewards == 'batch':
                rewards_std = rewards.std().item()
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

        if self.kl_in_reward and self.beta != 0.0:
            kl_list = []
            for batch_encoded in batch_encoded_inputs:
                old_per_token_logps = batch_encoded['old_per_token_logps']
                ref_per_token_logps = batch_encoded['ref_per_token_logps']
                completion_mask = batch_encoded['completion_mask']
                per_token_kl = old_per_token_logps - ref_per_token_logps
                kl = (per_token_kl * completion_mask).sum(-1)
                kl_list.append(kl)

            kl = torch.cat(kl_list, dim=0)
            kl = gather(kl)
            mode = 'train' if self.model.training else 'eval'
            self._metrics[mode]['kl'].append(kl.nanmean().item())
            rewards = rewards - self.beta * kl

        # --------------------------------------------------
        # Case 1: Default grouped mode
        # --------------------------------------------------
        if not self.dynamic_num_samples:
            grouped_rewards = rewards.view(-1, self.num_generations)
            K = self.num_generations

            # Compute group statistics
            group_rewards_mean = grouped_rewards.mean(dim=1)

            # Broadcast stats back to the original shape
            group_rewards_mean = group_rewards_mean.repeat_interleave(K)

            # Compute advantages based on estimation type
            if self.advantage_estimator == 'rloo':
                # RLOO: Leave-One-Out baseline
                # A_i = r_i - mean(r_j for j != i)
                # = r_i * K/(K-1) - mean_all * K/(K-1)
                advantages = rewards * K / (K - 1) - group_rewards_mean * K / (K - 1)
            else:  # 'grpo' or 'reinforce_plus_plus'
                # Both use group mean as baseline
                advantages = rewards - group_rewards_mean

            # Normalize advantages based on estimator and scale_rewards
            if self.advantage_estimator == 'reinforce_plus_plus':
                # REINFORCE++: Use std of advantages (not rewards)
                if self.scale_rewards == 'batch':
                    # Global whitening: std computed on advantages
                    # Note: advantages.mean() is mathematically 0, no need to subtract
                    advantages_std = advantages.std().expand_as(advantages)
                elif self.scale_rewards == 'group':
                    # Group-level whitening on advantages
                    advantages_grouped = advantages.view(-1, K)
                    advantages_std = advantages_grouped.std(dim=1).repeat_interleave(K)
                else:  # 'none'
                    advantages_std = None
                if advantages_std is not None:
                    advantages = normalize_advantages(advantages, advantages_std)
            else:  # 'grpo' or 'rloo'
                # GRPO/RLOO: Use std of original rewards
                if self.scale_rewards == 'batch':
                    rewards_std = rewards.std().expand_as(rewards)
                elif self.scale_rewards == 'group':
                    rewards_std = grouped_rewards.std(dim=1).repeat_interleave(K)
                else:  # 'none'
                    rewards_std = None
                if rewards_std is not None:
                    advantages = normalize_advantages(advantages, rewards_std)

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
            for pid, idxs in prompt_to_indices.items():
                idx_tensor = torch.tensor(idxs, device=device)
                r_group = unique_rewards[idx_tensor]
                prompt_means[idx_tensor] = r_group.mean()

            # Step 4. Compute advantages
            if self.advantage_estimator == 'rloo':
                # RLOO: Leave-One-Out baseline for dynamic mode
                request_advantages = torch.zeros_like(unique_rewards)
                for pid, idxs in prompt_to_indices.items():
                    K = len(idxs)
                    idx_tensor = torch.tensor(idxs, device=device)
                    r_group = unique_rewards[idx_tensor]
                    # A_i = r_i * K/(K-1) - mean * K/(K-1)
                    request_advantages[idx_tensor] = (r_group * K / (K - 1) - r_group.mean() * K / (K - 1))
            else:  # 'grpo' or 'reinforce_plus_plus'
                # Both use group mean as baseline
                request_advantages = unique_rewards - prompt_means

            # Step 5. Normalize advantages
            if self.advantage_estimator == 'reinforce_plus_plus':
                # REINFORCE++: Use std of advantages (not rewards)
                if self.scale_rewards == 'batch':
                    # Global whitening: std computed on advantages
                    # Note: advantages.mean() is mathematically 0, no need to subtract
                    advantages_std = request_advantages.std()
                    prompt_stds = torch.full_like(request_advantages, advantages_std)
                elif self.scale_rewards == 'group':
                    # Group-level whitening on advantages
                    prompt_stds = torch.zeros(len(unique_rewards), device=device)
                    for pid, idxs in prompt_to_indices.items():
                        idx_tensor = torch.tensor(idxs, device=device)
                        adv_group = request_advantages[idx_tensor]
                        prompt_stds[idx_tensor] = adv_group.std()
                else:  # 'none'
                    prompt_stds = None
                if prompt_stds is not None:
                    request_advantages = normalize_advantages(request_advantages, prompt_stds)
            else:  # 'grpo' or 'rloo'
                # GRPO/RLOO: Use std of original rewards
                if self.scale_rewards == 'batch':
                    rewards_std = unique_rewards.std()
                    prompt_stds = torch.full_like(unique_rewards, rewards_std)
                elif self.scale_rewards == 'group':
                    prompt_stds = torch.zeros(len(unique_rewards), device=device)
                    for pid, idxs in prompt_to_indices.items():
                        idx_tensor = torch.tensor(idxs, device=device)
                        r_group = unique_rewards[idx_tensor]
                        prompt_stds[idx_tensor] = r_group.std()
                else:  # 'none'
                    prompt_stds = None
                if prompt_stds is not None:
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

        while resample_count < self.max_resample_times:
            rewards_std = self.compute_std(inputs, rewards_per_func)
            valid_mask = (rewards_std > 0)
            all_inputs = gather_object(inputs)
            valid_samples.extend([inp for inp, mask in zip(all_inputs, valid_mask) if mask])
            valid_rewards_per_func.append(rewards_per_func[valid_mask])
            if len(valid_samples) >= self.args.generation_batch_size:
                break

            inputs = next(self.dynamic_resample_iterator)
            if self.template.truncation_strategy == 'raise':
                inputs = self.resample_encode_failed_inputs(inputs)
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
            logger.warning(f'There are still std=0 groups present after {self.max_resample_times} retries.')
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
            batch_size = len(batch)

            # Create completion_mask
            # In padding_free mode: labels shape is [1, total_seq_len] (rmpad format)
            # In non-padding_free mode: labels shape is [batch_size, seq_len] (batch format)
            completion_mask_raw = labels[:, -logits_to_keep:] != -100

            extra_kwargs = {
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

                # In padding_free mode, completion_mask_raw is [1, logits_to_keep] (rmpad format)
                # Pad it back to [batch_size, logits_to_keep] for consistency with per_token_logps
                completion_mask, _ = pad_logps_back_to_batch(
                    logps_rmpad=completion_mask_raw.float(),
                    logits_to_keep=logits_to_keep,
                    batch_size=batch_size,
                    seq_lengths=lengths,
                    pad_value=0.0)
                completion_mask = completion_mask.bool()
            else:
                # In non-padding_free mode, completion_mask is already [batch_size, logits_to_keep]
                completion_mask = completion_mask_raw

            extra_kwargs['completion_mask'] = completion_mask
            batch_encoded_inputs.update(extra_kwargs)

            with torch.no_grad():
                batch_encoded_inputs['old_per_token_logps'] = (
                    self._get_per_token_logps_and_entropies(self.model, batch_encoded_inputs)[0])
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

                # Extract rollout logprobs if available for importance sampling
                # rollout_logprobs is List[List[float]] - nested list where each inner list corresponds to
                # one assistant response turn. We need to align these with completion_mask positions.
                batch_encoded_inputs['rollout_per_token_logps'] = None
                if self.use_fast_infer:
                    rollout_logprobs_list = []
                    for data in batch:
                        if 'rollout_logprobs' in data and data['rollout_logprobs']:
                            rollout_logprobs_list.append(data['rollout_logprobs'])
                        else:
                            rollout_logprobs_list.append(None)

                    # Convert to tensor if all samples have rollout_logprobs
                    completion_mask = batch_encoded_inputs['completion_mask']
                    if all(lp is not None for lp in rollout_logprobs_list):
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
                            # Align rollout_logprobs with completion_mask for each sample
                            batch_size = completion_mask.shape[0]
                            seq_len = completion_mask.shape[1]

                            # Initialize with zeros (for prompt positions)
                            rollout_logps_tensor = torch.zeros(
                                batch_size, seq_len, dtype=torch.float32, device=self.accelerator.device)

                            for i, nested_lp in enumerate(rollout_logprobs_list):
                                # Flatten logprobs for this sample
                                flat_lps = [lp for turn_lps in nested_lp for lp in turn_lps]
                                if flat_lps:
                                    # Check for None values in flat_lps
                                    if any(lp is None for lp in flat_lps):
                                        logger.warning('Found None values in rollout_logprobs. '
                                                       'Skipping rollout importance sampling for this batch.')
                                        rollout_logps_tensor = None
                                        break
                                    # Get indices where completion_mask is True
                                    completion_indices = completion_mask[i].nonzero(as_tuple=True)[0]
                                    # Scatter logprobs to completion positions
                                    rollout_logps_tensor[i, completion_indices] = torch.tensor(
                                        flat_lps, dtype=torch.float32, device=self.accelerator.device)

                            batch_encoded_inputs['rollout_per_token_logps'] = rollout_logps_tensor

            ga_batch_encoded_inputs.append(batch_encoded_inputs)

        # --- log completion lengths ---
        mode = 'train' if self.model.training else 'eval'
        device = self.accelerator.device
        local_lengths = [inp['completion_mask'].sum(1).tolist() for inp in ga_batch_encoded_inputs]
        total_lengths = self._gather_and_flatten(local_lengths, dtype=torch.float32, device=device, flatten_level=1)

        # Store num_items_in_batch for DAPO loss (total completion tokens across all processes)
        num_items_in_batch = total_lengths.sum()
        for batch_encoded in ga_batch_encoded_inputs:
            batch_encoded['num_items_in_batch'] = num_items_in_batch

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
            remove_response(messages)
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
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, inputs, compute_entropy=self.compute_entropy)

        entropy_mask = None
        entropy_metrics = {}

        if self.compute_entropy:
            # fill the padded token with NaN
            entropies = entropies.masked_fill(completion_mask == 0, float('nan'))
            if self.args.log_entropy:
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
        if self.overlong_filter and any(truncated_mask):
            if all(truncated_mask):
                logger.info('All completions are overlong and truncated, '
                            'resulting in NaN some values for some metrics (e.g., KL)')
            truncated_mask = truncated_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & (~truncated_mask)

        # Compute the KL divergence between the model and the reference model
        # Only compute KL for loss if kl_in_reward=False (GRPO style)
        if self.beta != 0.0 and not self.kl_in_reward:
            ref_per_token_logps = inputs['ref_per_token_logps']
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)
        else:
            per_token_kl = None

        advantages = inputs['advantages']
        # When under on-policy training
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs['old_per_token_logps'] is None else inputs['old_per_token_logps'])

        # Compute rollout diagnostic metrics and apply IS correction if enabled
        rollout_correction_metrics = {}
        should_compute_rollout_metrics = (
            self.rollout_importance_sampling_mode is not None or self.log_rollout_offpolicy_metrics)

        local_has_rollout_per_token_logps = inputs.get('rollout_per_token_logps') is not None
        all_has_rollout_per_token_logps = gather_object([local_has_rollout_per_token_logps])

        should_compute_rollout_metrics = should_compute_rollout_metrics and all(all_has_rollout_per_token_logps)
        if (not self.disable_rollout_importance_sampling and should_compute_rollout_metrics):
            rollout_per_token_logps = inputs['rollout_per_token_logps']

            # Compute diagnostic metrics (KL, PPL, etc.) for monitoring off-policy gap
            rollout_correction_metrics = self._compute_rollout_offpolicy_metrics(old_per_token_logps,
                                                                                 rollout_per_token_logps,
                                                                                 completion_mask)

            # Apply importance sampling correction if mode is enabled
            if self.rollout_importance_sampling_mode is not None:
                # Compute the log ratio between policy model and rollout model
                # log π_θ(y|x) - log π_rollout(y|x)
                rollout_log_ratio = old_per_token_logps - rollout_per_token_logps

                # Apply importance sampling correction based on mode
                rollout_is_weights = self._apply_rollout_importance_sampling(rollout_log_ratio, completion_mask)

                # Compute additional IS-specific metrics (ESS, clipped_frac, is_weight_mean)
                is_metrics = self._compute_is_correction_metrics(rollout_log_ratio, rollout_is_weights, completion_mask)
                rollout_correction_metrics.update(is_metrics)

                # Store IS weights for loss computation
                inputs['rollout_is_weights'] = rollout_is_weights
            else:
                inputs['rollout_is_weights'] = None
        else:
            inputs['rollout_is_weights'] = None

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == 'token':
            log_importance_weights = log_ratio
        elif self.importance_sampling_level in ['sequence', 'sequence_token']:
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

        if self.loss_type == 'cispo':
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages.unsqueeze(1) * per_token_logps
        elif self.loss_type == 'sapo':
            advantages_expanded = advantages.unsqueeze(1)
            gate_pos = torch.sigmoid(self.tau_pos * (coef_1 - 1))
            gate_neg = torch.sigmoid(self.tau_neg * (coef_1 - 1))
            is_positive = advantages_expanded > 0
            soft_gate = torch.where(is_positive, gate_pos, gate_neg)

            per_token_loss = -soft_gate * advantages_expanded
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo']:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Apply vLLM importance sampling weights if available
        if inputs.get('rollout_is_weights') is not None and self.rollout_importance_sampling_mode is not None:
            rollout_is_weights = inputs['rollout_is_weights']
            per_token_loss = per_token_loss * rollout_is_weights

        if self.loss_type in ['grpo', 'sapo']:
            # completion_mask is now always [batch_size, seq_len] after pad_back
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == 'bnpo':
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == 'dr_grpo':
            batch_size = completion_mask.shape[0]
            loss = (per_token_loss * completion_mask).sum() / (batch_size * self.max_completion_length)
        elif self.loss_type in ['cispo', 'dapo']:
            # CISPO and DAPO: Normalize by total completion tokens across all processes
            normalizer = inputs['num_items_in_batch'] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
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

        if per_token_kl is not None:
            mean_kl = masked_batch_mean(per_token_kl)
            metrics_data['kl'] = self.accelerator.gather_for_metrics(mean_kl).nanmean().item()

        # Add rollout correction metrics
        if rollout_correction_metrics:
            metrics_data['rollout_correction'] = rollout_correction_metrics

        # Compute the clipped probability ratios
        if self.loss_type == 'cispo':
            # CISPO: Only track upper bound clipping
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather_for_metrics(cispo_clip_ratio)
            metrics_data['clipping'] = {'cispo_clip_ratio': gathered_cispo_clip_ratio.nanmean().item()}
        elif self.loss_type == 'sapo':
            pass
        else:
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

        # Update vLLM correction metrics
        if 'rollout_correction' in metrics_data:
            rollout_metrics = metrics_data['rollout_correction']
            for key, value in rollout_metrics.items():
                self._metrics[mode][f'rollout_correction/{key}'].append(value)

        # Update clipping metrics
        if 'clipping' in metrics_data:
            clipping = metrics_data['clipping']
            if 'cispo_clip_ratio' in clipping:
                # CISPO
                self._metrics[mode]['cispo_clip_ratio'].append(clipping['cispo_clip_ratio'])
            else:
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
        cispo_clip_values = []
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
                if 'cispo_clip_ratio' in clipping:
                    cispo_clip_values.append((clipping['cispo_clip_ratio'], weight))
                else:
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
        def weighted_avg(values):
            return sum(v * w for v, w in values) / sum(w for _, w in values)

        if cispo_clip_values:
            # CISPO specific metric
            aggregated_metrics['clipping'] = {'cispo_clip_ratio': weighted_avg(cispo_clip_values)}
        elif clip_values['low']:
            # Two-sided clipping metrics
            aggregated_metrics['clipping'] = {
                'low_clip_mean': weighted_avg(clip_values['low']),
                'low_clip_min': min(clip_values['low_min']),
                'high_clip_mean': weighted_avg(clip_values['high']),
                'high_clip_max': max(clip_values['high_max']),
                'region_clip_mean': weighted_avg(clip_values['region'])
            }

        # Update metrics
        self._update_metrics(aggregated_metrics)

    def _unpad_logps_and_entropies(self,
                                   logps: torch.Tensor,
                                   entropies: Optional[torch.Tensor],
                                   logits_to_keep: int,
                                   batch_size: int,
                                   seq_lengths: torch.Tensor,
                                   compute_entropy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Restore logps and entropies from rmpad format [1, total_nnz] to batch format [batch_size, max_seq_len].

        Args:
            logps: Per-token log probabilities in rmpad format [1, total_nnz]
            entropies: Per-token entropies in rmpad format [1, total_nnz] or None
            logits_to_keep: Number of tokens to keep per sequence
            batch_size: Number of sequences in the batch
            seq_lengths: Actual sequence lengths [batch_size]
            compute_entropy: Whether entropy was computed

        Returns:
            logps: Restored log probabilities [batch_size, logits_to_keep]
            entropies: Restored entropies [batch_size, logits_to_keep] or None
        """
        logps, _ = pad_logps_back_to_batch(
            logps_rmpad=logps, logits_to_keep=logits_to_keep, batch_size=batch_size, seq_lengths=seq_lengths)

        if compute_entropy and entropies is not None:
            entropies, _ = pad_logps_back_to_batch(
                logps_rmpad=entropies, logits_to_keep=logits_to_keep, batch_size=batch_size, seq_lengths=seq_lengths)

        return logps, entropies

    def _get_logps_via_sp(self,
                          model: torch.nn.Module,
                          inputs: 'DataType',
                          logits_to_keep: int,
                          input_ids: torch.Tensor,
                          compute_entropy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get per token logps via sequence parallel, returns rmpad format [1, total_nnz] for padding_free mode"""
        from swift.trainers.sequence_parallel.utils import GatherLoss
        from swift.trainers.sequence_parallel import sequence_parallel

        model_inputs = self._prepare_model_inputs(inputs)
        sequence_parallel.prepare_inputs(model_inputs)
        with self._template_context(self.template, inputs):
            output = model(**model_inputs)
            logits = output.logits
        # split input_ids to labels
        position_ids = sequence_parallel.real_position_ids
        _, _, labels, _, _, _, _ = sequence_parallel.pad_and_split_inputs(
            None, None, input_ids.clone(), None, None, None, real_position_ids=position_ids)

        labels = torch.where(labels == -100, self.processing_class.pad_token_id, labels)
        logits = logits / self.temperature
        per_token_logps = selective_log_softmax(logits, labels)
        entropies = None
        per_token_logps, _ = GatherLoss.apply(per_token_logps, labels, 1, position_ids)
        if compute_entropy:
            entropies = entropy_from_logits(logits)
            entropies, _ = GatherLoss.apply(entropies, labels, 1, position_ids)

        if self.template.padding_free:
            # In padding_free mode, we need to extract completion tokens from gathered data.
            # The behavior differs based on rp_world_size:
            # - rp_world_size > 1: Each sequence is padded to world_size * 2 multiple (per-sequence padding)
            # - rp_world_size == 1: Entire data is padded to world_size multiple (end padding only)
            seq_lengths = inputs['seq_lengths']
            batch_size = seq_lengths.shape[0]
            rp_world_size = sequence_parallel.rp_world_size

            from swift.utils import get_cu_seqlens_from_position_ids

            if rp_world_size > 1:
                # With ring parallel: GatherLoss pads each sequence to world_size * 2 multiple
                # Data layout after gather: [seq1_data, seq1_padding, seq2_data, seq2_padding, ...]
                # - Original data is at [offset:offset+orig_len]
                # - Padding is at [offset+orig_len:offset+padded_len]

                # Get original sequence boundaries (before padding)
                cu_seqlens_orig = get_cu_seqlens_from_position_ids(position_ids)

                # Get padded sequence boundaries (for offset calculation)
                padded_position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
                cu_seqlens_padded = get_cu_seqlens_from_position_ids(padded_position_ids)

                result_logps = []
                result_entropies = [] if compute_entropy else None
                gathered_logps = per_token_logps.squeeze(0)
                gathered_entropies = entropies.squeeze(0) if compute_entropy else None

                offset = 0
                for i in range(batch_size):
                    # Original sequence length (before SP padding)
                    orig_len = (cu_seqlens_orig[i + 1] - cu_seqlens_orig[i]).item()
                    # Padded sequence length (multiple of world_size * 2)
                    padded_len = (cu_seqlens_padded[i + 1] - cu_seqlens_padded[i]).item()
                    # Actual completion tokens for this sequence
                    actual_len = seq_lengths[i].item()

                    # Extract the last `actual_len` tokens from this sequence's ORIGINAL data region
                    # Due to label shifting (roll -1), per_token_logps[i] predicts token i+1
                    # So completion tokens [prompt_len, total_len) have logps at [prompt_len-1, total_len-1)
                    seq_start = offset + orig_len - actual_len - 1
                    seq_end = offset + orig_len - 1
                    result_logps.append(gathered_logps[seq_start:seq_end])
                    if compute_entropy:
                        result_entropies.append(gathered_entropies[seq_start:seq_end])

                    # Use padded_len for offset because gathered data includes padding
                    offset += padded_len

                per_token_logps = torch.cat(result_logps).unsqueeze(0)
                if compute_entropy:
                    entropies = torch.cat(result_entropies).unsqueeze(0)
            else:
                # Without ring parallel (rp_world_size == 1): Simple gather with end padding only
                # Use input_ids length directly as the authoritative original length
                original_total_len = input_ids.shape[-1]
                # Due to label shifting (roll -1), per_token_logps[i] predicts token i+1.
                start_idx = original_total_len - logits_to_keep - 1
                end_idx = original_total_len - 1
                per_token_logps = per_token_logps[:, start_idx:end_idx]
                if compute_entropy:
                    entropies = entropies[:, start_idx:end_idx]
        else:
            per_token_logps = per_token_logps[:, -logits_to_keep - 1:-1]
            if compute_entropy:
                entropies = entropies[:, -logits_to_keep - 1:-1]

        return per_token_logps, entropies

    def _get_logps_via_local_forward(self,
                                     model: torch.nn.Module,
                                     inputs: 'DataType',
                                     logits_to_keep: int,
                                     input_ids: torch.Tensor,
                                     compute_entropy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get per token logps via local forward pass, returns rmpad format [1, total_nnz] for padding_free mode"""
        model_inputs = self._prepare_model_inputs(inputs)
        if 'logits_to_keep' in self.model_kwarg_keys:
            model_inputs['logits_to_keep'] = logits_to_keep + 1

        # Forward pass
        logits = model(**model_inputs).logits

        # Extract relevant portion and apply temperature
        logits = logits[:, -(logits_to_keep + 1):-1, :] / self.temperature
        input_ids_for_logps = input_ids[:, -logits_to_keep:]

        is_padding_free = self.template.padding_free
        if is_padding_free:
            # In padding_free mode, compute logps on flattened tensors
            logits_rmpad = logits.squeeze(0)  # [total_nnz, vocab_size]
            input_ids_rmpad = input_ids_for_logps.squeeze(0)  # [total_nnz]

            # Compute logps on rmpad tensors
            logps = selective_log_softmax(logits_rmpad, input_ids_rmpad)  # [total_nnz]
            logps = logps.unsqueeze(0)  # [1, total_nnz]

            # Compute entropy if needed
            if compute_entropy:
                entropies = entropy_from_logits(logits_rmpad)  # [total_nnz]
                entropies = entropies.unsqueeze(0)  # [1, total_nnz]
            else:
                entropies = None
        else:
            logps = selective_log_softmax(logits, input_ids_for_logps)
            if compute_entropy:
                entropies = entropy_from_logits(logits)
            else:
                entropies = None

        return logps, entropies

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
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        is_padding_free = self.template.padding_free
        use_sp = self.template.sequence_parallel_size > 1

        # Store metadata for padding_free restoration
        if is_padding_free:
            original_seq_lengths = inputs.get('seq_lengths')
            batch_size = original_seq_lengths.shape[0]

        unwrapped_model = self.accelerator.unwrap_model(model)
        if is_peft_model(unwrapped_model):
            parameters = inspect.signature(unwrapped_model.base_model.model.forward).parameters
        else:
            parameters = inspect.signature(unwrapped_model.forward).parameters
        use_local_entropy = not hasattr(super(), '_get_per_token_logps_and_entropies') and compute_entropy

        # can_use_super only when not padding_free and not using SP
        can_use_super = (not self.is_multimodal and 'logits_to_keep' in parameters and not use_local_entropy
                         and not is_padding_free and not use_sp)

        if can_use_super:
            # Path 1: Use super() method (non-padding_free, non-SP)
            if hasattr(super(), '_get_per_token_logps_and_entropies'):
                logps, entropies = super()._get_per_token_logps_and_entropies(
                    model, input_ids, inputs['attention_mask'], logits_to_keep, compute_entropy=compute_entropy)
            else:
                logps = super()._get_per_token_logps(model, input_ids, inputs['attention_mask'], logits_to_keep)
                entropies = None
        elif use_sp:
            # Path 2: Use sequence parallel
            # In padding_free mode: returns [1, logits_to_keep] format (rmpad, needs unpad)
            # In non-padding_free mode: returns [batch_size, logits_to_keep] format
            logps, entropies = self._get_logps_via_sp(
                model, inputs, logits_to_keep, input_ids, compute_entropy=compute_entropy)
        else:
            # Path 3: Local forward pass (padding_free or multimodal or no logits_to_keep support)
            # Returns [1, total_nnz] in padding_free mode, or [batch_size, logits_to_keep] otherwise
            logps, entropies = self._get_logps_via_local_forward(
                model, inputs, logits_to_keep, input_ids, compute_entropy=compute_entropy)

        # Unpad for padding_free mode (both SP and non-SP paths need this)
        if is_padding_free:
            logps, entropies = self._unpad_logps_and_entropies(logps, entropies, logits_to_keep, batch_size,
                                                               original_seq_lengths, compute_entropy)

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
            model_inputs = self._prepare_model_inputs(inputs)
            if 'logits_to_keep' in self.model_kwarg_keys:
                model_inputs['logits_to_keep'] = logits_to_keep + 1

            last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state

        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        assert not self.template.padding_free
        assert self.advantage_estimator == 'grpo'
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

    def old_policy(self):
        if self.template.sequence_parallel_size == 1:
            return (self.num_iterations > 1
                    or self.args.gradient_accumulation_steps % self.args.steps_per_generation != 0)
        else:
            from swift.trainers.sequence_parallel import sequence_parallel
            return (self.num_iterations > 1 or self.args.gradient_accumulation_steps %
                    (self.args.steps_per_generation * sequence_parallel.world_size) != 0)

    @contextmanager
    def offload_context(self):
        if self.args.offload_model:
            self.offload_model(self.accelerator.unwrap_model(self.model))
            if self.ref_model:
                self.offload_model(self.ref_model)
        if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
            self.offload_optimizer()

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

    def _prepare_liger_loss(self):
        self.use_liger_loss = self.args.use_liger_kernel
        if self.use_liger_loss:
            from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
            kwargs = {}
            if 'importance_sampling_level' in inspect.signature(LigerFusedLinearGRPOLoss.__init__).parameters:
                kwargs['importance_sampling_level'] = self.importance_sampling_level
            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
                **kwargs,
            )
            self._forward_redirection = _ForwardRedirection()

    def _prepare_metrics(self):
        args = self.args
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

    def _collect_config_info(self) -> Dict[str, str]:
        config = {
            'dynamic_sample': str(self.dynamic_sample),
            'importance_sampling_level': str(self.importance_sampling_level),
            'advantage_estimator': str(self.advantage_estimator),
            'chord_sft_enabled': str(self.chord_sft_dataset is not None),
        }
        return config

    def _prepare_algorithm_params(self):
        args = self.args
        self.shuffle_dataset = args.dataset_shuffle

        self.loss_type = args.loss_type  # loss normalization
        self.scale_rewards = args.scale_rewards

        # GRPO, https://arxiv.org/abs/2402.03300
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper, Multi-step

        # DAPO, https://arxiv.org/abs/2503.14476
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.dynamic_sample = args.dynamic_sample
        self.max_resample_times = args.max_resample_times
        self.overlong_filter = args.overlong_filter

        # Entropy Mask, https://arxiv.org/abs/2506.01939
        self.top_entropy_quantile = args.top_entropy_quantile

        # GSPO, https://arxiv.org/abs/2507.18071
        self.importance_sampling_level = args.importance_sampling_level

        # SAPO, https://arxiv.org/abs/2511.20347
        self.tau_pos = args.tau_pos
        self.tau_neg = args.tau_neg

        # RLOO,
        self.advantage_estimator = args.advantage_estimator
        self.kl_in_reward = args.kl_in_reward

        # Rollout Importance Sampling Correction
        self.rollout_importance_sampling_mode = args.rollout_importance_sampling_mode
        self.rollout_importance_sampling_threshold = args.rollout_importance_sampling_threshold
        self.log_rollout_offpolicy_metrics = args.log_rollout_offpolicy_metrics

    def _prepare_chord_dataset(self):
        # CHORD, https://arxiv.org/abs/2508.11408
        self.chord_sft_iterator = None
        if self.chord_sft_dataset:
            self.chord_sft_iterator = make_chord_sft_dataset(self, self.chord_sft_dataset)

    def _prepare_rewards(self, reward_funcs, reward_model=None, reward_templates=None):
        args = self.args
        device = self.accelerator.device

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
            reward_plugins = args.reward_model_plugin
            if reward_plugins is None:
                reward_plugins = ['default'] * len(reward_model)
            assert len(reward_plugins) == len(reward_model), (
                f"The number of 'reward_model_plugin' ({len(reward_plugins)}) does not match "
                f"the number of 'reward_model' ({len(reward_model)}). "
                "Please provide a corresponding 'reward_model_plugin' for each 'reward_model'.")
            for rm, rm_plugin, rm_template in zip(reward_model, reward_plugins, reward_templates):
                # Set encoding mode train(see details in Template.encode).
                # Set max_length to None to disable truncation, as the input length has already been truncated earlier.
                rm_template.set_mode('train')
                rm_template.max_length = None
                if rm_plugin not in rm_plugins:
                    raise ValueError(f'rm_plugin {rm_plugin} is not implemented in swift.llm.plugin')
                self.reward_model_plugins.append(rm_plugins[rm_plugin](model=rm, template=rm_template))
                self.reward_funcs.append(rm)
                self.reward_func_names.append(rm.config._name_or_path.split('/')[-1])

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(f'Number of reward weights ({len(args.reward_weights)}) must match number of reward '
                                 f'functions ({len(reward_funcs)})')
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32).to(device)
        else:
            self.reward_weights = torch.ones(len(self.reward_func_names), dtype=torch.float32).to(device)

        # after init trainer
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True)

    def _prepare_resample_data_iterator(self):

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

    def _compute_sequence_level_ratios(self, is_ratio: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
        """
        Helper function to compute sequence-level importance sampling ratios.

        Args:
            is_ratio: Token-level IS ratios, shape [B, T]
            completion_mask: Boolean mask for completion tokens, shape [B, T]

        Returns:
            Sequence-level ratios as geometric mean of token-level ratios
        """
        log_ratio = torch.log(is_ratio.clamp(min=1e-10))
        seq_log_ratios = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        seq_ratios = torch.exp(seq_log_ratios)

        return seq_ratios

    def _apply_rollout_importance_sampling(self, rollout_log_ratio: torch.Tensor,
                                           completion_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply vLLM importance sampling correction using one of four modes.

        Args:
            rollout_log_ratio: log(π_θ / π_rollout) per token, shape [B, T]
            completion_mask: Boolean mask for completion tokens, shape [B, T]

        Returns:
            IS weights to multiply with loss, same shape as rollout_log_ratio
        """
        mode = self.rollout_importance_sampling_mode
        threshold = self.rollout_importance_sampling_threshold

        # Clamp log_ratio to prevent numerical overflow from padding values (-1e10)
        # A log_ratio of 20 corresponds to exp(20) ≈ 485 million, which is already extreme
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

            is_weights = clipped_seq_ratios.unsqueeze(-1).expand_as(is_ratio)

        elif mode == 'sequence_mask':
            # Sequence-level masked IS: mask entire sequences with ratio > threshold
            seq_ratios = self._compute_sequence_level_ratios(is_ratio, completion_mask)
            seq_mask = (seq_ratios <= threshold).float()

            # Apply mask to original token-level ratios
            is_weights = is_ratio * seq_mask.unsqueeze(-1)
        else:
            return is_ratio

        return is_weights

    def _compute_rollout_offpolicy_metrics(
        self,
        per_token_logps: torch.Tensor,
        rollout_per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute off-policy diagnostic metrics (always computed for monitoring).
        reference: verl/verl/trainer/ppo/rollout_corr_helper.py

        These metrics help diagnose the off-policy gap between rollout and training policies,
        which can arise from policy mismatch (e.g., vLLM BF16 vs FSDP FP32), model staleness,
        or general distribution shifts.

        Key metrics:
        - kl: Direct KL divergence estimator KL(π_rollout || π_training)
        - k3_kl: K3 KL estimator for stability (more stable for small KL)
        - training_ppl: Perplexity of training policy
        - rollout_ppl: Perplexity of rollout policy
        - log_ppl_diff: Difference in log perplexities
        - ppl_ratio: Ratio of training PPL to rollout PPL
        - chi2_token: Token-level χ² divergence E[ρ²] - 1
        - chi2_seq: Sequence-level χ² divergence E[(∏ρ_t)²] - 1

        Args:
            per_token_logps: Log probs from training policy model, shape [B, T]
            rollout_per_token_logps: Log probs from rollout policy, shape [B, T]
            completion_mask: Boolean mask for completion tokens, shape [B, T]

        Returns:
            Dictionary with off-policy diagnostic metrics
        """
        SAFETY_BOUND = 20.0
        metrics = {}

        # Helper function for masked mean
        def masked_mean(x, mask, axis=None):
            if axis is None:
                return (x * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                return (x * mask).sum(axis) / mask.sum(axis).clamp(min=1.0)

        # 1. Training policy perplexity (always computed)
        # Formula: exp(-1/|T| * Σ log π_training(y_t|y_<t))
        mean_log_prob_training = masked_mean(per_token_logps, completion_mask, axis=-1)  # (batch_size,)
        training_ppl = torch.exp(-mean_log_prob_training).mean()  # Batch mean of per-sequence PPL
        metrics['training_ppl'] = self.accelerator.gather_for_metrics(training_ppl).nanmean().item()

        # Also log log-ppl for easier analysis (avoids exponential scale)
        metrics['training_log_ppl'] = self.accelerator.gather_for_metrics(
            (-mean_log_prob_training).mean()).nanmean().item()

        # 2. Compute rollout off-policy metrics
        # All KL metrics estimate KL(π_training || π_rollout), which measures how much
        # the training policy deviates from the rollout policy. This is directly related
        # to the importance sampling ratio ρ = π_training / π_rollout.

        # log_ratio = log(π_training / π_rollout), used for both KL estimators
        log_ratio = per_token_logps - rollout_per_token_logps
        log_ratio *= completion_mask

        # 2a. kl: Direct estimator for KL(π_training || π_rollout)
        # Formula: KL(P||Q) = E_Q[log(P/Q)] when sampled from Q (rollout)
        # However, we use the identity: E_Q[log(P/Q)] = E_Q[log P] - E_Q[log Q]
        # Since data is from rollout, E_Q[log Q] ≈ E[rollout_logps], E_Q[log P] ≈ E[training_logps]
        # Positive value means training policy assigns higher probability than rollout
        kl = masked_mean(log_ratio, completion_mask)
        metrics['kl'] = self.accelerator.gather_for_metrics(kl).nanmean().item()

        # 2b. k3_kl: K3 estimator for KL(π_training || π_rollout)
        # More stable for small KL values
        # Formula: KL(P||Q) ≈ E_Q[P/Q - log(P/Q) - 1] where P=π_training, Q=π_rollout
        k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
        k3_kl = masked_mean(k3_kl_matrix, completion_mask)
        metrics['k3_kl'] = self.accelerator.gather_for_metrics(k3_kl).nanmean().item()

        # 2c. Rollout policy perplexity
        mean_log_prob_rollout = masked_mean(rollout_per_token_logps, completion_mask, axis=-1)  # (batch_size,)
        rollout_ppl = torch.exp(-mean_log_prob_rollout).mean()  # Batch mean of per-sequence PPL
        metrics['rollout_ppl'] = self.accelerator.gather_for_metrics(rollout_ppl).nanmean().item()
        metrics['rollout_log_ppl'] = self.accelerator.gather_for_metrics(
            (-mean_log_prob_rollout).mean()).nanmean().item()

        # 2d. Log PPL difference (sequence-level perplexity difference)
        # log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        # Since ppl = exp(-log_prob), we have:
        #   log(ppl_ratio) = log(training_ppl/rollout_ppl) = log_ppl_diff
        # Positive value means training assigns lower probability (higher PPL) than rollout
        log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        metrics['log_ppl_diff'] = self.accelerator.gather_for_metrics(log_ppl_diff.mean()).nanmean().item()
        metrics['log_ppl_abs_diff'] = self.accelerator.gather_for_metrics(log_ppl_diff.abs().mean()).nanmean().item()
        metrics['log_ppl_diff_max'] = self.accelerator.gather_for_metrics(log_ppl_diff.max()).max().item()
        metrics['log_ppl_diff_min'] = self.accelerator.gather_for_metrics(log_ppl_diff.min()).min().item()

        # 2e. PPL ratio (how much higher is training PPL vs rollout PPL)
        # IMPORTANT: Compute per-sequence ratio first, then average
        # For numerical stability, compute in log space using log_ppl_diff
        # Note: log_ppl_diff = log(ppl_ratio), so ppl_ratio = exp(log_ppl_diff)
        ppl_ratio = torch.exp(log_ppl_diff).mean()
        metrics['ppl_ratio'] = self.accelerator.gather_for_metrics(ppl_ratio).nanmean().item()

        # 2f. Chi-squared divergence: χ²(π_training || π_rollout) = E_μ[ρ²] - 1
        # where ρ = π_training / π_rollout and μ = π_rollout (rollout distribution)
        # This measures the variance of importance sampling weights
        # Token-level: E_token[ρ²] - 1 (averaged over all tokens)
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_token = torch.exp(log_ratio_safe)  # ρ = π_training / π_rollout (token-level)
        rho_squared_token = rho_token.square()
        chi2_token = masked_mean(rho_squared_token, completion_mask) - 1.0
        metrics['chi2_token'] = self.accelerator.gather_for_metrics(chi2_token).nanmean().item()

        # Sequence-level (geometric mean): E_seq[ρ_geo²] - 1
        # where ρ_geo = exp(mean(log ρ_t)) is the geometric mean of token-level ratios
        # This is more interpretable than the product-based chi2_seq, as it's normalized by sequence length
        # and comparable to other per-token metrics like chi2_token
        log_ratio_mean = masked_mean(log_ratio, completion_mask, axis=-1)  # mean(log ρ_t) per sequence
        log_ratio_mean_safe = torch.clamp(log_ratio_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_geo = torch.exp(log_ratio_mean_safe)  # geometric mean of ρ_t
        chi2_seq = (rho_geo.square().mean() - 1.0)
        metrics['chi2_seq'] = self.accelerator.gather_for_metrics(chi2_seq).nanmean().item()

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
            rollout_log_ratio: Log ratio log(π_policy / π_rollout), shape [B, T]
            is_weights: Importance sampling weights after correction, shape [B, T]
            completion_mask: Boolean mask for completion tokens, shape [B, T]

        Returns:
            Dictionary with IS-specific metrics:
                - is_weight_mean: Mean of IS weights
                - ess: Effective Sample Size = 1 / E[(w_i / E[w_i])²]
                - clipped_frac: Fraction of clipped/masked samples
        """
        metrics = {}
        SAFETY_BOUND = 20.0
        threshold = self.rollout_importance_sampling_threshold
        threshold_lower = 1.0 / threshold  # Default lower threshold (reciprocal of upper)

        # Helper function for masked mean
        def masked_mean(x, mask):
            return (x * mask).sum() / mask.sum().clamp(min=1.0)

        # Compute IS ratio with safety bounds
        log_ratio_safe = torch.clamp(rollout_log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        is_ratio = torch.exp(log_ratio_safe)

        # 1. IS weight statistics
        mean_is_weight = masked_mean(is_weights, completion_mask)
        metrics['is_weight_mean'] = self.accelerator.gather_for_metrics(mean_is_weight).nanmean().item()

        # 2. Compute Effective Sample Size (ESS) for IS weights
        # ESS = 1 / E[(w_i / E[w_i])²] (using clamped weights for stability)
        # This measures how many "effective" independent samples we have after IS weighting
        weights_for_ess = is_weights.clamp(min=threshold_lower, max=threshold)
        mean_for_ess = masked_mean(weights_for_ess, completion_mask)
        is_weights_normalized = weights_for_ess / (mean_for_ess + 1e-8)  # Avoid division by zero
        ess = 1.0 / masked_mean(is_weights_normalized.square(), completion_mask).clamp(min=1e-10)
        metrics['ess'] = self.accelerator.gather_for_metrics(ess).nanmean().item()

        # 3. Fraction of clipped/masked samples
        if self.rollout_importance_sampling_mode in ['token_truncate', 'token_mask']:
            # Token-level
            if self.rollout_importance_sampling_mode == 'token_truncate':
                clipped_frac = masked_mean((is_ratio > threshold).float(), completion_mask)
            else:  # token_mask
                clipped_frac = masked_mean((is_weights == 0).float(), completion_mask)
            metrics['clipped_frac'] = self.accelerator.gather_for_metrics(clipped_frac).nanmean().item()
        else:
            # Sequence-level (both truncate and mask)
            seq_ratios = self._compute_sequence_level_ratios(is_ratio, completion_mask)
            clipped_frac = (seq_ratios > threshold).float().mean()
            metrics['clipped_frac'] = self.accelerator.gather_for_metrics(clipped_frac).nanmean().item()

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
