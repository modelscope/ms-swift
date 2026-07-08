# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from accelerate.utils import broadcast_object_list
from collections import defaultdict, deque
from contextlib import contextmanager
from copy import copy, deepcopy
from functools import partial
from mcore_bridge import set_random_seed
from megatron.core import mpu
from megatron.core.rerun_state_machine import RerunDataIterator
from typing import Any, Dict, List, Optional, Tuple, Union

from swift.infer_engine.protocol import RequestConfig, RolloutInferRequest, RolloutOutput
from swift.megatron.arguments import MegatronArguments
from swift.megatron.utils import RouterReplayHelper, get_padding_to, set_router_replay_data
from swift.rl_core.advantage import (compute_advantages, compute_reward_metrics, compute_teacher_kl_per_token,
                                     expand_advantage_to_per_token)
from swift.rl_core.data import GRPOBatch, GRPOSample
from swift.rl_core.grpo_algorithm import score_completions
from swift.rl_core.resample import resample_encode_failed_inputs
from swift.rlhf_trainers.gkd_helpers import (assemble_teacher_completion_logprobs, build_opsd_samples,
                                             build_teacher_requests, encode_teacher_view,
                                             fetch_teacher_parsed_by_routing, remap_teacher_logps_to_student_frame,
                                             should_compute_local_teacher_logps)
from swift.rlhf_trainers.grpo_trainer import DataType
from swift.rlhf_trainers.utils import (collate_to_grpo_micro_batch, encode_sample, make_reward_weights,
                                       pad_logps_back_to_batch, profiling_context, profiling_decorator,
                                       resolve_reward_funcs)
from swift.template import Template
from swift.utils import get_logger
from .rlhf_mixin import MegatronRLHFTrainer
from .rollout_mixin import MegatronRolloutMixin
from .utils import gather, gather_object, reconstruct_tensor_cp
from .vocab_parallel_utils import compute_logps_and_entropy_from_logits

try:
    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
except ImportError:
    RouterReplay = None
    RouterReplayAction = None

logger = get_logger()


class MegatronGRPOTrainer(MegatronRolloutMixin, MegatronRLHFTrainer):

    # Per-sample container class used by MegatronRolloutMixin.to_samples.
    sample_cls = GRPOSample

    def __init__(self, args: MegatronArguments, template: Template, **kwargs):
        self.vllm_client = kwargs.pop('vllm_client')
        self.args = args
        self._setup_teacher()
        super().__init__(args, template)
        self.args = args
        self.hf_model_dir = args.model_info.model_dir
        self.processing_class = self.template.processor
        self._prepare_metrics()
        self._init_grpo_params()
        self._init_rollout_engine()
        self._prepare_rewards()
        self.resample_data_iterator = None

    def prepare_model(self):
        super().prepare_model()
        self._load_teacher_model()

    def train(self, train_dataset, val_dataset):
        if self.dynamic_sample or self.truncation_strategy == 'delete':
            self.resample_data_iterator = self._init_resample_data_iterator(train_dataset)
        super().train(train_dataset, val_dataset)

    def _init_grpo_params(self):
        """Initialize GRPO-specific parameters.

        Note: Common rollout params (world_size, process_index, device, request_config, etc.)
        are initialized by MegatronRolloutMixin._init_rollout_params().
        """
        args: MegatronArguments = self.args

        # GRPO algorithm params
        self.num_generations = args.num_generations  # G in the GRPO paper
        self.num_generations_eval = args.num_generations_eval or self.num_generations
        self.beta = args.beta
        self.loss_type = args.loss_type
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.top_entropy_quantile = args.top_entropy_quantile
        self.importance_sampling_level = args.importance_sampling_level

        # SAPO, https://arxiv.org/abs/2511.20347
        self.tau_pos = args.tau_pos
        self.tau_neg = args.tau_neg

        # REAL, https://arxiv.org/abs/2602.05630
        self.real_tau = args.real_tau

        # FIPO, https://arxiv.org/abs/2603.19835
        self.fipo_gamma = 2**(-1 / args.fipo_decay_rate)
        self.fipo_clip_range = args.fipo_clip_range
        self.fipo_clip_high_only = args.fipo_clip_high_only
        self.fipo_safety_threshold = args.fipo_safety_threshold

        # DAPO, https://arxiv.org/abs/2503.14476
        self.dynamic_sample = args.dynamic_sample
        self.max_resample_times = args.max_resample_times
        self.overlong_filter = args.overlong_filter

        # Dr. GRPO / RLOO / REINFORCE++
        self.scale_rewards = args.scale_rewards
        self.advantage_estimator = args.advantage_estimator
        self.kl_in_reward = args.kl_in_reward
        if self.scale_rewards == 'gdpo' and self.kl_in_reward:
            logger.warning('GDPO mode does not support kl_in_reward=True. Setting kl_in_reward=False.')
            self.kl_in_reward = False

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

        # truncation_strategy support
        self.truncation_strategy = args.truncation_strategy

        self.teacher_kl_coef = args.teacher_kl_coef

    def _init_rollout_engine(self):
        """Initialize rollout engine with GRPO-specific extensions."""
        super()._init_rollout_engine()

        # GRPO-specific initialization
        self.async_generate = self.args.async_generate  # TODO
        self.use_gym_env = self._resolve_use_gym_env()
        self._buffered_inputs = None

        # Rollout importance sampling requires vLLM >= 0.10.2
        self.disable_rollout_importance_sampling = not self.vllm_version_ge_0_10_2
        if not self.vllm_version_ge_0_10_2 and getattr(self.args, 'rollout_importance_sampling_mode', None) is not None:
            raise ValueError('rollout_importance_sampling_mode is not supported in vLLM version < 0.10.2, '
                             'please update vLLM to 0.10.2 or later.')

    def _resolve_use_gym_env(self) -> bool:
        """Resolve `use_gym_env` for the trainer, mirroring the HF rollout_mixin logic.

        Priority:
          1. Explicit `args.use_gym_env` (works for both server and colocate).
          2. In server mode, auto-detect from the connected vLLM rollout server.
          3. Default to False.
        """
        args = self.args
        explicit = getattr(args, 'use_gym_env', None)
        if explicit is not None:
            return bool(explicit)

        if not getattr(self, 'use_vllm', False) or self.vllm_mode != 'server':
            return False

        # super()._init_rollout_engine() has already called vllm_client.get_engine_type()
        # on the main rank, so vllm_client.use_gym_env is populated there.
        if self.is_main_process:
            value = [bool(getattr(self.vllm_client, 'use_gym_env', False))]
        else:
            value = [False]
        return broadcast_object_list(value, from_process=self.world_size - 1)[0]

    def _prepare_rewards(self):
        args = self.args
        reward_funcs_cfg = args.reward_funcs.copy()
        if not isinstance(reward_funcs_cfg, list):
            reward_funcs_cfg = [reward_funcs_cfg]

        self.reward_funcs, self.reward_func_names = resolve_reward_funcs(reward_funcs_cfg, args=self.args)

        # use_gym_env: gym total_reward is appended as an extra reward column so it can
        # blend with reward_funcs via reward_weights. When reward_funcs is empty, it becomes
        # the single reward source.
        if self.use_gym_env:
            self.reward_func_names.append('gym_reward')

        self.reward_weights = make_reward_weights(args.reward_weights, len(self.reward_func_names), self.device)

        self.reward_model_plugins = [None] * len(self.reward_funcs)

        assert self.reward_funcs or self.use_gym_env or self._has_teacher, \
            'reward_funcs is not set (or pass --use_gym_env true / a teacher for OPD-RL)'

    def _init_resample_data_iterator(self, train_dataset):
        """Initialize an independent data iterator for resampling.

        Uses a different seed (args.seed + 1) to avoid overlapping with training samples.

        Args:
            train_dataset: The training dataset to create the resample iterator from.

        Returns:
            The resample data iterator (first element of the iterator tuple).
        """
        args = self.args
        resample_seed = getattr(args, 'seed', 42) + 1
        try:
            set_random_seed(
                resample_seed,
                args.data_parallel_random_init,
                args.te_rng_tracker,
            )
            # TODO: VPP (Virtual Pipeline Parallelism)
            resample_data_iterator = self._prepare_data_iterator(train_dataset, use_origin_cyclic=True)[0]
        finally:
            set_random_seed(
                args.seed,
                args.data_parallel_random_init,
                args.te_rng_tracker,
            )
        return resample_data_iterator

    def _replace_data_iterator(self, data_iterator):
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

    def _generate_and_score_completions(self, inputs: DataType):
        # Get or create the rollout group (TP×PP×CP)

        rollout_group = self._get_rollout_group()

        # Resample for encoding failed data when truncation_strategy is 'delete'
        # This handles: (1) prompt length exceeds max_length, (2) multimodal encoding failures
        # Do this before get_local_rollout_batch to process prompt-level data.
        # Resample operates on dict rows; build per-sample objects only once the
        # training data is finalized (mirrors HF / Ray GRPO).
        if self.truncation_strategy == 'delete':
            inputs = self.resample_encode_failed_inputs(inputs)
        samples = self.to_samples(inputs)

        samples = self.get_local_rollout_batch(samples)

        samples = self._generate_completions(samples)

        rewards_per_func = self._score_completions(samples)

        # Dynamic sampling for std=0 groups (DAPO)
        if self.dynamic_sample:
            samples, rewards_per_func = self._dynamic_sampling(samples, rewards_per_func)

        # Log completions after all filtering so they align with rewards/advantages (issue #9533).
        self._log_completions_from_samples(samples)

        # Gather rollout data across rollout group
        total_samples = gather_object(samples, group=rollout_group)
        mini_batch_data = []
        template = self.template

        # OPSD: build each sample's teacher view when this batch uses privileged teacher_prompt.
        has_opsd_batch = build_opsd_samples(total_samples)
        is_opsd = (not self.use_teacher_api and has_opsd_batch
                   and (self._has_teacher_explicit or self._is_dynamic_self_distillation))

        # Step 1: Encode batches and compute logps first (unified flow like GRPOTrainer)
        with self._template_context(template):
            for s in total_samples:
                s.encoded = encode_sample(s, template)
                s.encoded.pop('_extra_kwargs', None)

            for idx in range(0, len(total_samples), self.micro_batch_size):
                sample_slice = total_samples[idx:idx + self.micro_batch_size]
                model_inputs, grpo_batch = collate_to_grpo_micro_batch(
                    sample_slice,
                    template,
                    device=self.device,
                    padding_to=get_padding_to(self.args),
                    router_replay_mode=self.args.router_replay_mode,
                )
                teacher_inputs = None
                if is_opsd:
                    teacher_inputs = self._collate_teacher_opsd_batch(sample_slice, template)
                # Wire format: model_inputs (dict) + grpo_batch (consumed by forward_step)
                data = model_inputs
                data['grpo_batch'] = grpo_batch
                with profiling_context(self, 'compute_ref_old_logps'):
                    data = self._maybe_compute_logps(data, teacher_inputs=teacher_inputs)
                mini_batch_data.append(data)

        if self._has_teacher and self.use_teacher_api:
            build_opsd_samples(total_samples)  # OPSD: populate teacher_messages for build_teacher_requests
            self._assemble_teacher_api_logps(total_samples, mini_batch_data)

        # Step 2: Compute KL from logps if kl_in_reward is enabled
        kl_values = None
        if self.kl_in_reward and self.beta != 0.0:
            kl_values = self._compute_kl_from_batches(mini_batch_data)

        # Step 3: Compute the per-sequence base advantage (with ref-KL penalty if kl_in_reward).
        advantages = self._compute_advantages(samples, rewards_per_func, kl_values=kl_values)
        total_advantages = gather(advantages, group=rollout_group)

        # Step 4: Write the advantage onto each batch, expanding the per-sequence base advantage to
        # per-token [B, T] here so the OPD-RL signed teacher log-ratio is added per token
        # (adv_t = base + coef * (teacher_logp - student_logp)).
        for idx, micro_batch_encoded in enumerate(mini_batch_data):
            grpo_batch = micro_batch_encoded['grpo_batch']
            start_idx = idx * self.micro_batch_size
            end_idx = start_idx + grpo_batch.completion_mask.shape[0]
            micro_batch_advantages = total_advantages[start_idx:end_idx]
            grpo_batch.advantages = expand_advantage_to_per_token(
                micro_batch_advantages,
                grpo_batch.completion_mask,
                teacher_per_token_logps=grpo_batch.teacher_per_token_logps,
                policy_per_token_logps=grpo_batch.old_per_token_logps
                if grpo_batch.teacher_per_token_logps is not None else None,
                teacher_kl_coef=self.teacher_kl_coef if grpo_batch.teacher_per_token_logps is not None else 0.0,
            )
        if any(m['grpo_batch'].teacher_per_token_logps is not None for m in mini_batch_data):
            self._log_teacher_kl_metric(mini_batch_data)

        if self.loss_type in ['cispo', 'dapo', 'fipo']:
            # Calculate num_items_in_batch
            # Count completion tokens from all mini_batch_data (this includes gathered data from rollout_group)
            # Use completion_mask.sum() for both padding_free and non-padding_free modes
            # since we want the count of actual completion tokens, not sequence lengths
            total_token_count = sum(batch_data['grpo_batch'].completion_mask.sum().item()
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
            # Store num_items_in_batch in each mini_batch_data for token-normalized losses
            for batch_data in mini_batch_data:
                batch_data['grpo_batch'].num_items_in_batch = num_items_in_batch

        return mini_batch_data

    def _compute_fipo_influence(self, log_ratio: torch.Tensor, coef_1: torch.Tensor, advantages: torch.Tensor,
                                completion_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute FIPO token-level influence weight from discounted Future-KL."""
        future_kl_delta = log_ratio.masked_fill(~completion_mask, 0.0)

        if self.args.delta is not None:
            delta = torch.as_tensor(self.args.delta, dtype=log_ratio.dtype, device=log_ratio.device)
            high_ratio_mask = coef_1 > delta
            future_kl_delta = torch.where(high_ratio_mask, torch.zeros_like(future_kl_delta), future_kl_delta)

        seq_len = future_kl_delta.shape[1]
        future_kl = torch.zeros_like(future_kl_delta)
        positions = torch.arange(seq_len, device=log_ratio.device).unsqueeze(1)
        gamma = torch.as_tensor(self.fipo_gamma, dtype=log_ratio.dtype, device=log_ratio.device)
        chunk_size = 128
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(seq_len, chunk_start + chunk_size)
            chunk_positions = torch.arange(chunk_start, chunk_end, device=log_ratio.device).unsqueeze(0)
            distance = chunk_positions - positions
            future_mask = distance >= 0
            decay_block = torch.pow(gamma, distance.clamp(min=0)) * future_mask.to(log_ratio.dtype)
            future_kl += torch.matmul(future_kl_delta[:, chunk_start:chunk_end], decay_block.t())
        future_kl = future_kl.masked_fill(~completion_mask, 0.0)

        influence_weight = torch.exp(future_kl)

        if self.fipo_clip_range:
            high = 1 + self.fipo_clip_range
            low = 1.0 if self.fipo_clip_high_only else 1 - self.fipo_clip_range
            influence_weight = torch.clamp(influence_weight, min=low, max=high)
        influence_weight = influence_weight.detach()

        safety_mask = torch.ones_like(completion_mask, dtype=torch.bool)
        if self.fipo_safety_threshold is not None:
            negative_advantage = advantages < 0
            high_is_ratio = coef_1 > self.fipo_safety_threshold
            safety_mask = ~(negative_advantage & high_is_ratio)
            influence_weight = torch.where(safety_mask, influence_weight,
                                           torch.clamp(influence_weight, min=0.8, max=1.0))

        metrics = {
            'future_kl': future_kl,
            'influence_weight': influence_weight,
            'safety_mask': safety_mask,
        }
        return influence_weight, metrics

    def _build_log_table(self) -> Dict[str, list]:
        """Extend base table with GRPO-specific rewards and advantages."""
        table = super()._build_log_table()
        table.update({k: list(v) for k, v in self._logs['rewards'].items()})
        table['advantages'] = list(self._logs['advantages'])
        return table

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
        infer_requests = self.samples2requests(inputs)

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

    @profiling_decorator
    def _score_completions(self, samples: List[GRPOSample]) -> torch.Tensor:
        """Score completions using all reward functions.

        Args:
            samples: List of on-policy samples carrying messages + rollout fields.

        Returns:
            rewards_per_func: Tensor of shape (num_examples, num_reward_funcs) with local reward values
        """
        return score_completions(
            samples,
            reward_funcs=self.reward_funcs,
            reward_model_plugins=self.reward_model_plugins,
            use_gym_env=self.use_gym_env,
            device=self.device,
            trainer_state=self.state,
        )

    def _compute_advantages(self,
                            samples: List[GRPOSample],
                            rewards_per_func: torch.Tensor,
                            kl_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the per-sequence base advantage for RL training.

        The OPD-RL teacher signal is not injected here; it is applied per-token when the base
        advantage is expanded to ``[B, T]`` onto each batch (see ``expand_advantage_to_per_token``).

        Args:
            samples: Local on-policy samples (only the count is used for slicing)
            rewards_per_func: Reward per function for local data samples
            kl_values: Optional KL values for kl_in_reward mode, shape [total_samples]

        Returns:
            advantages: Computed advantages for local batch, shape [local_batch_size]
        """
        mode = 'train' if self.unwrapped_models[0].training else 'eval'
        assert len(samples) == rewards_per_func.shape[0]
        num_generations = self.num_generations if mode == 'train' else self.num_generations_eval

        # Gather local rewards into a global tensor: GRPO group normalization needs every
        # completion of the same prompt visible on one rank (groups span DP ranks).
        # ``kl_values`` is already global (gathered in ``_compute_kl_from_batches``).
        # OPD-RL pure distillation: no reward_funcs -> a [N, 0] tensor.
        if rewards_per_func.shape[1] == 0:
            global_count = sum(gather_object([rewards_per_func.shape[0]]))
            total_rewards_per_func = torch.zeros((global_count, 0), dtype=torch.float32, device=self.device)
        else:
            total_rewards_per_func = gather(rewards_per_func)

        advantages, weighted_rewards = compute_advantages(
            rewards_per_func=total_rewards_per_func,
            reward_weights=self.reward_weights,
            num_generations=num_generations,
            advantage_estimator=self.advantage_estimator,
            scale_rewards=self.scale_rewards,
            kl_in_reward=self.kl_in_reward,
            beta=self.beta,
            kl_values=kl_values,
        )

        reward_metrics = compute_reward_metrics(
            rewards=weighted_rewards,
            rewards_per_func=total_rewards_per_func,
            reward_func_names=self.reward_func_names,
            num_generations=num_generations,
            scale_rewards=self.scale_rewards,
        )
        self._metrics[mode]['reward'].append(reward_metrics.reward_mean)
        self._metrics[mode]['reward_std'].append(reward_metrics.reward_std)
        self._metrics[mode]['frac_reward_zero_std'].append(reward_metrics.frac_reward_zero_std)
        if kl_values is not None:
            self._metrics[mode]['kl'].append(kl_values.nanmean().item())
        for name in self.reward_func_names:
            self._metrics[mode][f'rewards/{name}/mean'].append(reward_metrics.per_func_mean[name])
            self._metrics[mode][f'rewards/{name}/std'].append(reward_metrics.per_func_std[name])
        self._logs['advantages'].extend(advantages.tolist())
        for i, name in enumerate(self.reward_func_names):
            self._logs['rewards'][name].extend(total_rewards_per_func[:, i].tolist())

        # Slice the global advantages back to this rank's local samples.
        slice_start = self.process_index * len(samples)
        slice_end = slice_start + len(samples)
        return advantages[slice_start:slice_end]

    def _dynamic_sampling(self, rollout_batch: List[GRPOSample],
                          rewards_per_func: torch.Tensor) -> Tuple[List[GRPOSample], torch.Tensor]:
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
            if rewards_per_func.shape[1] == 0:
                global_rewards_per_func = torch.zeros((len(global_rollout_batch), 0),
                                                      dtype=torch.float32,
                                                      device=self.device)
            else:
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

            num_iters_per_step = self.get_num_iters_per_step()
            next_rollout_prompt_batch = []
            for _ in range(num_iters_per_step):
                next_rollout_prompt_batch.extend(next(self.resample_data_iterator))

            # Resample for encoding failed data when truncation_strategy is 'delete'
            if self.truncation_strategy == 'delete':
                next_rollout_prompt_batch = self.resample_encode_failed_inputs(next_rollout_prompt_batch)

            # Convert dict rows to per-sample objects before rollout
            next_rollout_prompt_batch = self.to_samples(next_rollout_prompt_batch)

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

    def _maybe_compute_logps(self, batch: Dict[str, Any], teacher_inputs: Optional[Tuple] = None) -> Dict[str, Any]:
        grpo_batch: GRPOBatch = batch.pop('grpo_batch')
        seq_lengths = grpo_batch.seq_lengths
        batch_size = grpo_batch.completion_mask.shape[0]
        max_seq_len = grpo_batch.completion_mask.shape[1]

        # batch is now clean model forward kwargs (template.encode guarantees this)
        inputs = batch
        if self.beta != 0.0:
            with self.null_ref_context() as ref_models:
                assert len(ref_models) == 1, 'GRPO currently does not support VPP.'
                ref_model = ref_models[0]
                ref_per_token_logps_packed, _ = self.compute_per_token_logps(
                    ref_model, iter([deepcopy(inputs)]), temperature=self.temperature)
                if self.template.padding_free:
                    ref_per_token_logps, _ = pad_logps_back_to_batch(
                        logps_rmpad=ref_per_token_logps_packed,
                        logits_to_keep=max_seq_len,
                        batch_size=batch_size,
                        seq_lengths=seq_lengths)
                else:
                    ref_per_token_logps = ref_per_token_logps_packed
                grpo_batch.ref_per_token_logps = ref_per_token_logps

        if should_compute_local_teacher_logps(
                has_teacher_explicit=self._has_teacher_explicit,
                is_dynamic_self_distillation=self._is_dynamic_self_distillation,
                use_teacher_api=self.use_teacher_api,
                has_opsd_batch=teacher_inputs is not None,
        ):
            grpo_batch.teacher_per_token_logps = self._compute_teacher_logps(
                inputs, grpo_batch, teacher_inputs=teacher_inputs)

        if self.enable_routing_replay:
            if self.args.router_replay_mode == 'R2':
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
            if self.args.router_replay_mode == 'R3':
                RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        old_per_token_logps_packed, routing_topk_idx = self.compute_per_token_logps(
            self.unwrapped_models[0], iter([deepcopy(inputs)]), temperature=self.temperature)
        if self.template.padding_free:
            old_per_token_logps, _ = pad_logps_back_to_batch(
                logps_rmpad=old_per_token_logps_packed,
                logits_to_keep=max_seq_len,
                batch_size=batch_size,
                seq_lengths=seq_lengths)
        else:
            old_per_token_logps = old_per_token_logps_packed
        grpo_batch.old_per_token_logps = old_per_token_logps

        if self.enable_routing_replay:
            batch['routed_experts'] = routing_topk_idx
            RouterReplay.clear_global_indices()
            RouterReplay.clear_global_router_replay_action()

        batch['grpo_batch'] = grpo_batch
        return batch

    def _collate_teacher_opsd_batch(self, sample_slice: List[GRPOSample], template) -> Tuple[Dict[str, Any], GRPOBatch]:
        """Encode + collate the OPSD teacher view (teacher_prompt + same response).

        Produces a teacher micro-batch whose completion frame may differ in length from
        the student's; ``_compute_teacher_logps`` remaps the teacher logps back onto the
        student frame. The student ``encoded`` is restored so downstream logps/loss keep
        the student frame. ``build_teacher_view`` is already done by ``build_opsd_samples``.
        """
        for s in sample_slice:
            s.encoded = encode_teacher_view(s, template)
        teacher_model_inputs, teacher_grpo_batch = collate_to_grpo_micro_batch(
            sample_slice,
            template,
            device=self.device,
            padding_to=get_padding_to(self.args),
            router_replay_mode=self.args.router_replay_mode,
        )
        teacher_model_inputs['grpo_batch'] = teacher_grpo_batch
        for s in sample_slice:
            s.encoded = encode_sample(s, template)
            s.encoded.pop('_extra_kwargs', None)
        return teacher_model_inputs, teacher_grpo_batch

    def _compute_teacher_logps(self,
                               inputs: Dict[str, Any],
                               grpo_batch: GRPOBatch,
                               teacher_inputs: Optional[Tuple] = None) -> torch.Tensor:
        """OPD-RL: per-token teacher logp on the sampled tokens via a local teacher forward.

        Reuses ``compute_per_token_logps`` (same path as old/ref logps) so the teacher logp
        frame matches the policy's (token-in-token-out). Same-model LoRA self-distillation runs
        the student under ``disable_adapter``; otherwise the separate teacher model is used.
        For OPSD, the teacher forwards its own ``teacher_inputs`` (teacher_prompt + same
        response) and the result is remapped onto the student's completion frame.
        """
        is_opsd = teacher_inputs is not None
        if is_opsd:
            teacher_model_inputs, teacher_grpo_batch = teacher_inputs
            fwd_inputs = {k: v for k, v in teacher_model_inputs.items() if k != 'grpo_batch'}
            fwd_batch = teacher_grpo_batch
        else:
            fwd_inputs = inputs
            fwd_batch = grpo_batch
        seq_lengths = fwd_batch.seq_lengths
        batch_size = fwd_batch.completion_mask.shape[0]
        max_seq_len = fwd_batch.completion_mask.shape[1]

        if self._teacher_use_disable_adapter:
            from contextlib import ExitStack
            with ExitStack() as stack:
                for m in self.peft_models:
                    stack.enter_context(m.disable_adapter())
                teacher_logps_packed, _ = self.compute_per_token_logps(
                    self.unwrapped_models[0], iter([deepcopy(fwd_inputs)]), temperature=self.temperature)
        else:
            # Dynamic self-distillation (teacher_models is None): teacher = student (same
            # weights including LoRA).
            models = self.teacher_models if self.teacher_models else self.unwrapped_models
            with self.load_teacher_model_context():
                teacher_logps_packed, _ = self.compute_per_token_logps(
                    models[0], iter([deepcopy(fwd_inputs)]), temperature=self.temperature)

        if self.template.padding_free:
            teacher_per_token_logps, _ = pad_logps_back_to_batch(
                logps_rmpad=teacher_logps_packed,
                logits_to_keep=max_seq_len,
                batch_size=batch_size,
                seq_lengths=seq_lengths)
        else:
            teacher_per_token_logps = teacher_logps_packed
        if is_opsd:
            teacher_per_token_logps = remap_teacher_logps_to_student_frame(teacher_per_token_logps,
                                                                           teacher_grpo_batch.completion_mask,
                                                                           grpo_batch.completion_mask)
        return teacher_per_token_logps

    def _assemble_teacher_api_logps(self, total_samples: List[GRPOSample], mini_batch_data: List[Dict[str,
                                                                                                      Any]]) -> None:
        """OPD-RL teacher API: fetch the sampled token's logp per response position
        (``prompt_logprobs=0``) and write it as ``teacher_per_token_logps`` (completion frame)
        on each micro-batch's GRPOBatch. Same sampled-token semantics as HF/Ray OPD-RL.

        Each sample routes to exactly one teacher by tag (single teacher = all samples);
        results are fetched globally then sliced per micro-batch.
        """
        requests = build_teacher_requests(total_samples, self.template)
        all_rti = [s.response_token_ids for s in total_samples]
        parsed = fetch_teacher_parsed_by_routing(
            total_samples,
            requests,
            self.teacher_configs,
            self.teacher_clients,
            gather_fn=self._gather_teacher_requests,
            infer_fn=lambda handle, client: self._infer_teacher_requests(handle, topk=0, teacher_client=client),
            scatter_fn=self._scatter_teacher_parsed,
            is_main_process=self.is_main_process,
            tag_key=getattr(self.args, 'teacher_tag_key', 'dataset'))

        offset = 0
        for data in mini_batch_data:
            grpo_batch: GRPOBatch = data['grpo_batch']
            device = grpo_batch.completion_mask.device
            n = grpo_batch.completion_mask.shape[0]
            teacher_out = assemble_teacher_completion_logprobs(
                parsed[offset:offset + n],
                grpo_batch.completion_mask,
                device,
                response_token_ids=all_rti[offset:offset + n])
            grpo_batch.teacher_per_token_logps = teacher_out.topk_logprobs[..., 0]
            offset += n

    def _log_teacher_kl_metric(self, mini_batch_data: List[Dict[str, Any]]) -> None:
        """OPD-RL: log the per-token teacher KL (k3) averaged over response tokens (monitoring only).

        Note the deliberate asymmetry: the advantage uses the *signed* k1 log-ratio
        (``teacher_logp - student_logp``, see ``expand_advantage_to_per_token``), but monitoring uses
        the non-negative k3 estimator because it is the better "distance from the teacher" gauge --
        it should decrease over training. They measure different things on purpose."""
        mode = 'train' if self.unwrapped_models[0].training else 'eval'
        kl_sum, tok_sum = 0.0, 0.0
        for data in mini_batch_data:
            grpo_batch: GRPOBatch = data['grpo_batch']
            k3 = compute_teacher_kl_per_token(grpo_batch.teacher_per_token_logps, grpo_batch.old_per_token_logps,
                                              grpo_batch.completion_mask)
            kl_sum += k3.sum().item()
            tok_sum += grpo_batch.completion_mask.sum().item()
        self._metrics[mode]['teacher_kl'].append(kl_sum / max(tok_sum, 1.0))

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
            grpo_batch = batch['grpo_batch']
            # Compute per-token KL: old_logp - ref_logp
            per_token_kl = grpo_batch.old_per_token_logps - grpo_batch.ref_per_token_logps  # [batch_size, max_seq_len]

            # Compute per-sample KL by summing over completion tokens
            sample_kl = (per_token_kl * grpo_batch.completion_mask).sum(-1)  # [batch_size]
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

    @property
    def on_policy(self):
        return self.steps_per_generation == 1

    @profiling_decorator
    def forward_step(self, data_iterator, model):
        args = self.args
        data = next(data_iterator)
        grpo_batch: GRPOBatch = data.pop('grpo_batch')
        data = self._prepare_batch(data)
        data.pop('loss_scale', None)

        if self.enable_routing_replay and RouterReplayHelper.is_replay_backward_action(model.config):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(model.config)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        if self.enable_routing_replay and RouterReplayHelper.is_replay_forward_action(model.config):
            layers_topk_idx = data.pop('routed_experts', None)
            set_router_replay_data(layers_topk_idx, model.config)

        labels = data.pop('labels', None)
        packed_seq_params = data.get('packed_seq_params')
        max_seq_len = grpo_batch.completion_mask.shape[1]
        micro_batch_size = self.micro_batch_size

        # data is now clean model forward kwargs (template.encode guarantees this;
        # grpo_batch / loss_scale / routed_experts / labels all popped above)
        is_pp_last_stage = mpu.is_pipeline_last_stage()
        output_tensor = model(**data)
        if is_pp_last_stage and output_tensor is not None:
            logits_packed = output_tensor
            if self.temperature != 1.0:
                logits_packed.div_(self.temperature)
            per_token_logps_packed, per_token_entropy_packed = compute_logps_and_entropy_from_logits(
                logits_packed, labels, compute_entropy=self.compute_entropy)

            if args.context_parallel_size > 1:
                num_samples = packed_seq_params.seq_lens.shape[0] if args.padding_free else micro_batch_size
                cp_size = args.context_parallel_size
                per_token_logps_packed = reconstruct_tensor_cp(cp_size, per_token_logps_packed, packed_seq_params,
                                                               num_samples)
                if per_token_entropy_packed is not None:
                    per_token_entropy_packed = reconstruct_tensor_cp(cp_size, per_token_entropy_packed,
                                                                     packed_seq_params, num_samples)

            if args.padding_free:
                # Pad from rmpad [1, total_tokens] to batch format [batch_size, max_seq_len]
                per_token_logps, _ = pad_logps_back_to_batch(
                    logps_rmpad=per_token_logps_packed,
                    logits_to_keep=max_seq_len,
                    batch_size=micro_batch_size,
                    seq_lengths=grpo_batch.seq_lengths)
                if per_token_entropy_packed is not None:
                    per_token_entropy, _ = pad_logps_back_to_batch(
                        logps_rmpad=per_token_entropy_packed,
                        logits_to_keep=max_seq_len,
                        batch_size=micro_batch_size,
                        seq_lengths=grpo_batch.seq_lengths,
                        pad_value=float('nan'))
                else:
                    per_token_entropy = None
            else:
                per_token_logps = per_token_logps_packed
                per_token_entropy = per_token_entropy_packed

            output_tensor = per_token_logps
            data['per_token_entropy'] = per_token_entropy

        if self.enable_routing_replay and RouterReplayHelper.is_replay_forward_action(model.config):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(model.config)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

        data['grpo_batch'] = grpo_batch
        data['labels'] = labels
        return output_tensor, partial(self.loss_func, data=data)

    @profiling_decorator
    def loss_func(self, output_tensor: torch.Tensor, data: Dict[str, Any]):
        grpo_batch: GRPOBatch = data['grpo_batch']
        # Get pre-padded data in batch format [batch_size, max_seq_len]
        advantages = grpo_batch.advantages  # [batch_size, max_seq_len] (per-token, expanded at batch construction)
        completion_mask = grpo_batch.completion_mask  # [batch_size, max_seq_len]
        truncated_mask = grpo_batch.truncated_mask  # [batch_size]
        micro_batch_size = self.micro_batch_size

        # Get pre-computed per_token_logps and per_token_entropy from forward_step
        # These are already in batch format [batch_size, max_seq_len]
        per_token_logps = output_tensor
        per_token_entropy = data.get('per_token_entropy')

        # Get pre-padded ref/old/rollout logps from grpo_batch
        ref_per_token_logps = grpo_batch.ref_per_token_logps  # [batch_size, max_seq_len] or None
        old_per_token_logps = grpo_batch.old_per_token_logps  # [batch_size, max_seq_len]
        rollout_per_token_logps = grpo_batch.rollout_per_token_logps  # [batch_size, max_seq_len] or None

        # Rollout importance sampling correction
        rollout_correction_metrics = {}
        rollout_is_weights = None
        should_compute_rollout_metrics = (not self.disable_rollout_importance_sampling
                                          and (self.rollout_importance_sampling_mode is not None
                                               or self.log_rollout_offpolicy_metrics))
        if should_compute_rollout_metrics:
            dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
            has_flag = torch.tensor([1 if rollout_per_token_logps is not None else 0],
                                    dtype=torch.int32,
                                    device=per_token_logps.device)
            torch.distributed.all_reduce(has_flag, op=torch.distributed.ReduceOp.MIN, group=dp_group)
            should_compute_rollout_metrics = has_flag.item() > 0
        if should_compute_rollout_metrics:
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

        # Compute completion lengths before any mask modifications for accurate logging
        completion_lengths = completion_mask.sum(dim=-1)

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
            safe_ratio = torch.clamp(ref_per_token_logps - per_token_logps, min=-20, max=20)
            per_token_kl = torch.clamp(torch.exp(safe_ratio) - safe_ratio - 1, min=-10, max=10)
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

        # advantages is per-token [B, T] (expanded at batch construction so the OPD-RL signed teacher
        # log-ratio is added per token). Edge loss types that need a per-sequence advantage (real / fipo /
        # off_policy_sequence_mask) are not supported with a teacher.
        if self._has_teacher and (self.loss_type in ['real', 'fipo']
                                  or self.off_policy_sequence_mask_delta is not None):
            raise ValueError(f'OPD-RL (teacher) does not support loss_type={self.loss_type!r} / '
                             'off_policy_sequence_mask.')

        fipo_metrics = None
        if self.loss_type == 'cispo':
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type == 'sapo':
            gate_pos = torch.sigmoid(self.tau_pos * (coef_1 - 1)) * (4.0 / self.tau_pos)
            gate_neg = torch.sigmoid(self.tau_neg * (coef_1 - 1)) * (4.0 / self.tau_neg)
            is_positive = advantages > 0
            soft_gate = torch.where(is_positive, gate_pos, gate_neg)
            per_token_loss = -soft_gate * advantages
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo', 'fipo']:
            if self.loss_type == 'fipo':
                fipo_weight, fipo_metrics = self._compute_fipo_influence(log_ratio, coef_1, advantages, completion_mask)

            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if self.loss_type == 'fipo':
                per_token_loss = per_token_loss * fipo_weight
        elif self.loss_type == 'real':
            per_token_loss = torch.zeros_like(per_token_logps)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        # Compute and apply entropy mask if enabled
        # entropy_mask zeros out gradients for low-entropy (confident) tokens
        entropy_mask = None
        entropy_metrics = {}
        if self.compute_entropy and per_token_entropy is not None:
            # Fill padded tokens with NaN for correct quantile computation
            entropies = per_token_entropy.masked_fill(completion_mask == 0, float('nan'))

            if self.log_entropy:
                # Log entropy statistics
                per_completion_entropies_mean = torch.nanmean(entropies, dim=1)
                global_entropies_mean = gather(
                    per_completion_entropies_mean, group=mpu.get_data_parallel_group(with_context_parallel=True))
                valid_mask = ~torch.isnan(global_entropies_mean)
                entropy_metrics = {
                    'entropy_mean':
                    global_entropies_mean.nanmean(),
                    'entropy_max':
                    global_entropies_mean[valid_mask].max() if valid_mask.any() else torch.tensor(
                        float('nan'), device=global_entropies_mean.device),
                    'entropy_min':
                    global_entropies_mean[valid_mask].min() if valid_mask.any() else torch.tensor(
                        float('nan'), device=global_entropies_mean.device),
                }

            # Compute entropy threshold and mask for top_entropy_quantile
            if self.top_entropy_quantile < 1.0:
                # Compute threshold across all completion tokens in the batch
                entropy_threshold = torch.nanquantile(entropies.flatten().float(), 1 - self.top_entropy_quantile)
                entropy_metrics['entropy_threshold'] = entropy_threshold
                # Only keep tokens with entropy >= threshold (high uncertainty tokens)
                entropy_mask = entropies >= entropy_threshold
                # Apply entropy mask to per_token_loss
                per_token_loss = per_token_loss * entropy_mask

        # Add KL penalty if needed
        if self.beta != 0.0 and per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Apply rollout importance sampling weights if available
        if rollout_is_weights is not None and self.rollout_importance_sampling_mode is not None:
            per_token_loss = per_token_loss * rollout_is_weights

        # Apply off-policy sequence masking if enabled
        # Mask out sequences where delta > threshold AND advantage < 0
        if self.off_policy_sequence_mask_delta is not None:
            old_policy_per_token_logps = rollout_per_token_logps if rollout_per_token_logps is not None \
                else old_per_token_logps
            # advantages is per-token [B, T]; the mask needs a per-sequence scalar.
            seq_advantages = (advantages * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            off_policy_seq_mask = self._compute_off_policy_sequence_mask(per_token_logps, old_policy_per_token_logps,
                                                                         completion_mask, seq_advantages)
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
        elif self.loss_type in ['cispo', 'dapo', 'fipo']:
            # CISPO, DAPO, and FIPO: Normalize by total completion tokens across all processes
            num_items_in_batch = grpo_batch.num_items_in_batch
            dp_size = mpu.get_data_parallel_world_size()
            normalizer = num_items_in_batch / dp_size
            loss = (per_token_loss * completion_mask).sum() / normalizer.clamp(min=1.0)
        elif self.loss_type == 'real':
            global_scores = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)

            group_scores = global_scores.view(-1, self.num_generations)
            # advantages is per-token [B, T] (constant across tokens without a teacher); reduce
            # to a per-sequence scalar for the group pos/neg split.
            seq_advantages = (advantages * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            group_rewards = seq_advantages.view(-1, self.num_generations)

            pos_mask = (group_rewards > 0)
            neg_mask = (group_rewards <= 0)
            valid_mask = (pos_mask.sum(dim=1) != 0) & (neg_mask.sum(dim=1) != 0)

            if not valid_mask.any():
                loss = torch.tensor(0., device=global_scores.device) * global_scores.mean()
            else:
                batch_scores = group_scores[valid_mask]
                batch_pos_mask = pos_mask[valid_mask]
                batch_neg_mask = neg_mask[valid_mask]

                scaled_scores = batch_scores / self.real_tau
                zeros = torch.zeros(batch_scores.size(0), 1, device=batch_scores.device, dtype=batch_scores.dtype)

                # Negative Loss: log(1 + sum(e^{S_neg}))
                neg_input = scaled_scores.masked_fill(~batch_neg_mask, float('-inf'))
                neg_loss = torch.logsumexp(torch.cat([neg_input, zeros], dim=1), dim=1)

                # Positive Loss: log(1 + sum(e^{-S_pos}))
                pos_input = (-scaled_scores).masked_fill(~batch_pos_mask, float('-inf'))
                pos_loss = torch.logsumexp(torch.cat([pos_input, zeros], dim=1), dim=1)

                loss = (neg_loss + pos_loss).sum() / group_rewards.size(0)

            if self.beta != 0.0 and per_token_kl is not None:
                kl_loss = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
                loss = loss + kl_loss * self.beta
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        avg_metric = {
            'loss': loss.clone().detach(),
        }
        custom_metrics = {}
        total_completion_lengths = gather(
            completion_lengths, group=mpu.get_data_parallel_group(with_context_parallel=True))
        custom_metrics = {
            'completions/mean_length': total_completion_lengths.float().mean(),
            'completions/max_length': total_completion_lengths.float().max(),
            'completions/min_length': total_completion_lengths.float().min(),
        }

        # Add entropy metrics to custom_metrics
        if entropy_metrics:
            for key, value in entropy_metrics.items():
                if isinstance(value, torch.Tensor):
                    custom_metrics[f'entropy/{key}'] = value.clone().detach()
                else:
                    custom_metrics[f'entropy/{key}'] = torch.tensor(value, device=loss.device)

        if self.beta != 0.0 and per_token_kl is not None:
            kl_value = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            avg_metric['kl'] = kl_value.clone().detach()

        mode = 'train' if self.unwrapped_models[0].training else 'eval'

        # Compute clipping metrics
        completion_token_count = completion_mask.sum().clamp(min=1.0)
        if fipo_metrics is not None:
            avg_metric['fipo/future_kl_mean'] = ((fipo_metrics['future_kl'] * completion_mask).sum()
                                                 / completion_token_count).clone().detach()
            avg_metric['fipo/influence_weight_mean'] = ((fipo_metrics['influence_weight'] * completion_mask).sum()
                                                        / completion_token_count).clone().detach()
            avg_metric['fipo/safety_keep_ratio'] = ((fipo_metrics['safety_mask'].float() * completion_mask).sum()
                                                    / completion_token_count).clone().detach()

        if self.loss_type == 'cispo':
            # CISPO: Only track upper bound clipping
            # coef_1 is [batch_size, max_seq_len] or [batch_size, 1] depending on importance_sampling_level
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = (is_cispo_clipped.float() * completion_mask).sum() / completion_token_count
            # Store local clip ratio, _all_reduce_metric will handle averaging across ranks
            self._metrics[mode]['cispo_clip_ratio'].append(cispo_clip_ratio)
        elif self.loss_type in ['sapo', 'real']:
            # SAPO / REAL: No hard clipping, skip clipping metrics
            pass
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo', 'fipo']:
            # coef_1 is [batch_size, max_seq_len] or [batch_size, 1] depending on importance_sampling_level
            # Use exp(log_importance_weights) to get the original ratios before clamping
            coef_1_for_metrics = torch.exp(log_importance_weights)
            is_low_clipped = (coef_1_for_metrics < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1_for_metrics > 1 + self.epsilon_high) & (advantages > 0)
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
            self._metrics[mode]['clip_ratio/low_mean'].append(low_clip.item())
            self._metrics[mode]['clip_ratio/high_mean'].append(high_clip.item())
            self._metrics[mode]['clip_ratio/region_mean'].append(clip_ratio.item())
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

        if (self._step - 1) % self.steps_per_generation == 0:
            self._flush_log_completions()

        return loss, reporting_metric

    @profiling_decorator
    def resample_encode_failed_inputs(self, inputs: DataType, max_resample_rounds: int = 10) -> DataType:
        """
        Attempt to encode each input using the template. If encoding fails,
        resample from a backup iterator until we have enough valid samples.

        This method handles two cases:
        1. Prompt length exceeds max_length
        2. Encoding failures (e.g., multimodal data processing errors)

        Args:
            inputs (DataType): A list of input data samples, each containing a `messages` field.
            max_resample_rounds (int, optional): Maximum number of resample rounds.
                Each round processes samples from pending_samples buffer. Defaults to 10.

        Returns:
            DataType: A list of successfully encoded input samples with the same length as inputs.

        Raises:
            RuntimeError: If we cannot collect enough valid samples after max_resample_rounds.
        """
        return resample_encode_failed_inputs(
            self.template,
            self.resample_data_iterator,
            inputs,
            max_resample_rounds=max_resample_rounds,
            strip_response=True,
        )

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

    def get_local_rollout_batch(self, batch: List[GRPOSample]) -> List[GRPOSample]:
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
        # Shared logging infrastructure (prompt/completion/jsonl/wandb/swanlab)
        self._prepare_logging()
        # GRPO-specific: add rewards and advantages columns
        self._logs['rewards'] = defaultdict(lambda: deque(maxlen=self.args.generation_batch_size))
        self._logs['advantages'] = deque(maxlen=self.args.generation_batch_size)

        self.init_custom_metric = False
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}

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

        # 2a. kl: Direct estimator for KL(π_rollout || π_training)
        kl = masked_mean(-log_ratio, completion_mask)
        metrics['kl'] = gather(kl.unsqueeze(0), group=dp_group).nanmean()

        # 2b. k3_kl: K3 estimator for KL
        log_ratio_safe = torch.clamp(log_ratio, min=-20, max=20)
        k3_kl_matrix = torch.clamp(torch.exp(log_ratio_safe) - log_ratio_safe - 1, min=-10, max=10)
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
