# Copyright (c) ModelScope Contributors. All rights reserved.
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

# https://github.com/modelscope/ms-swift/pull/8280
try:
    import trl.import_utils as _trl_import_utils
    _orig = _trl_import_utils.is_vllm_ascend_available
    if not isinstance(_orig(), bool):
        _trl_import_utils.is_vllm_ascend_available = lambda: bool(_orig()[0])
except Exception:
    pass
# fmt: on

import concurrent.futures
import inspect
import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from packaging import version
from transformers import PreTrainedModel
from transformers.trainer import Trainer as HfTrainer
from trl import GRPOTrainer as HFGRPOTrainer
from trl.models import prepare_deepspeed
from trl.trainer import grpo_trainer
from trl.trainer.grpo_trainer import RepeatSampler, nanmax, nanmin
from trl.trainer.utils import selective_log_softmax
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from swift.dataset import RowPreprocessor
from swift.infer_engine import TransformersEngine
from swift.rewards import orms, rm_plugins
from swift.rl_core.advantage import (compute_advantages, compute_advantages_dynamic, compute_reward_metrics,
                                     compute_teacher_kl_per_token, expand_advantage_to_per_token)
from swift.rl_core.data import GRPOBatch, GRPOSample
from swift.rl_core.grpo_algorithm import score_completions
from swift.rlhf_trainers.gkd_helpers import assemble_teacher_completion_logprobs, build_teacher_requests
from swift.sequence_parallel import GatherLoss, sequence_parallel
from swift.template import Template, TemplateInputs
from swift.trainers import SwiftMixin, disable_gradient_checkpointing
from swift.utils import (JsonlWriter, get_cu_seqlens_from_position_ids, get_logger, is_swanlab_available,
                         is_wandb_available, nanstd, remove_response, seed_worker, to_device,
                         unwrap_model_for_generation)
from .arguments import GRPOConfig
from .rollout_mixin import DataType, RolloutTrainerMixin, SyncRefModelCallback
from .utils import (_ForwardRedirection, collate_to_grpo_micro_batch, compute_chord_loss, encode_sample,
                    get_even_process_data, identity_data_collator, load_pil_img, make_chord_sft_dataset,
                    pad_logps_back_to_batch, patch_save_last_checkpoint, profiling_context, profiling_decorator,
                    replace_assistant_response_with_ids, swanlab_get_run)

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
    sample_cls = GRPOSample

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_model: Optional[List[Union[PreTrainedModel, nn.Module]]] = None,
                 reward_funcs: Optional[List[Union[str, Callable]]] = None,
                 *_args,
                 **kwargs):
        patch_save_last_checkpoint()
        args: GRPOConfig = kwargs['args']
        self.args = args
        self.ref_adapter_name = getattr(args, 'ref_adapter_name', None)
        self.model_adapter_name = None
        self.is_multimodal = model.model_meta.is_multimodal
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys() if not hasattr(model, 'get_base_model') else
            inspect.signature(model.get_base_model().forward).parameters.keys())

        self.vllm_client = kwargs.pop('vllm_client', None)
        self.chord_sft_dataset = kwargs.pop('chord_sft_dataset', None)
        reward_templates = kwargs.pop('reward_template', None)
        # Teacher kwargs (same set as GKD; setting one turns GRPO into OPD-RL: teacher KL as advantage).
        self._pop_teacher_kwargs(kwargs)
        self._prepare_algorithm_params()
        super().__init__(model, ref_model, *_args, **kwargs)
        self._prepare_chord_dataset()
        self.prepare_rollout()
        self._prepare_rewards(reward_funcs, reward_model, reward_templates)

        # A configured teacher enables OPD-RL (teacher KL as advantage); with no reward
        # functions it is the sole signal. Same gating as GKD: teacher set -> teacher branch.
        self._setup_teacher()
        self.teacher_kl_coef = args.teacher_kl_coef
        if not self.reward_funcs and not self.use_gym_env and not self._has_teacher:
            raise ValueError('You must specify reward_funcs or reward_model')

        if self.args.eval_strategy != 'no' and not self.args.eval_use_evalscope:
            total_eval_batch_size = self.args.per_device_eval_batch_size * \
                self.accelerator.num_processes // self.num_generations_eval
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
            infer_template = copy(self.template)
            infer_template.padding_free = False
            infer_template.sequence_parallel_size = 1
            infer_template.remove_unused_columns = True
            self.engine = TransformersEngine(self.model, template=infer_template, max_batch_size=0)  # 0: no limit

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(self))

        if self.args.dynamic_sample or self.template.truncation_strategy == 'raise':
            self._prepare_resample_data_iterator()
        # flag indicating whether the evaluation has started
        self.eval_flag = False

        if self.template.sequence_parallel_size > 1:
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
        self._current_train_step_time = 0.0

    def _get_data_collator(self, args, template):
        return identity_data_collator

    def _get_train_sampler(self, train_dataset=None):
        if self.template.sequence_parallel_size > 1:
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

    @profiling_decorator
    def _prepare_inputs(self, inputs: DataType) -> Dict[str, Union[torch.Tensor, Any]]:
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
                micro_batches = self._generate_and_score_completions(inputs)
                self._buffered_inputs = micro_batches
            micro_batch = self._buffered_inputs[self._step % num_rollout_samples]
        else:
            micro_batch = self._generate_and_score_completions(inputs)
        return micro_batch

    def _rollout_samples(self, inputs: DataType) -> List[GRPOSample]:
        """Convert raw rows to samples and generate completions."""
        if self.template.truncation_strategy == 'raise':
            inputs = self.resample_encode_failed_inputs(inputs)
        samples: List[GRPOSample] = self.to_samples(inputs)
        samples = self._generate_completions(samples)
        return samples

    @profiling_decorator
    def _score_completions(self, samples: List[GRPOSample]) -> List[GRPOSample]:
        """Score completions with reward functions and run DAPO dynamic sampling.

        Rewards are stashed on ``self._rewards_per_func`` for ``_postprocess_batch``
        (advantages depend on the encoded batches produced after this stage).
        Returns the (possibly resampled) samples.
        """
        total_rewards_per_func = self._compute_rewards_per_func(samples)
        mode = 'train' if self.model.training else 'eval'
        if self.dynamic_sample and mode == 'train':
            # dynamic sampling for std=0 groups
            samples, total_rewards_per_func = self._dynamic_sampling(samples, total_rewards_per_func)  # noqa
        self._rewards_per_func = total_rewards_per_func
        return samples

    def _postprocess_batch(self, samples: List[GRPOSample], batch_encoded_inputs: List[Dict[str, Any]]) -> None:
        """Compute advantages and write them into each ``grpo_batch``."""
        # OPD-RL teacher API: fetch single-token teacher logps before advantage computation.
        if self._has_teacher and self.use_teacher_api:
            self._assemble_teacher_api_logps(samples, batch_encoded_inputs)
        total_advantages = self._compute_advantages(samples, self._rewards_per_func, batch_encoded_inputs)

        local_advantages = get_even_process_data(self, total_advantages)
        assert len(local_advantages) == len(samples)
        for i, advantage in enumerate(local_advantages):
            samples[i].advantages = advantage
        # log metrics in samples
        self._logs['advantages'].extend(total_advantages.tolist())

        # Add advantages to each batch in batch_encoded_inputs
        gas_chunks = self.split_by_mini_batches(samples)
        assert len(gas_chunks) == len(batch_encoded_inputs), \
            f'Mismatch: {len(gas_chunks)} chunks vs {len(batch_encoded_inputs)} batches'

        for batch, batch_encoded in zip(gas_chunks, batch_encoded_inputs):
            # Under sequence parallel, split_by_mini_batches gathers samples across the SP group via
            # all_gather_object, so per-sample advantages may carry tensors from different ranks/devices;
            # move them onto the current device before stacking.
            device = self.accelerator.device
            grpo_batch: GRPOBatch = batch_encoded['grpo_batch']
            base_advantages = torch.stack([data.advantages.to(device) for data in batch])
            # Expand the per-sequence base advantage to per-token [B, T] here (not by broadcast
            # in the loss), so the OPD-RL signed teacher log-ratio is added per token
            # (adv_t = base + coef * (teacher_logp - student_logp)).
            grpo_batch.advantages = expand_advantage_to_per_token(
                base_advantages,
                grpo_batch.completion_mask,
                teacher_per_token_logps=grpo_batch.teacher_per_token_logps if self._has_teacher else None,
                policy_per_token_logps=grpo_batch.old_per_token_logps if self._has_teacher else None,
                teacher_kl_coef=self.teacher_kl_coef if self._has_teacher else 0.0,
            )
            if self._has_teacher:
                mode = 'train' if self.model.training else 'eval'
                # Monitoring uses the non-negative k3 estimator (a "distance from the teacher" gauge
                # that should decrease over training); the advantage above uses the signed k1 log-ratio.
                k3 = compute_teacher_kl_per_token(grpo_batch.teacher_per_token_logps, grpo_batch.old_per_token_logps,
                                                  grpo_batch.completion_mask)
                self._metrics[mode]['teacher_kl'].append(
                    (k3.sum() / grpo_batch.completion_mask.sum().clamp(min=1.0)).item())

    def _log_rollout(self, samples: List[GRPOSample]) -> None:
        """Log prompts/completions and extra gathered metrics (solution, num_turns)."""
        with profiling_context(self, 'log_metrics'):
            # --- logs (prompts + completions) ---
            messages = [s.messages[:-1] for s in samples]
            completions = [deepcopy(s.messages[-1]['content']) for s in samples]
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

            if all('solution' in s.extra for s in samples):
                metrics_for_logs_to_gather['solution'] = [s.extra['solution'] for s in samples]

            if all(s.rollout_infos and 'num_turns' in s.rollout_infos for s in samples):
                metrics_for_logs_to_gather['num_turns'] = [s.rollout_infos['num_turns'] for s in samples]

            if metrics_for_logs_to_gather:
                for key, value in metrics_for_logs_to_gather.items():
                    if key not in self._logs:
                        self._logs[key] = deque(maxlen=self.args.generation_batch_size)
                    self._logs[key].extend(self._gather_and_flatten(value, flatten_level=0))

    @profiling_decorator
    def _compute_rewards_per_func(self, samples: List[GRPOSample]) -> torch.Tensor:
        """Score completions using all reward functions.

        Args:
            samples: List of on-policy samples, each carrying messages with conversation history.

        Returns:
            rewards_per_func: Tensor of shape (num_examples, num_reward_funcs) with all reward values
        """
        with self._disable_sp_context():
            local_rewards_per_func = score_completions(
                samples,
                reward_funcs=self.reward_funcs,
                reward_model_plugins=self.reward_model_plugins,
                use_gym_env=self.use_gym_env,
                device=self.accelerator.device,
                trainer_state=self.state,
            )

        # OPD-RL pure distillation: no reward_funcs -> a [N, 0] tensor. accelerate.gather
        # can't reshape a 0-column tensor (view(-1, 0) is ambiguous), so gather only the
        # global sample count and rebuild an empty [N_global, 0] reward matrix. Downstream
        # advantages then come solely from the teacher KL injection.
        if local_rewards_per_func.shape[1] == 0:
            global_count = sum(gather_object([local_rewards_per_func.shape[0]]))
            return torch.zeros((global_count, 0), dtype=torch.float32, device=self.accelerator.device)

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

    def _compute_advantages(self, samples: List[GRPOSample], rewards_per_func: torch.Tensor,
                            batch_encoded_inputs: List[Dict[str, Any]]) -> torch.Tensor:
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
            samples (List[GRPOSample]):
                On-policy samples used to derive prompt_id / request_id grouping.
            rewards_per_func (torch.Tensor):
                Reward values for each reward function, shape `(N, num_reward_funcs)`.

        Returns:
            **advantages** (torch.Tensor):
                Computed advantages, shape `(N,)`.
        """

        def log_rewards_metrics(rewards: torch.Tensor, rewards_per_func_for_metrics: torch.Tensor):
            """Log reward statistics for monitoring. Only log once per unique request_id."""
            # rewards: [prompt_batch_size, num_generations]
            # rewards_per_func_for_metrics: [prompt_batch_size*num_generations, self.num_reward_funcs]
            mode = 'train' if self.model.training else 'eval'
            num_generations = self.num_generations if mode == 'train' else self.num_generations_eval
            group_rewards = rewards.view(-1, num_generations)
            rewards_mean = group_rewards.mean(-1).mean().item()
            if self.scale_rewards in ['group', 'none', 'gdpo']:
                # Handle edge case when num_generations_eval=1
                if num_generations > 1:
                    rewards_std = group_rewards.std(-1).mean().item()
                else:
                    rewards_std = 0.0
            elif self.scale_rewards == 'batch':
                rewards_std = rewards.std().item() if rewards.numel() > 1 else 0.0
            if num_generations > 1:
                is_std_zero = torch.isclose(group_rewards.std(dim=1), torch.zeros_like(group_rewards.std(dim=1)))
            else:
                is_std_zero = torch.ones(group_rewards.size(0), dtype=torch.bool, device=group_rewards.device)

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

        # Compute per-sample KL values if kl_in_reward is enabled.
        kl_values = None
        if self.kl_in_reward and self.beta != 0.0:
            kl_list = []
            for batch_encoded in batch_encoded_inputs:
                grpo_batch = batch_encoded['grpo_batch']
                per_token_kl = grpo_batch.old_per_token_logps - grpo_batch.ref_per_token_logps
                kl = (per_token_kl * grpo_batch.completion_mask).sum(-1)
                kl_list.append(kl)
            kl_values = torch.cat(kl_list, dim=0)
            kl_values = gather(kl_values)

        # Keep weighted rewards for the request-aware (multi-turn) path below.
        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)
        if self.kl_in_reward and self.beta != 0.0:
            rewards = rewards - self.beta * kl_values

        # --------------------------------------------------
        # Case 1: Default grouped mode
        # --------------------------------------------------
        mode = 'train' if self.model.training else 'eval'
        num_generations = self.num_generations if mode == 'train' else self.num_generations_eval
        if not self.dynamic_num_samples:
            advantages, weighted_rewards = compute_advantages(
                rewards_per_func=rewards_per_func,
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
                rewards_per_func=rewards_per_func,
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
            log_rewards_all(rewards_per_func)
            return advantages

        # --------------------------------------------------
        # Case 2: Request-aware mode
        # --------------------------------------------------
        else:
            prompt_ids = gather_object([s.prompt_id for s in samples])
            request_ids = gather_object([s.request_id for s in samples])
            assert rewards.shape[0] == len(prompt_ids) == len(request_ids)

            # Validate rewards consistency within the same request_id (fail-fast).
            for rid in set(request_ids):
                idxs = [i for i, r in enumerate(request_ids) if r == rid]
                if not torch.allclose(rewards[idxs], rewards[idxs[0]].expand(len(idxs)), atol=1e-6):
                    raise ValueError(f'Inconsistent rewards detected for request_id={rid}.')

            advantages, _ = compute_advantages_dynamic(
                rewards_per_func=rewards_per_func,
                reward_weights=self.reward_weights,
                prompt_ids=prompt_ids,
                request_ids=request_ids,
                advantage_estimator=self.advantage_estimator,
                scale_rewards=self.scale_rewards,
                kl_in_reward=self.kl_in_reward,
                beta=self.beta,
                kl_values=kl_values,
            )

            # Metrics are logged on request-deduplicated rewards.
            unique_indices = self._get_last_indices(request_ids)
            log_rewards_metrics(
                rewards=rewards[unique_indices], rewards_per_func_for_metrics=rewards_per_func[unique_indices])
            log_rewards_all(rewards_per_func)

            return advantages

    def _compute_teacher_logps(self, model_inputs, grpo_batch, origin_data=None) -> torch.Tensor:
        """OPD-RL: per-token teacher logp on the sampled tokens via a local teacher forward.

        Reuses ``_get_per_token_logps_and_entropies`` so the teacher logp frame matches
        the policy's old/ref logps (token-in-token-out, single token per position).
        LoRA same-model teacher runs the student under ``disable_adapter``.
        """
        with torch.no_grad():
            if self._teacher_use_disable_adapter:
                with self.accelerator.unwrap_model(self.model).disable_adapter(), disable_gradient_checkpointing(
                        self.model, self.args.gradient_checkpointing_kwargs):
                    return self._get_per_token_logps_and_entropies(
                        self.model, model_inputs, grpo_batch, origin_data=origin_data)[0]
            with self.load_teacher_model_context(), disable_gradient_checkpointing(
                    self.teacher_model, self.args.gradient_checkpointing_kwargs):
                return self._get_per_token_logps_and_entropies(
                    self.teacher_model, model_inputs, grpo_batch, origin_data=origin_data)[0]

    def _assemble_teacher_api_logps(self, samples: List[GRPOSample], batch_encoded_inputs: List[Dict[str,
                                                                                                     Any]]) -> None:
        """OPD-RL teacher API: fetch the *sampled* token's logp at each response position
        (``prompt_logprobs=0`` -> ``parse_prompt_logprobs(topk=0)``, the actually-present
        token, not the top-1) as a completion-frame ``TeacherOutput``, then read its single
        column as ``teacher_per_token_logps``. Future top-k RL reuses the same ``TeacherOutput``."""
        sample_chunks = self.split_by_mini_batches(samples)
        local_requests, chunk_sizes = [], []
        chunk_rti = []
        for chunk in sample_chunks:
            reqs = build_teacher_requests(chunk, self.template)
            local_requests.extend(reqs)
            chunk_sizes.append(len(reqs))
            chunk_rti.append([s.response_token_ids for s in chunk])
        parsed_local = self._fetch_teacher_logprobs(local_requests, topk=0)

        offset = 0
        for batch_encoded, cs, rti in zip(batch_encoded_inputs, chunk_sizes, chunk_rti):
            chunk_parsed = parsed_local[offset:offset + cs]
            offset += cs
            grpo_batch: GRPOBatch = batch_encoded['grpo_batch']
            teacher_out = assemble_teacher_completion_logprobs(
                chunk_parsed, grpo_batch.completion_mask, grpo_batch.completion_mask.device, response_token_ids=rti)
            grpo_batch.teacher_per_token_logps = teacher_out.topk_logprobs[..., 0]

    @profiling_decorator
    def _dynamic_sampling(self, samples: List[GRPOSample],
                          rewards_per_func: torch.Tensor) -> Tuple[List[GRPOSample], torch.Tensor]:
        """
        Perform dynamic sampling to replace samples with zero-reward-variance groups.

        This method implements DAPO (https://arxiv.org/abs/2503.14476) by replacing
        samples from groups with zero reward variance (std=0) through resampling.

        Args:
            samples: local on-policy samples
            rewards_per_func: reward per function for global samples

        Returns:
            tuple: (samples, rewards_per_func) with zero-variance groups replaced by resampled data
        """
        # DAPO https://arxiv.org/abs/2503.14476
        # Replaces samples with zero-reward-variance groups (std=0)
        resample_count = 0
        valid_samples = []
        valid_rewards_per_func = []
        origin_data = (samples, rewards_per_func)

        while resample_count < self.max_resample_times:
            rewards_std = self.compute_std(samples, rewards_per_func)
            valid_mask = (rewards_std > 0)
            all_samples = gather_object(samples)
            valid_samples.extend([s for s, mask in zip(all_samples, valid_mask) if mask])
            valid_rewards_per_func.append(rewards_per_func[valid_mask])
            if len(valid_samples) >= self.args.generation_batch_size:
                break

            inputs = next(self.dynamic_resample_iterator)
            if self.template.truncation_strategy == 'raise':
                inputs = self.resample_encode_failed_inputs(inputs)
            samples = self.to_samples(inputs)
            samples = self._generate_completions(samples)
            rewards_per_func = self._compute_rewards_per_func(samples)
            resample_count += 1

        if len(valid_samples) >= self.args.generation_batch_size:
            process_slice = slice(
                self.accelerator.process_index * len(samples),
                (self.accelerator.process_index + 1) * len(samples),
            )
            samples = valid_samples[:self.args.generation_batch_size][process_slice]
            rewards_per_func = torch.cat(valid_rewards_per_func)[:self.args.generation_batch_size]
        else:
            logger.warning(f'There are still std=0 groups present after {self.max_resample_times} retries.')
            samples, rewards_per_func = origin_data

        return samples, rewards_per_func

    def compute_std(self, samples: List[GRPOSample], rewards_per_func: torch.Tensor) -> torch.Tensor:
        """Compute the standard deviation of the rewards per function."""
        device = self.accelerator.device
        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)

        mode = 'train' if self.model.training else 'eval'
        num_generations = self.num_generations if mode == 'train' else self.num_generations_eval
        if not self.dynamic_num_samples:
            grouped_rewards = rewards.view(-1, num_generations)
            # Handle edge case when num_generations_eval=1
            if num_generations > 1:
                group_rewards_std = grouped_rewards.std(dim=1).repeat_interleave(num_generations)
            else:
                group_rewards_std = torch.zeros_like(rewards)
            return group_rewards_std
        else:
            prompt_ids = gather_object([s.prompt_id for s in samples])
            request_ids = gather_object([s.request_id for s in samples])
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
                # Edge case: when group size is 1
                prompt_stds[idx_tensor] = r_group.std() if len(idxs) > 1 else 0.0
            rid_to_idx = {rid: idx for idx, rid in enumerate(unique_request_ids)}
            indices_in_unique = torch.tensor([rid_to_idx[r] for r in request_ids], device=device)
            rewards_std = prompt_stds[indices_in_unique]

            return rewards_std

    @profiling_decorator
    def _prepare_batch_inputs(self, samples: List[GRPOSample]) -> List[Dict[str, Any]]:
        """
        Prepare the final batch inputs with ref/old_policy logps and other fields for RL training.

        Args:
            samples (List[GRPOSample]): List of local on-policy samples.

        Returns:
            List[Dict[str, Any]]: A list of prepared batch dicts (model inputs + ``grpo_batch``),
            organized as [steps_per_generation][batch_size]
        """
        template = self.template
        gas_chunks = self.split_by_mini_batches(samples)
        ga_batch_encoded_inputs: List[Dict[str, Any]] = []
        for batch in gas_chunks:
            with self._template_context(template):
                for s in batch:
                    encoded_inputs = encode_sample(s, template)
                    encoded_inputs.pop('_extra_kwargs', None)  # pop add_eos
                    s.encoded = encoded_inputs
                model_inputs, grpo_batch = collate_to_grpo_micro_batch(
                    batch, template, device=self.model.device, use_logits_to_keep=True)

            model_inputs.pop('labels', None)
            batch_encoded_inputs = {'model_inputs': model_inputs, 'grpo_batch': grpo_batch}
            if self.dynamic_num_samples and self.is_multimodal:
                batch_encoded_inputs['_origin_data'] = batch
            origin_data = batch if (self.dynamic_num_samples and self.is_multimodal) else None

            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                grpo_batch.old_per_token_logps = (
                    self._get_per_token_logps_and_entropies(
                        self.model, model_inputs, grpo_batch, origin_data=origin_data)[0])
                if self.beta == 0.0:
                    ref_per_token_logps = None
                elif self.ref_model is not None:
                    with disable_gradient_checkpointing(self.ref_model, self.args.gradient_checkpointing_kwargs):
                        ref_per_token_logps = \
                            self._get_per_token_logps_and_entropies(
                                self.ref_model, model_inputs, grpo_batch, origin_data=origin_data)[0]
                else:
                    with self.null_ref_context():
                        ref_per_token_logps = \
                            self._get_per_token_logps_and_entropies(
                                self.model, model_inputs, grpo_batch, origin_data=origin_data)[0]
                grpo_batch.ref_per_token_logps = ref_per_token_logps
                # OPD-RL: local teacher logp on the sampled tokens (API path filled later).
                if self._has_teacher and not self.use_teacher_api:
                    grpo_batch.teacher_per_token_logps = self._compute_teacher_logps(
                        model_inputs, grpo_batch, origin_data=origin_data)
            ga_batch_encoded_inputs.append(batch_encoded_inputs)

        # --- log completion lengths ---
        mode = 'train' if self.model.training else 'eval'
        device = self.accelerator.device
        local_lengths = [inp['grpo_batch'].completion_mask.sum(1).tolist() for inp in ga_batch_encoded_inputs]
        total_lengths = self._gather_and_flatten(local_lengths, dtype=torch.float32, device=device, flatten_level=1)

        # Store num_items_in_batch for DAPO loss (total completion tokens across all processes)
        num_items_in_batch = total_lengths.sum()
        for batch_encoded in ga_batch_encoded_inputs:
            batch_encoded['grpo_batch'].num_items_in_batch = num_items_in_batch

        self._metrics[mode]['completions/mean_length'].append(total_lengths.mean().item())
        self._metrics[mode]['completions/min_length'].append(total_lengths.min().item())
        self._metrics[mode]['completions/max_length'].append(total_lengths.max().item())

        # --- log completion clipped ratio ---
        local_trunc_masks = [inp['grpo_batch'].truncated_mask.tolist() for inp in ga_batch_encoded_inputs]
        total_trunc_masks = self._gather_and_flatten(
            local_trunc_masks, dtype=torch.bool, device=device, flatten_level=1)

        if not self.dynamic_num_samples:
            clipped_ratio = total_trunc_masks.sum().item() / total_lengths.shape[0]
            self._metrics[mode]['completions/clipped_ratio'].append(clipped_ratio)

            if all(s.rollout_infos and 'num_turns' in s.rollout_infos for s in samples):
                num_turns = torch.tensor(gather_object([s.rollout_infos['num_turns'] for s in samples]), device=device)
                self._metrics[mode]['num_turns'].append(num_turns.float().mean().item())
        else:
            request_ids = gather_object([s.request_id for s in samples])
            last_indices = self._get_last_indices(request_ids)

            final_trunc_masks = total_trunc_masks[last_indices]
            clipped_ratio = final_trunc_masks.sum().item() / final_trunc_masks.shape[0]
            self._metrics[mode]['completions/clipped_ratio'].append(clipped_ratio)

            if all(s.rollout_infos and 'num_turns' in s.rollout_infos for s in samples):
                num_turns_all = torch.tensor(
                    gather_object([s.rollout_infos['num_turns'] for s in samples]), device=device)
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

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Compute the per-token log probabilities for the model, return_outputs=True in mini-batch training
        if isinstance(inputs, list):
            assert len(inputs) == 1
            inputs = inputs[0]
        model_inputs = inputs['model_inputs']
        grpo_batch = inputs['grpo_batch']
        origin_data = inputs.get('_origin_data')
        if self.use_liger_loss:
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(
                model, unwrapped_model,
                lambda *_, **__: self.compute_liger_loss(unwrapped_model, model_inputs, grpo_batch), **model_inputs)
        else:
            return self._compute_loss(model, model_inputs, grpo_batch, origin_data)

    def _compute_loss(self, model, model_inputs, grpo_batch, origin_data=None):
        mode = 'train' if self.model.training else 'eval'

        # Check batch size and decide processing strategy
        batch_size = grpo_batch.seq_lengths.shape[0] if self.template.padding_free else model_inputs['input_ids'].shape[
            0]
        expected_bs = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size

        should_chunk = self.dynamic_num_samples and any(gather_object([batch_size > expected_bs]))
        if not should_chunk:
            return self._compute_loss_single(model, model_inputs, grpo_batch)
        else:
            # maybe dynamic rollout num for multi-turn training
            return self._compute_loss_chunked(model, model_inputs, grpo_batch, origin_data)

    def _compute_loss_single(self, model, model_inputs, grpo_batch):
        """Original loss computation logic for single batch processing."""
        loss, metrics_data = self._compute_loss_and_metrics(model, model_inputs, grpo_batch)
        self._update_metrics(metrics_data)
        return loss

    def _compute_fipo_influence(self, log_ratio: torch.Tensor, coef_1: torch.Tensor, advantages: torch.Tensor,
                                completion_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute FIPO token-level influence weight from Future-KL divergence."""
        future_kl_delta = log_ratio.masked_fill(~completion_mask, 0.0)

        # Dual-Clip participation mask: high-ratio tokens do not contribute to Future-KL.
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

        # avoid amplifying negative-advantage tokens with very high IS ratios.
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

    def _compute_loss_and_metrics(self, model, model_inputs: Dict[str, Any], grpo_batch: GRPOBatch):
        """Core loss computation without metrics recording."""
        mode = 'train' if self.model.training else 'eval'
        completion_mask = grpo_batch.completion_mask
        truncated_mask = grpo_batch.truncated_mask
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, model_inputs, grpo_batch, compute_entropy=self.compute_entropy)

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

        per_token_kl = None
        if self.beta != 0.0 and not self.kl_in_reward:
            ref_per_token_logps = grpo_batch.ref_per_token_logps
            safe_ratio = torch.clamp(ref_per_token_logps - per_token_logps, min=-20, max=20)
            per_token_kl = torch.clamp(torch.exp(safe_ratio) - safe_ratio - 1, min=-10, max=10)

        advantages = grpo_batch.advantages
        old_per_token_logps = (
            per_token_logps.detach() if grpo_batch.old_per_token_logps is None else grpo_batch.old_per_token_logps)

        # Compute rollout diagnostic metrics and apply IS correction if enabled
        rollout_correction_metrics = {}
        should_compute_rollout_metrics = (
            self.rollout_importance_sampling_mode is not None or self.log_rollout_offpolicy_metrics)

        local_has_rollout = grpo_batch.rollout_per_token_logps is not None
        should_compute_rollout_metrics = should_compute_rollout_metrics and all(gather_object([local_has_rollout]))
        rollout_is_weights = None
        if (not self.disable_rollout_importance_sampling and should_compute_rollout_metrics):
            rollout_per_token_logps = grpo_batch.rollout_per_token_logps

            # Compute diagnostic metrics (KL, PPL, etc.) for monitoring off-policy gap
            rollout_correction_metrics = self._compute_rollout_offpolicy_metrics(old_per_token_logps,
                                                                                 rollout_per_token_logps,
                                                                                 completion_mask)

            rollout_log_ratio, rollout_is_weights = self._get_rollout_is_correction(old_per_token_logps,
                                                                                    rollout_per_token_logps,
                                                                                    completion_mask)
            if rollout_log_ratio is not None:
                is_metrics = self._compute_is_correction_metrics(rollout_log_ratio, rollout_is_weights, completion_mask)
                rollout_correction_metrics.update(is_metrics)

            pass
        # rollout_is_weights is a local variable, initialized to None above

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

        # advantages is per-token [B, T] (expanded at batch construction so the OPD-RL signed
        # teacher log-ratio is added per token). Edge loss types that need a per-sequence
        # advantage (real / fipo / off_policy_sequence_mask) are not supported with a teacher.
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
        elif self.loss_type == 'real':
            per_token_loss = torch.zeros_like(per_token_logps)
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
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if rollout_is_weights is not None and self.rollout_importance_sampling_mode is not None:
            per_token_loss = per_token_loss * rollout_is_weights

        if self.off_policy_sequence_mask_delta is not None:
            old_policy_per_token_logps = (
                grpo_batch.rollout_per_token_logps
                if grpo_batch.rollout_per_token_logps is not None else old_per_token_logps)
            # advantages is per-token [B, T] (constant across tokens without a teacher); the mask
            # needs a per-sequence scalar.
            seq_advantages = (advantages * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            off_policy_seq_mask = self._compute_off_policy_sequence_mask(per_token_logps, old_policy_per_token_logps,
                                                                         completion_mask, seq_advantages)
            # Expand sequence mask to token level and apply to completion_mask
            off_policy_seq_mask_expanded = off_policy_seq_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & off_policy_seq_mask_expanded

        if self.loss_type in ['grpo', 'sapo']:
            # completion_mask is now always [batch_size, seq_len] after pad_back
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == 'bnpo':
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == 'dr_grpo':
            batch_size = completion_mask.shape[0]
            loss = (per_token_loss * completion_mask).sum() / (batch_size * self.max_completion_length)
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

            if self.beta != 0.0:
                kl_loss = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
                loss = loss + kl_loss * self.beta
        elif self.loss_type in ['cispo', 'dapo', 'fipo']:
            # CISPO, DAPO, and FIPO: Normalize by total completion tokens across all processes
            normalizer = grpo_batch.num_items_in_batch / self.accelerator.num_processes
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

        if fipo_metrics is not None:
            fipo_future_kl = masked_batch_mean(fipo_metrics['future_kl'])
            fipo_influence_weight = masked_batch_mean(fipo_metrics['influence_weight'])
            fipo_safety_keep = masked_batch_mean(fipo_metrics['safety_mask'].float())
            metrics_data['fipo'] = {
                'future_kl_mean': self.accelerator.gather_for_metrics(fipo_future_kl).nanmean().item(),
                'influence_weight_mean': self.accelerator.gather_for_metrics(fipo_influence_weight).nanmean().item(),
                'safety_keep_ratio': self.accelerator.gather_for_metrics(fipo_safety_keep).nanmean().item(),
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
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather_for_metrics(cispo_clip_ratio)
            metrics_data['clipping'] = {'cispo_clip_ratio': gathered_cispo_clip_ratio.nanmean().item()}
        elif self.loss_type in ['sapo', 'real']:
            pass
        else:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
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

        # Update FIPO metrics
        if 'fipo' in metrics_data:
            for key, value in metrics_data['fipo'].items():
                self._metrics[mode][f'fipo/{key}'].append(value)

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

    def _compute_loss_chunked(self, model, model_inputs: Dict[str, Any], grpo_batch: GRPOBatch, origin_data=None):
        """
        Compute loss in **fixed-size chunks** to reduce peak GPU memory.

        The function guarantees that **all ranks step through the same number of
        chunks**, so that collective communication remain synchronized
        even when local ``batch_size`` differs.
        """
        mode = 'train' if self.model.training else 'eval'
        chunk_size = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size
        batch_size = grpo_batch.seq_lengths.shape[0] if self.template.padding_free else model_inputs['input_ids'].shape[
            0]

        # Decide how many chunks every rank must run
        batch_sizes = gather_object([batch_size])
        chunks_per_device = [(bs + chunk_size - 1) // chunk_size for bs in batch_sizes]
        max_chunks = max(chunks_per_device)

        # Re-compute chunk size so that max_chunks * new_chunk_size covers entire batch
        new_chunk_size = (batch_size + max_chunks - 1) // max_chunks

        losses, weights = [], []
        all_metrics_data = []
        chunk_model_inputs, chunk_grpo_batch = {}, None
        for chunk_idx in range(max_chunks):
            start_idx = chunk_idx * new_chunk_size
            end_idx = min(start_idx + new_chunk_size, batch_size)

            is_dummy = False
            if start_idx < batch_size:
                chunk_model_inputs, chunk_grpo_batch = self.get_chunked_inputs(model_inputs, grpo_batch, start_idx,
                                                                               end_idx, origin_data)
                chunk_weight = end_idx - start_idx
            else:
                is_dummy = True
                chunk_weight = 0

            # Compute loss and metrics for this chunk
            chunk_loss, chunk_metrics_data = self._compute_loss_and_metrics(model, chunk_model_inputs, chunk_grpo_batch)

            if not is_dummy:
                losses.append(chunk_loss * chunk_weight)
                weights.append(chunk_weight)
                all_metrics_data.append((chunk_metrics_data, chunk_weight))
            else:
                # # Add dummy loss to computation graph to trigger ZeRO-3 backward hooks
                losses.append(chunk_loss * 0.0)

        # Compute weighted average loss
        total_weight = sum(weights)
        if total_weight > 0:
            final_loss = torch.stack(losses).sum() / total_weight
        else:
            final_loss = torch.stack(losses).sum()

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
        fipo_values = {}

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

            # Collect FIPO metrics (weighted by tokens)
            if 'fipo' in chunk_metrics:
                weight = chunk_tokens.item() if hasattr(chunk_tokens, 'item') else chunk_tokens
                for key, value in chunk_metrics['fipo'].items():
                    fipo_values.setdefault(key, []).append((value, weight))

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
            aggregated_metrics['entropy'] = {
                'entropy_logs': entropy_logs,
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

        if fipo_values:
            aggregated_metrics['fipo'] = {key: weighted_avg(values) for key, values in fipo_values.items()}

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
                          model_inputs: Dict[str, Any],
                          grpo_batch: GRPOBatch,
                          logits_to_keep: int,
                          input_ids: torch.Tensor,
                          compute_entropy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get per token logps via sequence parallel, returns rmpad format [1, total_nnz] for padding_free mode"""
        sequence_parallel.prepare_inputs(model_inputs)
        with self._template_context(self.template, model_inputs):
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
            seq_lengths = grpo_batch.seq_lengths
            batch_size = seq_lengths.shape[0]
            rp_world_size = sequence_parallel.rp_world_size

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
                                     model_inputs: Dict[str, Any],
                                     logits_to_keep: int,
                                     input_ids: torch.Tensor,
                                     compute_entropy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get per token logps via local forward pass, returns rmpad format [1, total_nnz] for padding_free mode"""
        if 'logits_to_keep' in self.model_kwarg_keys:
            model_inputs['logits_to_keep'] = logits_to_keep + 1

        # Forward pass
        logits = model(**model_inputs).logits

        logits = logits[:, -(logits_to_keep + 1):-1, :]
        logits.div_(self.temperature)

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

    @profiling_decorator
    def _get_per_token_logps_and_entropies(self,
                                           model,
                                           model_inputs: Dict[str, Any],
                                           grpo_batch: GRPOBatch,
                                           compute_entropy=False,
                                           origin_data=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute per-token log probabilities and entropies with memory-efficient batching.

        When rollout count is larger than expected, we process in smaller batches
        to control memory usage.
        """
        batch_size = grpo_batch.seq_lengths.shape[0] if self.template.padding_free else model_inputs['input_ids'].shape[
            0]
        mode = 'train' if self.model.training else 'eval'
        expected_bs = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size  # noqa
        should_chunk = self.dynamic_num_samples and any(gather_object([batch_size > expected_bs]))
        if not should_chunk:
            return self._get_per_token_logps_and_entropies_single(
                model, model_inputs, grpo_batch, compute_entropy=compute_entropy)
        else:
            return self._get_per_token_logps_and_entropies_chunked(
                model, model_inputs, grpo_batch, compute_entropy=compute_entropy, origin_data=origin_data)

    def _get_per_token_logps_and_entropies_single(self,
                                                  model,
                                                  model_inputs: Dict[str, Any],
                                                  grpo_batch: GRPOBatch,
                                                  compute_entropy=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits_to_keep = grpo_batch.logits_to_keep
        input_ids = model_inputs['input_ids']
        is_padding_free = self.template.padding_free
        use_sp = self.template.sequence_parallel_size > 1

        # Store metadata for padding_free restoration
        if is_padding_free:
            original_seq_lengths = grpo_batch.seq_lengths
            batch_size = original_seq_lengths.shape[0]

        if use_sp:
            # Sequence parallel path.
            # In padding_free mode: returns [1, logits_to_keep] format (rmpad, needs unpad)
            # In non-padding_free mode: returns [batch_size, logits_to_keep] format
            logps, entropies = self._get_logps_via_sp(
                model, model_inputs, grpo_batch, logits_to_keep, input_ids, compute_entropy=compute_entropy)
        else:
            # Local forward pass (sole non-SP path; the TRL super() path is equivalent and
            # has been removed to keep the shift logic in one place).
            # Returns [1, total_nnz] in padding_free mode, or [batch_size, logits_to_keep] otherwise
            logps, entropies = self._get_logps_via_local_forward(
                model, model_inputs, logits_to_keep, input_ids, compute_entropy=compute_entropy)

        # Unpad for padding_free mode (both SP and non-SP paths need this)
        if is_padding_free:
            logps, entropies = self._unpad_logps_and_entropies(logps, entropies, logits_to_keep, batch_size,
                                                               original_seq_lengths, compute_entropy)

        return logps, entropies

    def _get_per_token_logps_and_entropies_chunked(self,
                                                   model,
                                                   model_inputs: Dict[str, Any],
                                                   grpo_batch: GRPOBatch,
                                                   compute_entropy=False,
                                                   origin_data=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute per-token log-probabilities (and optionally entropies) in **fixed-size
        chunks** to bound peak GPU memory.

        This routine **guarantees that every rank executes the same number of
        chunks**, even when the local batch sizes differ.

        Parameters
        ----------
        model : torch.nn.Module
            The model used to compute log-probs and entropies.
        model_inputs : Dict[str, Any]
            Clean model-forward kwargs (input_ids, attention_mask, etc.).
        grpo_batch : GRPOBatch
            Batch-level RL signals (completion_mask, advantages, etc.).
        compute_entropy : bool, optional
            Whether to compute per-token entropies as well (default: False).
        origin_data : list, optional
            Original samples for multimodal re-encoding during chunking.

        Returns
        -------
        final_logps : torch.Tensor
            Concatenated per-token log-probabilities for the **entire batch**.
        final_entropies : torch.Tensor or None
            Concatenated per-token entropies, or ``None`` if ``compute_entropy`` is
            ``False``.
        """
        batch_size = grpo_batch.seq_lengths.shape[0] if self.template.padding_free else model_inputs['input_ids'].shape[
            0]
        mode = 'train' if self.model.training else 'eval'
        chunk_size = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size

        batch_sizes = gather_object([batch_size])  # list[int]
        chunks_per_device = [(bs + chunk_size - 1) // chunk_size for bs in batch_sizes]
        max_chunks = max(chunks_per_device)

        new_chunk_size = (batch_size + max_chunks - 1) // max_chunks

        all_logps, all_entropies = [], [] if compute_entropy else None

        # Process in chunks
        chunk_model_inputs, chunk_grpo_batch = {}, None
        for chunk_idx in range(max_chunks):
            start_idx = chunk_idx * new_chunk_size
            end_idx = min(start_idx + new_chunk_size, batch_size)

            if start_idx < end_idx:
                chunk_model_inputs, chunk_grpo_batch = self.get_chunked_inputs(model_inputs, grpo_batch, start_idx,
                                                                               end_idx, origin_data)

            chunk_logps, chunk_entropies = self._get_per_token_logps_and_entropies_single(
                model, chunk_model_inputs, chunk_grpo_batch, compute_entropy)

            if start_idx < end_idx:
                all_logps.append(chunk_logps)
                if compute_entropy and chunk_entropies is not None:
                    all_entropies.append(chunk_entropies)

        # Concatenate results
        final_logps = torch.cat(all_logps, dim=0)
        final_entropies = torch.cat(all_entropies, dim=0) if all_entropies else None

        return final_logps, final_entropies

    @profiling_decorator
    def _get_last_hidden_state(self, unwrapped_model, model_inputs, logits_to_keep):
        # unwrap the model to access the model.model
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        if not self.is_multimodal:
            last_hidden_state = unwrapped_model.model(
                input_ids=model_inputs['input_ids'], attention_mask=model_inputs['attention_mask']).last_hidden_state
        else:
            forward_inputs = dict(model_inputs)
            if 'logits_to_keep' in self.model_kwarg_keys:
                forward_inputs['logits_to_keep'] = logits_to_keep + 1

            last_hidden_state = unwrapped_model.model(**forward_inputs).last_hidden_state

        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def _get_rollout_is_correction(self, old_per_token_logps, rollout_per_token_logps, completion_mask):
        """Compute rollout importance sampling log-ratio and IS weights.

        Returns:
            (rollout_log_ratio, rollout_is_weights) if rollout IS correction is applicable,
            (None, None) otherwise.
        """
        if self.rollout_importance_sampling_mode is None or self.disable_rollout_importance_sampling:
            return None, None

        rollout_log_ratio = old_per_token_logps - rollout_per_token_logps
        rollout_is_weights = self._apply_rollout_importance_sampling(rollout_log_ratio, completion_mask)
        return rollout_log_ratio, rollout_is_weights

    def compute_liger_loss(self, unwrapped_model, model_inputs: Dict[str, Any], grpo_batch: GRPOBatch):
        assert not self.template.padding_free
        assert self.advantage_estimator == 'grpo'
        input_ids = model_inputs['input_ids']
        logits_to_keep = grpo_batch.logits_to_keep
        completion_ids = input_ids[:, -logits_to_keep:]
        completion_mask = grpo_batch.completion_mask

        last_hidden_state = self._get_last_hidden_state(unwrapped_model, model_inputs, logits_to_keep)

        old_per_token_logps = grpo_batch.old_per_token_logps
        local_has = grpo_batch.rollout_per_token_logps is not None
        vllm_is_ratio = None
        if all(gather_object([local_has])):
            _, vllm_is_ratio = self._get_rollout_is_correction(old_per_token_logps, grpo_batch.rollout_per_token_logps,
                                                               completion_mask)

        # LigerFusedLinearGRPOLoss expects [B]; expand_advantage_to_per_token always yields [B, T].
        advantages = grpo_batch.advantages
        advantages = (advantages * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)

        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=advantages,
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=old_per_token_logps,
            ref_per_token_logps=grpo_batch.ref_per_token_logps,
            vllm_is_ratio=vllm_is_ratio,
        )

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
                **{
                    k: list(v)[:seen_nums]
                    for k, v in self._logs['rewards'].items()
                },
                'advantages': list(self._logs['advantages'])[:seen_nums],
            }
            for key, value in self._logs.items():
                if key not in table and key not in ['image', 'rewards']:
                    table[key] = list(value)[:seen_nums]

            if self.args.log_entropy:
                table.update({'entropy': list(self._logs['entropy'])[:seen_nums]})

            report_to_wandb = self.args.report_to and 'wandb' in self.args.report_to and wandb.run is not None
            report_to_swanlab = self.args.report_to and 'swanlab' in self.args.report_to and swanlab_get_run(
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
        local_list : List[Any]
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

    def _get_last_indices(self, request_ids: List[str]) -> torch.Tensor:
        seen = {}
        for i, rid in enumerate(request_ids):
            seen[rid] = i
        return torch.tensor(list(seen.values()), dtype=torch.long, device=self.accelerator.device)

    def get_chunked_inputs(self, model_inputs, grpo_batch, start_idx, end_idx, origin_data=None):
        """Slice ``model_inputs`` and ``grpo_batch`` by batch dimension."""
        chunk_model_inputs = {}
        batch_size = grpo_batch.seq_lengths.shape[0] if self.template.padding_free else model_inputs['input_ids'].shape[
            0]
        for key, val in model_inputs.items():
            if isinstance(val, torch.Tensor):
                if val.ndim == 0:
                    chunk_model_inputs[key] = val
                elif self.is_multimodal and val.shape[0] != batch_size:
                    continue
                else:
                    chunk_model_inputs[key] = val[start_idx:end_idx]
            elif isinstance(val, list) and len(val) == batch_size:
                chunk_model_inputs[key] = val[start_idx:end_idx]
            else:
                chunk_model_inputs[key] = val

        chunk_grpo_batch = GRPOBatch(
            completion_mask=grpo_batch.completion_mask[start_idx:end_idx],
            truncated_mask=grpo_batch.truncated_mask[start_idx:end_idx],
            seq_lengths=(grpo_batch.seq_lengths[start_idx:end_idx]
                         if grpo_batch.seq_lengths.numel() > 0 else grpo_batch.seq_lengths),
            old_per_token_logps=(grpo_batch.old_per_token_logps[start_idx:end_idx]
                                 if grpo_batch.old_per_token_logps is not None else None),
            ref_per_token_logps=(grpo_batch.ref_per_token_logps[start_idx:end_idx]
                                 if grpo_batch.ref_per_token_logps is not None else None),
            rollout_per_token_logps=(grpo_batch.rollout_per_token_logps[start_idx:end_idx]
                                     if grpo_batch.rollout_per_token_logps is not None else None),
            advantages=grpo_batch.advantages[start_idx:end_idx] if grpo_batch.advantages is not None else None,
            num_items_in_batch=grpo_batch.num_items_in_batch,
            logits_to_keep=grpo_batch.logits_to_keep,
        )

        if self.is_multimodal and origin_data is not None:
            chunk_origin_data = origin_data[start_idx:end_idx]
            template = self.template
            current_length = model_inputs['input_ids'].shape[1]
            with self._template_context(template):
                encoded_data = [template.encode(data.to_template_dict()) for data in chunk_origin_data]
                for ed in encoded_data:
                    ed.pop('_extra_kwargs', None)
                chunk_model_inputs.update(
                    to_device(template.data_collator(encoded_data, padding_to=current_length), self.model.device))
                chunk_model_inputs.pop('labels', None)

        return chunk_model_inputs, chunk_grpo_batch

    def _prepare_liger_loss(self):
        self.use_liger_loss = self.args.use_liger_kernel
        if self.use_liger_loss:
            from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                compiled=False,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
                importance_sampling_level=self.importance_sampling_level,
                sapo_temperature_pos=self.tau_pos,
                sapo_temperature_neg=self.tau_neg,
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
            'offpolicy_sequence_mask': 'enable' if self.off_policy_sequence_mask_delta is not None else 'disable',
            'rollout_importance_sampling': 'enable' if self.rollout_importance_sampling_mode is not None else 'disable',
            'loss_type': str(self.loss_type),
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

        # REAL, https://arxiv.org/abs/2602.05630
        self.real_tau = args.real_tau

        # FIPO, https://arxiv.org/abs/2603.19835
        self.fipo_gamma = 2**(-1 / args.fipo_decay_rate)
        self.fipo_clip_range = args.fipo_clip_range
        self.fipo_clip_high_only = args.fipo_clip_high_only
        self.fipo_safety_threshold = args.fipo_safety_threshold

        # RLOO,
        self.advantage_estimator = args.advantage_estimator
        self.kl_in_reward = args.kl_in_reward
        if self.scale_rewards == 'gdpo' and self.kl_in_reward:
            logger.warning('GDPO mode does not support kl_in_reward=True. Setting kl_in_reward=False.')
            self.kl_in_reward = False

        # Rollout Importance Sampling Correction
        self.rollout_importance_sampling_mode = args.rollout_importance_sampling_mode
        self.rollout_importance_sampling_threshold = args.rollout_importance_sampling_threshold
        self.log_rollout_offpolicy_metrics = args.log_rollout_offpolicy_metrics

        # Off-Policy Sequence Masking
        self.off_policy_sequence_mask_delta = args.off_policy_sequence_mask_delta

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
                    reward_funcs[i] = reward_func_class(args=args)
                elif not callable(reward_func):
                    raise ValueError(f'reward_function {reward_func} is not implemented in swift.rewards')

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
                    raise ValueError(f'rm_plugin {rm_plugin} is not implemented in swift.rewards')
                self.reward_model_plugins.append(rm_plugins[rm_plugin](model=rm, template=rm_template))
                self.reward_funcs.append(rm)
                self.reward_func_names.append(rm.config._name_or_path.split('/')[-1])

        # use_gym_env: gym total_reward is appended as an extra reward column so it can
        # blend with reward_funcs via reward_weights. When reward_funcs is empty, it becomes
        # the single reward source.
        if self.use_gym_env:
            self.reward_func_names.append('gym_reward')

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(self.reward_func_names):
                raise ValueError(f'Number of reward weights ({len(args.reward_weights)}) must match number of reward '
                                 f'functions ({len(self.reward_func_names)})')
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32).to(device)
        else:
            self.reward_weights = torch.ones(len(self.reward_func_names), dtype=torch.float32).to(device)

        # after init trainer
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                elif self.is_fsdp_enabled:
                    from .utils import prepare_fsdp
                    self.reward_funcs[i] = prepare_fsdp(reward_func, self.accelerator)
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
            per_token_logps: Log probs from current policy, shape [B, T]
            old_policy_per_token_logps: Log probs from old/rollout policy, shape [B, T].
                Uses rollout_per_token_logps if available, otherwise old_per_token_logps.
            completion_mask: Boolean mask for completion tokens, shape [B, T]
            advantages: Advantage values per sample, shape [B]

        Returns:
            Sequence mask, shape [B], True = keep sequence, False = mask out
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
        # All KL metrics estimate KL(π_rollout || π_training), which measures how much
        # the training policy deviates from the rollout policy. This is directly related
        # to the importance sampling ratio ρ = π_training / π_rollout.

        # log_ratio = log(π_training / π_rollout), used for IS weights and KL estimators
        log_ratio = per_token_logps - rollout_per_token_logps
        log_ratio *= completion_mask

        # 2a. kl: Direct estimator for KL(π_rollout || π_training)
        # Formula: KL(P||Q) = E_P[log(P/Q)] where P=π_rollout, Q=π_training
        # = E_rollout[log(π_rollout) - log(π_training)] = E[-log_ratio]
        kl = masked_mean(-log_ratio, completion_mask)
        metrics['kl'] = self.accelerator.gather_for_metrics(kl).nanmean().item()

        # 2b. k3_kl: K3 estimator for KL(π_rollout || π_training)
        # More stable for small KL values
        log_ratio_safe = torch.clamp(log_ratio, min=-20, max=20)
        k3_kl_matrix = torch.clamp(torch.exp(log_ratio_safe) - log_ratio_safe - 1, min=-10, max=10)
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

    def _get_eval_sampler(self, eval_dataset):
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations_eval,
            seed=self.args.seed,
        )
