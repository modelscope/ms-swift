# Copyright (c) Alibaba, Inc. and its affiliates.
import gc
import inspect
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from copy import copy
from functools import partial
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from megatron.core import mpu
from megatron.core.inference.communication_utils import recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training import get_args, get_model, get_timers, training
from megatron.training.checkpointing import load_checkpoint
from megatron.training.training import cuda_graph_capture, cuda_graph_set_manual_hooks
from megatron.training.utils import (logical_and_across_model_parallel_group,
                                     reduce_max_stat_across_model_parallel_group, unwrap_model)
from torch.distributed.nn import all_reduce
from vllm.distributed import parallel_state as vllm_ps

from swift.llm import RequestConfig, RowPreprocessor, Template, to_device
from swift.llm.infer.protocol import RolloutOutput
from swift.plugin import orms
from swift.trainers.rlhf_trainer import GRPOTrainer, VLLMClient
from swift.trainers.rlhf_trainer.grpo_trainer import DataType
from swift.trainers.rlhf_trainer.utils import replace_assistant_response_with_ids
from swift.utils import get_current_device, get_logger, is_vllm_available, remove_response
from ..argument import MegatronRLHFArguments
from .rlhf_base import MegatronRLHFTrainer
from .trainer import MegatronTrainer
from .utils import get_batch, profiling_context

try:
    from mbridge import AutoBridge
except ImportError:
    pass

try:
    from megatron.post_training.algos.distillation import (
        get_tensor_shapes_adjust_fn_for_distillation, )

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

logger = get_logger()
"""
TODO:
    1. compute rewards
    2. compute advantages
    3. compute ref/old logps
    4. loss


FUTURE TODO:
    1. server mode
    2. offload model/optimizer
    3. DAPO (dynamic sampling)
    4. entropy mask
    5. reward model
    6. multi turn
"""


class MegatronGRPOTrainer(MegatronRLHFTrainer):

    def __init__(self, args: MegatronRLHFArguments, template: Template):
        super().__init__(args, template)
        self.args = args
        self.hf_model_dir = args.model_info.model_dir
        self.processing_class = self.template.processor
        # TODO: multi turn scheduler(colocate multi turn)
        self._init_grpo_params()
        self._prepare_rewards()
        self._prepare_rollout_engine()

        # debug: use mbridge to convert mcore to hf
        self.bridge = None

    def loss_func(self, output_tensor: torch.Tensor, *, ref_logps: torch.Tensor, labels: torch.Tensor,
                  packed_seq_params):
        # TODO：GRPO policy loss
        args: MegatronRLHFArguments = get_args()
        num_samples = packed_seq_params.num_samples

        logps = self.get_logps(output_tensor, labels, packed_seq_params)
        loss, chosen_rewards, rejected_rewards = self.dummy_dpo_trainer.dpo_loss(
            logps[:num_samples],
            logps[num_samples:],
            ref_logps[:num_samples],
            ref_logps[num_samples:],
        )
        if args.rpo_alpha:
            loss_mask = labels != -100
            num_tokens = packed_seq_params.cu_seqlens_q[num_samples] // args.context_parallel_size
            loss_mask[:, num_tokens:] = 0
            nll_loss = torch.concat([torch.sum(output_tensor * loss_mask)[None], loss_mask.sum()[None]])
            if args.context_parallel_size > 1:
                nll_loss = all_reduce(nll_loss, group=mpu.get_context_parallel_group())
            nll_loss = nll_loss[0] / nll_loss[1]
            loss = loss + args.rpo_alpha * nll_loss
        loss = loss.mean()
        metric = {
            'loss': loss.clone().detach(),
            'logps/chosen': logps[:num_samples].mean(),
            'logps/rejected': logps[num_samples:].mean(),
            'rewards/chosen': chosen_rewards.mean(),
            'rewards/rejected': rejected_rewards.mean(),
            'rewards/accuracies': (chosen_rewards > rejected_rewards).float().mean(),
            'rewards/margins': (chosen_rewards - rejected_rewards).mean(),
        }
        if args.rpo_alpha:
            metric['nll_loss'] = nll_loss.detach()
        reporting_metric = loss.new_tensor(list(metric.values()))
        torch.distributed.all_reduce(
            reporting_metric, torch.distributed.ReduceOp.AVG, group=mpu.get_data_parallel_group())
        reporting_metric = {k: reporting_metric[i] for i, k in enumerate(metric.keys())}
        # fix megatron-lm bug
        # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.12.0/megatron/core/pipeline_parallel/schedules.py#L291
        loss = loss / mpu.get_context_parallel_world_size()
        return loss, reporting_metric

    def _replace_data_iterator(self, data_iterator):
        # rollout the train_batch_size, split to mini-batches
        batch = next(data_iterator)  # [global_batch_size, ]

        batch = self._generate_and_score_completions(batch)

        args: MegatronRLHFArguments = get_args()
        num_iters_per_step = args.global_batch_size // (args.micro_batch_size * mpu.get_data_parallel_world_size())
        res = []
        with torch.no_grad(), self.null_ref_context() as ref_model:
            for i in range(num_iters_per_step):
                res.append(self.ref_forward(ref_model, data_iterator))
        return iter(res)

    def _init_grpo_params(self):
        args = self.args
        # distributed params
        self.world_size = torch.distributed.get_world_size()
        self.process_index = torch.distributed.get_rank()
        self.is_main_process = self.process_index == 0
        self.device = get_current_device()
        # algorithm params
        self.num_generations = args.num_generations  # G in the GRPO paper
        self.temperature = args.temperature
        self.loss_type = args.loss_type
        self.max_completion_length = args.max_completion_length
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.top_entropy_quantile = args.top_entropy_quantile
        self.importance_sampling_level = args.importance_sampling_level
        self.enable_offload = False
        # batch size
        self.global_batch_size = args.global_batch_size
        self.mini_batch_size = args.mini_batch_size
        self.micro_batch_size = args.micro_batch_size
        self.per_device_rollout_batch_size = self.global_batch_size // self.world_size * self.num_generations
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
        self.enable_server_multi_turn = False
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

    def prepare_vllm(self):
        from swift.llm.infer.infer_engine import GRPOVllmEngine
        args = self.args
        max_num_seqs = self.per_device_rollout_batch_size * self.vllm_tensor_parallel_size

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
            template=self.template,
            distributed_executor_backend='external_launcher',
        )
        if self.vllm_tensor_parallel_size > 1:
            self.vllm_tp_group = vllm_ps.get_tp_group().device_group
        return engine

    def _prepare_rewards(self):
        # TODO: reward model
        args = self.args
        reward_funcs = args.reward_funcs
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

        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(f'Number of reward weights ({len(args.reward_weights)}) must match number of reward '
                                 f'functions ({len(reward_funcs)})')
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32).to(self.device)
        else:
            self.reward_weights = torch.ones(len(self.reward_func_names), dtype=torch.float32).to(self.device)

    def _move_model_to_vllm(self):
        # TODO: LoRA, server
        if self.bridge is None:
            self.bridge = AutoBridge.from_pretrained(self.hf_model_dir)
        per_tensor_params = self.bridge.export_weights([self.unwrapped_model])
        self.engine.inner_model.load_weights(per_tensor_params)  # TODO: check tensor_model_parallel

    def forward_step(self, data_iterator, model):
        # train_batch_size
        # return: output_tensor, loss_func

        data = next(data_iterator)
        ref_logps = data.pop('logps')
        with self.stimer:
            output_tensor = model(**data)
        return output_tensor, partial(
            self.loss_func,
            ref_logps=ref_logps,
            labels=data.get('labels'),
            packed_seq_params=data.get('packed_seq_params'))

    def _patch_megatron(self):
        super()._patch_megatron()
        self._origin_train_step = self.train_step

    def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
        # borrowed from Megatron-LM 0.13
        # get train_batch_size Rollout / ref/old logps / reward / advantage
        # split to mini_batches (iter mini_batch)
        data_iterator = self._replace_data_iterator(data_iterator)

        args: MegatronRLHFArguments = get_args()
        timers = get_timers()
        batch = next(data_iterator)
        batch = self._generate_and_score_completions(batch)

        # split to mini-batches

        # CUDA Graph capturing only executes once, when it's the first training iteration.
        if args.curr_iteration == args.iteration and args.external_cuda_graph:
            cuda_graph_capture(model, config, args)

            # Set grad to zero.
            for model_chunk in model:
                model_chunk.zero_grad_buffer()
            optimizer.zero_grad()

            # Collect garbage and empty unused memory.
            gc.collect()
            torch.cuda.empty_cache()

        rerun_state_machine = get_rerun_state_machine()
        while rerun_state_machine.should_run_forward_backward(data_iterator):
            # Set grad to zero.
            for model_chunk in model:
                model_chunk.zero_grad_buffer()
            optimizer.zero_grad()

            if has_nvidia_modelopt:
                # [ModelOpt]: Pipeline-parallel Distillation stacks student and teacher tensors
                adjust_tensor_shapes_fn = get_tensor_shapes_adjust_fn_for_distillation(
                    model, args.seq_length, args.micro_batch_size, args.decoder_seq_length)
            else:
                adjust_tensor_shapes_fn = None

            # Forward pass.
            forward_backward_func = get_forward_backward_func()
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=False,
                adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
            )
        should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
        if should_exit:
            return {}, True, should_checkpoint, should_exit, exit_code, None, None

        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Vision gradients.
        if args.vision_pretraining and args.vision_pretraining_type == 'dino':
            unwrapped_model = unwrap_model(model[0])
            unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

        # Update parameters.

        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        timers('optimizer').stop()

        # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
        # so we must gather across mp ranks
        update_successful = logical_and_across_model_parallel_group(update_successful)
        # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
        # so we must gather across mp ranks
        grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
        if args.log_num_zeros_in_grad:
            num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)

        # Vision momentum.
        if args.vision_pretraining and args.vision_pretraining_type == 'dino':
            unwrapped_model = unwrap_model(model[0])
            unwrapped_model.update_momentum(args.curr_iteration)

        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
            opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        # Set the manual hooks when CUDA Graphs are enabled.
        if args.curr_iteration == args.iteration and args.external_cuda_graph:
            if args.use_distributed_optimizer and args.overlap_param_gather:
                cuda_graph_set_manual_hooks(model)

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}

            for key in losses_reduced[0].keys():
                val = [x[key].view(-1) for x in losses_reduced]
                if val[0].numel() == 2:
                    if args.sft:
                        # in mcore the normalization happens on micro batch instead of global
                        val = torch.vstack(val)
                        val = val[:, 0] / val[:, 1]
                        val = val.mean()
                        torch.distributed.all_reduce(val, group=mpu.get_data_parallel_group(with_context_parallel=True))
                        val /= torch.distributed.get_world_size(
                            group=mpu.get_data_parallel_group(with_context_parallel=True))
                        loss_reduced[key] = val
                    else:
                        # there is one dict per microbatch. in new reporting, we average
                        # over the total number of tokens across the global batch.
                        val = torch.vstack(val).sum(dim=0)
                        torch.distributed.all_reduce(val, group=mpu.get_data_parallel_group(with_context_parallel=True))
                        loss_reduced[key] = val[0] / val[1]
                elif val[0].numel() == 1:
                    # legacy behavior, we average over the number of microbatches
                    val = torch.cat(val).mean()
                    loss_reduced[key] = val
                else:
                    raise ValueError(f'Invalid value shape: {val[0].shape} for key {key}')
            return (
                loss_reduced,
                skipped_iter,
                should_checkpoint,
                should_exit,
                exit_code,
                grad_norm,
                num_zeros_in_grad,
            )
        return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad

    def _generate_and_score_completions(self, batch):

        def get_local_rollout_data(batch):
            # repeat num_generations times
            global_rollout_data = [item for item in batch for _ in range(self.num_generations)]
            # get local rollout data
            data_slice = slice(self.process_index * self.per_device_rollout_batch_size,
                               (self.process_index + 1) * self.per_device_rollout_batch_size)
            rollout_data = global_rollout_data[data_slice]
            return rollout_data

        rollout_data = get_local_rollout_data(batch)

        batch = self._generate_completions(rollout_data)

        rewards_per_func = self._score_completions(batch)

        batch['advantages'] = self._compute_advantages(batch, rewards_per_func)

        batch = self._maybe_compute_logps(batch)

        return batch

    def _generate_completions(self, batch):
        # TODO: server mode
        assert self.vllm_mode == 'colocate'
        # Step 1: Wake up the engine if it's sleeping (vLLM colocate mode)
        if self.engine.inner_model_executor.is_sleeping:
            wake_up_params = inspect.signature(self.engine.engine.wake_up).parameters
            # Load weights only (faster and reduces memory peak)
            kwargs = {'tags': ['weights']} if 'tags' in wake_up_params else {}
            self.engine.engine.wake_up(**kwargs)

        # Step 2: Load model weights
        self._move_model_to_vllm()

        if (self.engine.inner_model_executor.is_sleeping
                and 'tags' in inspect.signature(self.engine.engine.wake_up).parameters):
            self.engine.engine.wake_up(tags=['kv_cache'])

        batch = self.preprocess_rollout_data(batch)
        output: List[RolloutOutput] = self._rollout(batch)
        batch = self.postprocess_rollout_data(batch, output)
        return batch

    def preprocess_rollout_data(self, batch):
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

    def postprocess_rollout_data(self, batch, output):
        if self.vllm_tensor_parallel_size > 1:
            local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
            orig_size = len(output) // self.vllm_tensor_parallel_size
            tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
            output = output[tp_slice]

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

            return input_data

        assert len(batch) == len(output)
        return [merge_output_input_data(input_data, output) for input_data, output in zip(batch, output)]

    def _get_request_config(self) -> RequestConfig:
        request_config = copy(self.request_config)
        if self.args.vllm_mode == 'colocate' and self.vllm_tensor_parallel_size > 1:
            # Set request_config.seed
            # 1. Ensure that the seed for vLLM Engines within each TP (Tensor Parallelism) group is the same;
            #   otherwise, the program may hang.
            # 2. Ensure that the seed for vLLM Engines across different TP groups is different;
            #   otherwise, identical completions will be generated.
            batch_size = self.per_device_rollout_batch_size
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
        device = self.accelerator.device
        rewards_per_func = torch.zeros((len(batch), len(self.reward_funcs)), device=device)
        completions = [inp['messages'][-1]['content'] for inp in batch]
        for i, (reward_func, reward_model_plugin, reward_func_name) in enumerate(
                zip(self.reward_funcs, self.reward_model_plugins, self.reward_func_names)):
            with profiling_context(self, reward_func_name):
                # reward model
                reward_kwargs = {}  # TODO: step info
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

        slice_start = self.process_index * len(batch)
        slice_end = slice_start + len(batch)
        advantages = advantages[slice_start:slice_end]

        return advantages

    def _maybe_compute_logps(self, batch: DataType) -> DataType:
        # encode first
        template = self.template
        # get model forward kwargs from batch
        batch = self._maybe_replace_response_token(batch)
        with self._disable_maxlength_template_context(template):
            batch_encoded_inputs = [template.encode(data, return_length=True) for data in batch]
            batch_encoded_inputs = to_device(template.data_collator(batch_encoded_inputs), self.device)
            labels = batch_encoded_inputs.pop('labels')
            logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
            batch_encoded_inputs['logits_to_keep'] = logits_to_keep

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
