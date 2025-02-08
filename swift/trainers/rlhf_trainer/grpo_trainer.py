# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Optional, Union, Dict
from unittest.mock import patch
import os
import torch
import torch.nn as nn
from accelerate.utils import broadcast_object_list, gather, gather_object
from accelerate.utils.other import is_compiled_module
from transformers import GenerationConfig, PreTrainedModel
from trl import GRPOTrainer as HFGRPOTrainer
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import pad

from swift.llm import InferRequest, RequestConfig, to_device
from swift.utils import get_logger, is_vllm_available
import concurrent.futures
from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFGRPOTrainer.__init__
del HFGRPOTrainer._prepare_inputs

logger = get_logger()


class GRPOTrainer(RLHFTrainerMixin, SwiftMixin, HFGRPOTrainer):

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_funcs: Optional[Union[Callable, list[Callable]]] = None,
                 *_args,
                 **kwargs):

        args = kwargs['args']

        if reward_funcs is not None and not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs or []
        self.reward_templates = [None] * len(self.reward_funcs)
        if reward_model is not None:
            self.reward_templates.append(kwargs.pop('reward_template', None))
            self.reward_funcs.append(reward_model)
        if not self.reward_funcs:
            raise ValueError('You must specify reward_funcs or reward_model')

        self.num_generations = args.num_generations
        model.warnings_issued['estimate_tokens'] = True

        self._metrics = defaultdict(list)

        use_vllm = args.use_vllm

        super().__init__(model, ref_model, *_args, **kwargs)

        num_processes = self.accelerator.num_processes  # check: 梯度累加
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

        if use_vllm:
            if not is_vllm_available():
                raise ImportError('vLLM is not available and `use_vllm` is set to True. Please install vLLM with '
                                  '`pip install vllm` to use it.')
            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == 'auto':
                    vllm_device = f'cuda:{self.accelerator.num_processes}'  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(':')[0] == 'cuda' and int(vllm_device.split(':')[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f'The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM '
                        'without restricting the number of GPUs for training. Set the `--num_processes` argument to a '
                        'value lower than the number of GPUs available on your machine—typically, reducing it by one '
                        f'is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`.')
                # Check that the requested device is not also used for training
                if vllm_device in {f'cuda:{idx}' for idx in range(self.accelerator.num_processes)}:
                    logger.warning(
                        f'The requested device {vllm_device} is also used for training. This may lead to unexpected '
                        'behavior. It is recommended to use a dedicated device for vLLM.')
                from swift.llm import VllmEngine # , PtEngine
                world_size_patch = patch('torch.distributed.get_world_size', return_value=1)
                profiling_patch = patch(
                    'vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling', return_value=None)
                with world_size_patch, profiling_patch:
                    self.engine = VllmEngine(
                        model.model_dir,
                        device=vllm_device,
                        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                        enable_prefix_caching=True)
                self._last_loaded_step = 0
                self.accelerator.wait_for_everyone()
                self.engine.template = self.template
        else:
            from swift.llm import PtEngine
            self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=8)  # TODO: args.max_batch_size
        self.model_accepts_loss_kwargs = False
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    @contextmanager
    def set_padding_side_left(self):
        original_padding_side = self.template.padding_side
        self.template.padding_side = 'left'
        try:
            yield
        finally:
            self.template.padding_side = original_padding_side

    def _prepare_inputs(self, inputs) -> Dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                with unwrap_model_for_generation(
                        self.model, self.accelerator,
                        gather_deepspeed3_params=self.args.ds3_gather_for_generation) as unwrapped_model:
                    if is_compiled_module(unwrapped_model):
                        state_dict = unwrapped_model._orig_mod.state_dict()
                    else:
                        state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = self.engine.engine.engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights(state_dict.items())
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_messages = gather_object(messages)
            infer_requests = [InferRequest(message) for message in messages]
            if self.accelerator.is_main_process:
                outputs = self.engine.infer(
                    infer_requests, use_tqdm=False, return_outputs=True)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_messages)

            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(messages),
                (self.accelerator.process_index + 1) * len(messages),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            outputs = self.engine.infer(inputs, use_tqdm=False)
            for i, output in enumerate(outputs):
                inputs[i]['messages'].append({'role': 'assistant', 'content': output.choices[0].message.content})

        self.template.set_mode('train')
        max_workers = min(32, os.cpu_count(), len(inputs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.template.encode, infer_request)
                for infer_request in inputs
            ]
            concurrent.futures.wait(futures)
            batched_inputs = [future.result() for future in futures]
        inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)
        if self.model.model_meta.is_multimodal:
            _, inputs = self.template.pre_forward_hook(self.model, None, inputs)
        self.template.set_mode('pt')

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask,
                                                                logits_to_keep)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(self.model, prompt_completion_ids, attention_mask,
                                                                    logits_to_keep)

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        rewards_per_func = torch.zeros((len(messages), len(self.reward_funcs)), device=device)
        for i, (reward_func, reward_template) in enumerate(zip(self.reward_funcs, self.reward_templates)):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_inputs = []
                for completion in completions:
                    combined_message = messages + [{'role': 'assistant', 'content': completion}]
                    reward_input = StdTemplateInputs.from_dict({'messages': combined_message})
                    reward_template._preprocess_inputs(reward_input)
                    reward_input = self.reward_template._encode(reward_input)
                    reward_inputs.append(reward_input)
                reward_inputs = self.reward_template.data_collator(reward_inputs)
                reward_inputs.pop('labels', None)
                reward_inputs.pop('loss_scale', None)
                reward_inputs = super(type(self), self)._prepare_inputs(reward_inputs)

                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # Repeat all input columns (but "messages" and "completion") to match the number of generations
                reward_kwargs = {key: [example[key] for example in inputs] for key in inputs[0]}
                output_reward_func = reward_func(completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(messages),
            (self.accelerator.process_index + 1) * len(messages),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split('/')[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f'rewards/{reward_func_name}'].append(reward_per_func[i].item())

        self._metrics['reward'].append(rewards.mean().item())
        self._metrics['reward_std'].append(std_grouped_rewards.mean().item())

        return {
            'prompt_ids': prompt_ids,
            'prompt_mask': prompt_mask,
            'completion_ids': completion_ids,
            'completion_mask': completion_mask,
            'ref_per_token_logps': ref_per_token_logps,
            'advantages': advantages,
        }
