# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from transformers import GenerationConfig, PreTrainedModel
from trl import GRPOTrainer as HFGRPOTrainer
from trl.models import unwrap_model_for_generation

from swift.llm.template.template_inputs import StdTemplateInputs
from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFGRPOTrainer.__init__
del HFGRPOTrainer._prepare_inputs


class GRPOTrainer(RLHFTrainerMixin, SwiftMixin, HFGRPOTrainer):

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_funcs: Optional[Union[Callable, list[Callable]]] = None,
                 *_args,
                 **kwargs):

        args = kwargs['args']

        self.processing_class = kwargs.get('template').tokenizer

        if reward_funcs is not None and not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs or []
        self.reward_templates = [None] * len(self.reward_funcs)
        if reward_model is not None:
            self.reward_templates.append(kwargs.pop('reward_template', None))
            self.reward_funcs.append(reward_model)
        if not self.reward_funcs:
            raise ValueError('You must specify reward_funcs or reward_model')

        self.max_prompt_length = args.max_prompt_length
        self.num_generations = args.num_generations

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=self.processing_class.pad_token_id,
        )
        model.warnings_issued['estimate_tokens'] = True

        self._metrics = defaultdict(list)
        self.use_vllm = False  # just debug

        super().__init__(model, ref_model, *_args, **kwargs)

    def _prepare_inputs(self, inputs) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        inputs = StdTemplateInputs.from_dict(inputs[0])
        self.template._preprocess_inputs(inputs)
        if inputs.messages[-1]['role'] == 'assistant':
            inputs.messages[-1]['content'] = None  # remove response
        prompt_inputs = self.template._encode(inputs)
        if inputs.messages[-1]['role'] == 'assistant':
            inputs.messages.pop(-1)
        if 'attention_mask' not in prompt_inputs:
            prompt_inputs['attention_mask'] = attention_mask = [1] * len(prompt_inputs['input_ids'])
        self.template.mode = 'train'
        prompt_inputs = self.template.data_collator([prompt_inputs])
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs['input_ids'], prompt_inputs['attention_mask']

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            raise NotImplementedError
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config)

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0), ), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

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

        rewards_per_func = torch.zeros((self.num_generations, len(self.reward_funcs)), device=device)
        for i, (reward_func, reward_template) in enumerate(zip(self.reward_funcs, self.reward_templates)):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_inputs = []
                for completion in completions:
                    combined_message = inputs.messages + [{'role': 'assistant', 'content': completion}]
                    reward_input = StdTemplateInputs.from_dict({'messages': combined_message})
                    reward_template._preprocess_inputs(reward_input)
                    reward_input = self.reward_template._encode(reward_input)
                    reward_inputs.append(reward_input)
                reward_inputs = self.reward_template.data_collator(reward_inputs)
                reward_inputs.pop('labels', None)
                if 'attention_mask' not in reward_inputs:
                    reward_inputs['attention_mask'] = torch.ones_like(reward_inputs['input_ids'])
                reward_inputs = super(type(self), self)._prepare_inputs(reward_inputs)

                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "messages" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ['messages', 'completion']}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(messages=inputs.messages, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Log the metrics
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split('/')[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f'rewards/{reward_func_name}'].append(reward_per_func[i].item())

        self._metrics['reward'].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics['reward_std'].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return {
            'prompt_ids': prompt_ids,
            'prompt_mask': prompt_mask,
            'completion_ids': completion_ids,
            'completion_mask': completion_mask,
            'ref_per_token_logps': ref_per_token_logps,
            'advantages': advantages,
        }
