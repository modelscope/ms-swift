# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional, Union

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoTokenizer, GenerationConfig, PreTrainedModel
from trl import GRPOTrainer as HFGRPOTrainer
from trl.models import unwrap_model_for_generation

from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFGRPOTrainer.__init__


class GRPOTrainer(RLHFTrainerMixin, SwiftMixin, HFGRPOTrainer):

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 *_args,
                 **kwargs):

        args = kwargs['args']

        processing_class = kwargs.get('template').tokenizer
        # kwargs.get('tokenizer', None) or kwargs.get('processing_class', None)
        # reward_processing_class = kwargs.get('reward_processing_class', None)
        # if reward_processing_class is None:
        #     reward_processing_class = AutoTokenizer.from_pretrained(reward_model.config._name_or_path)

        # if reward_processing_class.pad_token_id is None:
        #     reward_processing_class.pad_token = reward_processing_class.eos_token
        # self.reward_processing_class = reward_processing_class  # TODO

        self.reward_model = reward_model
        # self.reward_model.config.pad_token_id = reward_processing_class.pad_token_id

        # kwargs['']
        self.max_prompt_length = args.max_prompt_length
        self.num_generations = args.num_generations

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.pad_token_id,
        )
        model.warnings_issued['estimate_tokens'] = True

        self._metrics = {'kl': [], 'reward': [], 'reward_std': []}

        super().__init__(model, ref_model, *_args, **kwargs)

        # self.model_template = kwargs['template']
        # self.reward_model_template = kwargs['reward_model_template']

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError('The GRPOTrainer does not support returning outputs')

        if self.max_prompt_length is not None:
            inputs['input_ids'] = inputs['input_ids'][:, -self.max_prompt_length:]
            inputs['attention_mask'] = inputs['attention_mask'][:, -self.max_prompt_length:]
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**inputs, generation_config=self.generation_config)
        prompt_length = inputs['input_ids'].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids):
            logits = model(input_ids).logits  # (B, L, V)
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0), ), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        reward_inputs = {}
        reward_inputs['input_ids'] = [p + c for p, c in zip(inputs['input_ids'], completion_ids)]
        reward_inputs['attention_mask'] = [
            torch.cat([p, torch.ones_like(c)]) for p, c in zip(inputs['attention_mask'], completion_ids)
        ]
        # Decode the generated completions
        # completions = self.template.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        # Compute the rewards
        # prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        # if is_conversational(inputs[0]):
        #     completions = [[{'role': 'assistant', 'content': completion}] for completion in completions]
        #     messages = [{'messages': p + c} for p, c in zip(prompts, completions)]
        #     texts = [apply_chat_template(x, self.reward_processing_class)['text'] for x in messages]
        #     reward_inputs = self.reward_processing_class(
        #         texts, return_tensors='pt', padding=True, padding_side='right', add_special_tokens=False)
        # else:
        #     texts = [p + c for p, c in zip(prompts, completions)]
        #     reward_inputs = self.reward_processing_class(
        #         texts, return_tensors='pt', padding=True, padding_side='right', add_special_tokens=False)
        reward_inputs = super()._prepare_inputs(reward_inputs)
        with torch.inference_mode():
            rewards = self.reward_model(**reward_inputs).logits[:, 0]  # Shape (B*G,)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        advantages = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(advantages - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        self._metrics['reward'].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics['reward_std'].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics['kl'].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss
