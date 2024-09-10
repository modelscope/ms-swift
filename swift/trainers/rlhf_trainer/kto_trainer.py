# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch
from trl import KTOTrainer as HFKTOTrainer
from trl.trainer import kto_trainer

from swift.trainers import PushToMsHubMixin, RLHFTrainerMixin, SwiftMixin
from swift.utils import get_logger

logger = get_logger()

del HFKTOTrainer.__init__


class KTOTrainer(PushToMsHubMixin, SwiftMixin, HFKTOTrainer):

    def __init__(self, *args, test_oom_error=False, **kwargs):
        is_vision = kwargs.pop('is_vision')
        super().__init__(*args, **kwargs)

        self.model.config.model_type = self.model.config.model_type[:-1]  # remove suffix
        self.is_vision_model = is_vision


# fix kto when tokenizer do not have a bos_token_id
def new_process_tokens(example: Dict[str, Any], model=None, **kwargs) -> Dict:
    """Process tokens of a KTO specific dataset.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    in case the prompt + completion responses is/are too long. First
        we truncate the prompt; if we're still too long, we truncate the completion.

    We also create the labels for the completion responses, which are of length equal to
        the sum of the length of the prompt and the completion response, with
        label_pad_token_id  for the prompt tokens.
    """
    prompt = example['prompt']
    completion = example['completion']

    batch = {
        f"{kwargs['prefix']}prompt": prompt,
        f"{kwargs['prefix']}completion": completion,
        f"{kwargs['prefix']}label": example['label'],
    }

    if not kwargs['is_encoder_decoder']:
        # Check issues below for more details
        #  1. https://github.com/huggingface/trl/issues/907
        #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        #  3. https://github.com/LianjiaTech/BELLE/issues/337

        if not isinstance(prompt, str):
            raise ValueError(f'prompt should be an str but got {type(prompt)}')

        if not isinstance(completion, str):
            raise ValueError(f'completion should be an str but got {type(completion)}')

        # keys of format prompt_* refers to just the prompt and answer_* refers to just the answer
        all_tokens = {
            'prompt_input_ids': example['prompt_input_ids'],
            'prompt_attention_mask': example['prompt_attention_mask'],
            'answer_input_ids': example['answer_input_ids'],
            'answer_attention_mask': example['answer_attention_mask'],
        }

        # calculate max length by checking if BOS/EOS is already there
        max_length = kwargs['max_length']
        eos_token_id = kwargs['tokenizer'].eos_token_id
        if eos_token_id != all_tokens['answer_input_ids'][-1]:
            max_length -= 1

        # if combined sequence is too long (> max_length - 1 for BOS token - 1 for EOS), truncate the prompt
        if len(all_tokens['prompt_input_ids']) + len(all_tokens['answer_input_ids']) > max_length:
            for k in ['prompt_input_ids', 'prompt_attention_mask']:
                if kwargs['truncation_mode'] == 'keep_start':
                    all_tokens[k] = all_tokens[k][:kwargs['max_prompt_length']]
                elif kwargs['truncation_mode'] == 'keep_end':
                    all_tokens[k] = all_tokens[k][-kwargs['max_prompt_length']:]
                else:
                    raise ValueError(f"Unknown truncation mode: {kwargs['truncation_mode']}")

        # if that's still too long, truncate the response
        if len(all_tokens['prompt_input_ids']) + len(all_tokens['answer_input_ids']) > max_length:
            for k in ['answer_input_ids', 'answer_attention_mask']:
                all_tokens[k] = all_tokens[k][:max_length - kwargs['max_prompt_length']]

        # all input_ids and attention mask as is. We then check if we need to add BOS/EOS tokens
        batch[f"{kwargs['prefix']}prompt_input_ids"] = all_tokens['prompt_input_ids']
        batch[f"{kwargs['prefix']}prompt_attention_mask"] = all_tokens['prompt_attention_mask']
        batch[f"{kwargs['prefix']}completion_input_ids"] = (
            all_tokens['prompt_input_ids'] + all_tokens['answer_input_ids'])
        batch[f"{kwargs['prefix']}completion_attention_mask"] = (
            all_tokens['prompt_attention_mask'] + all_tokens['answer_attention_mask'])

        # add EOS, which affects only the full completion
        if len(all_tokens['answer_input_ids']) == 0 or eos_token_id != all_tokens['answer_input_ids'][-1]:
            batch[f"{kwargs['prefix']}completion_input_ids"] = batch[f"{kwargs['prefix']}completion_input_ids"] + [
                eos_token_id
            ]
            batch[f"{kwargs['prefix']}completion_attention_mask"] = batch[
                f"{kwargs['prefix']}completion_attention_mask"] + [1]

        batch[f"{kwargs['prefix']}completion_labels"] = batch[f"{kwargs['prefix']}completion_input_ids"][:]
        batch[f"{kwargs['prefix']}completion_labels"][:len(batch[f"{kwargs['prefix']}prompt_input_ids"])] = [
            kwargs['label_pad_token_id']
        ] * len(batch[f"{kwargs['prefix']}prompt_input_ids"])
    else:
        completion_tokens = kwargs['tokenizer'](
            completion, truncation=True, max_length=kwargs['max_completion_length'], add_special_tokens=True)
        prompt_tokens = kwargs['tokenizer'](
            prompt, truncation=True, max_length=kwargs['max_prompt_length'], add_special_tokens=True)

        batch[f"{kwargs['prefix']}prompt_input_ids"] = prompt_tokens['input_ids']
        batch[f"{kwargs['prefix']}prompt_attention_mask"] = prompt_tokens['attention_mask']

        batch[f"{kwargs['prefix']}completion_labels"] = completion_tokens['input_ids']
        batch[f"{kwargs['prefix']}completion_attention_mask"] = completion_tokens['attention_mask']
        if model is not None and hasattr(model, 'prepare_decoder_input_ids_from_labels'):
            batch[f"{kwargs['prefix']}completion_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                labels=torch.tensor(batch['completion_labels']))

    return batch


if not hasattr(kto_trainer, '_original_process_tokens'):
    kto_trainer._original_process_tokens = kto_trainer._process_tokens
    kto_trainer._process_tokens = new_process_tokens
