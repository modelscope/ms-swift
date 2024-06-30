from typing import Any, Dict, List, Optional

import torch
from transformers import trainer
from trl import KTOTrainer as HFKTOTrainer
from trl.trainer import kto_trainer

from swift.llm.utils.template import Context, History, Template
from swift.llm.utils.utils import sort_by_max_length
from swift.utils import get_logger
from .callback import DefaultFlowCallbackNew, PrinterCallbackNew, ProgressCallbackNew
from .mixin import PushToMsHubMixin, SwiftMixin

logger = get_logger()


def encode_batch(batch: Dict[str, List[Any]], template: Template):
    """
    Encode a batch from KTO specific dataset with given template

    Args:
    batch: A dictionary containing:
        - prompt: The main prompt string
        - completion: The completion string
        - label: The label data
        - history (optional): A list of historical queries/responses
        - system (optional): A system string to use

    template: swift Template object

    Returns:
    A dictionary with encoded prompt, completion, and label.
    """

    query: Optional[str] = batch.get('query', None)
    history: Optional[History] = batch.get('history', None)
    system: Optional[str] = batch.get('system', None)
    if history is None:
        history = []
    if system is None:
        if template.use_default_system:
            system = template.default_system
    else:
        assert template.system_prefix is not None, 'not support `system`'

    res_context_list: List[Context] = []
    compute_loss_idx: List[float] = []

    if system is None:
        assert template.prefix != template.system_prefix, f'template.prefix: {template.prefix}'
        prefix = template.prefix
    else:
        prefix = template.system_prefix

    template._concat_context_list(prefix, res_context_list, compute_loss_idx, system=system)

    for i, (q, r) in enumerate(history):
        template._concat_context_list([*template.prompt, '{{RESPONSE}}', *template.chat_sep],
                                      res_context_list,
                                      compute_loss_idx,
                                      query=q,
                                      response=r,
                                      round0=i)
    template._concat_context_list(template.prompt, res_context_list, compute_loss_idx, query=query, round0=len(history))
    res_context_list, compute_loss_idx = template._simplify_context_list(res_context_list, compute_loss_idx)
    prompt = ''.join(res_context_list)

    return {'prompt': prompt, 'completion': batch['response'], 'label': batch['label']}


class KTOTrainer(PushToMsHubMixin, SwiftMixin, HFKTOTrainer):

    def __init__(self, *args, template: Template, test_oom_error=False, **kwargs):
        eval_dataset = kwargs.get('eval_dataset', None)
        kwargs['train_dataset'] = kwargs['train_dataset'].map(
            encode_batch,
            fn_kwargs={'template': template},
            desc='Encode dataset with template',
        )
        if eval_dataset is not None:
            kwargs['eval_dataset'] = eval_dataset.map(
                encode_batch,
                fn_kwargs={'template': template},
                desc='Encode dataset with template',
            )
        super().__init__(*args, **kwargs)
        train_ds_info = self.stat_dataset(self.train_dataset)
        val_ds_info = self.stat_dataset(self.eval_dataset)
        self.dataset_info = {'train_dataset': train_ds_info, 'val_dataset': val_ds_info}
        if test_oom_error:
            self.train_dataset = sort_by_max_length(self.train_dataset, 20000)
        # performance
        self.perf: Dict[str, Any] = {
            'gen_time': 0.,
            'gen_len': 0,
            'memory': {},
            'model': self.model.get_trainable_parameters() if hasattr(self.model, 'get_trainable_parameters') else None,
        }

    def train(self, *args, **kwargs) -> torch.Tensor:
        res = super().train(*args, **kwargs)
        for i in range(torch.cuda.device_count()):
            self.perf['memory'][f'cuda:{i}'] = f'{torch.cuda.max_memory_reserved(i)/1024/1024/1024:.2f}GiB'
        return res

    @staticmethod
    def stat_dataset(llm_dataset) -> Any:
        _token_len = []
        from datasets import Dataset as HfDataset
        from swift.utils.np_utils import stat_array
        if isinstance(llm_dataset, HfDataset):
            prompt_input_ids = llm_dataset['prompt_input_ids']
            answer_input_ids = llm_dataset['answer_input_ids']
            for pi, ai in zip(prompt_input_ids, answer_input_ids):
                _token_len.append(len(pi) + len(ai))
        _, stat_str = stat_array(_token_len)
        logger.info(f'Dataset Token Length: {stat_str}')
        return stat_str


# monkey patching
trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]
trainer.PrinterCallback = PrinterCallbackNew


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
