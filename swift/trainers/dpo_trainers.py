from typing import Dict, List, Optional, Union

from torch import nn
from transformers import PreTrainedModel
from trl import DPOTrainer as HFDPOTrainer

from swift.llm.utils.template import Context, Template
from swift.llm.utils.utils import sort_by_max_length
from swift.trainers.mixin import PushToMsHubMixin, SwiftMixin
from swift.utils import get_logger

logger = get_logger()


class DPOTrainer(PushToMsHubMixin, SwiftMixin, HFDPOTrainer):

    def __init__(self,
                 *args,
                 template: Template,
                 test_oom_error=False,
                 **kwargs):
        self.template = template
        super().__init__(*args, **kwargs)
        self.stat_dataset(self.train_dataset)
        self.stat_dataset(self.eval_dataset)
        if test_oom_error:
            self.train_dataset = sort_by_max_length(self.train_dataset, 20000)

    def concat_template(self, feature):
        query: Optional[str] = feature.get('query', None)
        system: Optional[str] = feature.get('system', None)
        if system is None:
            if self.template.use_default_system:
                system = self.template.default_system
        else:
            assert self.template.prefix_has_system is not None, 'not support `system`'
        res_context_list: List[Context] = []
        compute_loss_idx: List[int] = []
        if system is None:
            assert self.template.prefix != self.template.prefix_has_system, f'template.prefix: {self.template.prefix}'
            prefix = self.template.prefix
        else:
            prefix = self.template.prefix_has_system
        self.template._concat_context_list(
            prefix, res_context_list, compute_loss_idx, system=system)
        self.template._concat_context_list(
            self.template.prompt,
            res_context_list,
            compute_loss_idx,
            query=query,
            round0=True)
        res_context_list, compute_loss_idx = self.template._simplify_context_list(
            res_context_list, compute_loss_idx)

        return res_context_list, feature['response'], feature[
            'rejected_response']

    def build_tokenized_answer(self, prompt, answer):
        input_ids, labels, kwargs = self.template._encode_context_list(
            prompt, [])
        tgt_input_ids = self.template._encode_context_list([answer], [])[0]
        tgt_input_ids += self.template._encode_context_list(
            self.template.suffix, [])[0]
        return dict(
            prompt_input_ids=input_ids,
            prompt_attention_mask=[1] * len(input_ids),
            input_ids=tgt_input_ids,
            attention_mask=[1] * len(tgt_input_ids),
        )

    def tokenize_row(self,
                     feature,
                     model: Union[PreTrainedModel, nn.Module] = None) -> Dict:
        batch = {}
        if not self.is_encoder_decoder:
            prompt, chosen, rejected = self.concat_template(feature)

            prompt_tokens, _, _ = self.template._encode_context_list(
                prompt, [])
            prompt_tokens = {
                'input_ids': prompt_tokens,
                'attention_mask': [1] * len(prompt_tokens),
            }
            prompt_tokens = {
                f'prompt_{k}': v
                for k, v in prompt_tokens.items()
            }

            if not isinstance(chosen, str):
                raise ValueError(
                    f'chosen should be an str but got {type(chosen)}')
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(
                    f'rejected should be an str but got {type(rejected)}')
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            longer_response_length = max(
                len(chosen_tokens['input_ids']),
                len(rejected_tokens['input_ids']))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [
                    chosen_tokens, rejected_tokens, prompt_tokens
            ]:
                if len(answer_tokens['prompt_input_ids']
                       ) + longer_response_length > self.max_length:
                    if self.truncation_mode == 'keep_start':
                        for k in ['prompt_input_ids', 'prompt_attention_mask']:
                            answer_tokens[k] = answer_tokens[
                                k][:self.max_prompt_length]
                    elif self.truncation_mode == 'keep_end':
                        for k in ['prompt_input_ids', 'prompt_attention_mask']:
                            answer_tokens[k] = answer_tokens[k][
                                -self.max_prompt_length:]
                    else:
                        raise ValueError(
                            f'Unknown truncation mode: {self.truncation_mode}')

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens['prompt_input_ids']
                       ) + longer_response_length > self.max_length:
                    for k in ['input_ids', 'attention_mask']:
                        answer_tokens[k] = answer_tokens[k][:self.max_length
                                                            - self.
                                                            max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f'prompt_{k}'] + chosen_tokens[k]
                for k in ['input_ids', 'attention_mask']
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f'prompt_{k}'] + rejected_tokens[k]
                for k in ['input_ids', 'attention_mask']
            }
            chosen_sequence_tokens['labels'] = chosen_sequence_tokens[
                'input_ids'][:]
            _paddings = [self.label_pad_token_id] * len(
                chosen_tokens['prompt_input_ids'])
            chosen_sequence_tokens[
                'labels'][:len(chosen_tokens['prompt_input_ids'])] = _paddings
            rejected_sequence_tokens['labels'] = rejected_sequence_tokens[
                'input_ids'][:]
            _paddings = [self.label_pad_token_id] * len(
                rejected_tokens['prompt_input_ids'])
            rejected_sequence_tokens['labels'][:len(
                rejected_tokens['prompt_input_ids'])] = _paddings

            for k, toks in {
                    'chosen_': chosen_sequence_tokens,
                    'rejected_': rejected_sequence_tokens,
                    '': prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == 'token_type_ids':
                        continue
                    batch[f'{k}{type_key}'] = tokens

        else:
            # encoder-decoder
            batch = super().tokenize_row(feature, model)

        return batch

    @staticmethod
    def stat_dataset(llm_dataset) -> None:
        _token_len = []
        from datasets import Dataset as HfDataset
        from swift.utils.np_utils import stat_array
        if isinstance(llm_dataset, HfDataset):
            chosen = llm_dataset['chosen_input_ids']
            rejected = llm_dataset['rejected_input_ids']
            for cc, rr in zip(chosen, rejected):
                _token_len.append(max(len(cc), len(rr)))
        else:
            for d in llm_dataset:
                _token_len.append(
                    max(
                        len(d['chosen_input_ids']),
                        len(d['rejected_input_ids'])))
        _, stat_str = stat_array(_token_len)
        logger.info(f'Dataset Token Length: {stat_str}')
