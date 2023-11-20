# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from transformers import PreTrainedTokenizerBase

DEFAULT_SYSTEM = 'you are a helpful assistant!'
History = List[Union[Tuple[str, str], List[str]]]


class TemplateType:
    # text-generation
    default_generation = 'default-generation'
    default_generation_bos = 'default-generation-bos'
    chatglm_generation = 'chatglm-generation'
    # chat
    default = 'default'
    chatml = 'chatml'
    qwen = chatml
    baichuan = 'baichuan'
    chatglm2 = 'chatglm2'
    chatglm3 = 'chatglm3'
    llama = 'llama'
    openbuddy = 'openbuddy'
    internlm = 'internlm'
    xverse = 'xverse'
    ziya = 'ziya'
    skywork = 'skywork'
    bluelm = 'bluelm'


Prompt = List[Union[str, List[Union[str, int]]]]


class Template:

    def __init__(self, prefix: Prompt, prompt: Prompt,
                 chat_sep: Optional[Prompt], suffix: Prompt) -> None:
        self.prefix = prefix
        self._has_system = False
        for p in prefix:
            if isinstance(p, str) and '{{SYSTEM}}' in p:
                self._has_system = True
        self.prompt = prompt
        self.chat_sep = chat_sep
        self._support_multi_round = self.chat_sep is not None
        self.suffix = suffix
        self._is_init = False

    def init_template(
        self,
        tokenizer: PreTrainedTokenizerBase,
        system: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation_strategy: Literal['ignore',
                                     'truncation_left'] = 'truncation_left'
    ) -> None:
        self._is_init = True
        self.tokenizer = tokenizer
        self.system = system
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

    def encode(self, example: Dict[str,
                                   Any]) -> Dict[str, Optional[List[int]]]:
        if not self._is_init:
            raise ValueError(
                'Template has not been initialized, please call init_template(...) first.'
            )
        query: Optional[str] = example.get('query', None)
        response: Optional[str] = example.get('response', None)
        history: Optional[History] = example.get('history', None)
        system = example.get('system', None)
        if query is None:
            query = ''
        if history is None:
            history = []
        if len(history) > 0:
            assert self._support_multi_round, 'the template not support multi-round chat'
        if system is None:
            system = self.system
        else:
            assert self._has_system, 'not support `system`'
        return _encode(self, query, response, history, system,
                       self.truncation_strategy)


TEMPLATE_MAPPING: Dict[str, Template] = {}


def register_template(template_type: str, template: Template) -> None:
    if template_type in TEMPLATE_MAPPING:
        raise ValueError(
            f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.'
        )
    TEMPLATE_MAPPING[template_type] = template


register_template(
    TemplateType.default,
    Template(['{{SYSTEM}}\n\n'],
             ['### Human:\n', '{{QUERY}}\n\n', '### Assistant:\n'], ['\n\n'],
             [['eos_token_id']]))

# You can set the query as '' to serve as a template for pre-training.
register_template(TemplateType.default_generation,
                  Template([], ['{{QUERY}}'], None, [['eos_token_id']]))
register_template(
    TemplateType.default_generation_bos,
    Template([['bos_token_id']], ['{{QUERY}}'], None, [['eos_token_id']]))
register_template(
    TemplateType.chatml,
    Template(
        ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
        ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        ['<|im_end|>\n'], ['<|im_end|>\n<|endoftext|>']))
register_template(
    TemplateType.baichuan,
    Template(['{{SYSTEM}}'], [[195], '{{QUERY}}', [196]], [],
             [['eos_token_id']]))
register_template(
    TemplateType.chatglm2,
    Template([[64790, 64792]], ['[Round {{ROUND1}}]\n\n问：{{QUERY}}\n\n答：'],
             ['\n\n'], [['eos_token_id']]))

register_template(
    TemplateType.chatglm_generation,
    Template([[64790, 64792]], ['{{QUERY}}'], None, [['eos_token_id']]))

register_template(
    TemplateType.chatglm3,
    Template([[64790, 64792]], [[64795], '\n {{QUERY}}', [64796], '\n '], [],
             [['eos_token_id']]))

# ref: https://github.com/facebookresearch/llama/blob/main/llama/generation.py
register_template(
    TemplateType.llama,
    Template([['bos_token_id'], '[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n'],
             ['{{QUERY}} [/INST] '],
             [' ', ['eos_token_id', 'bos_token_id'], '[INST] '],
             [['eos_token_id']]))
register_template(
    TemplateType.openbuddy,
    Template([['bos_token_id'], '{{SYSTEM}}\n\n'],
             ['User: {{QUERY}}\nAssistant: '], ['\n'], [['eos_token_id']]))

register_template(
    TemplateType.internlm,
    Template(['<s>'], ['<|User|>:{{QUERY}}<eoh>\n<|Bot|>:'], ['<eoa>\n'],
             ['<eoa>\n</s>']))
register_template(
    TemplateType.xverse,
    Template([], ['Human: {{QUERY}}\n\nAssistant: '], [['eos_token_id']],
             [['eos_token_id']]))
register_template(
    TemplateType.ziya,
    Template([['bos_token_id']], ['<human>:{{QUERY}}\n<bot>:'], ['\n'],
             [['eos_token_id']]))

register_template(
    TemplateType.skywork,
    Template([], ['</s><s>[USER]{{QUERY}}[SEP][BOT]'], None, ['[SEP]</s>']))

register_template(
    TemplateType.bluelm,
    Template([['bos_token_id']], ['[|Human|]:{{QUERY}}[|AI|]:'], [],
             [['eos_token_id']]))
Context = Union[str, List[int]]


def _simplify_context_list(
        context_list: List[Context],
        compute_loss_idx: List[int]) -> Tuple[List[Context], List[int]]:
    res: List[Context] = []
    res_idx: List[int] = []
    temp: List[str] = []
    compute_loss_idx = set(compute_loss_idx)
    for i, c in enumerate(context_list):
        if isinstance(c, str) and i not in compute_loss_idx:
            temp.append(c)
        else:
            if len(temp) > 0:
                res.append(''.join(temp))
                temp.clear()
            res.append(c)
            if i in compute_loss_idx:
                res_idx.append(len(res) - 1)
    if len(temp) > 0:
        res.append(''.join(temp))
    return res, res_idx


def _concat_context_list(
    context_list: List[Context],
    res_context_list: List[Context],
    compute_loss_idx: List[int],
    system: Optional[str] = None,
    query: Optional[str] = None,
    response: Optional[str] = None,
    round0: Optional[int] = None,
) -> None:
    # concat context list and replace placeholder
    round1 = None
    if round0 is not None:
        round1 = str(round0 + 1)
        round0 = str(round0)
    for context in context_list:
        if isinstance(context, str):
            if '{{RESPONSE}}' == context:
                assert response is not None
                res_context_list.append(response)
                compute_loss_idx.append(len(res_context_list) - 1)
                continue
            old_str_list = [
                '{{SYSTEM}}', '{{QUERY}}', '{{ROUND0}}', '{{ROUND1}}'
            ]
            new_str_list = [system, query, round0, round1]
            for (old_str, new_str) in zip(old_str_list, new_str_list):
                if new_str is not None and old_str in context:
                    context = context.replace(old_str, new_str)
        res_context_list.append(context)


def _encode_context_list(
        tokenizer: PreTrainedTokenizerBase,
        context_list: List[Context],
        compute_loss_idx: Optional[List[int]] = None) -> List[int]:
    input_ids: List[int] = []
    labels: List[int] = []
    if compute_loss_idx is not None:
        compute_loss_idx = set(compute_loss_idx)
    for i, context in enumerate(context_list):
        if isinstance(context, list):
            for c in context:
                if isinstance(c, str):
                    token = getattr(tokenizer, c)
                    assert token is not None
                else:
                    token = c
                input_ids.append(token)
                labels.append(-100)
        elif isinstance(context, str):
            token_list = tokenizer(
                context, return_attention_mask=False,
                add_special_tokens=False)['input_ids']
            input_ids += token_list
            if compute_loss_idx is None:
                continue
            if i in compute_loss_idx:
                labels += token_list
            else:
                labels += [-100] * len(token_list)
    if compute_loss_idx is None:
        return input_ids
    else:
        return input_ids, labels


def _encode(template: Template, query: str, response: Optional[str],
            history: History, system: str,
            truncation_strategy: str) -> Dict[str, Optional[List[int]]]:
    res_context_list: List[Context] = []
    compute_loss_idx: List[int] = []
    _concat_context_list(
        template.prefix, res_context_list, compute_loss_idx, system=system)
    for i, (q, r) in enumerate(history):
        _concat_context_list(
            [*template.prompt, '{{RESPONSE}}', *template.chat_sep],
            res_context_list,
            compute_loss_idx,
            query=q,
            response=r,
            round0=i)
    _concat_context_list(
        template.prompt,
        res_context_list,
        compute_loss_idx,
        query=query,
        round0=len(history))
    res_context_list, compute_loss_idx = _simplify_context_list(
        res_context_list, compute_loss_idx)
    input_ids, labels = _encode_context_list(template.tokenizer,
                                             res_context_list,
                                             compute_loss_idx)

    if response is not None:
        tgt_input_ids = _encode_context_list(template.tokenizer, [response])
        tgt_input_ids += _encode_context_list(template.tokenizer,
                                              template.suffix)
        labels = labels + tgt_input_ids
        input_ids += tgt_input_ids
    else:
        labels = None

    if template.max_length is not None:
        if truncation_strategy == 'ignore' and len(
                input_ids) > template.max_length:
            return None
        input_ids = input_ids[-template.max_length:]
        if labels is not None:
            labels = labels[-template.max_length:]

    return {'input_ids': input_ids, 'labels': labels}


def get_template(
    template_type: str,
    tokenizer: PreTrainedTokenizerBase,
    system: str = DEFAULT_SYSTEM,
    max_length: Optional[int] = None,
    truncation_strategy: Literal['ignore',
                                 'truncation_left'] = 'truncation_left'
) -> Template:
    template = TEMPLATE_MAPPING[template_type]
    template.init_template(tokenizer, system, max_length, truncation_strategy)
    return template
