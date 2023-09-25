# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer

DEFAULT_SYSTEM = 'you are a helpful assistant!'
History = List[Tuple[str, str]]

TEMPLATE_MAPPING = {
    'default': {
        'prefix': ['{{SYSTEM}}\n\n'],
        'prompt': ['### Human:\n', '{{QUERY}}\n\n', '### Assistant:\n'],
        'chat_sep': ['\n\n'],
        'suffix': [['eos_token_id']],
    },
    # You can set the query as '' to serve as a template for pre-training.
    'default-generation': {
        'prefix': [],
        'prompt': ['{{QUERY}}'],
        'suffix': [['eos_token_id']],
    },
    'chatml': {
        'prefix': ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
        'prompt':
        ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        'chat_sep': ['<|im_end|>\n'],
        'suffix': ['<|im_end|><|endoftext|>'],
    },
    'baichuan': {
        'prefix': [],
        'prompt': [[195], '{{QUERY}}', [196]],
        'chat_sep': [],
        'suffix': [['eos_token_id']],
    },
    'chatglm2': {
        'prefix': [[64790, 64792]],
        'prompt': ['[Round {{ROUND}}]\n\n问：{{QUERY}}\n\n答：'],
        'chat_sep': ['\n\n'],
        'suffix': [['eos_token_id']],
    },
    'chatglm2-generation': {
        'prefix': [[64790, 64792]],
        'prompt': ['{{query}}'],
        'suffix': [['eos_token_id']],
    },
    'llama': {
        'prefix': [['bos_token_id'],
                   '[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n'],
        'prompt': ['{{QUERY}} [/INST] '],
        'chat_sep': [' ', ['eos_token_id', 'bos_token_id'], '[INST] '],
        'suffix': [['eos_token_id']],
    },
    'openbuddy-llama': {
        'prefix': ['{{SYSTEM}}\n\n'],
        'prompt': ['User: {{QUERY}}\nAssistant: '],
        'chat_sep': ['\n'],
        'suffix': [['eos_token_id']],
    },
    'internlm': {
        'prefix': ['<s>'],
        'prompt': ['<|User|>:{{QUERY}}<eoh>\n<|Bot|>:'],
        'chat_sep': ['<eoa>\n'],
        'suffix': ['<eoa></s>'],
    }
}
Context = Union[str, List[int]]


def simplify_context_list(context_list: List[Context]) -> List[Context]:
    res: List[Context] = []
    temp: List[str] = []
    for c in context_list:
        if isinstance(c, str):
            temp.append(c)
        else:
            if len(temp) > 0:
                res.append(''.join(temp))
                temp.clear()
            res.append(c)
    if len(temp) > 0:
        res.append(''.join(temp))
    return res


def concat_context_list(
    context_list: List[Context],
    new_context_list: List[Context],
    system: Optional[str] = None,
    query: Optional[str] = None,
    round: Optional[str] = None,
) -> None:
    # concat context list and replace placeholder
    for context in context_list:
        if isinstance(context, str):
            for (old_str,
                 new_str) in zip(['{{SYSTEM}}', '{{QUERY}}', '{{ROUND}}'],
                                 [system, query, round]):
                if new_str is not None and old_str in context:
                    context = context.replace(old_str, new_str)
        new_context_list.append(context)


def _encode(tokenizer: PreTrainedTokenizer,
            context_list: List[Context]) -> List[int]:
    input_ids: List[int] = []
    for context in context_list:
        if isinstance(context, list):
            for c in context:
                if isinstance(c, str):
                    token = getattr(tokenizer, c)
                    assert token is not None
                else:
                    token = c
                input_ids.append(token)
        elif isinstance(context, str):
            input_ids += tokenizer(
                context, return_attention_mask=False,
                add_special_tokens=False)['input_ids']
    return input_ids


def _preprocess(
    template_type: str,
    tokenizer: PreTrainedTokenizer,
    query: str,
    response: Optional[str] = None,
    history: Optional[History] = None,
    system: Optional[str] = None,
    max_length: Optional[int] = None,
    # do cross-validation with `model.generate()`
    generation_mode: bool = False,
) -> Dict[str, List[int]]:
    if history is None:
        history = []

    template_config = TEMPLATE_MAPPING[template_type]
    if system is None:
        system = DEFAULT_SYSTEM
    total_context_list: List[Context] = []
    concat_context_list(
        template_config['prefix'], total_context_list, system=system)
    for i, (q, r) in enumerate(history):
        assert 'chat_sep' in template_config, 'not support multi-round chat'
        concat_context_list(
            [*template_config['prompt'], r, *template_config['chat_sep']],
            total_context_list,
            query=q,
            round=str(i + 1))
    concat_context_list(
        template_config['prompt'],
        total_context_list,
        query=query,
        round=str(len(history) + 1))
    total_context_list = simplify_context_list(total_context_list)
    input_ids = _encode(tokenizer, total_context_list)

    labels = None
    if response is not None:
        tgt_input_ids = _encode(tokenizer, [response])
        tgt_input_ids += _encode(tokenizer, template_config['suffix'])

        if not generation_mode:
            # train, or validate with `loss`
            labels = [-100] * len(input_ids) + tgt_input_ids
            input_ids += tgt_input_ids
        else:
            labels = tgt_input_ids

    if max_length is not None:
        input_ids = input_ids[-max_length:]
        if labels is not None:
            labels = labels[-max_length:]

    return {'input_ids': input_ids, 'labels': labels}


def get_preprocess(
    template_type: str,
    tokenizer: PreTrainedTokenizer,
    system: Optional[str] = None,
    max_length: Optional[int] = None,
) -> Callable[[Dict[str, Any]], Dict[str, List[int]]]:

    def preprocess(example: Dict[str, Any],
                   generation_mode: bool = False) -> Dict[str, List[int]]:
        history: Optional[History] = example.get('history', None)
        query: str = example['query']
        response: str = example.get('response', None)
        custom_system = example.get('system', system)
        return _preprocess(template_type, tokenizer, query, response, history,
                           custom_system, max_length, generation_mode)

    return preprocess
