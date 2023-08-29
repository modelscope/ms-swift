from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer

DEFAULT_SYSTEM = 'you are a helpful assistant!'

TEMPLATE_MAPPING = {
    'default': {
        'prefix': ['{{system}}\n\n'],
        'prompt': ['### Human:\n', '{{query}}\n\n', '### Assistant:\n'],
        'chat_sep': ['\n\n'],
        'suffix': [['eos_token_id']],
    },
    'chatml': {
        'prefix': [['im_start_id'], 'system\n{{system}}', ['im_end_id'], '\n'],
        'prompt': [['im_start_id'], 'user\n{{query}}', ['im_end_id'], '\n',
                   ['im_start_id'], 'assistant\n'],
        'chat_sep': [
            ['im_end_id'],
            '\n',
        ],
        'suffix': [['im_end_id'], ['eod_id']],
    },
    'baichuan': {
        'prefix': [],
        'prompt': [[195], '{{query}}', [196]],
        'chat_sep': [],
        'suffix': [['eos_token_id']],
    },
    'chatglm2': {
        'prefix': [[64790, 64792]],
        'prompt': ['[Round {{round}}]\n\n问：{{query}}\n\n答：'],
        'chat_sep': ['\n\n'],
        'suffix': [['eos_token_id']],
    },
    'llama': {
        'prefix': [['bos_token_id'],
                   '[INST] <<SYS>>\n{{system}}\n<</SYS>>\n\n'],
        'prompt': ['{{query}} [/INST] '],
        'chat_sep': [' ', ['eos_token_id', 'bos_token_id'], '[INST] '],
        'suffix': [['eos_token_id']],
    },
    'openbuddy_llama': {
        'prefix': ['{{system}}\n\n'],
        'prompt': ['User: {{query}}\nAssistant: '],
        'chat_sep': ['\n'],
        'suffix': [['eos_token_id']],
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
    if len(res) > 0 and isinstance(res[-1], str):
        # avoid two spaces
        res[-1] = res[-1].rstrip(' ')
    return res


def concat_context_list(
    context_list: List[Context],
    new_context_list: List[Context],
    placeholder_list: List[str],
    system: Optional[str] = None,
    query: Optional[str] = None,
    round: Optional[str] = None,
) -> None:
    for context in context_list:
        if isinstance(context, str):
            for (old_str,
                 new_str) in zip(['{{system}}', '{{query}}', '{{round}}'],
                                 [system, query, round]):
                if new_str is not None and old_str in context:
                    placeholder_list.append(new_str)
        new_context_list.append(context)


def _encode(tokenizer: PreTrainedTokenizer, context_list: List[Context],
            placeholder_list: List[str]) -> List[int]:
    input_ids: List[int] = []
    placeholder_it = iter(placeholder_list)
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
            for old_str in ['{{system}}', '{{query}}', '{{round}}']:
                if old_str in context:
                    new_str = next(placeholder_it)
                    context = context.replace(old_str, new_str)
            input_ids += tokenizer(
                context, return_attention_mask=False,
                add_special_tokens=False)['input_ids']
    return input_ids


def _preprocess(
    template_type: str,
    tokenizer: PreTrainedTokenizer,
    query: str,
    response: Optional[str] = None,
    history: Optional[List[Tuple[str, str]]] = None,
    system: Optional[str] = None,
    max_length: Optional[int] = None,
) -> Dict[str, List[int]]:
    if history is None:
        history = []

    template_config = TEMPLATE_MAPPING[template_type]
    if system is None:
        system = DEFAULT_SYSTEM
    total_context_list: List[Context] = []
    placeholder_list: List[str] = []
    concat_context_list(
        template_config['prefix'],
        total_context_list,
        placeholder_list,
        system=system)
    for i, (q, r) in enumerate(history):
        concat_context_list(
            [*template_config['prompt'], r, *template_config['chat_sep']],
            total_context_list,
            placeholder_list,
            query=q,
            round=str(i + 1))
    concat_context_list(
        template_config['prompt'],
        total_context_list,
        placeholder_list,
        query=query,
        round=str(len(history) + 1))
    total_context_list = simplify_context_list(total_context_list)
    input_ids = _encode(tokenizer, total_context_list, placeholder_list)

    labels = None
    if response is not None:
        labels = [-100] * len(input_ids)
        tgt_input_ids = _encode(tokenizer, [response], [])
        tgt_input_ids += _encode(tokenizer, template_config['suffix'], [])
        input_ids += tgt_input_ids
        labels += tgt_input_ids

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
        batched: bool = False
) -> Callable[[Dict[str, Any]], Dict[str, List[int]]]:

    def preprocess(example: Dict[str, Any]) -> Dict[str, List[int]]:
        history: Optional[List[Tuple[str, str]]] = example.get('history', None)
        query: str = example['query']
        response: str = example.get('response', None)
        custom_system = example.get('system', system)
        return _preprocess(template_type, tokenizer, query, response, history,
                           custom_system, max_length)

    if batched:
        # Avoid tqdm printing too much logs when dataset.map(...)
        def batched_preprocess(
                batched_examples: Dict[str,
                                       List[Any]]) -> Dict[str, List[Any]]:
            n = len(batched_examples['query'])
            res: List[Dict[str, Any]] = []
            for i in range(n):
                example = {k: v[i] for k, v in batched_examples.items()}
                res.append(preprocess(example))
            batched_res: Dict[List[str, Any]] = {}
            assert len(res) > 0
            for k in res[0].keys():
                batched_res[k] = [r[k] for r in res]
            return batched_res

        return batched_preprocess
    return preprocess
