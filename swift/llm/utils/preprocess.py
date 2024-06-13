# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import os.path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from tqdm import tqdm

from .media import MediaTag
from .template import History

PreprocessFunc = Callable[[HfDataset], HfDataset]


def parse_medias(d, media_key=None):
    if isinstance(media_key, str):
        if media_key in d:
            medias = d[media_key]
        else:
            medias = None
    elif media_key:
        medias = media_key(d)
    else:
        medias = None
    return medias


class MediaMixin:

    def __init__(self,
                 media_key: Union[str, Callable] = 'image',
                 media_tag: str = '<image>',
                 media_type: Literal['image', 'audio', 'video'] = None):
        self.media_key = media_key
        self.media_tag = media_tag
        self.media_type = media_type
        self.media_replacer = MediaTag(media_type, media_tag)

    @property
    def media_name(self):
        if not self.media_type:
            return None
        return self.media_replacer.media_keys[self.media_type]

    def parse_medias(self, d):
        return parse_medias(d, self.media_key)


class RowPreprocessMixin:

    def preprocess(self, d):
        raise NotImplementedError


class SwiftPreprocessor:

    def __call__(self, dataset: HfDataset) -> HfDataset:
        if 'history' in dataset.features:
            old_history = dataset['history']
            has_history = False
            history: List[History] = []
            for h in tqdm(old_history):
                if isinstance(h, str):
                    h = ast.literal_eval(h)
                elif h is None:
                    h = []
                if len(h) > 0:
                    has_history = True
                history.append(h)
            dataset = dataset.remove_columns(['history'])
            if has_history:
                dataset = dataset.add_column('history', history)
        if 'system' in dataset.features:
            system = dataset['system']
            has_system = len([sys for sys in system if sys not in {None, ''}]) > 0
            if not has_system:
                dataset = dataset.remove_columns(['system'])
        return dataset


class AlpacaPreprocessor(MediaMixin, RowPreprocessMixin):

    def __init__(self, concat_inst_inp: Optional[Callable[[str, str], str]] = None, **kwargs):
        self.concat_inst_inp = concat_inst_inp
        super().__init__(**kwargs)

    def preprocess(self, d):
        inst, inp = d['instruction'], d.get('input', None)
        h, output = d.pop('history', None), d['output']
        sys = d.pop('system', None)
        tool = d.pop('tools', None)
        if output is None:
            return {
                'query': '',
                'response': '',
            }
        if inp is None or len(inp) == 0:
            q = inst
        elif self.concat_inst_inp is not None:
            q = self.concat_inst_inp(inst, inp)
        else:
            q = f'{inst}\n{inp}'
        row = {
            'history': h,
            'query': q,
            'system': sys,
            'response': output,
            'tools': tool,
        }
        self.media_replacer(row, self.parse_medias(d))
        return row

    def __call__(self, dataset: HfDataset) -> HfDataset:
        query: List[str] = []
        response = []
        system = None
        history = None
        medias = None
        tools = None
        for i, d in enumerate(tqdm(dataset)):
            d = self.preprocess(d)
            q = d['query']
            r = d['response']
            if not q and not r:
                continue
            h = d.get('history')
            sys = d.get('system')
            med = d.get(self.media_name)
            tool = d.get('tools', None)
            if history is None and h is not None:
                history = [None for _ in range(i - 1)]
            if system is None and sys is not None:
                system = [None for _ in range(i - 1)]
            if tools is None and tool is not None:
                tools = [None for _ in range(i - 1)]
            if medias is None and med is not None:
                medias = [None for _ in range(i - 1)]
            query.append(q)
            response.append(r)
            if history is not None:
                history.append(h)
            if system is not None:
                system.append(sys)
            if medias is not None:
                medias.append(med)
            if tools is not None:
                tools.append(tool)

        d_dict = {'query': query, 'response': response}
        if history is not None:
            d_dict['history'] = history
        if system is not None:
            d_dict['system'] = system
        if medias is not None:
            d_dict[self.media_name] = medias
        if tools is not None:
            d_dict['tools'] = tools
        dataset = HfDataset.from_dict(d_dict)
        return dataset


def _default_repair_conversations(s: Union[str, Any]) -> Any:
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s


class ConversationsPreprocessor(MediaMixin, RowPreprocessMixin):

    def __init__(self,
                 user_role: str = 'user',
                 assistant_role: str = 'assistant',
                 system_role: str = 'system',
                 conversations_key: str = 'conversations',
                 from_key: str = 'from',
                 value_key: str = 'value',
                 repair_conversations: Callable[[Union[str, Dict[str, str]]],
                                                Optional[Dict[str, str]]] = _default_repair_conversations,
                 error_strategy: Literal['delete', 'raise'] = 'raise',
                 **kwargs):
        self.user_role = user_role
        self.assistant_role = assistant_role
        self.system_role = system_role
        self.conversations_key = conversations_key
        self.from_key = from_key
        self.value_key = value_key
        self.repair_conversations = repair_conversations
        self.error_strategy = error_strategy
        super().__init__(**kwargs)

    def preprocess(self, d):
        try:
            conversations = d[self.conversations_key]
            conversations = self.repair_conversations(conversations)
            if conversations is None:
                return {
                    'query': '',
                    'response': '',
                }
            lo = 0
            sys = None
            h: History = []
            assert len(conversations) >= 2
            if conversations[0][self.from_key] == self.system_role:
                lo += 1
                sys = conversations[0][self.value_key]
            assert conversations[-2][self.from_key] == self.user_role
            assert conversations[-1][self.from_key] == self.assistant_role

            for q, r in zip(conversations[lo:-2:2], conversations[lo + 1:-2:2]):
                assert q[self.from_key] == self.user_role
                assert r[self.from_key] == self.assistant_role
                h.append([q[self.value_key], r[self.value_key]])
            query = conversations[-2][self.value_key]
            response = conversations[-1][self.value_key]
            system = sys
            history = h
            tool = d.get('tools', [])
            kwargs = {'system': system, 'history': history}
            kwargs.update({
                'query': query,
                'response': response,
                'tools': tool,
            })
            self.media_replacer(kwargs, self.parse_medias(d))
            return kwargs
        except (AssertionError, SyntaxError):
            if self.error_strategy == 'raise':
                raise ValueError(f'conversations: {conversations}')
            else:
                return {
                    'query': '',
                    'response': '',
                }

    def __call__(self, dataset: HfDataset) -> HfDataset:
        query: List[str] = []
        response: List[str] = []
        system: List[Optional[str]] = []
        tools: List[List[Dict[str, Any]]] = []
        has_system = False
        history: List[History] = []
        has_history = False
        medias: List = []
        has_medias = False
        has_tools = False

        for d in tqdm(dataset):
            d = self.preprocess(d)
            q = d['query']
            r = d['response']
            if not q and not r:
                continue
            h = d.get('history')
            sys = d.get('system')
            tool = d.get('tools')
            med = d.get(self.media_name)
            if h:
                has_history = True
            if sys:
                has_system = True
            if med:
                has_medias = True
            if tool:
                has_tools = True
            query.append(q)
            response.append(r)
            system.append(sys)
            history.append(h)
            medias.append(med)

        kwargs = {}
        if has_system:
            kwargs['system'] = system
        if has_medias:
            kwargs[self.media_name] = medias
        kwargs.update({
            'query': query,
            'response': response,
        })
        if has_history:
            kwargs['history'] = history
        if has_tools:
            kwargs['tools'] = tools
        if has_tools:
            kwargs['tools'] = tools
        dataset = HfDataset.from_dict(kwargs)
        return dataset


class ListPreprocessor(MediaMixin, RowPreprocessMixin):

    def __init__(self,
                 query_key: str = 'user',
                 response_key: str = 'assistant',
                 conversations_key: str = 'conversations',
                 inner_key: str = None,
                 repair_conversations: Callable[[Union[str, Dict[str, str]]],
                                                Optional[Dict[str, str]]] = _default_repair_conversations,
                 error_strategy: Literal['delete', 'raise'] = 'raise',
                 **kwargs):
        self.query_key = query_key
        self.response_key = response_key
        self.conversations_key = conversations_key
        self.inner_key = inner_key
        self.repair_conversations = repair_conversations
        self.error_strategy = error_strategy
        super().__init__(**kwargs)

    def preprocess(self, d):
        conversations = None
        try:
            conversations = d[self.conversations_key]
            if self.inner_key is not None:
                conversations = conversations[self.inner_key]
            history = []
            for c in conversations:
                history.append([c[self.query_key], c[self.response_key]])

            query, response = history.pop(-1)
            d_dict = {
                'history': history,
                'query': query,
                'response': response,
            }
            self.media_replacer(d_dict, self.parse_medias(d))
        except Exception:
            if self.error_strategy == 'raise':
                raise ValueError(f'conversations: {conversations}')
            else:
                return {
                    'query': '',
                    'response': '',
                }
        return d_dict

    def __call__(self, dataset: HfDataset):
        return dataset.map(self.preprocess, load_from_cache_file=False).filter(lambda d: d.get('query'))


class ComposePreprocessor:

    def __init__(self, preprocessor_list: List[PreprocessFunc]) -> None:
        self.preprocessor_list = preprocessor_list

    def __call__(self, dataset: HfDataset) -> HfDataset:
        for preprocessor in self.preprocessor_list:
            dataset = preprocessor(dataset)
        return dataset


class RenameColumnsPreprocessor:

    def __init__(self, rename_mapping: Dict[str, str]) -> None:
        self.rename_mapping = rename_mapping

    def __call__(self, dataset: HfDataset) -> HfDataset:
        for old_name, new_name in self.rename_mapping.items():
            dataset = dataset.rename_column(old_name, new_name)
        return dataset


def preprocess_sharegpt(dataset: HfDataset) -> HfDataset:
    query = []
    response = []
    system: List[Optional[str]] = []
    has_system = False
    history: List[History] = []
    has_history = False
    for d in tqdm(dataset):
        if isinstance(d['conversation'], str):
            try:
                conversation = ast.literal_eval(d['conversation'])
            except SyntaxError:
                continue
        else:
            conversation = d['conversation']
        query.append(conversation[-1]['human'])
        response.append(conversation[-1]['assistant'])
        h = []
        for c in conversation[:-1]:
            h.append([c['human'], c['assistant']])
        if len(h) > 0:
            has_history = True
        history.append(h)
        sys = d.get('system')
        if sys is not None:
            has_system = True
        system.append(sys)
    kwargs = {'query': query, 'response': response}
    if has_history:
        kwargs['history'] = history
    if has_system:
        kwargs['system'] = system
    return HfDataset.from_dict(kwargs)


class SmartPreprocessor:

    def __init__(self) -> None:
        self.preprocessor_mapping = {
            'swift': {
                'required': ['response'],
                'preprocessor': SwiftPreprocessor()
            },
            'alpaca': {
                'required': ['instruction', 'output'],
                'preprocessor': AlpacaPreprocessor()
            },
            'conversations': {  # qwen
                'required': ['conversations'],
                'preprocessor': ConversationsPreprocessor()
            },
            'chatml': {
                'required': ['messages'],
                'preprocessor':
                ConversationsPreprocessor(conversations_key='messages', from_key='role', value_key='content')
            },
            'sharegpt': {
                'required': ['conversation'],
                'preprocessor': preprocess_sharegpt
            }
        }

    def _get_preprocessor(self, dataset: HfDataset) -> PreprocessFunc:
        keys = set(dataset.features.keys())
        required_keys_mapping = {k: v['required'] for k, v in self.preprocessor_mapping.items()}
        for k, required_keys in required_keys_mapping.items():
            if len(set(required_keys) - keys) == 0:
                return self.preprocessor_mapping[k]['preprocessor']
        raise ValueError(f"""dataset.features.keys(): {dataset.features.keys()}
required_keys_mapping: {required_keys_mapping}""")

    def __call__(self, dataset: HfDataset) -> HfDataset:
        preprocessor = self._get_preprocessor(dataset)
        return preprocessor(dataset)


class TextGenerationPreprocessor:

    def __init__(self, prompt: str, query_key: str = 'query', response_key: str = 'response') -> None:
        self.prompt = prompt
        self.query_key = query_key
        self.response_key = response_key

    def __call__(self, dataset: HfDataset) -> HfDataset:
        query = []
        for d in tqdm(dataset):
            query.append(self.prompt.format(query=d[self.query_key]))
        return HfDataset.from_dict({'query': query, 'response': dataset[self.response_key]})


class ClsPreprocessor:

    def __init__(self, labels: List[str], task_name: str, is_pair_seq: bool = False) -> None:
        self.labels = labels
        category = ', '.join(labels)
        if is_pair_seq:
            inputs = 'Sentence1: {sentence1}\nSentence2: {sentence2}'
        else:
            inputs = 'Sentence: {sentence}'
        self.prompt = f"""Task: {task_name}
{inputs}
Category: {category}
Output:"""
        self.task_name = task_name
        self.is_pair_seq = is_pair_seq

    def __call__(self, dataset: HfDataset) -> HfDataset:
        query = []
        response = []
        for d in tqdm(dataset):
            if d['label'] is None:  # ignore dataset error
                continue
            if self.is_pair_seq:
                q = self.prompt.format(sentence1=d['sentence1'], sentence2=d['sentence2'])
            else:
                q = self.prompt.format(sentence=d['sentence'])
            query.append(q)
            response.append(self.labels[int(d['label'])])
        return HfDataset.from_dict({'query': query, 'response': response})
