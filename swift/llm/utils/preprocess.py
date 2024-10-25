# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import os
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from tqdm import tqdm
from transformers.utils import strtobool

from swift.utils import get_logger
from .media import MediaTag
from .template import History

dataset_enable_cache = strtobool(os.environ.get('DATASET_ENABLE_CACHE', 'False'))

DATASET_TYPE = Union[HfDataset, HfIterableDataset]
PreprocessFunc = Callable[[DATASET_TYPE], DATASET_TYPE]

logger = get_logger()


def _reduce_columns(cls: type) -> type:
    # Remove unnecessary columns from the output dataset.
    if getattr(cls, '_patching', False) or dataset_enable_cache:
        return cls

    call_func = cls.__call__
    preprocess = cls.preprocess
    cls._patching = True

    def new_call_func(self, dataset: DATASET_TYPE) -> DATASET_TYPE:
        self.key_mapping = {k: i for i, k in enumerate(self.empty_row.keys())}
        num_proc = int(os.environ.get('DATASET_MAP_NPROC', '1'))
        self.shared_shm_name = None
        shm, buffer = None, None
        if num_proc > 1:  # multiprocess
            shm = shared_memory.SharedMemory(create=True, size=len(self.key_mapping))
            self.shared_shm_name = shm.name
            buffer = shm.buf
        self.column_state = np.ndarray((len(self.key_mapping), ), dtype=np.bool_, buffer=buffer)
        self.column_state[:] = False
        dataset = call_func(self, dataset)
        if isinstance(dataset, HfIterableDataset) and dataset.features is None:
            features = next(iter(dataset)).keys()
        else:
            features = dataset.features.keys()
        for k in features:
            if k in ['images', 'videos', 'audios']:
                continue
            k_i = self.key_mapping.get(k, -1)
            if k_i == -1 or not self.column_state[k_i]:
                dataset = dataset.remove_columns([k])
        if shm:
            shm.close()
            shm.unlink()
        return dataset

    def new_preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if self.shared_shm_name is not None:  # multiprocess
            shm = shared_memory.SharedMemory(name=self.shared_shm_name)
            column_state = np.ndarray((len(self.key_mapping), ), dtype=np.bool_, buffer=shm.buf)
        else:
            column_state = self.column_state
        row = preprocess(self, row)
        for k, v in row.items():
            if k in ['images', 'videos', 'audios']:
                continue
            k_i = self.key_mapping[k]
            if column_state[k_i]:
                continue
            if k == 'query_role':
                if v and v != 'user':
                    column_state[k_i] = True
            elif k == 'history_roles':
                if v and any(_v[0] != 'user' or _v[1] != 'assistant' for _v in v):
                    column_state[k_i] = True
            elif v:
                column_state[k_i] = True
        return row

    cls.__call__ = new_call_func
    cls.preprocess = new_preprocess

    return cls


def parse_medias(d: Dict[str, Any], media_key=None):
    if isinstance(media_key, str):
        if media_key in d:
            medias = d[media_key]
        else:
            medias = None
    elif media_key:  # function
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

    def parse_medias(self, d: Dict[str, Any]):
        return parse_medias(d, self.media_key)

    @property
    def empty_row(self):
        empty_row = {
            'query': None,
            'response': None,
            'tools': None,
            'system': None,
            'history': None,
        }
        if self.media_type and not isinstance(self.media_key, str):
            empty_row[self.media_name] = None
        return empty_row


class RowPreprocessMixin:

    def preprocess(self, d: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class SwiftPreprocessor:

    def __call__(self, dataset: DATASET_TYPE) -> DATASET_TYPE:
        if isinstance(dataset, HfIterableDataset):
            return dataset
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


@_reduce_columns
class AlpacaPreprocessor(MediaMixin, RowPreprocessMixin):

    def __init__(self, concat_inst_inp: Optional[Callable[[str, str], str]] = None, **kwargs):
        self.concat_inst_inp = concat_inst_inp
        super().__init__(**kwargs)

    def preprocess(self, d: Dict[str, Any]) -> Dict[str, Any]:
        inst = d['instruction']
        inp: Optional[str] = d.get('input', None)
        h, output = d.pop('history', None), d['output']
        sys = d.pop('system', None)
        tool = d.pop('tools', None)
        if output is None:
            return self.empty_row
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
        medias = self.parse_medias(d)
        self.media_replacer(row, medias)
        return row

    def __call__(self, dataset: DATASET_TYPE) -> DATASET_TYPE:
        kwargs = {}
        if not isinstance(dataset, HfIterableDataset):
            kwargs['load_from_cache_file'] = dataset_enable_cache
        dataset = dataset.map(self.preprocess, **kwargs).filter(lambda row: row.get('response'))
        if self.media_type and isinstance(self.media_key, str) and self.media_key != self.media_name:
            dataset = dataset.rename_columns({self.media_key: self.media_name})
        return dataset


def _default_repair_conversations(s: Union[str, Any]) -> Any:
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s


@_reduce_columns
class ConversationsPreprocessor(MediaMixin, RowPreprocessMixin):

    def __init__(self,
                 user_role: str = 'user',
                 assistant_role: str = 'assistant',
                 system_role: str = 'system',
                 conversations_key: str = 'conversations',
                 from_key: str = 'from',
                 value_key: str = 'value',
                 tool_role: str = 'tool',
                 repair_conversations: Callable[[Union[str, List[Dict[str, str]]]],
                                                Optional[List[Dict[str, str]]]] = _default_repair_conversations,
                 error_strategy: Literal['delete', 'raise'] = 'raise',
                 **kwargs):
        self.user_role = user_role
        self.assistant_role = assistant_role
        self.system_role = system_role
        self.conversations_key = conversations_key
        self.from_key = from_key
        self.value_key = value_key
        self.tool_role = tool_role
        self.repair_conversations = repair_conversations
        self.error_strategy = error_strategy
        super().__init__(**kwargs)

    @property
    def empty_row(self):
        empty_row = super().empty_row
        empty_row['history_roles'] = None
        empty_row['query_role'] = None
        empty_row['tools'] = None
        return empty_row

    def preprocess(self, d: Dict[str, Any]) -> Dict[str, Any]:
        try:
            conversations = d[self.conversations_key]
            conversations = self.repair_conversations(conversations)
            if conversations is None:
                return self.empty_row
            lo = 0
            sys = None
            h: History = []
            hr: History = []
            assert len(conversations) >= 2
            if conversations[0][self.from_key] == self.system_role:
                lo += 1
                sys = conversations[0][self.value_key]
            assert conversations[-2][self.from_key] in [self.user_role, self.tool_role]
            assert conversations[-1][self.from_key] == self.assistant_role

            for q, r in zip(conversations[lo:-2:2], conversations[lo + 1:-2:2]):
                assert q[self.from_key] in [self.user_role, self.tool_role]
                assert r[self.from_key] == self.assistant_role
                h.append([q[self.value_key], r[self.value_key]])
                _q_role = q[self.from_key]
                _r_role = r[self.from_key]
                _q_role = _q_role if _q_role == 'tool' else 'user'
                _r_role = _r_role if _r_role == 'tool' else 'assistant'
                hr.append([_q_role, _r_role])
            query = conversations[-2][self.value_key]
            query_role = conversations[-2][self.from_key]
            query_role = query_role if query_role == 'tool' else 'user'
            response = conversations[-1][self.value_key]
            system = sys
            history = h
            tools = d.get('tools') or []
            row = {'system': system, 'history': history, 'history_roles': hr}
            row.update({
                'query': query,
                'query_role': query_role,
                'response': response,
                'tools': tools,
            })
            medias = self.parse_medias(d)
            self.media_replacer(row, medias)
            return row
        except (AssertionError, SyntaxError) as e:
            logger.error(e)
            if self.error_strategy == 'raise':
                raise ValueError(f'conversations: {conversations}')
            else:
                return self.empty_row

    def __call__(self, dataset: DATASET_TYPE) -> DATASET_TYPE:
        kwargs = {}
        if not isinstance(dataset, HfIterableDataset):
            kwargs['load_from_cache_file'] = dataset_enable_cache
        dataset = dataset.map(self.preprocess, **kwargs).filter(lambda row: row.get('response') is not None)
        if self.media_type and isinstance(self.media_key, str) and self.media_key != self.media_name:
            dataset = dataset.rename_columns({self.media_key: self.media_name})
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

    def preprocess(self, d: Dict[str, Any]) -> Dict[str, Any]:
        conversations = None
        try:
            conversations = d[self.conversations_key]
            if self.inner_key is not None:
                conversations = conversations[self.inner_key]
            history = []
            for c in conversations:
                history.append([c[self.query_key], c[self.response_key]])

            query, response = history.pop(-1)
            row = {
                'history': history,
                'query': query,
                'response': response,
            }
            medias = self.parse_medias(d)
            self.media_replacer(row, medias)
        except Exception:
            if self.error_strategy == 'raise':
                raise ValueError(f'conversations: {conversations}')
            else:
                return self.empty_row
        return row

    def __call__(self, dataset: DATASET_TYPE) -> DATASET_TYPE:
        kwargs = {}
        if not isinstance(dataset, HfIterableDataset):
            kwargs['load_from_cache_file'] = dataset_enable_cache
        dataset = dataset.map(self.preprocess, **kwargs).filter(lambda d: d.get('response'))
        if self.media_type and isinstance(self.media_key, str) and self.media_key != self.media_name:
            dataset = dataset.rename_columns({self.media_key: self.media_name})
        return dataset


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
            if old_name in dataset.features:
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
            },
            'pretrain': {
                'required': ['text'],
                'preprocessor': RenameColumnsPreprocessor({
                    'prompt': 'query',
                    'text': 'response'
                })
            }
        }

    def _get_preprocessor(self, dataset: DATASET_TYPE) -> PreprocessFunc:
        if isinstance(dataset, HfIterableDataset) and dataset.features is None:
            keys = set(next(iter(dataset)).keys())
        else:
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
