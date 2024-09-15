# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import os
from copy import copy
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from tqdm import tqdm
from transformers.utils import strtobool
from swift.utils import get_logger
from swift.llm.utils.template import History

dataset_enable_cache = strtobool(os.environ.get('DATASET_ENABLE_CACHE', 'False'))

DATASET_TYPE = Union[HfDataset, HfIterableDataset]
PreprocessFunc = Callable[[DATASET_TYPE], DATASET_TYPE]

logger = get_logger()


class GroundingMixin:

    _grounding_task_type = None

    _grounding_prompts = {
        'grounding': {
            'en': [('<ref-object>', '<bbox>'), ('The positions of <ref-object> is', '<bbox>'),
                   ('Find the positions of <ref-object>', '<bbox>'), ('Where is <ref-object>', '<bbox>'),
                   ('Find <ref-object>', '<bbox>'), ('Show me <ref-object>', '<bbox>'),
                   ('Detect <ref-object>', '<bbox>'), ('Locate <ref-object>', '<bbox>'),
                   ('Tell me the location of <ref-object>', '<bbox>'), ('Give the location of <ref-object>', '<bbox>'),
                   ('Provide the bounding box coordinate of <ref-object>', '<bbox>')],
            'zh': [('<ref-object>', '<bbox>'), ('<ref-object>的位置在图片中', '<bbox>'),
                   ('<ref-object>在图片中', '<bbox>'),
                   ('<ref-object>在', '<bbox>'), ('找到<ref-object>的位置', '<bbox>'), ('<ref-object>在哪里', '<bbox>'),
                   ('提供<ref-object>的坐标位置', '<bbox>')]
        },
        'caption': {
            'en': [
                ('<bbox>', '<ref-object>'),
                ('The object at position <bbox>', '<ref-object>'),
                ('This <bbox> is', '<ref-object>'),
                ('What is the object at <bbox>', '<ref-object>'),
                ('Describe <bbox>', '<ref-object>'),
                ('<bbox> is', '<ref-object>'),
                ('The bounding box coordinate <bbox> contains', '<ref-object>'),
            ],
            'zh': [
                ('<bbox>', '<ref-object>'),
                ('<bbox>是什么', '<ref-object>'),
                ('<bbox>的位置包含', '<ref-object>'),
                ('描述<bbox>', '<ref-object>'),
                ('<bbox>中是', '<ref-object>'),
                ('坐标<bbox>描述了什么', '<ref-object>'),
                ('描述<bbox>中的事物', '<ref-object>'),
            ]
        },
    }

    @classmethod
    def construct_grounding_prompt(cls):
        lang = np.random.choice(['en', 'zh'], p=[0.8, 0.2])
        prompts = cls._grounding_prompts[cls._grounding_prompts][lang]
        query, response = prompts[np.random.choice(range(len(prompts)))]
        return query, response


class RowPreprocessor(GroundingMixin):

    _has_history = False
    _has_system = False
    _has_tool = False
    _column_mapping = {}
    _mapping_kwargs = {
        'load_from_cache_file': False,
        'num_proc': 8,
    }

    _modals = []
    _modal_tags = []
    _modal_keys = []
    _grounding_language_mixin = [0.8, 0.2]
    _standard_tags = {
        'image': '<image>',
        'audio': '<audio>',
        'video': '<video>',
    }
    _standard_keys = {
        'audio': 'audios',
        'image': 'images',
        'video': 'videos',
    }

    @classmethod
    def replace_standard_tag(cls, messages, medias, modal):
        assert len(cls._modal_tags) == len(cls._modals)
        _modal_tag = None
        for _modal, _tag in zip(cls._modals, cls._modal_tags):
            if modal == _modal:
                _modal_tag = _tag
        assert _modal_tag is not None
        media_cnt = len(medias) if isinstance(medias, (tuple, list)) else 1 if medias else 0
        # like <image>, etc
        standard_tag = cls._standard_tags[modal]
        all_content = ''.join([m['content'] for m in messages])
        if _modal_tag in all_content:
            # If the messages already have placeholders like `<image>`
            assert all_content.count(_modal_tag) == media_cnt
            for m in messages:
                # Replace to standard tag
                m['content'] = m['content'].replace(_modal_tag, standard_tag)
        else:
            for m in messages:
                if m['role'] not in ('tool', 'system', 'assistant'):
                    m['content'] = ''.join([standard_tag] * media_cnt) + m['content']

        return messages

    @classmethod
    def parse_media_from_row(cls, row: Dict[str, Any], modal):
        modal_key = cls._modal_keys[modal]
        if isinstance(modal_key, str):
            if modal_key in row:
                medias = row[modal_key]
            else:
                medias = None
        elif modal_key:  # function
            medias = modal_key(row)
        else:
            medias = None
        return medias

    @classmethod
    def preprocess(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        return row

    @classmethod
    def filter(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        return True

    @classmethod
    def rename_columns(cls, dataset: HfDataset, column_mapping: Dict[str, str]):
        return dataset.rename_columns(column_mapping)

    @classmethod
    def __call__(cls, dataset: HfDataset, **kwargs):
        if cls._modal_keys or cls._modals:
            assert len(cls._modal_keys) == len(cls._modals)

        column_mapping = copy(cls._column_mapping)
        # Replace un-standard media keys to standard keys
        for idx, _modal in enumerate(cls._modals):
            modal_key = cls._modal_keys[idx]
            standard_key = cls._standard_keys[idx]
            column_mapping[modal_key] = standard_key

        if column_mapping:
            dataset = cls.rename_columns(dataset, column_mapping)
        if cls.preprocess is not RowPreprocessor.preprocess:
            dataset = dataset.map(cls.preprocess, **kwargs)
        if cls.filter is not RowPreprocessor.filter:
            dataset = dataset.filter(cls.filter, **kwargs)
        return dataset


class SwiftPreprocessor(RowPreprocessor):

    @classmethod
    def preprocess(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        output = {}
        if cls._has_history:
            history = row['history']
            if isinstance(history, str):
                history = ast.literal_eval(history)
                output['history'] = history
        return output


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
        if self.media_type:
            if not isinstance(self.media_key, str):
                row[self.media_name] = medias
            else:
                row[self.media_key] = medias
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
            if self.media_type:
                if not isinstance(self.media_key, str):
                    row[self.media_name] = medias
                else:
                    row[self.media_key] = medias
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
            if self.media_type:
                if not isinstance(self.media_key, str):
                    row[self.media_name] = medias
                else:
                    row[self.media_key] = medias
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
