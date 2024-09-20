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


DATASET_TYPE = Union[HfDataset, HfIterableDataset]
PreprocessFunc = Callable[[DATASET_TYPE, ...], DATASET_TYPE]

logger = get_logger()


class GroundingMixin:
    task_type: Optional[str] = None

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

    def construct_grounding_prompt(self):
        # TODO Only support one bbox to one object
        lang = np.random.choice(['en', 'zh'], p=[0.8, 0.2])
        prompts = GroundingMixin._grounding_prompts[self.task_type][lang]
        query, response = prompts[np.random.choice(range(len(prompts)))]
        return query, response


standard_tags = {
    'image': '<image>',
    'audio': '<audio>',
    'video': '<video>',
}
standard_keys = {
    'audio': 'audios',
    'image': 'images',
    'video': 'videos',
}


class RowPreprocessor(GroundingMixin):

    _mapping_kwargs = {
        'load_from_cache_file': False,
        'num_proc': 8,
    }

    _grounding_language_mixin = [0.8, 0.2]

    has_tool: bool = False
    column_mapping: Dict[str, str] = {}
    modals: List[str] = []
    modal_tags: List[str] = []
    modal_keys: List[str] = []

    def __init__(self, **kwargs):
        if 'has_tool' in kwargs:
            self.has_tool = kwargs.pop('has_tool')
        if 'column_mapping' in kwargs:
            self.column_mapping = kwargs.pop('column_mapping')
        if 'modals' in kwargs:
            self.modals = kwargs.pop('modals')
        if 'modal_tags' in kwargs:
            self.modal_tags = kwargs.pop('modal_tags')
        if 'modal_keys' in kwargs:
            self.modal_keys = kwargs.pop('modal_keys')

    def replace_standard_tag(self, messages, medias, modal):
        assert len(self.modal_tags) == len(self.modals)
        _modal_tag = None
        for _modal, _tag in zip(self.modals, self.modal_tags):
            if modal == _modal:
                _modal_tag = _tag
        assert _modal_tag is not None
        media_cnt = len(medias) if isinstance(medias, (tuple, list)) else 1 if medias else 0
        # like <image>, etc
        standard_tag = standard_tags[modal]
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

    def query_to_message(self, row):
        messages = []
        if 'query' in row:
            messages.append({'role': 'user', 'content': row['query']})
        if 'response' in row:
            messages.append({'role': 'assistant', 'content': row['response']})
        old_messages = row.get('messages', [])
        old_messages.extend(messages)
        row['messages'] = old_messages

    def parse_media_from_row(self, row: Dict[str, Any], modal):
        modal_key = self.modal_keys[modal]
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

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return row

    def empty_row(self):
        row = {'messages': None}
        if self.has_tool:
            row['tools'] = None
        for _modal in self.modals:
            row[standard_keys[_modal]] = None
        return row

    def map(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = self.preprocess(row)
        self.query_to_message(row)
        return row

    def filter(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return row.get('messages')

    def rename_columns(self, dataset: HfDataset, column_mapping: Dict[str, str]):
        return dataset.rename_columns(column_mapping)

    def prepare_multi_modal(self, row):
        for modal in self.modals:
            medias = self.parse_media_from_row(row, modal)
            if medias:
                row['messages'] = self.replace_standard_tag(row['messages'], medias, modal)
                modal_key = self.modal_keys[modal]
                if not isinstance(modal_key, str):
                    row[modal_key] = medias
                else:
                    row[standard_keys[modal]] = medias
        if self.task_type in self._grounding_prompts.keys():
            query, response = self.construct_grounding_prompt()
            row['messages'][-2]['content'] = query
            row['messages'][-1]['content'] = response
        return row

    def prepare_map_kwargs(self, dataset: DATASET_TYPE, **kwargs):
        _kwargs = {}
        if not isinstance(dataset, HfIterableDataset):
            _kwargs.update(self._mapping_kwargs)
        _kwargs.update(kwargs)
        return _kwargs

    def prepare_downloading(self, dataset):
        pass

    def __call__(self,
                 dataset: DATASET_TYPE,
                 *,
                 history: bool = False,
                 system: bool = False,
                 tool: bool = False,
                 column_mapping: Optional[Dict[str, str]] = None,
                 modals: Optional[List[str]] = None,
                 modal_tags: Optional[List[str]] = None,
                 modal_keys: Optional[List[str]] = None,
                 grounding_task_type: Optional[str] = None,
                 **kwargs) -> DATASET_TYPE:
        kwargs = self.prepare_map_kwargs(**kwargs)
        self.prepare_downloading(dataset)
        if self.modal_keys or self.modals:
            assert len(self.modal_keys) == len(self.modals)

        if self.preprocess is not RowPreprocessor.preprocess:
            dataset = dataset.map(self.preprocess, **kwargs)
        if self.modals:
            dataset = dataset.map(self.prepare_multi_modal, **kwargs)
        if self.filter is not RowPreprocessor.filter:
            dataset = dataset.filter(self.filter, **kwargs)

        column_mapping = copy(self.column_mapping)
        # Replace un-standard media keys to standard keys
        for idx, _modal in enumerate(self.modals):
            modal_key = self.modal_keys[idx]
            standard_key = standard_keys[idx]
            if standard_key not in dataset.features:
                column_mapping[modal_key] = standard_key

        if column_mapping:
            dataset = self.rename_columns(dataset, column_mapping)
        return dataset


class SwiftPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        output = {}
        if self.has_history:
            history = row['history']
            if isinstance(history, str):
                history = ast.literal_eval(history)
                output['history'] = history
        return output


class AlpacaPreprocessor(RowPreprocessor):
    concat_inst_inp = None

    def __init__(self,
                 *,
                 concat_inst_inp: Optional[Callable[[str, str], str]] = None, **kwargs):
        """Alpaca format preprocessor

        Args:
            concat_inst_inp: The concat sep between instruction and input
        """
        self.concat_inst_inp = concat_inst_inp
        super().__init__(**kwargs)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        instruction = row['instruction']
        input = row.get('input', None)
        output = row['output']
        history = row.get('history', [])
        system = row.get('system', None)
        tools = row.get('tools', None)
        if output is None:
            return self.empty_row()
        if not input or not instruction:
            query = (instruction or '') + (input or '')
        elif self.concat_inst_inp:
            if isinstance(self.concat_inst_inp, str):
                query = instruction + self.concat_inst_inp + input
            else:
                query = self.concat_inst_inp(instruction, input)
        else:
            query = f'{instruction}\n{input}'

        messages = []
        if system:
            messages.append({
                'role': 'system',
                'content': system
            })
        for h in history:
            messages.append({
                'role': 'user',
                'content': h[0]
            })
            messages.append({
                'role': 'assistant',
                'content': h[1]
            })
        messages.append({
            'role': 'user',
            'content': query
        })
        messages.append({
            'role': 'assistant',
            'content': output
        })

        row = {
            'messages': messages,
        }
        if tools:
            row['tools'] = tools
        return row


def _default_repair_conversations(s: Union[str, Any]) -> Any:
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s


class ConversationsPreprocessor(RowPreprocessor):

    def __init__(self,
                 *,
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

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            conversations = row[self.conversations_key]
            conversations = self.repair_conversations(conversations)
            if conversations is None:
                return self.empty_row

            messages = []
            system_message = None
            if conversations[0][self.from_key] == self.system_role:
                system_message = conversations.pop(0)

            if system_message:
                messages.append({
                    'role': 'system',
                    'content': system_message[self.value_key]
                })
            for idx, c in enumerate(conversations):
                if idx % 2 == 0:
                    assert c[self.from_key] in [self.user_role, self.tool_role]
                    messages.append({
                        'role': 'user' if c[self.from_key] == self.user_role else 'tool',
                        'content': c[self.value_key]
                    })
                else:
                    assert c[self.from_key] == self.assistant_role
                    messages.append({
                        'role': 'assistant',
                        'content': c[self.value_key]
                    })

            row = {
                'messages': messages,
            }

            if self.has_tool:
                row['tools'] = row['tools']
            return row
        except (AssertionError, SyntaxError) as e:
            logger.error(e)
            if self.error_strategy == 'raise':
                raise ValueError(f'Unsupported row: {row}')
            else:
                return self.empty_row


class ListPreprocessor(RowPreprocessor):

    def __init__(self,
                 *,
                 user_key: str = 'user',
                 tool_key: str = 'tool',
                 system_key: str = 'system',
                 assistant_key: str = 'assistant',
                 conversations_key: str = 'conversations',
                 inner_key: str = None,
                 repair_conversations: Callable[[Union[str, Dict[str, str]]],
                                                Optional[Dict[str, str]]] = _default_repair_conversations,
                 error_strategy: Literal['delete', 'raise'] = 'raise',
                 **kwargs):
        self.user_key = user_key
        self.tool_key = tool_key
        self.system_key = system_key
        self.assistant_key = assistant_key
        self.conversations_key = conversations_key
        self.inner_key = inner_key
        self.repair_conversations = repair_conversations
        self.error_strategy = error_strategy
        super().__init__(**kwargs)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            conversations = row[self.conversations_key]
            if self.inner_key is not None:
                conversations = conversations[self.inner_key]
            messages = []
            if self.system_key and row.get(self.system_key):
                messages.append({
                    'role': 'system',
                    'content': row[self.system_key]
                })
            for c in conversations:
                if self.user_key in c:
                    messages.append(
                        {
                            'role': 'user',
                            'content': c[self.user_key],
                        }
                    )
                else:
                    messages.append(
                        {
                            'role': 'tool',
                            'content': c[self.tool_key],
                        }
                    )
                messages.append(
                    {
                        'role': 'assistant',
                        'content': c[self.assistant_key],
                    }
                )

            row = {
                'messages': messages,
            }

            if self.has_tool:
                row['tools'] = row['tools']
            return row
        except Exception:
            if self.error_strategy == 'raise':
                raise ValueError(f'Unsupported row: {row}')
            else:
                return self.empty_row


class ComposePreprocessor:

    def __init__(self, preprocessor_list: List[PreprocessFunc]) -> None:
        self.preprocessor_list = preprocessor_list

    def __call__(self, dataset: DATASET_TYPE, **kwargs) -> DATASET_TYPE:
        for preprocessor in self.preprocessor_list:
            dataset = preprocessor(dataset, **kwargs)
        return dataset


class RenameColumnsPreprocessor:

    def __init__(self, rename_mapping: Dict[str, str]) -> None:
        self.rename_mapping = rename_mapping

    def query_to_message(self, row):
        messages = []
        if 'query' in row:
            messages.append({'role': 'user', 'content': row['query']})
        if 'response' in row:
            messages.append({'role': 'assistant', 'content': row['response']})
        old_messages = row.get('messages', [])
        old_messages.extend(messages)
        row['messages'] = old_messages

    def __call__(self, dataset: DATASET_TYPE, **kwargs) -> DATASET_TYPE:
        for old_name, new_name in self.rename_mapping.items():
            if old_name in dataset.features:
                dataset = dataset.rename_column(old_name, new_name)
        if 'query' in dataset.features or 'response' in dataset.features:
            dataset = dataset.map(self.query_to_message, **kwargs)
        return dataset


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
                'preprocessor': ListPreprocessor(conversations_key='conversation', user_key='human',
                                                 assistant_key='assistant')
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
        raise ValueError(f'dataset.features.keys(): {dataset.features.keys()} '
                         f'required_keys_mapping: {required_keys_mapping}')

    def __call__(self, dataset: HfDataset, **kwargs) -> HfDataset:
        preprocessor = self._get_preprocessor(dataset)
        return preprocessor(dataset, **kwargs)


class TextGenerationPreprocessor(RowPreprocessor):

    def __init__(self,
                 *,
                 prompt: str,
                 query_key: str = 'query',
                 response_key: str = 'response',
                 **kwargs) -> None:
        self.prompt = prompt
        self.query_key = query_key
        self.response_key = response_key
        super().__init__(**kwargs)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = self.prompt.format(query=row[self.query_key])
        response = row[self.response_key]
        messages = [
            {
                'role': 'user',
                'content': query,
            },
            {
                'role': 'assistant',
                'content': response,
            }
        ]
        row = {
            'messages': messages,
        }
        return row


class ClsPreprocessor:
    # TODO

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
