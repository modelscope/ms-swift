# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
from contextlib import contextmanager
from copy import copy
from functools import partial
from multiprocessing import shared_memory
from typing import Any, Callable, Counter, Dict, List, Literal, Optional, Set, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from tqdm import tqdm

from swift.llm import Messages, history_to_messages
from swift.utils import get_logger

DATASET_TYPE = Union[HfDataset, HfIterableDataset]
PreprocessFunc = Callable[[DATASET_TYPE, ...], DATASET_TYPE]

logger = get_logger()


class GroundingMixin:
    """This class offers prompts to the grounding task"""
    task_type: Optional[str] = None

    _grounding_language_mixin = [0.8, 0.2]
    _grounding_prompts = {
        'grounding': {
            'en': [('<ref-object>', '<bbox>'), ('The positions of <ref-object> is', '<bbox>'),
                   ('Find the positions of <ref-object>', '<bbox>'), ('Where is <ref-object>', '<bbox>'),
                   ('Find <ref-object>', '<bbox>'), ('Show me <ref-object>', '<bbox>'),
                   ('Detect <ref-object>', '<bbox>'), ('Locate <ref-object>', '<bbox>'),
                   ('Tell me the location of <ref-object>', '<bbox>'), ('Give the location of <ref-object>', '<bbox>'),
                   ('Provide the bounding box coordinate of <ref-object>', '<bbox>')],
            'zh': [('<ref-object>', '<bbox>'), ('<ref-object>的位置在图片中', '<bbox>'), ('<ref-object>在图片中', '<bbox>'),
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


multimodal_tags = {
    'image': '<image>',
    'audio': '<audio>',
    'video': '<video>',
}
multimodal_keys = {
    'audio': 'audios',
    'image': 'images',
    'video': 'videos',
}


def get_dataset_features(dataset: DATASET_TYPE) -> Set[str]:
    if isinstance(dataset, HfIterableDataset) and dataset.features is None:
        features = next(iter(dataset)).keys()
    else:
        features = dataset.features.keys()
    return set(features)


standard_keys = ['messages', 'rejected_response', 'label', 'images', 'videos', 'audios', 'tools', 'objects']


def remove_useless_columns(dataset: DATASET_TYPE) -> DATASET_TYPE:
    features = get_dataset_features(dataset)
    k_list = [k for k in features if k in standard_keys]
    if len(k_list) != len(features):
        dataset = dataset.select_columns(k_list)
    return dataset


class RowPreprocessor:

    def __init__(self, columns_mapping: Optional[Dict[str, str]] = None) -> None:
        self.columns_mapping = columns_mapping or {}
        self._shared_shm_name = None
        self._column_state = None

    def empty_row(self) -> Dict[str, Any]:
        return {k: None for k in standard_keys}

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def filter(self, row: Dict[str, Any]) -> bool:
        return row['messages'] is not None

    def _row_map(self, row: Dict[str, Any], strict: bool) -> Dict[str, Any]:
        if self._shared_shm_name is not None:
            shm = shared_memory.SharedMemory(name=self._shared_shm_name)
            column_state = np.ndarray((len(standard_keys), ), dtype=np.bool_, buffer=shm.buf)
        else:
            column_state = self._column_state

        try:
            row = self.preprocess(row)
            if row is None:
                row = self.empty_row()
            elif column_state is not None:
                for i, k in enumerate(standard_keys):
                    if k in row:
                        column_state[i] = True
        except Exception:
            if strict:
                raise
            row = self.empty_row()
        return row

    @contextmanager
    def _shared_column_state(self, num_proc: int):
        """Used to remove unnecessary columns, this function is compatible with multi-processing."""
        if num_proc == 1:
            self._column_state = np.zeros((len(standard_keys), ), dtype=np.bool_)
            yield self._column_state
            self._column_state = None
            return

        shm = shared_memory.SharedMemory(create=True, size=len(standard_keys))
        self._shared_shm_name = shm.name
        column_state = np.ndarray((len(standard_keys), ), dtype=np.bool_, buffer=shm.buf)
        column_state[:] = False
        try:
            yield column_state
        finally:
            # clear resources
            shm.close()
            shm.unlink()
            self._shared_shm_name = None

    def _filter_columns(self, dataset: HfDataset, column_state: np.ndarray) -> HfDataset:
        features = get_dataset_features(dataset)
        remove_keys = []
        for i, k in enumerate(standard_keys):
            if k in features and not column_state[i]:
                remove_keys.append(k)
        dataset = dataset.remove_columns(remove_keys)
        return dataset

    def _safe_rename_columns(self, dataset: HfDataset) -> HfDataset:
        features = get_dataset_features(dataset)
        safe_columns_mapping = {k: v for k, v in self.columns_mapping.items() if k in features}
        if safe_columns_mapping:
            dataset = dataset.rename_columns(safe_columns_mapping)
        return dataset

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        strict: bool = True,
        load_from_cache_file: bool = False,
    ) -> DATASET_TYPE:
        dataset = self._safe_rename_columns(dataset)
        with self._shared_column_state(num_proc) as column_state:
            try:
                _row_map = partial(self._row_map, strict=strict)
                dataset = dataset.map(_row_map, num_proc=num_proc, load_from_cache_file=load_from_cache_file)
                dataset = dataset.filter(self.filter, num_proc=num_proc, load_from_cache_file=load_from_cache_file)
                dataset = self._filter_columns(dataset, column_state)
            except NotImplementedError:
                pass
        return dataset


class ExtraRowPreprocessor(GroundingMixin):
    """
    This class owns the data processing scenario.
    """

    has_tool: bool = False
    columns_mapping: Dict[str, str] = {}
    modals: List[str] = []  # image/video/audio
    modal_tags: List[str] = {}
    modal_keys: List[str] = {}

    def __init__(self, **kwargs):
        if 'has_tool' in kwargs:
            self.has_tool = kwargs.pop('has_tool')
        if 'columns_mapping' in kwargs:
            self.columns_mapping = kwargs.pop('columns_mapping')
        if 'modals' in kwargs:
            self.modals = kwargs.pop('modals')
        if 'modal_tags' in kwargs:
            self.modal_tags = kwargs.pop('modal_tags')
        if 'modal_keys' in kwargs:
            self.modal_keys = kwargs.pop('modal_keys')
        if self.modals:
            if not self.modal_tags:
                self.modal_tags = {modal: multimodal_tags[modal] for modal in self.modals}
            if not self.modal_keys:
                self.modal_keys = {modal: multimodal_keys[modal] for modal in self.modals}

    def replace_standard_tag(self, messages: Messages, medias: List[Any], modal: str):
        """Replace tags to standard tags,
            for example:
            query: <img>What is in the image?
            to:
            query: <image>What is in the image?
            Meanwhile, if the conversation shorts of <image>,
            this method will add equal ones to the head of the messages.
        Args:
            messages: The messages input
            medias: The medias, like images or videos
            modal: The modal, like image/video/audio
        Returns:
            Messages
        """
        assert len(self.modal_tags) == len(self.modals)
        _modal_tag = self.modal_tags[modal]
        assert _modal_tag is not None
        media_cnt = len(medias) if isinstance(medias, (tuple, list)) else 1 if medias else 0
        # like <image>, etc
        standard_tag = multimodal_tags[modal]
        all_content = ''.join([m['content'] for m in messages])
        if _modal_tag in all_content:
            # If the messages already have placeholders like `<image>`
            assert all_content.count(_modal_tag) == media_cnt
            for m in messages:
                # Replace to standard tag
                m['content'] = m['content'].replace(_modal_tag, standard_tag)
        elif '<img>' in all_content and '</img>' in all_content:
            pass
        else:
            for m in messages:
                if m['role'] not in ('tool', 'system', 'assistant'):
                    m['content'] = ''.join([standard_tag] * media_cnt) + m['content']
                    break

        return messages

    def query_to_message(self, row: Dict[str, Any]):
        """A compatible method to turn query/response to messages, this is used to fit the existing dataset_info.json"""
        messages = []
        if 'query' in row:
            query = row['query']
            if isinstance(query, list):
                query = query[np.random.choice(range(len(query)))]
            messages.append({'role': 'user', 'content': query})
        if 'response' in row:
            response = row['response']
            if isinstance(response, list):
                response = response[np.random.choice(range(len(response)))]
            messages.append({'role': 'assistant', 'content': response})
        old_messages = row.get('messages') or []
        old_messages.extend(messages)
        row['messages'] = old_messages

    def parse_media_from_row(self, row: Dict[str, Any], modal: str):
        """Parse media from a row
        Args:
            row: The row in `Dict`
            modal: The modal
        """
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
        """Please override this method"""
        return row

    def empty_row(self) -> Dict[str, Any]:
        """Generate an empty row, for later filtering"""
        row = {'messages': None}
        if self.has_tool:
            row['tools'] = None
        for _modal in self.modals:
            row[multimodal_keys[_modal]] = None
        return row

    def filter(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Filter unwanted row, by default `messages` must exist"""
        return row.get('messages')

    def rename_columns(self, dataset: DATASET_TYPE, columns_mapping: Dict[str, str]) -> DATASET_TYPE:
        """Rename columns"""
        return dataset.rename_columns(columns_mapping)

    def prepare_multi_modal(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare multi-modal.
        1. Replace tag to standard ones of all modals
        2. Construct grounding prompt and set them
        Args:
            row: The row
        Returns:
            The output dict
        """
        for modal in self.modals:
            medias = self.parse_media_from_row(row, modal)
            if 'messages' not in row:
                row['messages'] = []
            if self.task_type in self._grounding_prompts.keys():
                query, response = self.construct_grounding_prompt()
                row['messages'].extend([
                    {
                        'role': 'user',
                        'content': query
                    },
                    {
                        'role': 'assistant',
                        'content': response
                    },
                ])
            if medias:
                row['messages'] = self.replace_standard_tag(row['messages'], medias, modal)
                modal_key = self.modal_keys[modal]
                if not isinstance(modal_key, str):
                    row[multimodal_keys[modal]] = medias
                else:
                    row[modal_key] = medias
        return row

    def prepare_downloading(self, dataset: DATASET_TYPE) -> None:
        """Override this method to prepare extra resources downloading"""
        pass

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        strict: bool = True,
        load_from_cache_file: bool = False,
    ) -> DATASET_TYPE:
        """Preprocess a dataset.
        Args:
            dataset: The dataset to be mapped and filtered.
            **kwargs:
                Extra kwargs for mapping&filtering
        Returns:
            The processed dataset
        """
        maybe_multi_modal_keys = ['image', 'images', 'audio', 'audios', 'video', 'videos']
        maybe_multi_modal = any([key in dataset.features for key in maybe_multi_modal_keys])
        self.prepare_downloading(dataset)

        columns_mapping = copy(self.columns_mapping)
        # Replace un-standard media keys to standard keys
        for idx, _modal in enumerate(self.modals):
            modal_key = self.modal_keys[_modal]
            standard_key = multimodal_keys[_modal]
            if standard_key not in dataset.features:
                columns_mapping[modal_key] = standard_key

        if columns_mapping:
            dataset = self.rename_columns(dataset, columns_mapping)

        dataset = dataset.map(self.preprocess, num_proc=num_proc, load_from_cache_file=load_from_cache_file)
        dataset = dataset.filter(self.filter, **kwargs)

        all_keys = list(multimodal_keys.values())
        if (maybe_multi_modal and not self.modals) or (maybe_multi_modal
                                                       and not any([key in dataset.features for key in all_keys])):
            logger.warn('FOUND MULTI MODAL IN DATASETS, BUT NO KEY IN DATASET, MAYBE THE DATASET IS NOT CORRECT')
        if self.modals and not any([key in dataset.features for key in all_keys]):
            raise ValueError('Modals are set and no media keys set')
        return dataset


class ResponsePreprocessor(RowPreprocessor):
    """Dataset compatible with older versions of ms-swift"""

    def __init__(self, columns_mapping: Optional[Dict[str, str]] = None) -> None:
        system_keys = ['system', 'system_prompt']
        query_keys = ['query', 'prompt', 'input', 'instruction', 'question']
        response_keys = ['response', 'answer', 'output', 'targets', 'answer_key', 'text', 'completion', 'content']
        self.row_mapping = {}
        for key in system_keys:
            self.row_mapping[key] = 'system'
        for key in query_keys:
            self.row_mapping[key] = 'query'
        for key in response_keys:
            self.row_mapping[key] = 'response'
        super().__init__(columns_mapping=columns_mapping)

    @staticmethod
    def row_keys_map(row: Dict[str, Any], row_mapping: Dict[str, str]) -> Dict[str, Any]:
        # If there are multiple mappings to the same keys, then delete them.
        row_mapping = {k: v for k, v in row_mapping.items() if k in row}
        counter = Counter(row_mapping.values())

        for k, new_k in row_mapping.items():
            if counter[new_k] > 1:
                continue
            row[new_k] = row.pop(k)

        return row

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = self.row_keys_map(row, self.row_mapping)
        response = row.pop('response')
        history = row.pop('history', None) or []
        query = row.pop('query', None)
        system = row.pop('system', None)
        history.append([query, response])

        row.update({'messages': history_to_messages(history, system)})
        return row


class AlpacaPreprocessor(ResponsePreprocessor):

    def __init__(self, *, concat_inst_input: Union[Callable[[str, str], str]] = '\n', **kwargs):
        """Alpaca format preprocessor

        Args:
            concat_inst_input: The concat sep between instruction and input
        """
        self.concat_inst_input = concat_inst_input
        super().__init__(**kwargs)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        instruction = row.pop('instruction', None)
        input_ = row.pop('input', None)
        output = row.pop('output', None)
        if output is not None:
            row['response'] = output

        if instruction is None and input_ is None:
            query = None
        else:
            instruction = instruction or ''
            input_ = input_ or ''
            if isinstance(self.concat_inst_input, str):
                query = instruction + self.concat_inst_input + input_
            else:
                query = self.concat_inst_input(instruction, input_)
            row['query'] = query
        return super().preprocess(row)


def _default_repair_conversations(s: Union[str, Any]) -> Any:
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s


class MessagesPreprocessor(RowPreprocessor):

    def __init__(
            self,
            *,
            role_key: str = 'auto',  # 'role' or 'from'
            content_key: str = 'auto',  # 'content' or 'value'
            user_role: str = 'auto',  # 'user', 'human'
            assistant_role: str = 'auto',  # 'assistant', 'gpt', 'bot', ''
            system_role: str = 'system',
            tool_role: str = 'tool',
            # 'conversation', 'conversations' -> 'messages'
            columns_mapping: Union[Dict[str, str], str, None] = 'auto',
            repair_conversations: Callable[[Union[str, List[Dict[str, str]]]],
                                           Optional[List[Dict[str, str]]]] = _default_repair_conversations,
            error_strategy: Literal['delete', 'raise'] = 'raise',
            **kwargs):
        self.role_key = role_key
        self.content_key = content_key
        self.user_role = user_role
        self.assistant_role = assistant_role
        self.system_role = system_role
        self.tool_role = tool_role
        self.repair_conversations = repair_conversations
        self.error_strategy = error_strategy
        super().__init__(columns_mapping, **kwargs)

    def query_to_message(self, row: Dict[str, Any]):
        return row

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            conversations = row[self.conversations_key]
            conversations = self.repair_conversations(conversations)
            if conversations is None:
                return self.empty_row()

            messages = []
            system_message = None
            if conversations[0][self.role_key] == self.system_role:
                system_message = conversations.pop(0)

            if system_message:
                messages.append({'role': 'system', 'content': system_message[self.content_key]})
            for idx, c in enumerate(conversations):
                if idx % 2 == 0:
                    assert c[self.role_key] in [self.user_role, self.tool_role]
                    messages.append({
                        'role': 'user' if c[self.role_key] == self.user_role else 'tool',
                        'content': c[self.content_key]
                    })
                else:
                    assert c[self.role_key] == self.assistant_role
                    messages.append({'role': 'assistant', 'content': c[self.content_key]})

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
                return self.empty_row()


class SharegptPreprocessor(RowPreprocessor):

    def __init__(self,
                 *,
                 user_key: str = 'user',
                 assistant_key: str = 'assistant',
                 system_key: str = 'system',
                 tool_key: str = 'tool',
                 inner_key: str = None,
                 repair_conversations: Callable[[Union[str, Dict[str, str]]],
                                                Optional[Dict[str, str]]] = _default_repair_conversations,
                 error_strategy: Literal['delete', 'raise'] = 'raise',
                 **kwargs):
        self.user_key = user_key
        self.assistant_key = assistant_key
        self.system_key = system_key
        self.tool_key = tool_key
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
                messages.append({'role': 'system', 'content': row[self.system_key]})
            for c in conversations:
                if self.user_key in c:
                    messages.append({
                        'role': 'user',
                        'content': c[self.user_key],
                    })
                else:
                    messages.append({
                        'role': 'tool',
                        'content': c[self.tool_key],
                    })
                messages.append({
                    'role': 'assistant',
                    'content': c[self.assistant_key],
                })

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

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        strict: bool = True,
        load_from_cache_file: bool = False,
    ) -> DATASET_TYPE:
        for preprocessor in self.preprocessor_list:
            dataset = preprocessor(dataset, **kwargs)
        return dataset


class RenameColumnsPreprocessor:

    def __init__(self, rename_mapping: Dict[str, str]) -> None:
        self.rename_mapping = rename_mapping

    def query_to_message(self, row):
        messages = []
        if 'query' in row:
            query = row['query']
            if isinstance(query, list):
                query = query[np.random.choice(range(len(query)))]
            messages.append({'role': 'user', 'content': query})
        if 'response' in row:
            response = row['response']
            if isinstance(response, list):
                response = response[np.random.choice(range(len(response)))]
            messages.append({'role': 'assistant', 'content': response})
        old_messages = row.get('messages', [])
        old_messages.extend(messages)
        return {'messages': old_messages}

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        strict: bool = True,
        load_from_cache_file: bool = False,
    ) -> DATASET_TYPE:
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
                'preprocessor': ResponsePreprocessor()
            },
            'alpaca': {
                'required': ['instruction', 'input'],
                'preprocessor': AlpacaPreprocessor()
            },
            'conversations': {  # qwen
                'required': ['conversations'],
                'preprocessor': MessagesPreprocessor()
            },
            'chatml': {
                'required': ['messages'],
                'preprocessor': MessagesPreprocessor()
            },
            'sharegpt': {
                'required': ['conversation'],
                'preprocessor': SharegptPreprocessor(user_key='human', assistant_key='assistant')
            },
            'pretrain': {
                'required': ['text'],
                'preprocessor': RowPreprocessor(columns_mapping={
                    'prompt': 'query',
                    'text': 'response'
                })
            }
        }

    def _get_preprocessor(self, dataset: DATASET_TYPE) -> PreprocessFunc:
        features = get_dataset_features(dataset)
        required_keys_mapping = {k: v['required'] for k, v in self.preprocessor_mapping.items()}
        for k, required_keys in required_keys_mapping.items():
            if len(set(required_keys) - features) == 0:
                return self.preprocessor_mapping[k]['preprocessor']
        raise ValueError(f'dataset.features.keys(): {dataset.features.keys()} '
                         f'required_keys_mapping: {required_keys_mapping}')

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        strict: bool = True,
        load_from_cache_file: bool = False,
    ) -> DATASET_TYPE:
        preprocessor = self._get_preprocessor(dataset)
        return preprocessor(dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)


class TextGenerationPreprocessor(RowPreprocessor):

    def __init__(self, *, prompt: str, query_key: str = 'query', response_key: str = 'response', **kwargs) -> None:
        self.prompt = prompt
        self.query_key = query_key
        self.response_key = response_key
        super().__init__(**kwargs)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = self.prompt.format(query=row[self.query_key])
        response = row[self.response_key]
        messages = [{
            'role': 'user',
            'content': query,
        }, {
            'role': 'assistant',
            'content': response,
        }]
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

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        strict: bool = True,
        load_from_cache_file: bool = False,
    ) -> DATASET_TYPE:
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
