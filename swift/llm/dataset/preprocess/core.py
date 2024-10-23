# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
from collections import Counter
from contextlib import contextmanager
from copy import copy
from functools import partial
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset

from swift.llm import Messages, history_to_messages
from swift.utils import get_logger

DATASET_TYPE = Union[HfDataset, HfIterableDataset]

logger = get_logger()


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

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

    @staticmethod
    def _filter_columns(dataset: HfDataset, column_state: np.ndarray) -> HfDataset:
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

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.row_keys_map(row, self.row_mapping)
        response = row.pop('response', None)
        if response is None:
            return
        history = row.pop('history', None) or []
        query = row.pop('query', None)
        system = row.pop('system', None)
        if isinstance(history, str):  # e.g. "[['query1', 'response1']]"
            history = ast.literal_eval(history)
        history.append([query, response])

        row.update({'messages': history_to_messages(history, system)})
        return row


class AlpacaPreprocessor(ResponsePreprocessor):

    def __init__(self,
                 *,
                 concat_inst_input: Union[Callable[[str, str], str]] = '\n',
                 columns_mapping: Optional[Dict[str, str]] = None) -> None:
        """Alpaca format preprocessor

        Args:
            concat_inst_input: The concat sep between instruction and input
        """
        self.concat_inst_input = concat_inst_input
        super().__init__(columns_mapping=columns_mapping)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        instruction = row.pop('instruction', None)
        input_ = row.pop('input', None)
        output = row.pop('output', None)
        if output is not None:
            row['response'] = output

        if instruction is not None or input_ is not None:
            instruction = instruction or ''
            input_ = input_ or ''
            if isinstance(self.concat_inst_input, str):
                query = instruction + self.concat_inst_input + input_
            else:
                query = self.concat_inst_input(instruction, input_)
            row['query'] = query
        return super().preprocess(row)


def default_repair_messages(s: Union[str, Any]) -> Any:
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s


class MessagesPreprocessor(RowPreprocessor):

    def __init__(
        self,
        *,
        # If set to None, automatic matching will be performed.
        role_key: Optional[str] = None,  # 'role', 'from'
        content_key: Optional[str] = None,  # 'content', 'value'
        user_role: Optional[str] = None,  # 'user', 'human'
        assistant_role: Optional[str] = None,  # 'assistant', 'gpt', 'bot'
        system_role: str = 'system',
        tool_role: str = 'tool',
        # 'conversation', 'conversations' -> 'messages'
        columns_mapping: Optional[Dict[str, str]] = None,
        repair_messages: Callable[[Union[str, List[Dict[str, str]]]],
                                  Optional[List[Dict[str, str]]]] = default_repair_messages,
        inner_key: Optional[str] = None,
    ):
        self.role_keys = ['role', 'from'] if role_key is None else [role_key]
        self.content_keys = ['content', 'value'] if content_key is None else [content_key]
        self.user_roles = ['user', 'human'] if user_role is None else [user_role]
        self.assistant_roles = ['assistant', 'gpt', 'bot'] if assistant_role is None else [assistant_role]

        self.system_role = system_role
        self.tool_role = tool_role
        self.repair_messages = repair_messages
        self.inner_key = inner_key
        if columns_mapping is None:
            columns_mapping = {
                'conversation': 'messages',
                'conversations': 'messages',
                # sharegpt
                self.system_role: 'system'
            }
        super().__init__(columns_mapping=columns_mapping)

    @staticmethod
    def _is_sharegpt_format(message: Dict[str, str]) -> bool:
        if 'role' in message or 'content' in message:
            return False
        return True

    @staticmethod
    def check_message(user_message: Dict[str, str], assistant_message: Dict[str, str]) -> None:
        assert user_message['role'] in {'user', 'tool'}, f'user_message: {user_message}'
        assert assistant_message['role'] in {'assistant'}, f'assistant_message: {assistant_message}'

    def sharegpt_to_messages(self, messages: List[Dict[str, str]], system: Optional[str]) -> List[Dict[str, str]]:
        self._to_std_key(messages, 'user', self.user_roles)
        self._to_std_key(messages, 'assistant', self.assistant_roles)
        new_messages = []
        if system is not None:
            new_messages.append({'role': 'system', 'content': system})
        for message in messages:
            if self.tool_role in message:
                user_message = {'role': 'tool', 'content': message[self.tool_role]}
            else:
                user_message = {'role': 'user', 'content': message['user']}
            assistant_message = {'role': 'assistant', 'content': message['assistant']}
            new_messages.append(user_message)
            new_messages.append(assistant_message)
        return new_messages

    def to_std_messages(self, messages: List[Dict[str, str]]) -> None:
        start_idx = 0
        if messages[0]['role'] == self.system_role:
            messages[0]['role'] = 'system'
            start_idx = 1
        for user_message, assistant_message in zip(messages[start_idx::2], messages[start_idx + 1::2]):
            user_role = user_message['role']
            assistant_role = assistant_message['role']
            if user_role in self.user_roles:
                user_message['role'] = 'user'
            elif user_role == self.tool_role:
                user_message['role'] = 'tool'
            if assistant_role in self.assistant_roles:
                assistant_message['role'] = 'assistant'
            self.check_message(user_message, assistant_message)

    @staticmethod
    def _to_std_key(messages: List[Dict[str, str]], std_key: str, optional_keys: List[str]) -> None:
        for message in messages:
            for key in optional_keys:
                if key in messages:
                    message[std_key] = message.pop(key)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        messages = row['messages']
        if self.inner_key is not None:
            messages = messages[self.inner_key]
        messages: Optional[List[Dict[str, str]]] = self.repair_messages(messages)
        if messages:
            return
        self._to_std_key(messages, 'role', self.role_keys)
        self._to_std_key(messages, 'content', self.content_keys)
        if self._is_sharegpt_format(messages[0]):
            system = row.pop('system', None)
            messages = self.sharegpt_to_messages(messages, system)
        else:
            self.to_std_messages(messages)  # inplace
        row['messages'] = messages
        return row


class AutoPreprocessor:

    def _get_preprocessor(self, dataset: DATASET_TYPE) -> RowPreprocessor:
        features = get_dataset_features(dataset)
        for key in ['conversation', 'conversations', 'messages']:
            if key in features:
                return MessagesPreprocessor()
        if 'instruction' in features and 'input' in features:
            return AlpacaPreprocessor()
        return ResponsePreprocessor()

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
