# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import multiprocessing
from collections import Counter
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Union

from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset

from swift.llm import history_to_messages
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


class RowPreprocessor:

    def __init__(self,
                 *,
                 columns_mapping: Optional[Dict[str, str]] = None,
                 remove_useless_columns: bool = True) -> None:
        self.columns_mapping = columns_mapping or {}
        self.remove_useless_columns = remove_useless_columns
        self.shared_list = None

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        return dataset

    def _row_map(self, row: Dict[str, Any], idx: int, *, strict: bool) -> Dict[str, Any]:
        try:
            row = self.preprocess(row)
        except Exception as e:
            row = None
            if strict:
                raise
            logger.error(f'There are errors in the dataset, the data will be deleted. error: {e}')
        if row is None:
            self.shared_list.append(idx)

        return row or {'messages': None}

    def _safe_rename_columns(self, dataset: HfDataset) -> HfDataset:
        features = get_dataset_features(dataset)
        safe_columns_mapping = {k: v for k, v in self.columns_mapping.items() if k in features}
        if safe_columns_mapping:
            dataset = dataset.rename_columns(safe_columns_mapping)
        return dataset

    @staticmethod
    def _remove_useless_columns(dataset: DATASET_TYPE) -> DATASET_TYPE:
        features = get_dataset_features(dataset)
        k_list = [k for k in features if k in standard_keys]
        if len(k_list) != len(features):
            dataset = dataset.select_columns(k_list)
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
        dataset = self.prepare_dataset(dataset)
        if num_proc == 1:
            # to remove
            self.shared_list = []
        else:
            self.shared_list = multiprocessing.Manager().list()
        try:
            _row_map = partial(self._row_map, strict=strict)
            dataset = dataset.map(
                _row_map, num_proc=num_proc, load_from_cache_file=load_from_cache_file, with_indices=True)
            if len(self.shared_list) > 0:
                self.shared_list = set(self.shared_list)
                self.shared_list = [i for i in range(len(dataset)) if i not in self.shared_list]
                dataset = dataset.select(self.shared_list)
        except NotImplementedError:
            pass
        return self._remove_useless_columns(dataset)


class ResponsePreprocessor(RowPreprocessor):
    """Dataset compatible with older versions of ms-swift"""

    def __init__(self,
                 *,
                 columns_mapping: Optional[Dict[str, str]] = None,
                 remove_useless_columns: bool = True) -> None:
        system_keys = ['system', 'system_prompt']
        query_keys = ['query', 'prompt', 'input', 'instruction', 'question']
        response_keys = [
            'response', 'answer', 'output', 'targets', 'target', 'answer_key', 'text', 'completion', 'content'
        ]
        self.row_mapping = {}
        for key in system_keys:
            self.row_mapping[key] = 'system'
        for key in query_keys:
            self.row_mapping[key] = 'query'
        for key in response_keys:
            self.row_mapping[key] = 'response'
        super().__init__(columns_mapping=columns_mapping, remove_useless_columns=remove_useless_columns)

    @staticmethod
    def row_keys_map(row: Dict[str, Any], row_mapping: Dict[str, str]) -> Dict[str, Any]:
        # If there are multiple mappings to the same keys, then delete them.
        row_mapping = {k: v for k, v in row_mapping.items() if k in row}
        counter = Counter(row_mapping.values())

        for k, new_k in row_mapping.items():
            if counter[new_k] > 1:
                # For example, if "response" and "answer" match, then no processing is done.
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
                 columns_mapping: Optional[Dict[str, str]] = None,
                 remove_useless_columns: bool = True) -> None:
        """Alpaca format preprocessor

        Args:
            concat_inst_input: The concat sep between instruction and input
        """
        self.concat_inst_input = concat_inst_input
        super().__init__(columns_mapping=columns_mapping, remove_useless_columns=remove_useless_columns)

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
        remove_useless_columns: bool = True,
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
        super().__init__(columns_mapping=columns_mapping, remove_useless_columns=remove_useless_columns)

    @staticmethod
    def _is_sharegpt_format(message: Dict[str, str]) -> bool:
        if 'role' in message or 'content' in message:
            return False
        return True

    @staticmethod
    def check_message(user_message: Dict[str, str], assistant_message: Dict[str, str]) -> None:
        assert (user_message['role'] in {'user', 'tool'} and 'content' in user_message), f'user_message: {user_message}'
        assert (assistant_message['role'] in {'assistant'}
                and 'content' in assistant_message), f'assistant_message: {assistant_message}'

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
                if key in message:
                    message[std_key] = message.pop(key)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        messages = row['messages']
        if self.inner_key is not None:
            messages = messages[self.inner_key]
        messages: Optional[List[Dict[str, str]]] = self.repair_messages(messages)
        if not messages:
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

    def __init__(self,
                 *,
                 columns_mapping: Optional[Dict[str, str]] = None,
                 remove_useless_columns: bool = True) -> None:
        self.columns_mapping = columns_mapping
        self.remove_useless_columns = remove_useless_columns

    def _get_preprocessor(self, dataset: DATASET_TYPE) -> RowPreprocessor:
        features = get_dataset_features(dataset)
        for key in ['conversation', 'conversations', 'messages']:
            if key in features:
                return MessagesPreprocessor(
                    columns_mapping=self.columns_mapping, remove_useless_columns=self.remove_useless_columns)
        if 'instruction' in features and 'input' in features:
            return AlpacaPreprocessor(
                columns_mapping=self.columns_mapping, remove_useless_columns=self.remove_useless_columns)
        return ResponsePreprocessor(
            columns_mapping=self.columns_mapping, remove_useless_columns=self.remove_useless_columns)

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
