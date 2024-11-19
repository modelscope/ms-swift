# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import multiprocessing
from collections import Counter
from contextlib import contextmanager
from functools import partial
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import Features, Image
from datasets import IterableDataset as HfIterableDataset
from datasets import Value

from swift.llm import history_to_messages
from swift.utils import get_logger

DATASET_TYPE = Union[HfDataset, HfIterableDataset]

logger = get_logger()


def get_dataset_features(dataset: DATASET_TYPE) -> Dict[str, Any]:
    if isinstance(dataset, HfIterableDataset) and dataset.features is None:
        features = next(iter(dataset))
    else:
        features = dataset.features
    return features


standard_keys = ['messages', 'rejected_response', 'label', 'images', 'videos', 'audios', 'tools', 'objects']


class RowPreprocessor:

    standard_keys = standard_keys
    cast_mm_data = True

    def __init__(self,
                 *,
                 columns_mapping: Optional[Dict[str, str]] = None,
                 remove_useless_columns: bool = True,
                 dataset_sample: Optional[int] = None,
                 random_state: Union[np.random.RandomState, int, None] = None,
                 traceback_limit: int = 10) -> None:
        self.columns_mapping = columns_mapping or {}
        self.remove_useless_columns = remove_useless_columns
        self.row_mapping = {}
        self.traceback_limit = traceback_limit
        self._traceback_counter = 0
        self.dataset_sample = dataset_sample
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

    @staticmethod
    def check_messages(row: Dict[str, Any]) -> None:
        if 'messages' not in row:
            return
        messages = row['messages']
        assert len(messages) > 0, f'messages: {messages}'
        if messages[0]['role'] == 'system':
            messages = messages[1:]
        for user_message, assistant_message in zip(messages[::2], messages[1::2]):
            assert (user_message['role'] in {'user', 'tool'} and 'content' in user_message
                    and user_message['content'] is not None), f'user_message: {user_message}'
            assert (assistant_message['role'] in {'assistant'} and 'content' in assistant_message
                    and assistant_message['content']), f'assistant_message: {assistant_message}'

    @staticmethod
    def check_rejected_response(row: Dict[str, Any]) -> None:
        if 'rejected_messages' in row:
            chosen_messages = row['messages']
            rejected_messages = row['rejected_messages']
            messages = []
            rejected_response = None
            for chosen_user, chosen_assistant, rejected_user, rejected_assistant in zip(
                    chosen_messages[::2], chosen_messages[1::2], rejected_messages[::2], rejected_messages[1::2]):
                assert chosen_user == rejected_user
                messages.append(chosen_user)
                messages.append(chosen_assistant)
                if chosen_assistant != rejected_assistant:
                    rejected_response = rejected_assistant['content']
            row['messages'] = messages
            row['rejected_response'] = rejected_response

        if 'rejected_response' in row:
            messages = row['messages']
            rejected_response = row['rejected_response']
            assert rejected_response is not None and rejected_response != messages[-1][
                'content'], f'rejected_response: {rejected_response}'

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        return dataset

    def _row_map(self, batched_row: Dict[str, Any], *, strict: bool) -> Dict[str, Any]:
        batched_row = dict(batched_row)
        self.row_keys_map(batched_row, self.row_mapping)
        keys = list(batched_row.keys())
        if len(keys) == 0:
            return {}

        batch_size = len(batched_row[keys[0]])
        res = {}
        num_samples = 0
        for i in range(batch_size):
            row = {key: batched_row[key][i] for key in keys}

            try:
                row = self.preprocess(row)
                if row is not None:
                    self.check_rejected_response(row)
                    self.check_messages(row)
            except Exception:
                if strict:
                    logger.warning('To avoid errors, you can pass `strict=False`.')
                    raise
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    import traceback
                    print(traceback.format_exc())
                    logger.error('👆👆👆There are errors in the dataset, the data will be deleted')
                    self._traceback_counter += 1
                row = None
            if row is None:
                continue

            for k, v in row.items():
                if k not in res:
                    res[k] = [None] * num_samples
                res[k].append(v)

            num_samples += 1

        return res

    @staticmethod
    def safe_rename_columns(dataset: HfDataset, columns_mapping: Dict[str, Any]) -> HfDataset:
        features = get_dataset_features(dataset).keys()
        safe_columns_mapping = {k: v for k, v in columns_mapping.items() if k in features}
        if safe_columns_mapping:
            dataset = dataset.rename_columns(safe_columns_mapping)
        return dataset

    @classmethod
    def _remove_useless_columns(cls, dataset: DATASET_TYPE) -> DATASET_TYPE:
        features = get_dataset_features(dataset).keys()
        k_list = [k for k in features if k in cls.standard_keys]
        if len(k_list) != len(features):
            dataset = dataset.select_columns(k_list)
        return dataset

    @staticmethod
    def row_keys_map(row: Dict[str, Any], row_mapping: Dict[str, str]) -> None:
        # If there are multiple mappings to the same keys, then delete them.
        row_keys = {k.lower(): k for k in row.keys()}
        row_mapping = {row_keys[k]: v for k, v in row_mapping.items() if k in row_keys}
        counter = Counter(row_mapping.values())

        for k, new_k in row_mapping.items():
            if counter[new_k] > 1:
                # For example, if "response" and "answer" match, then no processing is done.
                continue
            row[new_k] = row.pop(k)

    @staticmethod
    @contextmanager
    def _patch_arrow_writer():
        # fix AI-ModelScope/ms_agent_for_agentfabric:all
        from datasets.arrow_writer import ArrowWriter

        def _new_init(self, schema=None, features=None, *args, **kwargs):

            if features is not None:
                features['messages'] = [{
                    'role': Value(dtype='string', id=None),
                    'content': Value(dtype='string', id=None)
                }]
            ArrowWriter.__origin_init__(self, schema, features, *args, **kwargs)

        ArrowWriter.__origin_init__ = ArrowWriter.__init__
        ArrowWriter.__init__ = _new_init
        yield
        ArrowWriter.__init__ = ArrowWriter.__origin_init__
        del ArrowWriter.__origin_init__

    def _cast_mm_data(self, dataset, decode: bool):
        if not self.cast_mm_data:
            return dataset
        features = get_dataset_features(dataset)
        for key in ['images', 'videos', 'audios']:
            if key in features and isinstance(features[key], Image) and features[key].decode != decode:
                dataset = dataset.cast_column(key, Image(decode=decode))
        return dataset

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        strict: bool = True,
        load_from_cache_file: bool = False,
    ) -> DATASET_TYPE:
        from ..utils import sample_dataset
        if self.dataset_sample is not None:
            dataset = sample_dataset(dataset, self.dataset_sample, self.random_state)
        dataset = self.safe_rename_columns(dataset, self.columns_mapping)
        dataset = self.prepare_dataset(dataset)
        _row_map = partial(self._row_map, strict=strict)

        dataset = self._cast_mm_data(dataset, False)
        with self._patch_arrow_writer():
            try:
                dataset_mapped = dataset.map(
                    _row_map,
                    num_proc=num_proc,
                    load_from_cache_file=load_from_cache_file,
                    batched=True,
                    remove_columns=list(get_dataset_features(dataset).keys()))
            except NotImplementedError:
                pass
        dataset_mapped = self._cast_mm_data(dataset_mapped, True)

        if hasattr(dataset, '__len__') and len(dataset) != len(dataset_mapped):
            logger.info(
                f'Dataset filtered, origin length: {len(dataset)}, filtered dataset length: {len(dataset_mapped)}')
        if self.remove_useless_columns:
            dataset_mapped = self._remove_useless_columns(dataset_mapped)
        return dataset_mapped


class ResponsePreprocessor(RowPreprocessor):
    """Dataset compatible with older versions of ms-swift"""

    def __init__(self,
                 *,
                 columns_mapping: Optional[Dict[str, str]] = None,
                 remove_useless_columns: bool = True,
                 **kwargs) -> None:
        super().__init__(columns_mapping=columns_mapping, remove_useless_columns=remove_useless_columns, **kwargs)
        system_keys = ['system', 'system_prompt']
        query_keys = ['query', 'prompt', 'input', 'instruction', 'question']
        response_keys = ['response', 'answer', 'output', 'targets', 'target', 'answer_key', 'solution'
                         ] + ['text', 'completion', 'content']
        for key in system_keys:
            self.row_mapping[key] = 'system'
        for key in query_keys:
            self.row_mapping[key] = 'query'
        for key in response_keys:
            self.row_mapping[key] = 'response'

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = row.pop('response', None)
        if response is None:
            row.pop('query', None)
            row.pop('history', None)
            row.pop('system', None)
            return
        if isinstance(response, (list, tuple)):
            # sometimes response is a list, pick one randomly
            response = self.random_state.choice(response)
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
                 remove_useless_columns: bool = True,
                 **kwargs) -> None:
        """Alpaca format preprocessor

        Args:
            concat_inst_input: The concat sep between instruction and input
        """
        super().__init__(columns_mapping=columns_mapping, remove_useless_columns=remove_useless_columns, **kwargs)
        self.concat_inst_input = concat_inst_input

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
            **kwargs):
        super().__init__(columns_mapping=columns_mapping, remove_useless_columns=remove_useless_columns, **kwargs)
        self.role_keys = ['role', 'from'] if role_key is None else [role_key]
        self.content_keys = ['content', 'value'] if content_key is None else [content_key]
        self.user_roles = ['user', 'human'] if user_role is None else [user_role]
        self.assistant_roles = ['assistant', 'gpt', 'bot'] if assistant_role is None else [assistant_role]

        self.system_role = system_role
        self.tool_role = tool_role
        self.repair_messages = repair_messages
        self.inner_key = inner_key

        message_keys = ['messages', 'conversation', 'conversations']
        for key in message_keys:
            self.row_mapping[key] = 'messages'
        # sharegptq
        system_keys = ['system', 'system_prompt']
        if system_role not in system_keys:
            system_keys.append(system_role)
        for key in system_keys:
            self.row_mapping[key] = 'system'

    @staticmethod
    def _is_sharegpt_format(message: Dict[str, str]) -> bool:
        if 'role' in message or 'content' in message:
            return False
        return True

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
        if start_idx == 1 and len(messages) % 2 == 0:
            raise ValueError(f'The messages length is not even: {messages}')
        for user_message, assistant_message in zip(messages[start_idx::2], messages[start_idx + 1::2]):
            user_role = user_message['role']
            assistant_role = assistant_message['role']
            if user_role in self.user_roles:
                user_message['role'] = 'user'
            elif user_role == self.tool_role:
                user_message['role'] = 'tool'
            if assistant_role in self.assistant_roles:
                assistant_message['role'] = 'assistant'

    @staticmethod
    def _to_std_key(messages: List[Dict[str, str]], std_key: str, optional_keys: List[str]) -> None:
        for message in messages:
            for key in optional_keys:
                if key in message:
                    message[std_key] = message.pop(key)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if 'rejected_messages' in row:
            row['rejected_messages'] = MessagesPreprocessor.preprocess(
                self, {'messages': row['rejected_messages']})['messages']
        messages = row['messages']
        if self.inner_key is not None:
            messages = messages[self.inner_key]
        messages: Optional[List[Dict[str, str]]] = self.repair_messages(messages)
        if not messages or isinstance(messages, str):
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
                 remove_useless_columns: bool = True,
                 **kwargs) -> None:
        self.columns_mapping = columns_mapping or {}
        kwargs['remove_useless_columns'] = remove_useless_columns
        self.kwargs = kwargs

    def _get_preprocessor(self, dataset: DATASET_TYPE) -> RowPreprocessor:
        features = get_dataset_features(dataset).keys()
        for key in ['conversation', 'conversations', 'messages']:
            if key in features:
                return MessagesPreprocessor(**self.kwargs)
        if 'instruction' in features and 'input' in features:
            return AlpacaPreprocessor(**self.kwargs)
        return ResponsePreprocessor(**self.kwargs)

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        strict: bool = True,
        load_from_cache_file: bool = False,
    ) -> DATASET_TYPE:
        dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns_mapping)
        preprocessor = self._get_preprocessor(dataset)
        return preprocessor(dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)
