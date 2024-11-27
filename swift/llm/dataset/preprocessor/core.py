# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
from collections import Counter
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import Image
from datasets import IterableDataset as HfIterableDataset
from datasets import Value

from swift.llm import history_to_messages
from swift.utils import get_logger

DATASET_TYPE = Union[HfDataset, HfIterableDataset]

logger = get_logger()


def get_dataset_features(dataset: DATASET_TYPE) -> Dict[str, Any]:
    features = dataset.features
    if features is None:
        assert isinstance(dataset, HfIterableDataset)
        dataset = dataset._resolve_features()
        features = dataset.features
    return features


class RowPreprocessor:
    cast_mm_data = True

    def __init__(self,
                 *,
                 columns_mapping: Optional[Dict[str, str]] = None,
                 dataset_sample: Optional[int] = None,
                 random_state: Union[np.random.RandomState, int, None] = None,
                 traceback_limit: int = 10) -> None:
        self.columns_mapping = columns_mapping or {}
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
            if (user_message['role'] not in {'user', 'tool'} or 'content' not in user_message
                    or user_message['content'] is None):
                raise ValueError(f'user_message: {user_message}')
            if (assistant_message['role'] not in {'assistant'} or 'content' not in assistant_message
                    or assistant_message['content'] in {'', None}):
                raise ValueError(f'assistant_message: {assistant_message}')

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
            if rejected_response is None or rejected_response == messages[-1]['content']:
                raise ValueError(f'rejected_response: {rejected_response}')

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        return dataset

    @staticmethod
    def batched_to_rows(batched_row: Dict[str, Any]):
        keys = list(batched_row.keys())
        batch_size = len(batched_row[keys[0]])
        return [{key: batched_row[key][i] for key in keys} for i in range(batch_size)]

    @staticmethod
    def rows_to_batched(rows: List[Dict[str, Any]]):
        batched = {}
        for i, row in enumerate(rows):
            for k, v in row.items():
                if k not in batched:
                    batched[k] = [None] * i
                batched[k].append(v)
        return batched

    def batched_preprocess(self, batched_row: Dict[str, Any], *, strict: bool) -> Dict[str, Any]:
        batched_row = dict(batched_row)
        assert len(batched_row) > 0
        self.row_keys_map(batched_row, self.row_mapping)
        rows = self.batched_to_rows(batched_row)

        new_rows = []
        for row in rows:
            try:
                row = self.preprocess(row)
                if row is not None:
                    self.check_messages(row)
                    self.check_rejected_response(row)
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
            new_rows.append(row)
        res = self.rows_to_batched(new_rows)

        if len(res) == 0:
            res['messages'] = []

        return res

    @staticmethod
    def safe_rename_columns(dataset: HfDataset, columns_mapping: Dict[str, Any]) -> HfDataset:
        features = get_dataset_features(dataset).keys()
        safe_columns_mapping = {k: v for k, v in columns_mapping.items() if k in features}
        if safe_columns_mapping:
            dataset = dataset.rename_columns(safe_columns_mapping)
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
        try:
            yield
        finally:
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
        batch_size: int = 1000,
    ) -> DATASET_TYPE:
        from ..utils import sample_dataset
        if self.dataset_sample is not None:
            dataset = sample_dataset(dataset, self.dataset_sample, self.random_state)
        dataset = self.safe_rename_columns(dataset, self.columns_mapping)
        dataset = self.prepare_dataset(dataset)
        dataset = self._cast_mm_data(dataset, False)
        map_kwargs = {}
        if isinstance(dataset, HfDataset):
            map_kwargs.update({'num_proc': num_proc, 'load_from_cache_file': load_from_cache_file})
        with self._patch_arrow_writer():
            try:
                dataset_mapped = dataset.map(
                    self.batched_preprocess,
                    batched=True,
                    batch_size=batch_size,
                    fn_kwargs={'strict': strict},
                    remove_columns=list(get_dataset_features(dataset).keys()),
                    **map_kwargs)
            except NotImplementedError:
                pass
        if isinstance(dataset_mapped, HfDataset) and len(dataset) != len(dataset_mapped):
            logger.info(
                f'Dataset filtered, origin length: {len(dataset)}, filtered dataset length: {len(dataset_mapped)}')

        dataset_mapped = self._cast_mm_data(dataset_mapped, True)
        return dataset_mapped


class ResponsePreprocessor(RowPreprocessor):
    """Dataset compatible with older versions of ms-swift"""

    def __init__(self, *, columns_mapping: Optional[Dict[str, str]] = None, **kwargs) -> None:
        super().__init__(columns_mapping=columns_mapping, **kwargs)
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
                 **kwargs) -> None:
        """Alpaca format preprocessor

        Args:
            concat_inst_input: The concat sep between instruction and input
        """
        super().__init__(columns_mapping=columns_mapping, **kwargs)
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
            **kwargs):
        super().__init__(columns_mapping=columns_mapping, **kwargs)
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

    def __init__(self, *, columns_mapping: Optional[Dict[str, str]] = None, **kwargs) -> None:
        self.columns_mapping = columns_mapping or {}
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
