# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from datasets import Dataset as HfDataset
from tqdm import tqdm

from .template import History

PreprocessFunc = Callable[[HfDataset], HfDataset]


class SwiftPreprocessor:

    def __call__(self, dataset: HfDataset) -> HfDataset:
        if 'history' in dataset.features:
            old_history = dataset['history']

            history: List[History] = []
            for old_h in tqdm(old_history):
                if isinstance(old_h, list):
                    break
                h = None
                if old_h is not None:
                    h = ast.literal_eval(old_h)
                history.append(h)
            else:
                dataset = dataset.remove_columns(['history']).add_column('history', history)
        return dataset


class AlpacaPreprocessor:

    def __init__(self, concat_inst_inp: Optional[Callable[[str, str], str]] = None):
        self.concat_inst_inp = concat_inst_inp

    def __call__(self, dataset: HfDataset) -> HfDataset:
        query: List[str] = []
        response = []
        system = None
        history = None
        for i, d in enumerate(tqdm(dataset)):
            inst, inp = d['instruction'], d.get('input', None)
            h, output = d.pop('history', None), d['output']
            sys = d.pop('system', None)
            if history is None and h is not None:
                history = [None for _ in range(i - 1)]
            if system is None and sys is not None:
                system = [None for _ in range(i - 1)]
            if output is None:
                continue
            if inp is None or len(inp) == 0:
                q = inst
            elif self.concat_inst_inp is not None:
                q = self.concat_inst_inp(inst, inp)
            else:
                q = f'{inst}\n{inp}'
            query.append(q)
            response.append(output)
            if history is not None:
                history.append(h)
            if system is not None:
                system.append(sys)
        d_dict = {'query': query, 'response': response}
        if history is not None:
            d_dict['history'] = history
        if system is not None:
            d_dict['system'] = system
        dataset = HfDataset.from_dict(d_dict)
        return dataset


def _default_repair_conversations(s: Union[str, Any]) -> Any:
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s


class ConversationsPreprocessor:

    def __init__(self,
                 user_role: str = 'user',
                 assistant_role: str = 'assistant',
                 system_role: str = 'system',
                 conversations_key: str = 'conversations',
                 from_key: str = 'from',
                 value_key: str = 'value',
                 repair_conversations: Callable[[Union[str, Dict[str, str]]],
                                                Optional[Dict[str, str]]] = _default_repair_conversations,
                 error_strategy: Literal['delete', 'raise'] = 'raise'):
        self.user_role = user_role
        self.assistant_role = assistant_role
        self.system_role = system_role
        self.conversations_key = conversations_key
        self.from_key = from_key
        self.value_key = value_key
        self.repair_conversations = repair_conversations
        self.error_strategy = error_strategy

    def __call__(self, dataset: HfDataset) -> HfDataset:
        query: List[str] = []
        response: List[str] = []
        system: List[Optional[str]] = []
        has_system = False
        history: List[History] = []
        has_history = False

        for d in tqdm(dataset):
            try:
                conversations = d[self.conversations_key]
                conversations = self.repair_conversations(conversations)
                if conversations is None:
                    continue
                lo = 0
                sys = None
                h: History = []
                assert len(conversations) >= 2
                if conversations[0][self.from_key] == self.system_role:
                    has_system = True
                    lo += 1
                    sys = conversations[0][self.value_key]
                assert conversations[-2][self.from_key] == self.user_role
                assert conversations[-1][self.from_key] == self.assistant_role

                for q, r in zip(conversations[lo:-2:2], conversations[lo + 1:-2:2]):
                    assert q[self.from_key] == self.user_role
                    assert r[self.from_key] == self.assistant_role
                    h.append([q[self.value_key], r[self.value_key]])
                if len(h) > 0:
                    has_history = True
                query.append(conversations[-2][self.value_key])
                response.append(conversations[-1][self.value_key])
                system.append(sys)
                history.append(h)
            except (AssertionError, SyntaxError):
                if self.error_strategy == 'raise':
                    raise ValueError(f'conversations: {conversations}')
        kwargs = {}
        if has_system:
            kwargs['system'] = system
        if has_history:
            kwargs['history'] = history
        kwargs.update({
            'query': query,
            'response': response,
        })
        dataset = HfDataset.from_dict({**kwargs})
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
            dataset = dataset.rename_column(old_name, new_name)
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
            'conversations': {
                'required': ['conversations'],
                'preprocessor': ConversationsPreprocessor()
            },
            'chatml': {
                'required': ['messages'],
                'preprocessor':
                ConversationsPreprocessor(conversations_key='messages', from_key='role', value_key='content')
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
