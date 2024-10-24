import ast
import os
import re
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import json
from datasets import Dataset as HfDataset

from ..constant import LLMDatasetName
from ..loader import DatasetLoader, DatasetSyntax
from ..preprocess import (AlpacaPreprocessor, AutoPreprocessor, ClsPreprocessor, MessagesPreprocessor,
                          ResponsePreprocessor, RowPreprocessor, TextGenerationPreprocessor)
from ..register import DatasetMeta, SubsetDataset, register_dataset


def _concat_inst_inp_alpaca_zh(inst: str, inp: str) -> str:
    if inp.startswith('输入：'):
        inp = inp[3:]
    return f'{inst}\n{inp}'


register_dataset(
    LLMDatasetName.alpaca_zh,
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/alpaca-gpt4-data-zh',
        hf_dataset_id='llm-wizard/alpaca-gpt4-data-zh',
        preprocess_func=AlpacaPreprocessor(concat_inst_input=_concat_inst_inp_alpaca_zh),
        tags=['chat', 'general', '🔥'],
    ))


class LongAlpacaPreprocessor(AlpacaPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = row['output']
        prefix_prompt = 'Answer: '
        if response and response.startswith(prefix_prompt):
            response = response[len(prefix_prompt):].strip()
            row['output'] = response
        return super().preprocess(row)


register_dataset(
    LLMDatasetName.long_alpaca_12k,
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/LongAlpaca-12k',
        hf_dataset_id='Yukang/LongAlpaca-12k',
        preprocess_func=LongAlpacaPreprocessor(),
        tags=['longlora', 'QA'],
    ))


class RuozhibaPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        title = row['title'] if row.get('title', None) is not None else row['content']
        abs = row['abs'] if 'abs' in row else None
        if abs and abs != title:
            title = title + '，' + abs

        pattern = r'\d+[\.,\s,\、](.+)'
        match = re.search(pattern, title)
        if match:
            title = match.group(1)
        if title:
            return {'messages': {'role': 'assistant', 'content': title}}


register_dataset(
    LLMDatasetName.ruozhiba,
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/ruozhiba',
        subsets=['post-annual', 'title-good', 'title-norm'],
        preprocess_func=RuozhibaPreprocessor(),
        tags=['pretrain', '🔥']))


def _repair_ms_bench(messages: str) -> Optional[List[Dict[str, str]]]:
    if isinstance(messages, str):
        messages = ast.literal_eval(messages)
    default_system = 'You are a helpful assistant.'
    messages: List[Dict[str, str]]
    if messages[0]['from'] == 'system' and messages[0]['value'] == default_system:
        messages.pop(0)
    # skip MOSS
    for c in messages:
        value = c['value'].lower()
        if 'moss' in value or 'human:' in value or 'assistant:' in value or 'user:' in value:
            return
    return messages


register_dataset(
    LLMDatasetName.ms_bench,
    DatasetMeta(
        ms_dataset_id='iic/ms_bench',
        preprocess_func=MessagesPreprocessor(repair_messages=_repair_ms_bench),
        tags=['chat', 'general', 'multi-round', '🔥']))


def _repair_agent_messages(messages: List[Dict[str, str]], use_mini: bool) -> Optional[List[Dict[str, str]]]:
    if use_mini:
        pattern = r'\d\. {"plugin_name": "(.+?)"'
        if messages[0]['from'] != 'system':
            return
        system = messages[0]['value']
        find_list = re.findall(pattern, system)
        if len(set(find_list)) <= 1:
            return
    return messages


register_dataset(
    LLMDatasetName.damo_agent_zh,
    DatasetMeta(
        ms_dataset_id='damo/MSAgent-Bench',
        subsets=[
            SubsetDataset(
                preprocess_func=MessagesPreprocessor(repair_messages=partial(_repair_agent_messages, use_mini=False))),
            SubsetDataset(
                name='mini',
                subset='default',
                preprocess_func=MessagesPreprocessor(repair_messages=partial(_repair_agent_messages, use_mini=True)),
                is_weak_subset=True)
        ],
        tags=['chat', 'agent', 'multi-round']))

advertise_gen_prompt = """Task: Generating advertisements based on keywords.
Keywords: {{QUERY}}
Advertisements:"""

register_dataset(
    LLMDatasetName.advertise_gen_zh,
    DatasetMeta(
        ms_dataset_id='lvjianjin/AdvertiseGen',
        hf_dataset_id='shibing624/AdvertiseGen',
        preprocess_func=TextGenerationPreprocessor(
            prompt=advertise_gen_prompt, columns_mapping={
                'content': 'query',
                'summary': 'response'
            }),
        tags=['text-generation', '🔥'],
    ))


class FireflyPreprocessor(ResponsePreprocessor):

    _firefly_kind_list = {
        'ProseGeneration', 'MRC', 'JinYongGeneration', 'TextCorrection', 'ClassicalChinese', 'BELLE', 'StoryGeneration',
        'Couplet', 'Cot', 'Dictionary', 'Translation', 'Program', 'SentimentAnalyze', 'OpenQA', 'AncientPoem',
        'TextMatching', 'NLI', 'Summary', 'KeywordRecognition', 'ProductDesc', 'LyricGeneration', 'Composition',
        'MusicComment', 'NER'
    }

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if row['kind'] not in FireflyPreprocessor._firefly_kind_list:
            return
        return super().preprocess(row)


register_dataset(
    LLMDatasetName.firefly_zh,
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/firefly-train-1.1M',
        hf_dataset_id='YeungNLP/firefly-train-1.1M',
        preprocess_func=FireflyPreprocessor(),
        tags=['chat', 'general'],
    ),
)

register_dataset(
    LLMDatasetName.cmnli_zh,
    DatasetMeta(
        ms_dataset_id='modelscope/clue',
        hf_dataset_id='clue',
        subsets=['cmnli'],
        preprocess_func=ClsPreprocessor(['neutral', 'entailment', 'contradiction'],
                                        task='Natural Language Inference',
                                        is_pair_seq=True),
        tags=['text-generation', 'classification'],
    ))

register_dataset(
    LLMDatasetName.jd_sentiment_zh,
    DatasetMeta(
        ms_dataset_id='DAMO_NLP/jd',
        preprocess_func=ClsPreprocessor(['negative', 'positive'], task='Sentiment Classification', is_pair_seq=False),
        tags=['text-generation', 'classification', '🔥']))
