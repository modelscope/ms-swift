# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import re
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import numpy as np

from ...template import split_str_parts_by
from ..preprocessor import (AlpacaPreprocessor, ClsGenerationPreprocessor, ClsPreprocessor, MessagesPreprocessor,
                            ResponsePreprocessor, RowPreprocessor, TextGenerationPreprocessor)
from ..register import DatasetMeta, SubsetDataset, register_dataset


class AlpacaZhPreprocessor(AlpacaPreprocessor):

    @classmethod
    def concat_inst_input(cls, instruction, input_):
        if input_ and input_.startswith('输入：'):
            input_ = input_[3:]
        return super().concat_inst_input(instruction, input_)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/alpaca-gpt4-data-zh',
        hf_dataset_id='llm-wizard/alpaca-gpt4-data-zh',
        preprocess_func=AlpacaZhPreprocessor(),
        tags=['chat', 'general', '🔥'],
    ))


class LongAlpacaPreprocessor(AlpacaPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = row['response']
        prefix_prompt = 'Answer: '
        if response and response.startswith(prefix_prompt):
            response = response[len(prefix_prompt):].strip()
            row['output'] = response
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/LongAlpaca-12k',
        hf_dataset_id='Yukang/LongAlpaca-12k',
        preprocess_func=LongAlpacaPreprocessor(),
        tags=['long-sequence', 'QA'],
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
            return {'messages': [{'role': 'assistant', 'content': title}]}


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/ruozhiba',
        subsets=['post-annual', 'title-good', 'title-norm'],
        preprocess_func=RuozhibaPreprocessor(),
        tags=['pretrain', '🔥']))


class MathTrnPreprocessor(ResponsePreprocessor):

    def preprocess(self, row):
        query = row['query']
        output = row['response']
        row = {
            'query': query,
            'response': output,
        }
        return super().preprocess(row)


register_dataset(
    DatasetMeta(ms_dataset_id='AI-ModelScope/math-trn-format', preprocess_func=MathTrnPreprocessor(), tags=['math']))


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
    DatasetMeta(
        ms_dataset_id='damo/MSAgent-Bench',
        subsets=[
            SubsetDataset(
                preprocess_func=MessagesPreprocessor(repair_messages=partial(_repair_agent_messages, use_mini=False))),
            SubsetDataset(
                name='mini',
                preprocess_func=MessagesPreprocessor(repair_messages=partial(_repair_agent_messages, use_mini=True)),
                is_weak_subset=True)
        ],
        split=['train', 'validation'],
        tags=['chat', 'agent', 'multi-round']))

advertise_gen_prompt = """Task: Generating advertisements based on keywords.
Keywords: {{QUERY}}
Advertisements:"""

register_dataset(
    DatasetMeta(
        ms_dataset_id='lvjianjin/AdvertiseGen',
        hf_dataset_id='shibing624/AdvertiseGen',
        preprocess_func=TextGenerationPreprocessor(
            prompt=advertise_gen_prompt, columns={
                'content': 'query',
                'summary': 'response'
            }),
        tags=['text-generation', '🔥'],
        split=['train', 'validation'],
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
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/firefly-train-1.1M',
        hf_dataset_id='YeungNLP/firefly-train-1.1M',
        preprocess_func=FireflyPreprocessor(),
        tags=['chat', 'general'],
    ))

register_dataset(
    DatasetMeta(
        ms_dataset_id='modelscope/clue',
        hf_dataset_id='clue',
        subsets=['cmnli'],
        preprocess_func=ClsGenerationPreprocessor(['neutral', 'entailment', 'contradiction'],
                                                  task='Natural Language Inference',
                                                  is_pair_seq=True),
        tags=['text-generation', 'classification'],
        split=['train', 'validation'],
    ))

register_dataset(
    DatasetMeta(
        ms_dataset_id='DAMO_NLP/jd',
        subsets=[
            SubsetDataset(
                'default',
                'default',
                preprocess_func=ClsGenerationPreprocessor(['negative', 'positive'],
                                                          task='Sentiment Classification',
                                                          is_pair_seq=False)),
            SubsetDataset(
                'cls',
                'default',
                preprocess_func=ClsPreprocessor(columns={'sentence': 'query'}),
            ),
        ],
        tags=['text-generation', 'classification', '🔥'],
        split=['train', 'validation'],
    ))


class SyntheticText2SqlPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        sql_prompt = row['sql_prompt']
        sql_context = row['sql_context']
        sql = row['sql']
        sql_explanation = row['sql_explanation']
        query = f'Sql Table information:\n{sql_context}\n{sql_prompt}'
        response = f'Let\'s think step by step:\n{sql_explanation}\nSo the final sql is:\n{sql}'
        return super().preprocess({'query': query, 'response': response})


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/synthetic_text_to_sql',
        hf_dataset_id='gretelai/synthetic_text_to_sql',
        preprocess_func=SyntheticText2SqlPreprocessor(),
        tags=['nl2sql', 'en']))


def _repair_toolbench(conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
    assert len(conversations) == 2
    if conversations[1]['from'] in {'caller', 'conclusion'}:
        conversations[1]['from'] = 'assistant'
    return conversations


register_dataset(
    DatasetMeta(
        ms_dataset_id='shenweizhou/alpha-umi-toolbench-processed-v2',
        subsets=['backbone', 'caller', 'planner', 'summarizer'],
        preprocess_func=MessagesPreprocessor(repair_messages=_repair_toolbench),
        tags=['chat', 'agent', '🔥'],
        huge_dataset=True))


class BlossomMathPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        output, answer = row['output'], row['answer']
        return super().preprocess({'query': row['query'], 'response': f'{output}\n\nAnswer: {answer}'})


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/blossom-math-v2',
        hf_dataset_id='Azure99/blossom-math-v2',
        preprocess_func=BlossomMathPreprocessor(),
        tags=['chat', 'math', '🔥']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/sql-create-context',
        hf_dataset_id='b-mc2/sql-create-context',
        preprocess_func=AlpacaPreprocessor(columns={
            'question': 'instruction',
            'context': 'input',
            'answer': 'output'
        }),
        tags=['chat', 'sql', '🔥']))


class TigerBotLawPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = """{type}
{title}
"""
        cur_prompt = prompt.format(type=row['type'], title=row['title'])
        for i in range(1, 4):
            chapter = row[f'chapter{i}']
            if chapter is not None:
                cur_prompt += f'{chapter}'
        cur_prompt += f'{row["response"]}'
        return super().preprocess({'response': cur_prompt})


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/tigerbot-law-plugin',
        hf_dataset_id='TigerResearch/tigerbot-law-plugin',
        preprocess_func=TigerBotLawPreprocessor(),
        tags=['text-generation', 'law', 'pretrained']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='codefuse-ai/CodeExercise-Python-27k',
        preprocess_func=MessagesPreprocessor(columns={'chat_rounds': 'messages'}),
        tags=['chat', 'coding', '🔥']))


class LeetcodePythonPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        code_with_problem = row['code_with_problem']
        idx = code_with_problem.find('```python')
        problem = code_with_problem[:idx]
        if problem.startswith('# '):
            problem = problem[2:]
        code = code_with_problem[idx:].strip()
        explanation = row['explanation_only']
        return super().preprocess({'query': problem, 'response': f'{code}\n\n{explanation}'})


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/leetcode-solutions-python',
        preprocess_func=LeetcodePythonPreprocessor(),
        tags=['chat', 'coding', '🔥']))


class StsbPreprocessor(ResponsePreprocessor):

    def __init__(self, sim_threshold: Optional[float] = None):
        self.sim_threshold = sim_threshold
        super().__init__()

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            'query': row['sentence1'],
            'response': row['sentence2'],
            'label': row['score'],
        }
        if self.sim_threshold is None or float(row['label']) >= self.sim_threshold:
            return super().preprocess(row)
        else:
            return None


class StsbGeneratePreprocessor(ResponsePreprocessor):
    prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 1.0.
Sentence 1: {text1}
Sentence 2: {text2}
Similarity score: """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return super().preprocess({
            'query': self.prompt.format(text1=row['sentence1'], text2=row['sentence2']),
            'response': f"{row['score']:.1f}"
        })


class StsbRegressionPreprocessor(StsbGeneratePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return super(StsbGeneratePreprocessor, self).preprocess({
            'query':
            self.prompt.format(text1=row['sentence1'], text2=row['sentence2']),
            'label':
            row['score']
        })


register_dataset(
    DatasetMeta(
        ms_dataset_id='sentence-transformers/stsb',
        hf_dataset_id='sentence-transformers/stsb',
        subsets=[
            SubsetDataset('default', preprocess_func=StsbPreprocessor()),  # embedding
            SubsetDataset('positive', preprocess_func=StsbPreprocessor(sim_threshold=0.75)),  # infonce
            SubsetDataset('generate', preprocess_func=StsbGeneratePreprocessor()),
            SubsetDataset('reg', preprocess_func=StsbRegressionPreprocessor()),
        ],
        tags=['similarity', '🔥']))


class MTEBRerankPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = row['query']
        positives = row['positive'] if isinstance(row['positive'], list) else [row['positive']]
        negatives = row['negative'] if isinstance(row['negative'], list) else [row['negative']]

        expanded_rows = []
        for positive in positives:
            expanded_row = {'query': query, 'response': positive, 'rejected_response': negatives}
            expanded_rows.append(super().preprocess(expanded_row))

        return expanded_rows


register_dataset(
    DatasetMeta(
        ms_dataset_id='MTEB/scidocs-reranking',
        hf_dataset_id='mteb/scidocs-reranking',
        split=['validation', 'test'],
        preprocess_func=MTEBRerankPreprocessor(),
        tags=['rerank', '🔥']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='MTEB/stackoverflowdupquestions-reranking',
        hf_dataset_id='mteb/stackoverflowdupquestions-reranking',
        split=['train', 'test'],
        preprocess_func=MTEBRerankPreprocessor(),
        tags=['rerank', '🔥']))


def _repair_conversations_agent_instruct(s: str) -> List[Dict[str, Any]]:
    s = s.replace('}\n {', '},\n {')
    if isinstance(s, str):
        s = ast.literal_eval(s)
    return s


register_dataset(
    DatasetMeta(
        ms_dataset_id='huangjintao/AgentInstruct_copy',
        subsets=['alfworld', 'db', 'kg', 'mind2web', 'os', 'webshop'],
        preprocess_func=MessagesPreprocessor(repair_messages=_repair_conversations_agent_instruct),
        tags=['chat', 'agent', 'multi-round']))


class MultiRoleAgentPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        conv = row['conversations']
        res_prompt = '\n\n【注意事项】\n1. 这是聊天室，不要发送私信给任何人\n2. 仅代表你个人说话,不要扮演其他人，只根据对话历史进行回复\n3. 长话短说，不要说太多话，不要超过50字 '
        history_prompt = '\n\n【chat history】'
        conv_prompt = '\n {name}:{content}'
        query, response = '', conv[-1]['value']
        system = conv[0]['value'] if conv[0]['from'] == 'system' else ''
        if conv[0]['from'] == 'user':
            query = conv[0]['value']
        elif 'next_speakers:' not in system:
            if '【注意事项】' not in system and system:
                system += res_prompt
            system += history_prompt
            system += ''.join([conv_prompt.format(name=c['from'], content=c['value']) for c in conv[1:-1]])

        if not query or not response:
            return

        return {
            'messages': [{
                'role': 'system',
                'content': system
            }, {
                'role': 'user',
                'content': query
            }, {
                'role': 'assistant',
                'content': response
            }],
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='iic/MSAgent-MultiRole',
        preprocess_func=MultiRoleAgentPreprocessor(),
        tags=['chat', 'agent', 'multi-round', 'role-play', 'multi-agent']))

register_dataset(DatasetMeta(ms_dataset_id='swift/ToolBench', tags=['chat', 'agent', 'multi-round']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='tastelikefeet/competition_math',
        subsets=[
            SubsetDataset(
                name='default',
                subset='default',
                split=['train', 'test'],
            ),
        ],
        tags=['qa', 'math']))

register_dataset(DatasetMeta(ms_dataset_id='modelscope/gsm8k', subsets=['main'], split=['train'], tags=['qa', 'math']))

register_dataset(
    DatasetMeta(ms_dataset_id='modelscope/MathR', subsets=['default', 'clean'], split=['train'], tags=['qa', 'math']))

register_dataset(
    DatasetMeta(ms_dataset_id='modelscope/MathR-32B-Distill', subsets=['data'], split=['train'], tags=['qa', 'math']))


class CoundownTaskPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        numbers = row['nums']
        target = row.pop('response', None)
        query = (f'Using the numbers {numbers}, create an equation that equals {target}.\n'
                 'You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.\n'
                 'Show your work in <think> </think> tags. And return the final equation and answer '
                 'in <answer> </answer> tags, for example <answer> (1 + 2) / 3 * 4 = 4 </answer>.')
        row.update({'target': target, 'query': query})
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='zouxuhong/Countdown-Tasks-3to4',
        subsets=['default'],
        preprocess_func=CoundownTaskPreprocessor(),
        tags=['math']))


class HC3Preprocessor(ResponsePreprocessor):
    prompt = """Classification Task: Are the following responses from a human or from ChatGPT?
Question: {question}
Answer: {answer}
Category: Human, ChatGPT
Output:"""

    def preprocess(self, row):
        rows = []
        for response in ['Human', 'ChatGPT']:
            query = self.prompt.format(
                question=row['query'], answer=self.random_state.choice(row[f'{response.lower()}_answers']))
            rows.append(super().preprocess({'query': query, 'response': response}))
        return rows


class HC3ClsPreprocessor(HC3Preprocessor):

    def preprocess(self, row):
        rows = []
        for i, response in enumerate(['Human', 'ChatGPT']):
            query = self.prompt.format(
                question=row['query'], answer=self.random_state.choice(row[f'{response.lower()}_answers']))
            rows.append(ResponsePreprocessor.preprocess(self, {'query': query, 'label': i}))
        return rows


hc3_subset_names = ['baike', 'open_qa', 'nlpcc_dbqa', 'finance', 'medicine', 'law', 'psychology']
hc3_subsets: List[SubsetDataset] = []
for hc3_subset_name in hc3_subset_names:
    hc3_subsets.append(
        SubsetDataset(
            name=hc3_subset_name,
            subset=hc3_subset_name,
            preprocess_func=HC3Preprocessor(),
        ))
    hc3_subsets.append(
        SubsetDataset(
            name=f'{hc3_subset_name}_cls',
            subset=hc3_subset_name,
            preprocess_func=HC3ClsPreprocessor(),
        ))

register_dataset(
    DatasetMeta(
        ms_dataset_id='simpleai/HC3-Chinese',
        hf_dataset_id='Hello-SimpleAI/HC3-Chinese',
        subsets=hc3_subsets,
        tags=['text-generation', 'classification', '🔥']))

hc3_subset_names = ['finance', 'medicine']
hc3_subsets: List[SubsetDataset] = []
for hc3_subset_name in hc3_subset_names:
    hc3_subsets.append(
        SubsetDataset(
            name=hc3_subset_name,
            subset=hc3_subset_name,
            preprocess_func=HC3Preprocessor(),
        ))
    hc3_subsets.append(
        SubsetDataset(
            name=f'{hc3_subset_name}_cls',
            subset=hc3_subset_name,
            preprocess_func=HC3ClsPreprocessor(),
        ))

register_dataset(
    DatasetMeta(
        ms_dataset_id='simpleai/HC3',
        hf_dataset_id='Hello-SimpleAI/HC3',
        subsets=hc3_subsets,
        preprocess_func=HC3Preprocessor(),
        tags=['text-generation', 'classification', '🔥']))


class DureaderPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = """Task: Question Generation
Context: {context}
Answer: {answer}
Question:"""
        answer, context = row['text1'].split('[SEP]')
        return {
            'messages': [{
                'role': 'user',
                'content': prompt.format(context=context, answer=answer)
            }, {
                'role': 'assistant',
                'content': row['text2']
            }]
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='modelscope/DuReader_robust-QG',
        preprocess_func=DureaderPreprocessor(),
        split=['train', 'validation', 'test'],
        tags=['text-generation', '🔥']))


class HHRLHFPreprocessor(RowPreprocessor):

    @staticmethod
    def _to_messages(data):
        messages = []
        for query, response in zip(data[::2], data[1::2]):
            messages.append({'role': 'user', 'content': query})
            messages.append({'role': 'assistant', 'content': response})
        return messages

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        chosen = row['chosen'].strip()
        rejected = row['rejected'].strip()
        parts_chosen = [s.strip() for s in re.split('\n\nHuman:|\n\nAssistant:|\n\nHum:', chosen)]
        parts_rejected = [s.strip() for s in re.split('\n\nHuman:|\n\nAssistant:|\n\nHum:', rejected)]
        if parts_chosen[0].startswith('Human:'):
            assert parts_rejected[0].startswith('Human:')
            parts_chosen[0] = parts_chosen[0][6:].strip()
            parts_rejected[0] = parts_rejected[0][6:].strip()
        row['messages'] = self._to_messages(parts_chosen)
        row['rejected_messages'] = self._to_messages(parts_rejected)
        return row


# TODO meta file broken
register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/hh-rlhf',
        subsets=['helpful-base', 'helpful-online', 'helpful-rejection-sampled'],
        preprocess_func=HHRLHFPreprocessor(),
        split=['train', 'test'],
        tags=['rlhf', 'dpo'],
        huge_dataset=True))


class XlamFunctionCallingPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        messages = [{'role': 'user', 'content': row['query']}]
        response = row['answers']
        response = json.loads(response)
        messages += [{'role': 'tool_call', 'content': json.dumps(content)} for content in response]
        return {'messages': messages, 'tools': row['tools']}


class XlamFunctionCallingGRPOPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row['query']
        answers = row['response']
        if isinstance(answers, str):
            answers = json.loads(answers)
        answer = np.random.choice(answers)
        name = answer['name']
        args = json.dumps(answer['arguments'])
        response = f'Action: {name}\nAction Input: {args}'
        row = {'query': query, 'response': response, 'solution': response, 'tools': row['tools']}
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='LLM-Research/xlam-function-calling-60k',
        hf_dataset_id='Salesforce/xlam-function-calling-60k',
        subsets=[
            SubsetDataset('default', 'dataset', preprocess_func=XlamFunctionCallingPreprocessor()),
            SubsetDataset('grpo', 'dataset', preprocess_func=XlamFunctionCallingGRPOPreprocessor())
        ],
        tags=['agent', 'grpo', '🔥']))


class HHRLHFCNPreprocessor(MessagesPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row['messages'].append(row.pop('chosen'))
        row['rejected_response'] = row['rejected']['text']
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/hh_rlhf_cn',
        subsets=['hh_rlhf', 'harmless_base_cn', 'harmless_base_en', 'helpful_base_cn', 'helpful_base_en'],
        preprocess_func=HHRLHFCNPreprocessor(columns={'context': 'messages'}, content_key='text'),
        split=['train', 'test'],
        tags=['rlhf', 'dpo', '🔥']))


def repair_conversations(s: Union[str, Any]) -> Any:
    if isinstance(s, str):
        s = s.replace('}\n {', '},{')
        s = s.replace('}\n{', '},{')
        s = s.replace('}{', '},{')
        s = s.replace('}\n  {', '},{')
        return ast.literal_eval(s)
    return s


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/lmsys-chat-1m',
        hf_dataset_id='lmsys/lmsys-chat-1m',
        preprocess_func=MessagesPreprocessor(repair_messages=repair_conversations),
        tags=['chat', 'em']))


class EmojiPreprocessr(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Remove dirty characters
        row['query'] = row['query'].replace('️', '')
        row['response'] = row['response'].replace('️', '')
        row['rejected_response'] = row['rejected_response'].replace('️', '')
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='hjh0119/shareAI-Llama3-DPO-zh-en-emoji',
        hf_dataset_id='shareAI/DPO-zh-en-emoji',
        preprocess_func=EmojiPreprocessr(columns={
            'answer_zh': 'response',
            'answer_en': 'rejected_response'
        }),
        tags=['rlhf', 'dpo']))

register_dataset(
    DatasetMeta(ms_dataset_id='AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto', tags=['rlhf', 'kto']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='OmniData/Zhihu-KOL-More-Than-100-Upvotes',
        hf_dataset_id='bzb2023/Zhihu-KOL-More-Than-100-Upvotes',
        tags=['zhihu', 'qa']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='OmniData/Zhihu-KOL',
        hf_dataset_id='wangrui6/Zhihu-KOL',
        huge_dataset=True,
        tags=['zhihu', 'qa'],
    ))


class GuanacoPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        instruction = row['instruction']
        input = row['input']
        output = row['output']
        history = []
        if instruction:
            parts = split_str_parts_by(
                instruction, ['User:', 'User：', 'Assistant：', 'Assistant:', 'Asssistent:', 'Assistent:', 'Assistenz:'])
            for idx, part in enumerate(parts):
                if idx % 2 == 0:
                    if 'user' not in part['key'].lower():
                        return
                    history.append([part['content'], None])
                else:
                    if 'assist' not in part['key'].lower() and 'asssist' not in part['key'].lower():
                        return
                    history[-1][-1] = part['content']
        if input.startswith('User:'):
            input = input[len('User:'):].strip()
        if any([not h[0] or not h[1] for h in history]):
            return

        messages = []
        for h in history:
            messages.append({'role': 'user', 'content': h[0]})
            messages.append({'role': 'assistant', 'content': h[1]})
        messages.append({'role': 'user', 'content': input})
        messages.append({'role': 'assistant', 'content': output})
        return {
            'messages': messages,
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/GuanacoDataset',
        hf_dataset_id='JosephusCheung/GuanacoDataset',
        preprocess_func=GuanacoPreprocessor(),
        tags=['chat', 'zh']))


class FunctionCallChatmlPreprocessor(MessagesPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        res = super().preprocess(row)

        if res['function_description']:
            res['tools'] = res['function_description'].split('\n\n')
        messages = res['messages']
        if messages[0]['role'] == 'system':
            messages.pop(0)
        return res


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/function-calling-chatml',
        hf_dataset_id='Locutusque/function-calling-chatml',
        preprocess_func=FunctionCallChatmlPreprocessor(),
        tags=['agent', 'en', 'sft', '🔥']))


class Dolly15kPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        instruction = row['instruction']
        context = row['context']
        response = row['response']
        query = ''
        if context:
            query = 'Here gives some useful information:\n'
            query += context
            query += '\n'
        query += instruction
        return {
            'messages': [{
                'role': 'user',
                'content': query
            }, {
                'role': 'assistant',
                'content': response
            }],
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/databricks-dolly-15k',
        hf_dataset_id='databricks/databricks-dolly-15k',
        preprocess_func=Dolly15kPreprocessor(),
        tags=['multi-task', 'en', 'quality']))


class OrpoDPOMix40kPreprocessor(MessagesPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if row['source'] == 'toxic-dpo-v0.2':
            return
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/orpo-dpo-mix-40k',
        hf_dataset_id='mlabonne/orpo-dpo-mix-40k',
        preprocess_func=OrpoDPOMix40kPreprocessor(columns={
            'chosen': 'messages',
            'rejected': 'rejected_messages'
        }),
        tags=['dpo', 'orpo', 'en', 'quality']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/sharegpt',
        subsets=['common-zh', 'unknow-zh', 'common-en'],
        tags=['chat', 'general', 'multi-round']))


class SelfCognitionPreprocessor(ResponsePreprocessor):

    def __init__(self, *args, query_suffix: str = '', response_prefix: str = '', **kwargs):
        self.query_suffix = query_suffix
        self.response_prefix = response_prefix
        self.name: Optional[Tuple[str, str]] = None
        self.author: Optional[Tuple[str, str]] = None
        super().__init__(*args, **kwargs)

    def set_name_author(self, name, author):
        self.name = name
        self.author = author

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        for key in ['name', 'author']:
            val = getattr(self, key)
            if val is None:
                continue
            val = val[0] if row['tag'] == 'zh' else val[1]
            if val is None:
                continue
            placeholder = '{{' + key.upper() + '}}'
            row['query'] = row['query'].replace(placeholder, val)
            row['response'] = row['response'].replace(placeholder, val)

        row['query'] = row['query'] + self.query_suffix
        row['response'] = self.response_prefix + row['response']
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/self-cognition',
        hf_dataset_id='modelscope/self-cognition',
        subsets=[
            SubsetDataset(preprocess_func=SelfCognitionPreprocessor()),
            SubsetDataset(
                'qwen3',
                preprocess_func=SelfCognitionPreprocessor(
                    query_suffix=' /no_think', response_prefix='<think>\n\n</think>\n\n')),
            SubsetDataset(
                'empty_think', preprocess_func=SelfCognitionPreprocessor(response_prefix='<think>\n\n</think>\n\n')),
        ],
        dataset_name='self-cognition',
        tags=['chat', 'self-cognition', '🔥']))
