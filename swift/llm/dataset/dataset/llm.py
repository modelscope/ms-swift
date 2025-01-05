# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import json
import re
from copy import copy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

from ..preprocessor import (AlpacaPreprocessor, ClsGenerationPreprocessor, ClsPreprocessor, MessagesPreprocessor,
                            ResponsePreprocessor, RowPreprocessor, TextGenerationPreprocessor)
from ..register import DatasetMeta, SubsetDataset, register_dataset
from ...template import split_str_parts_by


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
            prompt=advertise_gen_prompt, columns_mapping={
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
                preprocess_func=ClsPreprocessor(columns_mapping={'sentence': 'query'}),
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
        preprocess_func=AlpacaPreprocessor(columns_mapping={
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
        preprocess_func=MessagesPreprocessor(columns_mapping={'chat_rounds': 'messages'}),
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
        res_prompt = """\n\n【注意事项】\n1. 这是聊天室，不要发送私信给任何人\n2. 仅代表你个人说话,不要扮演其他人，
        只根据对话历史进行回复\n3. 长话短说，不要说太多话，不要超过50字 """
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


class ToolBenchPPOPreprocessor(MessagesPreprocessor):

    def batched_preprocess(self, batched_row: Dict[str, Any], *, strict: bool) -> Dict[str, Any]:
        batched_row = dict(batched_row)
        assert len(batched_row) > 0
        self._fix_streaming_keys(batched_row)
        rows = self.batched_to_rows(batched_row)

        new_rows = []
        for row in rows:
            rows = self.preprocess(row)
            new_rows.extend(rows)
        res = self.rows_to_batched(new_rows)

        if len(res) == 0:
            res['messages'] = []

        return res

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        import numpy as np
        row = super().preprocess(row)
        has_error = False
        http_errno = ['404', '403', '400', '401', '500', '502', '503']
        for m in row['messages']:
            if m['role'] == 'tool':
                try:
                    content = json.loads(m['content'])
                except:
                    has_error = True
                    continue
                if content.get('error') or any(errno in m['content'] for errno in http_errno) or not content.get('response'):
                    has_error = True
        if 'give_up_and_restart' in row['messages'][-1]['content'] or 'unfortunately' in row['messages'][-1]['content'].lower() or 'failed' in row['messages'][-1]['content'].lower() or 'sorry' in row['messages'][-1]['content'].lower():
            has_error = True

        if len(row['messages']) > 7:
            has_error = True

        if has_error:
            return []

        all_rows = []
        action_rounds = sum([1 if 'Action Input:' in message['content'] and message['role'] == 'assistant' else 0 for message in row['messages']])
        n_round = np.random.choice(action_rounds)
        _round = 0
        for i, message in enumerate(row['messages']):
            if message['role'] == 'assistant':
                if 'Action:' in message['content'] and 'Action Input:' in message['content']:
                    if _round == n_round:
                        new_row = copy(row)
                        new_row['messages'] = new_row['messages'][:i]
                        new_row['ground_truth'] = message['content']
                        # new_row['messages'] = [m for m in new_row['messages'] if m['role'] != 'system']
                        # new_row.pop('tools', None)
                        all_rows.append(new_row)
                        break
                    else:
                        _round += 1
        return all_rows


class ToolBenchFilteredPreprocessor(MessagesPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        import numpy as np
        row = super().preprocess(row)
        has_error = False
        http_errno = ['404', '403', '400', '401', '500', '502', '503']
        for m in row['messages']:
            if m['role'] == 'tool':
                try:
                    content = json.loads(m['content'])
                except:
                    has_error = True
                    continue
                if content.get('error') or any(errno in m['content'] for errno in http_errno) or not content.get('response'):
                    has_error = True
        if 'give_up_and_restart' in row['messages'][-1]['content'] or 'unfortunately' in row['messages'][-1]['content'].lower() or 'failed' in row['messages'][-1]['content'].lower() or 'sorry' in row['messages'][-1]['content'].lower():
            has_error = True

        if len(row['messages']) > 7:
            has_error = True

        if has_error:
            return None

        return row

        


register_dataset(DatasetMeta(
    ms_dataset_id='swift/ToolBench',
    subsets=[
        SubsetDataset(
            name='default',
        ),
        SubsetDataset(
            name='ppo',
            preprocess_func=ToolBenchPPOPreprocessor(),
        ),
        SubsetDataset(
            name='filtered',
            preprocess_func=ToolBenchFilteredPreprocessor(),
        ),
    ],
    tags=['chat', 'agent', 'multi-round']))


class XlamPreprocessor(ResponsePreprocessor):

    def batched_preprocess(self, batched_row: Dict[str, Any], *, strict: bool) -> Dict[str, Any]:
        tools = batched_row['tools']
        all_tools = {}
        for _tools in tools:
            _tools = json.loads(_tools)
            for _tool in _tools:
                name = _tool['name']
                if name not in all_tools:
                    all_tools[name] = _tool

        return super().batched_preprocess(batched_row, strict=strict, all_tools=all_tools)

    def preprocess(self, row: Dict[str, Any], all_tools) -> Optional[Dict[str, Any]]:
        query = row['query']
        answers = row['response']
        answers = json.loads(answers)
        tools = row['tools']
        tools = json.loads(tools)
        if len(answers) > 1:
            # parallelism tool calling not supported
            return None

        answer = answers[0]
        if len(tools) < 5:
            for tool in tools:
                all_tools.pop(tool['name'], None)
        all_tools = list(all_tools.values())
        import numpy as np
        choices = np.random.permutation(len(all_tools))[:(5-len(tools))]
        all_tools = [all_tools[i] for i in choices]
        tools.extend(all_tools)
        action = answer['name']
        action_input = answer['arguments']
        if len(tools) > 10:
            return None
        answer = f'Action: {action}\nAction Input:{json.dumps(action_input)}'
        row = {
            'query': query,
            'tools': json.dumps(tools),
            'ground_truth': answer,
        }
        return super().preprocess(row)


register_dataset(DatasetMeta(
    ms_dataset_id='LLM-Research/xlam-function-calling-60k',
    subsets=[
        SubsetDataset(
            name='default',
            subset='dataset',
            preprocess_func=XlamPreprocessor(),
        ),
    ],
    tags=['chat', 'agent']))


class CompetitionMathRLPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any], all_tools=None) -> Optional[Dict[str, Any]]:
        query = row['problem']
        solution = row['response']
        row = {
            'query': query,
            'ground_truth': solution,
        }
        return super().preprocess(row)


class CompetitionMathPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any], all_tools=None) -> Optional[Dict[str, Any]]:
        query = row['problem']
        response = row['response']
        row = {
            'query': query,
            'response': response,
        }
        return super().preprocess(row)


register_dataset(DatasetMeta(
    ms_dataset_id='modelscope/competition_math',
    subsets=[
        SubsetDataset(
            name='default',
            subset='default',
            preprocess_func=CompetitionMathPreprocessor(),
        ),
        SubsetDataset(
            name='rl',
            subset='default',
            preprocess_func=CompetitionMathRLPreprocessor(),
        ),
    ],
    tags=['qa', 'math']))


register_dataset(DatasetMeta(
    ms_dataset_id='modelscope/gsm8k',
    subsets=['main'],
    split=['train'],
    tags=['qa', 'math']))


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


class HHRLHFCNPreprocessor(MessagesPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row['messages'].append(row.pop('chosen'))
        row['rejected_response'] = row['rejected']['text']
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/hh_rlhf_cn',
        subsets=['hh_rlhf', 'harmless_base_cn', 'harmless_base_en', 'helpful_base_cn', 'helpful_base_en'],
        preprocess_func=HHRLHFCNPreprocessor(columns_mapping={'context': 'messages'}, content_key='text'),
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

register_dataset(
    DatasetMeta(
        ms_dataset_id='hjh0119/shareAI-Llama3-DPO-zh-en-emoji',
        subsets=[
            SubsetDataset(
                'zh',
                preprocess_func=ResponsePreprocessor(columns_mapping={
                    'answer_zh': 'response',
                    'answer_en': 'rejected_response'
                })),
            SubsetDataset(
                'en',
                preprocess_func=ResponsePreprocessor(columns_mapping={
                    'answer_en': 'response',
                    'answer_zh': 'rejected_response'
                }))
        ],
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
            from swift.llm.template import split_str_parts_by
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
        preprocess_func=OrpoDPOMix40kPreprocessor(columns_mapping={
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
    name: Optional[Tuple[str, str]] = None
    author: Optional[Tuple[str, str]] = None

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
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/self-cognition',
        hf_dataset_id='modelscope/self-cognition',
        preprocess_func=SelfCognitionPreprocessor(),
        tags=['chat', 'self-cognition', '🔥']))
