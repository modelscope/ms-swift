import ast
import re
from functools import partial
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset as HfDataset
from datasets import IterableDataset

from ..preprocess import (DATASET_TYPE, AlpacaPreprocessor, ClsPreprocessor, MessagesPreprocessor, ResponsePreprocessor,
                          RowPreprocessor, TextGenerationPreprocessor)
from ..register import DatasetMeta, SubsetDataset, register_dataset


def _concat_inst_inp_alpaca_zh(inst: str, inp: str) -> str:
    if inp.startswith('输入：'):
        inp = inp[3:]
    return f'{inst}\n{inp}'


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/alpaca-gpt4-data-zh',
        hf_dataset_id='llm-wizard/alpaca-gpt4-data-zh',
        preprocess_func=AlpacaPreprocessor(concat_inst_input=_concat_inst_inp_alpaca_zh),
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
                subset='default',
                preprocess_func=MessagesPreprocessor(repair_messages=partial(_repair_agent_messages, use_mini=True)),
                is_weak_subset=True)
        ],
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
        preprocess_func=ClsPreprocessor(['neutral', 'entailment', 'contradiction'],
                                        task='Natural Language Inference',
                                        is_pair_seq=True),
        tags=['text-generation', 'classification'],
    ))

register_dataset(
    DatasetMeta(
        ms_dataset_id='DAMO_NLP/jd',
        preprocess_func=ClsPreprocessor(['negative', 'positive'], task='Sentiment Classification', is_pair_seq=False),
        tags=['text-generation', 'classification', '🔥']))


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

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
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

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/ToolBench',
        preprocess_func=MessagesPreprocessor(remove_useless_columns=False, ),
        tags=['chat', 'agent', 'multi-round']))


def _preprocess_hc3(dataset: DATASET_TYPE, **kwargs) -> DATASET_TYPE:
    prompt = """Classification Task: Are the following responses from a human or from ChatGPT?
Question: {question}
Answer: {answer}
Category: Human, ChatGPT
Output:"""
    if isinstance(dataset, IterableDataset):

        def generate_example(dataset):
            for example in dataset:
                question = example['question']
                for h in example['human_answers']:
                    yield {
                        'messages': [{
                            'role': 'user',
                            'content': prompt.format(question=question, answer=h)
                        }, {
                            'role': 'assistant',
                            'content': 'Human'
                        }]
                    }
                for c in example['chatgpt_answers']:
                    yield {
                        'messages': [{
                            'role': 'user',
                            'content': prompt.format(question=question, answer=c)
                        }, {
                            'role': 'assistant',
                            'content': 'ChatGPT'
                        }]
                    }

        return IterableDataset.from_generator(generate_example, gen_kwargs={'dataset': dataset})

    messages = []
    for d in dataset:
        question = d['question']
        for h in d['human_answers']:
            messages.append({
                'messages': [{
                    'role': 'user',
                    'content': prompt.format(question=question, answer=h)
                }, {
                    'role': 'assistant',
                    'content': 'Human'
                }]
            })
        for c in d['chatgpt_answers']:
            messages.append({
                'messages': [{
                    'role': 'user',
                    'content': prompt.format(question=question, answer=c)
                }, {
                    'role': 'assistant',
                    'content': 'ChatGPT'
                }]
            })
    return HfDataset.from_list(messages)


register_dataset(
    DatasetMeta(
        ms_dataset_id='simpleai/HC3-Chinese',
        hf_dataset_id='Hello-SimpleAI/HC3-Chinese',
        subsets=['baike', 'open_qa', 'nlpcc_dbqa', 'finance', 'medicine', 'law', 'psychology'],
        preprocess_func=_preprocess_hc3,
        tags=['text-generation', 'classification', '🔥']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='simpleai/HC3',
        hf_dataset_id='Hello-SimpleAI/HC3',
        subsets=['finance', 'medicine'],
        preprocess_func=_preprocess_hc3,
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

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        chosen = row['chosen'].strip()
        rejected = row['rejected'].strip()
        parts_chosen = [s.strip() for s in re.split('\n\nHuman:|\n\nAssistant:|\n\nHum:', chosen)]
        parts_rejected = [s.strip() for s in re.split('\n\nHuman:|\n\nAssistant:|\n\nHum:', rejected)]
        if parts_chosen[0].startswith('Human:'):
            assert parts_rejected[0].startswith('Human:')
            parts_chosen[0] = parts_chosen[0][6:].strip()
            parts_rejected[0] = parts_rejected[0][6:].strip()
        history = []
        idx, s1, s2 = None, None, None
        for idx, (s1, s2) in enumerate(zip(parts_chosen, parts_rejected)):
            if s1 == s2:
                if idx % 2 == 0:
                    history.append([s1, None])
                else:
                    history[-1][-1] = s1
            else:
                break

        if idx % 2 == 0:
            return

        messages = []
        for h in history:
            messages.append({'role': 'user', 'content': h[0]})
            messages.append({'role': 'assistant', 'content': h[1]})

        messages[-1]['content'] = s1
        return {
            'messages': messages,
            'rejected_response': s2,
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/hh-rlhf',
        subsets=['harmless-base', 'helpful-base', 'helpful-online', 'helpful-rejection-sampled'],
        preprocess_func=HHRLHFPreprocessor(),
        split=['train'],
        tags=['rlhf', 'dpo', 'pairwise']))


class HHRLHFCNPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        history = []
        try:
            if isinstance(row['context'], str):
                row['context'] = ast.literal_eval(row['context'])
            if isinstance(row['chosen'], str):
                row['chosen'] = ast.literal_eval(row['chosen'])
            if isinstance(row['rejected'], str):
                row['rejected'] = ast.literal_eval(row['rejected'])
            for idx, h in enumerate(row['context']):
                if idx % 2 == 0 and h['role'] != 'human':
                    raise ValueError()
                if idx % 2 != 0 and h['role'] != 'assistant':
                    raise ValueError()
                if idx % 2 == 0:
                    history.append([h['text'], None])
                else:
                    history[-1][-1] = h['text']

            if history[-1][-1] is not None:
                raise ValueError()
        except:  # noqa
            return
        else:
            messages = []
            for h in history:
                messages.append({'role': 'user', 'content': h[0]})
                messages.append({'role': 'assistant', 'content': h[1]})

            messages[-1]['content'] = row['chosen']['text']
            return {
                'messages': messages,
                'rejected_response': row['rejected']['text'],
            }

    def filter_valid_row(self, row):
        try:
            if isinstance(row['context'], str):
                row['context'] = ast.literal_eval(row['context'])
            if isinstance(row['chosen'], str):
                row['chosen'] = ast.literal_eval(row['chosen'])
            if isinstance(row['rejected'], str):
                row['rejected'] = ast.literal_eval(row['rejected'])
            return True
        except:  # noqa
            return False

    def __call__(self, dataset, **kwargs):
        filter_kwargs = kwargs.copy()
        filter_kwargs.pop('strict', None)
        dataset = dataset.filter(self.filter_valid_row, **filter_kwargs)
        return super(HHRLHFCNPreprocessor, self).__call__(dataset, **kwargs)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/hh_rlhf_cn',
        subsets=['hh_rlhf', 'harmless_base_cn', 'harmless_base_en', 'helpful_base_cn', 'helpful_base_en'],
        preprocess_func=HHRLHFCNPreprocessor(),
        split=['train', 'test'],
        tags=['rlhf', 'dpo', 'pairwise', '🔥']))


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


class ShareAIDPOPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'messages': [
                {
                    'role': 'user',
                    'content': row['question']
                },
                {
                    'role': 'assistant',
                    'content': row['answer_zh']
                },
            ],
            'rejected_response': row['answer_en'],
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='hjh0119/shareAI-Llama3-DPO-zh-en-emoji',
        preprocess_func=ShareAIDPOPreprocessor(),
        tags=['rlhf', 'dpo', 'pairwise']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto',
        preprocess_func=ResponsePreprocessor(
            columns_mapping={
                'prompt': 'query',
                'completion': 'response',
            }, remove_useless_columns=False),
        tags=['rlhf', 'kto']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='OmniData/Zhihu-KOL-More-Than-100-Upvotes',
        hf_dataset_id='bzb2023/Zhihu-KOL-More-Than-100-Upvotes',
        preprocess_func=ResponsePreprocessor(columns_mapping={
            'INSTRUCTION': 'query',
            'RESPONSE': 'response'
        }),
        tags=['zhihu', 'qa']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='OmniData/Zhihu-KOL',
        hf_dataset_id='wangrui6/Zhihu-KOL',
        preprocess_func=ResponsePreprocessor(columns_mapping={
            'INSTRUCTION': 'query',
            'RESPONSE': 'response'
        }),
        huge_dataset=True,
        tags=['zhihu', 'qa']))


class GuanacoPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        instruction = row['instruction']
        input = row['input']
        output = row['output']
        history = []
        if instruction:
            from swift.llm.template.agent import split_str_parts_by
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

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
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


class OrpoDPOMix40kPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        chosen_history = row['chosen']
        rejected_history = row['rejected']
        history = []
        query = None
        response = None
        rejected_response = None
        if row['source'] == 'toxic-dpo-v0.2':
            return
        try:
            for i, (chosen, rejected) in enumerate(zip(chosen_history, rejected_history)):
                role = chosen['role']
                content = chosen['content']
                rejected_role = rejected['role']
                rejected_content = rejected['content']
                assert role == rejected_role
                if i % 2 == 0:
                    assert role == 'user'
                else:
                    assert role == 'assistant'

                if content != rejected_content:
                    assert role == 'assistant'
                    response = content
                    rejected_response = rejected_content
                    query = history.pop(-1)[0]
                else:
                    if role == 'user':
                        history.append([content, None])
                    else:
                        history[-1][-1] = content

        except (AssertionError, IndexError):
            return

        messages = []
        for h in history:
            messages.append({'role': 'user', 'content': h[0]})
            messages.append({'role': 'assistant', 'content': h[1]})

        if query is None or response is None:
            return

        messages.append({'role': 'user', 'content': query})
        messages.append({'role': 'assistant', 'content': response})
        return {
            'messages': messages,
            'rejected_response': rejected_response,
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/orpo-dpo-mix-40k',
        hf_dataset_id='mlabonne/orpo-dpo-mix-40k',
        preprocess_func=OrpoDPOMix40kPreprocessor(),
        tags=['dpo', 'orpo', 'en', 'quality']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/sharegpt',
        subsets=['common-zh', 'unknow-zh', 'common-en'],
        preprocess_func=MessagesPreprocessor(),
        tags=['chat', 'general', 'multi-round']))
