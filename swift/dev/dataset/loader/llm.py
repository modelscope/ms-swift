# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pure-text dataset registrations, mirroring the model side's ``loader/llm.py``.

The families here are grouped by *how much they have to declare*, which is also the order to read
them in and the order to add a new dataset in:

1. Declaration only -- ids, subsets, splits. No preprocessor at all: the format layer detects the
   row shape and the base :class:`Preprocessor` runs it.
2. Declaration plus :attr:`Preprocessor.columns` -- the row shape is standard but the fields are
   named oddly, so a rename is all that is needed.
3. A :class:`Preprocessor` subclass -- the row needs real work: fields merged into one turn, rows
   filtered out, text cleaned.

A preprocessor is always declared as a *class*, never an instance: the loader's ``preprocessor``
attribute is shared by every load, and a preprocessor carries per-load state.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from swift.template.utils import split_str_parts_by

from .base import DatasetLoader, SubsetMeta, register_dataset
from ..preprocessor import MessagesRepairPreprocessor, Preprocessor


# ============================================================================================
# 1. Declaration only -- no preprocessor; the format layer detects the row shape.
# ============================================================================================


@register_dataset
class ToolBenchLoader(DatasetLoader):
    dataset_type = 'toolbench'
    datasets = [('swift/ToolBench', None)]
    tags = ['chat', 'agent', 'multi-round']


@register_dataset
class ShareGptLoader(DatasetLoader):
    dataset_type = 'sharegpt'
    datasets = [('swift/sharegpt', None)]
    subsets = ['common-zh', 'unknow-zh', 'common-en']
    tags = ['chat', 'general', 'multi-round']


@register_dataset
class Gsm8kLoader(DatasetLoader):
    dataset_type = 'gsm8k'
    datasets = [('modelscope/gsm8k', None)]
    subsets = ['main']
    tags = ['qa', 'math']


@register_dataset
class MathRLoader(DatasetLoader):
    dataset_type = 'MathR'
    datasets = [('modelscope/MathR', None)]
    subsets = ['default', 'clean']
    tags = ['qa', 'math']


@register_dataset
class MathRDistillLoader(DatasetLoader):
    dataset_type = 'MathR-32B-Distill'
    datasets = [('modelscope/MathR-32B-Distill', None)]
    subsets = ['data']
    tags = ['qa', 'math']


@register_dataset
class DapoMath17kLoader(DatasetLoader):
    dataset_type = 'DAPO-Math-17k'
    datasets = [('open-r1/DAPO-Math-17k-Processed', 'open-r1/DAPO-Math-17k-Processed')]
    subsets = ['all']
    tags = ['math', 'rlvr']


@register_dataset
class CompetitionMathLoader(DatasetLoader):
    dataset_type = 'competition_math'
    datasets = [('tastelikefeet/competition_math', None)]
    # This dataset's test split is worth training on too, so the split list is widened for the one
    # subset rather than for the class -- which is what per-subset `split` is for.
    subsets = [SubsetMeta('default', split=['train', 'test'])]
    tags = ['qa', 'math']


@register_dataset
class UltrafeedbackKtoLoader(DatasetLoader):
    dataset_type = 'ultrafeedback-kto'
    datasets = [('AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto', None)]
    tags = ['rlhf', 'kto']


@register_dataset
class ZhihuKolLoader(DatasetLoader):
    dataset_type = 'zhihu-kol'
    datasets = [('OmniData/Zhihu-KOL', 'wangrui6/Zhihu-KOL')]
    huge_dataset = True
    tags = ['zhihu', 'qa']


@register_dataset
class ZhihuKolFilteredLoader(DatasetLoader):
    dataset_type = 'zhihu-kol-filtered'
    datasets = [('OmniData/Zhihu-KOL-More-Than-100-Upvotes', 'bzb2023/Zhihu-KOL-More-Than-100-Upvotes')]
    tags = ['zhihu', 'qa']


# ============================================================================================
# 2. Declaration plus a column rename -- standard row shape, non-standard field names.
# ============================================================================================


class SqlCreateContextPreprocessor(Preprocessor):
    """Alpaca in all but naming: the question is the instruction and the schema is the input."""

    format_name = 'alpaca'
    columns = {'question': 'instruction', 'context': 'input', 'answer': 'output'}


@register_dataset
class SqlCreateContextLoader(DatasetLoader):
    dataset_type = 'sql-create-context'
    datasets = [('AI-ModelScope/sql-create-context', 'b-mc2/sql-create-context')]
    preprocessor = SqlCreateContextPreprocessor
    tags = ['chat', 'sql', '🔥']


class CodeExercisePreprocessor(Preprocessor):
    """A dialogue dataset that calls its dialogue column ``chat_rounds``."""

    format_name = 'openai'
    columns = {'chat_rounds': 'messages'}


@register_dataset
class CodeExerciseLoader(DatasetLoader):
    dataset_type = 'code-exercise-python'
    datasets = [('codefuse-ai/CodeExercise-Python-27k', None)]
    preprocessor = CodeExercisePreprocessor
    tags = ['chat', 'coding', '🔥']


# ============================================================================================
# 3. A preprocessor subclass -- fields merged, rows filtered, text cleaned.
# ============================================================================================


class AlpacaZhPreprocessor(Preprocessor):
    """Alpaca, minus the ``'输入：'`` lead-in some rows prepend to the ``input`` half of the turn."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        input_ = row.get('input')
        if isinstance(input_, str) and input_.startswith('输入：'):
            row['input'] = input_[len('输入：'):]
        return super().preprocess(row)


@register_dataset
class AlpacaZhLoader(DatasetLoader):
    dataset_type = 'alpaca-zh'
    datasets = [('AI-ModelScope/alpaca-gpt4-data-zh', 'llm-wizard/alpaca-gpt4-data-zh')]
    preprocessor = AlpacaZhPreprocessor
    tags = ['chat', 'general', '🔥']


class LongAlpacaPreprocessor(Preprocessor):
    """Alpaca, minus the ``'Answer: '`` prefix this dataset puts on every answer."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        output = row.get('output')
        prefix = 'Answer: '
        if isinstance(output, str) and output.startswith(prefix):
            row['output'] = output[len(prefix):].strip()
        return super().preprocess(row)


@register_dataset
class LongAlpacaLoader(DatasetLoader):
    dataset_type = 'long-alpaca-12k'
    datasets = [('AI-ModelScope/LongAlpaca-12k', 'Yukang/LongAlpaca-12k')]
    preprocessor = LongAlpacaPreprocessor
    tags = ['long-sequence', 'QA']


class RuozhibaPreprocessor(Preprocessor):
    """A pretrain dataset: each row is one completion, emitted as a lone assistant turn.

    Builds the standard row itself (no format converter applies): it stitches ``title``/``content``
    and an optional ``abs``, strips a leading list-item number, and drops rows left empty.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        title = row['title'] if row.get('title') is not None else row.get('content')
        abstract = row.get('abs')
        if abstract and abstract != title:
            title = f'{title}，{abstract}'
        match = re.search(r'\d+[\.,\s,\、](.+)', title)
        if match:
            title = match.group(1)
        if title:
            return {'messages': [{'role': 'assistant', 'content': title}]}


@register_dataset
class RuozhibaLoader(DatasetLoader):
    dataset_type = 'ruozhiba'
    datasets = ['AI-ModelScope/ruozhiba']
    subsets = ['post-annual', 'title-good', 'title-norm']
    preprocessor = RuozhibaPreprocessor
    tags = ['pretrain', '🔥']


class MathTrnPreprocessor(Preprocessor):
    """Question and answer only: this dataset carries extra bookkeeping columns to be ignored."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return super().preprocess({'query': row['query'], 'response': row['response']})


@register_dataset
class MathTrnLoader(DatasetLoader):
    dataset_type = 'math-trn-format'
    datasets = [('AI-ModelScope/math-trn-format', None)]
    preprocessor = MathTrnPreprocessor
    tags = ['math']


class FireflyPreprocessor(Preprocessor):
    """Keep only the task kinds worth training on; Firefly mixes in a long tail of others."""

    KINDS = {
        'ProseGeneration', 'MRC', 'JinYongGeneration', 'TextCorrection', 'ClassicalChinese', 'BELLE', 'StoryGeneration',
        'Couplet', 'Cot', 'Dictionary', 'Translation', 'Program', 'SentimentAnalyze', 'OpenQA', 'AncientPoem',
        'TextMatching', 'NLI', 'Summary', 'KeywordRecognition', 'ProductDesc', 'LyricGeneration', 'Composition',
        'MusicComment', 'NER'
    }

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if row.get('kind') not in self.KINDS:
            return None
        return super().preprocess(row)


@register_dataset
class FireflyLoader(DatasetLoader):
    dataset_type = 'firefly-zh'
    datasets = [('AI-ModelScope/firefly-train-1.1M', 'YeungNLP/firefly-train-1.1M')]
    preprocessor = FireflyPreprocessor
    tags = ['chat', 'general']


class BlossomMathPreprocessor(Preprocessor):
    """Append the bare numeric answer to the worked solution, as the model should produce both.

    ``answer`` is popped first so it cannot be mistaken for the response itself -- the response
    aliases cover ``answer``, and the worked solution in ``output`` is the one meant.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        answer = row.pop('answer', None)
        new_row = super().preprocess(row)
        if new_row and answer is not None:
            new_row['messages'][-1]['content'] += f'\n\nAnswer: {answer}'
        return new_row


@register_dataset
class BlossomMathLoader(DatasetLoader):
    dataset_type = 'blossom-math-zh'
    datasets = [('AI-ModelScope/blossom-math-v2', 'Azure99/blossom-math-v2')]
    preprocessor = BlossomMathPreprocessor
    tags = ['chat', 'math', '🔥']


class SyntheticText2SqlPreprocessor(Preprocessor):
    """Fold the schema into the question, and the explanation into the answer, as chain-of-thought."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        query = f"Sql Table information:\n{row['sql_context']}\n{row['sql_prompt']}"
        response = (f"Let's think step by step:\n{row['sql_explanation']}\n"
                    f"So the final sql is:\n{row['sql']}")
        return super().preprocess({'query': query, 'response': response})


@register_dataset
class SyntheticText2SqlLoader(DatasetLoader):
    dataset_type = 'synthetic-text2sql'
    datasets = [('AI-ModelScope/synthetic_text_to_sql', 'gretelai/synthetic_text_to_sql')]
    preprocessor = SyntheticText2SqlPreprocessor
    tags = ['nl2sql', 'en']


class LeetcodePythonPreprocessor(Preprocessor):
    """Split one column that holds problem statement and solution into the two halves of a turn."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        code_with_problem = row['code_with_problem']
        index = code_with_problem.find('```python')
        problem = code_with_problem[:index]
        if problem.startswith('# '):
            problem = problem[2:]
        code = code_with_problem[index:].strip()
        response = f"{code}\n\n{row['explanation_only']}"
        return super().preprocess({'query': problem, 'response': response})


@register_dataset
class LeetcodePythonLoader(DatasetLoader):
    dataset_type = 'leetcode-python-en'
    datasets = [('AI-ModelScope/leetcode-solutions-python', None)]
    preprocessor = LeetcodePythonPreprocessor
    tags = ['chat', 'coding', '🔥']


class TigerBotLawPreprocessor(Preprocessor):
    """A pretrain corpus: concatenate the statute's heading, chapters and body into one passage."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        body = self.pop_first(row, 'response', 'content', 'text')
        passage = f"{row['type']}\n{row['title']}\n"
        for i in range(1, 4):
            chapter = row.get(f'chapter{i}')
            if chapter is not None:
                passage += chapter
        passage += body
        return super().preprocess({'response': passage})


@register_dataset
class TigerBotLawLoader(DatasetLoader):
    dataset_type = 'tigerbot-law-zh'
    datasets = [('AI-ModelScope/tigerbot-law-plugin', 'TigerResearch/tigerbot-law-plugin')]
    preprocessor = TigerBotLawPreprocessor
    tags = ['text-generation', 'law', 'pretrained']


class Dolly15kPreprocessor(Preprocessor):
    """Prefix the instruction with its reference passage, when the row has one."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        query = row['instruction']
        context = row.get('context')
        if context:
            query = f'Here gives some useful information:\n{context}\n{query}'
        return super().preprocess({'query': query, 'response': row['response']})


@register_dataset
class Dolly15kLoader(DatasetLoader):
    dataset_type = 'dolly-15k'
    datasets = [('AI-ModelScope/databricks-dolly-15k', 'databricks/databricks-dolly-15k')]
    preprocessor = Dolly15kPreprocessor
    tags = ['multi-task', 'en', 'quality']


class EmojiDpoPreprocessor(Preprocessor):
    """A flat preference pair, once a stray variation selector is stripped from the text.

    The fields are cleaned under their *raw* names: renaming happens inside the converter, which
    :meth:`Preprocessor.preprocess` has not reached yet.
    """

    columns = {'answer_zh': 'response', 'answer_en': 'rejected_response'}
    DIRTY_CHAR = '️'

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for key in ['question', 'answer_zh', 'answer_en']:
            value = row.get(key)
            if isinstance(value, str):
                row[key] = value.replace(self.DIRTY_CHAR, '')
        return super().preprocess(row)


@register_dataset
class EmojiDpoLoader(DatasetLoader):
    dataset_type = 'shareai-dpo-emoji'
    datasets = [('hjh0119/shareAI-Llama3-DPO-zh-en-emoji', 'shareAI/DPO-zh-en-emoji')]
    preprocessor = EmojiDpoPreprocessor
    tags = ['rlhf', 'dpo']


class OrpoDpoMix40kPreprocessor(Preprocessor):
    """A preference dataset of mixed provenance, minus the deliberately toxic slice."""

    columns = {'chosen': 'messages', 'rejected': 'rejected_messages'}

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if row.get('source') == 'toxic-dpo-v0.2':
            return None
        return super().preprocess(row)


@register_dataset
class OrpoDpoMix40kLoader(DatasetLoader):
    dataset_type = 'orpo-dpo-mix-40k'
    datasets = [('AI-ModelScope/orpo-dpo-mix-40k', 'mlabonne/orpo-dpo-mix-40k')]
    preprocessor = OrpoDpoMix40kPreprocessor
    tags = ['dpo', 'orpo', 'en', 'quality']


# ============================================================================================
# 4. Dialogue datasets whose `messages` column needs repairing first -- see
#    `MessagesRepairPreprocessor`. Each `repair` runs before any conversion, and may drop the row.
# ============================================================================================


class MsBenchPreprocessor(MessagesRepairPreprocessor):
    """Drop the boilerplate system turn, and drop rows that leak another assistant's persona."""

    default_system = 'You are a helpful assistant.'
    # Marks of a transcript that was pasted in rather than spoken: a rival model's name, or role
    # labels left inline where a real turn boundary should be.
    leaked_markers = ('moss', 'human:', 'assistant:', 'user:')

    def repair(self, messages: Any) -> Optional[List[Dict[str, Any]]]:
        messages = self.converter.parse_literal(messages)
        if messages[0].get('from') == 'system' and messages[0].get('value') == self.default_system:
            messages.pop(0)
        for message in messages:
            value = message['value'].lower()
            if any(marker in value for marker in self.leaked_markers):
                return None
        return messages


@register_dataset
class MsBenchLoader(DatasetLoader):
    dataset_type = 'ms-bench'
    datasets = ['iic/ms_bench']
    preprocessor = MsBenchPreprocessor
    tags = ['chat', 'general', 'multi-round', '🔥']


class AgentInstructPreprocessor(MessagesRepairPreprocessor):
    """The dialogue is a repr of a dict list that lost the commas between entries."""

    def repair(self, messages: Any) -> Optional[List[Dict[str, Any]]]:
        if isinstance(messages, str):
            messages = self.converter.parse_literal(messages.replace('}\n {', '},\n {'))
        return messages


@register_dataset
class AgentInstructLoader(DatasetLoader):
    dataset_type = 'agent-instruct'
    datasets = ['huangjintao/AgentInstruct_copy']
    subsets = ['alfworld', 'db', 'kg', 'mind2web', 'os', 'webshop']
    preprocessor = AgentInstructPreprocessor
    tags = ['chat', 'agent', 'multi-round']


class LmsysChat1mPreprocessor(MessagesRepairPreprocessor):
    """Same lost-comma damage as ``AgentInstruct``, in four more spellings of the gap."""

    def repair(self, messages: Any) -> Optional[List[Dict[str, Any]]]:
        if isinstance(messages, str):
            for broken in ('}\n {', '}\n{', '}{', '}\n  {'):
                messages = messages.replace(broken, '},{')
            messages = self.converter.parse_literal(messages)
        return messages


@register_dataset
class LmsysChat1mLoader(DatasetLoader):
    dataset_type = 'lmsys-chat-1m'
    datasets = [('AI-ModelScope/lmsys-chat-1m', 'lmsys/lmsys-chat-1m')]
    preprocessor = LmsysChat1mPreprocessor
    tags = ['chat', 'em']


class AlphaUmiToolbenchPreprocessor(MessagesRepairPreprocessor):
    """Relabel this dataset's own role names for the answering turn as ``assistant``."""

    def repair(self, messages: Any) -> Optional[List[Dict[str, Any]]]:
        assert len(messages) == 2, f'messages: {messages}'
        if messages[1]['from'] in {'caller', 'conclusion'}:
            messages[1]['from'] = 'assistant'
        return messages


@register_dataset
class AlphaUmiToolbenchLoader(DatasetLoader):
    dataset_type = 'alpha-umi-toolbench'
    datasets = ['shenweizhou/alpha-umi-toolbench-processed-v2']
    subsets = ['backbone', 'caller', 'planner', 'summarizer']
    preprocessor = AlphaUmiToolbenchPreprocessor
    huge_dataset = True
    tags = ['chat', 'agent', '🔥']


class MSAgentBenchPreprocessor(MessagesRepairPreprocessor):
    """Pass every row through; the ``mini`` subset narrows this (see the subclass below)."""

    def repair(self, messages: Any) -> Optional[List[Dict[str, Any]]]:
        return messages


class MSAgentBenchMiniPreprocessor(MSAgentBenchPreprocessor):
    """``mini``: keep only rows whose system prompt offers a genuine choice between plugins.

    A row advertising one plugin (or none) teaches selection nothing, so the smaller subset is the
    multi-plugin rows rather than a random sample.
    """

    plugin_pattern = r'\d\. {"plugin_name": "(.+?)"'

    def repair(self, messages: Any) -> Optional[List[Dict[str, Any]]]:
        if messages[0].get('from') != 'system':
            return None
        plugins = re.findall(self.plugin_pattern, messages[0]['value'])
        if len(set(plugins)) <= 1:
            return None
        return messages


@register_dataset
class MSAgentBenchLoader(DatasetLoader):
    dataset_type = 'ms-agent-bench'
    datasets = ['damo/MSAgent-Bench']
    subsets = [
        SubsetMeta('default', preprocessor=MSAgentBenchPreprocessor),
        SubsetMeta('default', name='mini', preprocessor=MSAgentBenchMiniPreprocessor, is_weak_subset=True),
    ]
    split = ['train', 'validation']
    tags = ['chat', 'agent', 'multi-round']


# ============================================================================================
# 5. Datasets built from scratch or from several columns at once: transcripts to split, prompts to
#    template, one row fanning out into several.
# ============================================================================================


class MultiRoleAgentPreprocessor(Preprocessor):
    """A group chat flattened to one turn: the earlier speakers become part of the system prompt.

    Only the last utterance is the target, and everything before it is context spoken by *other*
    people -- so it cannot be modelled as alternating user/assistant turns. Rows whose system prompt
    already carries ``next_speakers:`` are left alone: that variant is prompted differently upstream.
    """

    rules_prompt = ('\n\n【注意事项】\n1. 这是聊天室，不要发送私信给任何人\n'
                    '2. 仅代表你个人说话,不要扮演其他人，只根据对话历史进行回复\n'
                    '3. 长话短说，不要说太多话，不要超过50字 ')
    history_prompt = '\n\n【chat history】'
    turn_prompt = '\n {name}:{content}'

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        conv = row['conversations']
        query, response = '', conv[-1]['value']
        system = conv[0]['value'] if conv[0]['from'] == 'system' else ''
        if conv[0]['from'] == 'user':
            query = conv[0]['value']
        elif 'next_speakers:' not in system:
            if '【注意事项】' not in system and system:
                system += self.rules_prompt
            system += self.history_prompt
            system += ''.join(self.turn_prompt.format(name=c['from'], content=c['value']) for c in conv[1:-1])
        if not query or not response:
            return None
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
            }]
        }


@register_dataset
class MultiRoleAgentLoader(DatasetLoader):
    dataset_type = 'ms-agent-multirole'
    datasets = ['iic/MSAgent-MultiRole']
    preprocessor = MultiRoleAgentPreprocessor
    tags = ['chat', 'agent', 'multi-round', 'role-play', 'multi-agent']


class FunctionCallChatmlPreprocessor(Preprocessor):
    """The tool schemas ship as one blank-line-separated blob in their own column.

    The system turn is dropped because it only restates those schemas in prose; once they are proper
    ``tools``, the template renders them itself and a second copy would double the prompt.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = super().preprocess(row)
        if row is None:
            return None
        description = row.pop('function_description', None)
        if description:
            row['tools'] = description.split('\n\n')
        messages = row['messages']
        if messages[0]['role'] == 'system':
            messages.pop(0)
        return row


@register_dataset
class FunctionCallChatmlLoader(DatasetLoader):
    dataset_type = 'function-calling-chatml'
    datasets = [('AI-ModelScope/function-calling-chatml', 'Locutusque/function-calling-chatml')]
    preprocessor = FunctionCallChatmlPreprocessor
    tags = ['agent', 'en', 'sft', '🔥']


class GuanacoPreprocessor(Preprocessor):
    """Earlier turns are a transcript inside ``instruction``, labelled with speaker prefixes.

    The prefixes are misspelt several ways (``Assistenz:``, ``Asssistent:``), so splitting is by the
    whole set of spellings and each part is then checked to be the speaker its position requires --
    strictly alternating user, assistant. A row that breaks the alternation, or that leaves a turn
    without its answer, is dropped rather than guessed at.
    """

    speaker_prefixes = [
        'User:', 'User：', 'Assistant：', 'Assistant:', 'Asssistent:', 'Assistent:', 'Assistenz:'
    ]

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        instruction, input_, output = row['instruction'], row['input'], row['output']
        history = []
        if instruction:
            for idx, part in enumerate(split_str_parts_by(instruction, self.speaker_prefixes)):
                key = part['key'].lower()
                if idx % 2 == 0:
                    if 'user' not in key:
                        return None
                    history.append([part['content'], None])
                else:
                    if 'assist' not in key and 'asssist' not in key:
                        return None
                    history[-1][-1] = part['content']
        if any(not turn[0] or not turn[1] for turn in history):
            return None
        if input_.startswith('User:'):
            input_ = input_[len('User:'):].strip()

        messages = self.converter.history_to_messages(history)
        messages.append({'role': 'user', 'content': input_})
        messages.append({'role': 'assistant', 'content': output})
        return {'messages': messages}


@register_dataset
class GuanacoLoader(DatasetLoader):
    dataset_type = 'guanaco'
    datasets = [('AI-ModelScope/GuanacoDataset', 'JosephusCheung/GuanacoDataset')]
    preprocessor = GuanacoPreprocessor
    tags = ['chat', 'zh']


class HHRLHFPreprocessor(Preprocessor):
    """Preference pair stored as two full transcripts, turns marked by ``\\n\\nHuman:`` and friends.

    Both sides share the same prompt, so they are split the same way and become ``messages`` and
    ``rejected_messages``. ``Hum:`` is a truncation that appears in the data, hence the third
    delimiter.
    """

    turn_pattern = '\n\nHuman:|\n\nAssistant:|\n\nHum:'

    @staticmethod
    def transcript_to_messages(parts: List[str]) -> List[Dict[str, str]]:
        """Alternating utterances, starting with the user, to standard messages."""
        messages = []
        for query, response in zip(parts[::2], parts[1::2]):
            messages.append({'role': 'user', 'content': query})
            messages.append({'role': 'assistant', 'content': response})
        return messages

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        chosen = [s.strip() for s in re.split(self.turn_pattern, row.pop('chosen').strip())]
        rejected = [s.strip() for s in re.split(self.turn_pattern, row.pop('rejected').strip())]
        # The very first turn keeps its label, the split having had no preceding blank line to match.
        if chosen[0].startswith('Human:'):
            assert rejected[0].startswith('Human:'), f'rejected: {rejected}'
            chosen[0] = chosen[0][len('Human:'):].strip()
            rejected[0] = rejected[0][len('Human:'):].strip()
        row['messages'] = self.transcript_to_messages(chosen)
        row['rejected_messages'] = self.transcript_to_messages(rejected)
        return row


@register_dataset
class HHRLHFLoader(DatasetLoader):
    dataset_type = 'hh-rlhf'
    datasets = ['AI-ModelScope/hh-rlhf']
    subsets = ['helpful-base', 'helpful-online', 'helpful-rejection-sampled']
    split = ['train', 'test']
    preprocessor = HHRLHFPreprocessor
    huge_dataset = True
    tags = ['rlhf', 'dpo']


class HHRLHFCNPreprocessor(Preprocessor):
    """The Chinese port splits the pair differently: shared prefix in ``context``, then two endings.

    So the chosen ending is appended to the prefix to make the dialogue, and the rejected one is left
    flat as ``rejected_response`` for the template to expand. Messages here spell content ``text``.
    """

    columns = {'context': 'messages'}
    converter_kwargs = {'content_key': 'text'}

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        row['messages'].append(row.pop('chosen'))
        row['rejected_response'] = row.pop('rejected')['text']
        return super().preprocess(row)


@register_dataset
class HHRLHFCNLoader(DatasetLoader):
    dataset_type = 'hh-rlhf-cn'
    datasets = ['AI-ModelScope/hh_rlhf_cn']
    subsets = ['hh_rlhf', 'harmless_base_cn', 'harmless_base_en', 'helpful_base_cn', 'helpful_base_en']
    split = ['train', 'test']
    preprocessor = HHRLHFCNPreprocessor
    tags = ['rlhf', 'dpo', '🔥']


class HC3Preprocessor(Preprocessor):
    """Human-or-ChatGPT detection, posed as generation: one source row becomes two labelled rows.

    Each row holds a question plus several answers from each source, so it yields one example per
    source -- balanced by construction. Which of a source's answers is used is drawn from
    :attr:`Preprocessor.random_state`, seeded, so a rerun picks the same ones.
    """

    prompt = """Classification Task: Are the following responses from a human or from ChatGPT?
Question: {question}
Answer: {answer}
Category: Human, ChatGPT
Output:"""
    sources = ('Human', 'ChatGPT')

    def build_query(self, row: Dict[str, Any], source: str) -> str:
        answers = row[f'{source.lower()}_answers']
        return self.prompt.format(question=row['query'], answer=self.random_state.choice(answers))

    def preprocess(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        row = self.standardise(row)
        rows = []
        for source in self.sources:
            rows.append(super().preprocess({'query': self.build_query(row, source), 'response': source}))
        return rows


class HC3ClsPreprocessor(HC3Preprocessor):
    """The same two rows as a classification task: an ``int`` label instead of a spelled-out answer."""

    def preprocess(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        row = self.standardise(row)
        rows = []
        for label, source in enumerate(self.sources):
            rows.append(Preprocessor.preprocess(self, {'query': self.build_query(row, source), 'label': label}))
        return rows


@register_dataset
class HC3ChineseLoader(DatasetLoader):
    dataset_type = 'hc3-zh'
    datasets = [('simpleai/HC3-Chinese', 'Hello-SimpleAI/HC3-Chinese')]
    # Every domain is offered twice, as generation and as classification, so a run can pick the task
    # without the dataset having to be registered twice.
    subsets = [
        meta for sub in ('baike', 'open_qa', 'nlpcc_dbqa', 'finance', 'medicine', 'law', 'psychology')
        for meta in (SubsetMeta(sub, preprocessor=HC3Preprocessor),
                     SubsetMeta(sub, name=f'{sub}_cls', preprocessor=HC3ClsPreprocessor))
    ]
    tags = ['text-generation', 'classification', '🔥']


@register_dataset
class HC3Loader(DatasetLoader):
    dataset_type = 'hc3'
    datasets = [('simpleai/HC3', 'Hello-SimpleAI/HC3')]
    subsets = [
        meta for sub in ('finance', 'medicine')
        for meta in (SubsetMeta(sub, preprocessor=HC3Preprocessor),
                     SubsetMeta(sub, name=f'{sub}_cls', preprocessor=HC3ClsPreprocessor))
    ]
    tags = ['text-generation', 'classification', '🔥']


class DureaderPreprocessor(Preprocessor):
    """Reverse the QA task: given a passage and its answer, produce the question.

    The source packs answer and context into one column separated by ``[SEP]``; the question, which
    is the target here, is the second column.
    """

    prompt = """Task: Question Generation
Context: {context}
Answer: {answer}
Question:"""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        answer, context = row['text1'].split('[SEP]')
        return {
            'messages': [{
                'role': 'user',
                'content': self.prompt.format(context=context, answer=answer)
            }, {
                'role': 'assistant',
                'content': row['text2']
            }]
        }


@register_dataset
class DureaderLoader(DatasetLoader):
    dataset_type = 'dureader-robust'
    datasets = ['modelscope/DuReader_robust-QG']
    split = ['train', 'validation', 'test']
    preprocessor = DureaderPreprocessor
    tags = ['text-generation', '🔥']


class CountdownTaskPreprocessor(Preprocessor):
    """Reinforcement-learning task: the answer column becomes the reward ``target``, not a turn.

    The row is deliberately left without an assistant message -- the model has to produce the
    equation, and ``target`` is what its attempt is scored against.
    """

    prompt = ('Using the numbers {numbers}, create an equation that equals {target}.\n'
              'You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.\n'
              'Show your work in <think> </think> tags. And return the final equation and answer '
              'in <answer> </answer> tags, for example <answer> (1 + 2) / 3 * 4 = 4 </answer>.')

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        target = row.pop('response', None)
        # `target` is itself an alias of `response`, so it must not go back on the row before
        # conversion -- the alias pass would promote it and invent an assistant turn out of the answer.
        # The row is deliberately left without one: the model has to produce the equation.
        converted = super().preprocess({'query': self.prompt.format(numbers=row['nums'], target=target)})
        if converted is None:
            return None
        converted['target'] = target
        return converted


@register_dataset
class CountdownTaskLoader(DatasetLoader):
    dataset_type = 'countdown-tasks-3to4'
    datasets = ['zouxuhong/Countdown-Tasks-3to4']
    subsets = ['default']
    preprocessor = CountdownTaskPreprocessor
    tags = ['math']


class SudokuPreprocessor(Preprocessor):
    """Both grids arrive as one 81-character line; wrap them to 9 rows so the layout is readable."""

    prompt = ('Solve the following 9x9 Sudoku puzzle. '
              "Empty cells are marked with '0'. "
              'Provide the completed grid as your answer.\n\n'
              'Puzzle:\n{puzzle}')

    @staticmethod
    def format_grid(line: str) -> str:
        return '\n'.join(line[i:i + 9] for i in range(0, len(line), 9))

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        puzzle = self.format_grid(row['query'].replace('.', '0'))
        return super().preprocess({
            'query': self.prompt.format(puzzle=puzzle),
            'response': self.format_grid(row['response'])
        })


@register_dataset
class SudokuLoader(DatasetLoader):
    dataset_type = 'sudoku-extreme-1k'
    datasets = [('sapientinc/sudoku-extreme-1k', 'sapientinc/sudoku-extreme-1k')]
    preprocessor = SudokuPreprocessor
    tags = ['math']


class XlamFunctionCallingPreprocessor(Preprocessor):
    """Tool calls as the target: each requested call becomes its own ``tool_call`` message."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        messages = [{'role': 'user', 'content': row['query']}]
        for call in json.loads(row['answers']):
            messages.append({'role': 'tool_call', 'content': json.dumps(call)})
        return {'messages': messages, 'tools': row['tools']}


class XlamFunctionCallingGRPOPreprocessor(Preprocessor):
    """The ``grpo`` view: one call, rendered as the text a policy is expected to emit.

    ``solution`` repeats the answer because the reward function reads it from there while ``response``
    is what training compares against. The pick is seeded, unlike legacy's use of the global numpy
    generator, so two runs over this subset agree.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        answers = row['response']
        if isinstance(answers, str):
            answers = json.loads(answers)
        answer = self.random_state.choice(answers)
        response = f"Action: {answer['name']}\nAction Input: {json.dumps(answer['arguments'])}"
        # `solution` is set after conversion: it is an alias of `response`, so handing it to the
        # converter would make the alias pass fight over which one is the answer.
        converted = super().preprocess({'query': row['query'], 'response': response, 'tools': row['tools']})
        if converted is None:
            return None
        converted['solution'] = response
        return converted


@register_dataset
class XlamFunctionCallingLoader(DatasetLoader):
    dataset_type = 'xlam-function-calling-60k'
    datasets = [('LLM-Research/xlam-function-calling-60k', 'Salesforce/xlam-function-calling-60k')]
    subsets = [
        SubsetMeta('dataset', name='default', preprocessor=XlamFunctionCallingPreprocessor),
        SubsetMeta('dataset', name='grpo', preprocessor=XlamFunctionCallingGRPOPreprocessor),
    ]
    tags = ['agent', 'grpo', '🔥']


# ============================================================================================
# 4. Rows that are not a dialogue at all -- a label, a score, or a set of candidate answers.
#
# What these have in common is that the standard row they produce is *not* the usual
# ``messages``-and-nothing-else: a classification row carries an integer ``label`` and no assistant
# turn, an embedding row carries ``positive_messages`` / ``negative_messages`` beside ``messages``.
# Every one of those columns is in :attr:`Preprocessor.MESSAGE_COLUMNS`, so they survive Arrow as
# they are; nothing about the shape needs special handling below this file.
#
# Each also comes in several views of the same rows -- generation, classification, regression -- which
# is why they are registered as sibling subsets with different preprocessors rather than as separate
# datasets: the choice is a task decision, not a data one.
# ============================================================================================


class ClsGenerationPreprocessor(Preprocessor):
    """A classification dump posed as generation: the label set is stated in the prompt, its name is
    the answer.

    The alternative view -- keep the integer and let a classification head read it -- is
    :class:`JdClsPreprocessor` below. Both are offered for the datasets that have them, because which
    one is right depends on the ``task_type`` of the run and not on the dataset.

    Legacy passed ``labels`` / ``task`` / ``is_pair_seq`` to a constructor and stored a pre-formatted
    prompt on the instance. Here they are class attributes and the prompt is built per row: the number
    of sentence columns is what ``is_pair_seq`` really encoded, so naming them says it directly.
    """

    labels: Sequence[str] = ()
    task: str = ''
    sentence_keys: Sequence[str] = ('sentence', )

    def build_query(self, row: Dict[str, Any]) -> str:
        if len(self.sentence_keys) > 1:
            sentences = '\n'.join(f'Sentence{i}: {row[key]}' for i, key in enumerate(self.sentence_keys, start=1))
        else:
            sentences = f'Sentence: {row[self.sentence_keys[0]]}'
        return f"Task: {self.task}\n{sentences}\nCategory: {', '.join(self.labels)}\nOutput:"

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        label = row.get('label')
        if label is None:
            # The test split ships unlabelled; without the answer the row teaches nothing.
            return None
        return super().preprocess({'query': self.build_query(row), 'response': self.labels[int(label)]})


class AdvertiseGenPreprocessor(Preprocessor):
    """Keywords in, ad copy out. The task is only legible once it is spelled out in the prompt.

    ``content`` has to be renamed explicitly: :class:`ResponseConverter` reads it as an alias of
    ``response``, which is the opposite of what it holds here. Caller renames win over a format's own
    aliases, which is what makes that fixable by declaration.
    """

    columns = {'content': 'query', 'summary': 'response'}
    prompt = """Task: Generating advertisements based on keywords.
Keywords: {keywords}
Advertisements:"""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        row['query'] = self.prompt.format(keywords=row['query'])
        return super().preprocess(row)


@register_dataset
class AdvertiseGenLoader(DatasetLoader):
    dataset_type = 'advertise-gen-zh'
    datasets = [('lvjianjin/AdvertiseGen', 'shibing624/AdvertiseGen')]
    split = ['train', 'validation']
    preprocessor = AdvertiseGenPreprocessor
    tags = ['text-generation', '🔥']


class CmnliPreprocessor(ClsGenerationPreprocessor):
    labels = ('neutral', 'entailment', 'contradiction')
    task = 'Natural Language Inference'
    sentence_keys = ('sentence1', 'sentence2')


@register_dataset
class ClueLoader(DatasetLoader):
    dataset_type = 'cmnli'
    datasets = [('modelscope/clue', 'clue')]
    subsets = ['cmnli']
    split = ['train', 'validation']
    preprocessor = CmnliPreprocessor
    tags = ['text-generation', 'classification']


class JdSentimentPreprocessor(ClsGenerationPreprocessor):
    labels = ('negative', 'positive')
    task = 'Sentiment Classification'


class JdClsPreprocessor(Preprocessor):
    """The same reviews as a classification task: the label stays an ``int``, and there is no answer
    turn at all."""

    columns = {'sentence': 'query'}

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        converted = super().preprocess(row)
        if converted is None:
            return None
        converted['label'] = int(converted['label'])
        return converted


@register_dataset
class JdSentimentLoader(DatasetLoader):
    dataset_type = 'jd-sentiment-zh'
    datasets = [('DAMO_NLP/jd', None)]
    subsets = [
        SubsetMeta('default', preprocessor=JdSentimentPreprocessor),
        SubsetMeta('default', name='cls', preprocessor=JdClsPreprocessor),
    ]
    split = ['train', 'validation']
    tags = ['text-generation', 'classification', '🔥']


class StsbPreprocessor(Preprocessor):
    """A sentence pair and how alike they are, as an embedding example.

    The second sentence becomes a ``positive_messages`` entry rather than an answer: an embedding loss
    compares the two encodings, so the pair is the example and the score is its target.
    """

    # Keep every pair, however unalike, since the score is the label. A subclass raises the bar for
    # the InfoNCE view, where a "positive" that is only loosely related is a mislabelled positive.
    sim_threshold: Optional[float] = None

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        score = float(row['score'])
        if self.sim_threshold is not None and score < self.sim_threshold:
            return None
        return {
            'messages': [{
                'role': 'user',
                'content': row['sentence1']
            }],
            'positive_messages': [[{
                'role': 'user',
                'content': row['sentence2']
            }]],
            'label': score,
        }


class StsbInfoncePreprocessor(StsbPreprocessor):
    sim_threshold = 0.75


class StsbGeneratePreprocessor(Preprocessor):
    """The same pairs as generation: the score is the text to produce, to one decimal place."""

    prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 1.0.
Sentence 1: {text1}
Sentence 2: {text2}
Similarity score: """

    def build_query(self, row: Dict[str, Any]) -> str:
        return self.prompt.format(text1=row['sentence1'], text2=row['sentence2'])

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return super().preprocess({'query': self.build_query(row), 'response': f"{row['score']:.1f}"})


class StsbRegressionPreprocessor(StsbGeneratePreprocessor):
    """The same prompt, but the score stays a float for a regression head instead of being written out."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return Preprocessor.preprocess(self, {'query': self.build_query(row), 'label': row['score']})


@register_dataset
class StsbLoader(DatasetLoader):
    dataset_type = 'stsb'
    datasets = ['sentence-transformers/stsb']
    subsets = [
        SubsetMeta('default', preprocessor=StsbPreprocessor),
        SubsetMeta('default', name='positive', preprocessor=StsbInfoncePreprocessor),
        SubsetMeta('default', name='generate', preprocessor=StsbGeneratePreprocessor),
        SubsetMeta('default', name='reg', preprocessor=StsbRegressionPreprocessor),
    ]
    tags = ['similarity', '🔥']


class MTEBRerankPreprocessor(Preprocessor):
    """A query with its relevant and irrelevant documents: a reranker's example.

    Both document columns may hold one string or a list of them, and both become lists of one-turn
    dialogues -- the ``assistant`` role because a document is what the query is answered *with*.
    """

    @staticmethod
    def as_messages(documents: Any) -> List[List[Dict[str, str]]]:
        if not isinstance(documents, list):
            documents = [documents]
        return [[{'role': 'assistant', 'content': document}] for document in documents]

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return {
            'messages': [{
                'role': 'user',
                'content': row['query']
            }],
            'positive_messages': self.as_messages(row['positive']),
            'negative_messages': self.as_messages(row['negative']),
        }


@register_dataset
class ScidocsRerankingLoader(DatasetLoader):
    dataset_type = 'scidocs-reranking'
    datasets = [('MTEB/scidocs-reranking', 'mteb/scidocs-reranking')]
    split = ['validation', 'test']
    preprocessor = MTEBRerankPreprocessor
    tags = ['rerank', '🔥']


@register_dataset
class StackoverflowRerankingLoader(DatasetLoader):
    dataset_type = 'stackoverflowdupquestions-reranking'
    datasets = [('MTEB/stackoverflowdupquestions-reranking', 'mteb/stackoverflowdupquestions-reranking')]
    split = ['train', 'test']
    preprocessor = MTEBRerankPreprocessor
    tags = ['rerank', '🔥']


class SelfCognitionPreprocessor(Preprocessor):
    """Rows that teach a model who it is, with the identity left as a placeholder to fill in.

    The dataset ships ``{{NAME}}`` and ``{{AUTHOR}}`` in both the question and the answer, so the
    substituted values come from the *run* (``--model_name`` / ``--model_author``) rather than from the
    data. That makes this the one preprocessor here that cannot be configured by declaration alone:
    :meth:`SelfCognitionLoader.build_preprocessor` passes them in.

    A row's ``tag`` says which language it is written in, which is what picks between the Chinese and
    English form of a name. An unset name is left as-is rather than blanked: the placeholder surviving
    into the training text is at least visible, whereas an empty string is not.
    """

    # Appended / prepended to shape the answer for a specific thinking convention; see the subsets.
    query_suffix: str = ''
    response_prefix: str = ''

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name: Optional[Tuple[str, str]] = None
        self.author: Optional[Tuple[str, str]] = None

    def set_name_author(self, name: Any, author: Any) -> None:
        self.name = self.as_language_pair(name)
        self.author = self.as_language_pair(author)

    @staticmethod
    def as_language_pair(value: Any) -> Optional[Tuple[str, str]]:
        """Normalise what the command line gave into a ``(zh, en)`` pair, or ``None`` for unset.

        One value means the same name is right in both languages -- true of most model names, which
        are not translated -- so it is duplicated rather than left half-empty.
        """
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if not value or value[0] is None:
            return None
        if len(value) == 1 or value[1] is None:
            return (value[0], value[0])
        return (value[0], value[1])

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        for key in ('name', 'author'):
            pair = getattr(self, key)
            if pair is None:
                continue
            value = pair[0] if row['tag'] == 'zh' else pair[1]
            placeholder = '{{' + key.upper() + '}}'
            row['query'] = row['query'].replace(placeholder, value)
            row['response'] = row['response'].replace(placeholder, value)
        row['query'] = row['query'] + self.query_suffix
        row['response'] = self.response_prefix + row['response']
        return super().preprocess(row)


class Qwen3SelfCognitionPreprocessor(SelfCognitionPreprocessor):
    """For a model whose thinking is switched by a token in the question: ask it not to think, and
    supply the empty thinking block its answers are expected to open with."""

    query_suffix = ' /no_think'
    response_prefix = '<think>\n\n</think>\n\n'


class EmptyThinkSelfCognitionPreprocessor(SelfCognitionPreprocessor):
    """The same empty thinking block, for a model that needs no switch in the question."""

    response_prefix = '<think>\n\n</think>\n\n'


@register_dataset
class SelfCognitionLoader(DatasetLoader):
    dataset_type = 'self-cognition'
    datasets = [('swift/self-cognition', 'modelscope/self-cognition')]
    # Declared at family level as well as per subset, so every subset resolves to *some* subclass of
    # `SelfCognitionPreprocessor`: `build_preprocessor` below calls a setter only this class has.
    preprocessor = SelfCognitionPreprocessor
    subsets = [
        SubsetMeta('default'),
        SubsetMeta('default', name='qwen3', preprocessor=Qwen3SelfCognitionPreprocessor),
        SubsetMeta('default', name='empty_think', preprocessor=EmptyThinkSelfCognitionPreprocessor),
    ]
    tags = ['chat', 'self-cognition', '🔥']

    def build_preprocessor(self, subset: SubsetMeta) -> Any:
        """Hand the run's model identity to the preprocessor -- the one thing this dataset cannot
        declare.

        Legacy reached the other way: its loader walked every ``preprocess_func`` of every dataset
        looking for instances of this class and called a setter on them, so a dataset-specific need
        was implemented in the shared load path. Overriding this hook keeps it in the one family that
        has it.
        """
        preprocessor = super().build_preprocessor(subset)
        preprocessor.set_name_author(self._kwargs.get('model_name'), self._kwargs.get('model_author'))
        return preprocessor



