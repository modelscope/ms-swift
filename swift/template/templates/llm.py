# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from swift.utils import get_env_args
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Prompt, get_last_user_round
from .llama import Llama3_2TemplateMeta
from .qwen import Qwen2VLTemplate, QwenTemplateMeta
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta

register_template(
    TemplateMeta(
        LLMTemplateType.default,
        prefix=[],
        prompt=['### Human:\n{{QUERY}}\n\n### Assistant:\n'],
        chat_sep=['\n\n'],
        default_system=DEFAULT_SYSTEM,
        system_prefix=['{{SYSTEM}}\n\n'],
        auto_add_bos=True))

register_template(
    TemplateMeta(
        LLMTemplateType.modelscope_agent,
        prefix=[],
        prompt=[' \n\n<|user|>:{{QUERY}} \n\n<|assistant|>:'],
        chat_sep=[],
        suffix=[' \n\n</s>'],
        system_prefix=[' \n\n<|system|>:{{SYSTEM}}'],
        default_system=DEFAULT_SYSTEM,
    ))


class GMETemplate(Qwen2VLTemplate):

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        if inputs.messages[-1]['role'] != 'assistant':
            inputs.messages.append({'role': 'assistant', 'content': ''})
        return inputs


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_gme, template_cls=GMETemplate, suffix=['<|endoftext|>']))


class JinaRerankerM0Template(Qwen2VLTemplate):

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        instruction = ''
        if inputs.system is not None:
            instruction = inputs.system
            inputs.system = None
        query = inputs.messages[0]['content']
        document = inputs.messages[1]['content']
        user_message = instruction + '\n' + '**Query**:\n' + query + '\n' + '**Document**:\n' + document
        inputs.messages = [{'role': 'user', 'content': user_message}]
        return inputs


register_template(
    TemplateMeta(
        MLLMTemplateType.jina_reranker_m0,
        template_cls=JinaRerankerM0Template,
        prefix=[],
        chat_sep=[],
        prompt=['{{QUERY}}']))

register_template(
    TemplateMeta(LLMTemplateType.baichuan, prefix=['{{SYSTEM}}'], prompt=[[195], '{{QUERY}}', [196]], chat_sep=[]))

register_template(
    TemplateMeta(
        LLMTemplateType.baichuan_m1,
        prefix=[],
        prompt=['<C_Q>{{QUERY}}<C_A>'],
        chat_sep=[],
        suffix=['<C_A>'],
        system_prefix=['<B_SYS>{{SYSTEM}}'],
        default_system=DEFAULT_SYSTEM,
    ))

register_template(
    TemplateMeta(
        LLMTemplateType.numina,
        prefix=[['bos_token_id']],
        prompt=['### Problem: {{QUERY}}\n### Solution: '],
        chat_sep=['\n'],
        system_prefix=[['bos_token_id'], '{{SYSTEM}}']))

register_template(
    TemplateMeta(
        LLMTemplateType.mistral_nemo,
        prefix=['<s>[INST] '],
        prompt=['{{SYSTEM}}\n\n', '{{QUERY}}[/INST]'],
        chat_sep=['</s>[INST] '],
        suffix=['</s>']))

register_template(
    TemplateMeta(
        LLMTemplateType.xverse,
        prefix=['{{SYSTEM}}'],
        prompt=['Human: {{QUERY}}\n\nAssistant: '],
        chat_sep=[['eos_token_id']]))

register_template(TemplateMeta(LLMTemplateType.yuan, prefix=[], prompt=['{{QUERY}}<sep>'], chat_sep=None))
register_template(
    TemplateMeta(
        LLMTemplateType.ziya,
        prefix=[['bos_token_id'], '{{SYSTEM}}'],
        prompt=['<human>:{{QUERY}}\n<bot>:'],
        chat_sep=['\n']))

register_template(
    TemplateMeta(
        LLMTemplateType.skywork,
        prefix=['<s>{{SYSTEM}}'],
        prompt=['</s><s>[USER]{{QUERY}}[SEP][BOT]'],
        chat_sep=None,
        suffix=['[SEP]</s>']))

register_template(
    Llama3_2TemplateMeta(
        LLMTemplateType.skywork_o1,
        default_system=(
            'You are Skywork-o1, a thinking model developed by Skywork AI, specializing in solving complex problems '
            "involving mathematics, coding, and logical reasoning through deep thought. When faced with a user's "
            'request, you first engage in a lengthy and in-depth thinking process to explore possible solutions to '
            'the problem. After completing your thoughts, you then provide a detailed explanation of the solution '
            'process in your response.'),
    ))

register_template(
    TemplateMeta(
        LLMTemplateType.bluelm,
        prefix=[['bos_token_id'], '{{SYSTEM}}'],
        prompt=['[|Human|]:{{QUERY}}[|AI|]:'],
        chat_sep=[]))

register_template(
    TemplateMeta(
        LLMTemplateType.codefuse_codellama,
        prefix=['{{SYSTEM}}'],
        prompt=['<|role_start|>human<|role_end|>{{QUERY}}<|role_start|>bot<|role_end|>'],
        chat_sep=[]))

register_template(
    TemplateMeta(
        LLMTemplateType.codefuse,
        prefix=[],
        prompt=['<s>human\n{{QUERY}}\n<s>bot\n'],
        chat_sep=[['eos_token_id'], '\n'],
        system_prefix=['<s>system\n{{SYSTEM}}\n']))

register_template(
    TemplateMeta(
        LLMTemplateType.zephyr,
        prefix=[],
        prompt=['<|user|>\n{{QUERY}}</s>\n<|assistant|>\n'],
        chat_sep=['</s>\n'],
        suffix=['</s>'],
        system_prefix=['<|system|>\n{{SYSTEM}}</s>\n']))

register_template(
    TemplateMeta(
        LLMTemplateType.sus,
        prefix=['{{SYSTEM}}'],
        prompt=['### Human: {{QUERY}}\n\n### Assistant: '],
        chat_sep=['<|endoftext|>'],
        suffix=['<|endoftext|>']))

register_template(
    TemplateMeta(
        LLMTemplateType.orion,
        prefix=['<s>{{SYSTEM}}'],
        prompt=['Human: {{QUERY}}\n\nAssistant: </s>'],
        chat_sep=['</s>'],
        suffix=['</s>']))


@dataclass
class TeleChatTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: [['user_token_id'], '{{QUERY}}', ['bot_token_id']])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: [['eos_token_id']])
    suffix: Prompt = field(default_factory=lambda: [['eos_token_id']])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<_system>{{SYSTEM}}\n'])
    auto_add_bos: bool = True


register_template(TeleChatTemplateMeta(LLMTemplateType.telechat))

telechat_system = '你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。'
register_template(TeleChatTemplateMeta(LLMTemplateType.telechat2, default_system=telechat_system))


class TeleChat3Template(Template):

    class _TruthyEmptySystem(str):

        def __bool__(self):
            return True

    def _get_system(self, inputs):
        system = super()._get_system(inputs)
        if system == '' and not inputs.tools:
            system = self._TruthyEmptySystem('')
        return system

    def _get_response_prefix(self, inputs=None):
        response_prefix = None if inputs is None else inputs.chat_template_kwargs.get('response_prefix')
        if response_prefix is None:
            response_prefix = self.response_prefix
        if response_prefix is not None:
            return response_prefix
        if not self.use_chat_template:
            return ''
        return self.template_meta.thinking_prefix

    @staticmethod
    def _to_content_segments(content):
        if isinstance(content, list) and (not content or not isinstance(content[0], int)):
            return content.copy()
        return [content]

    @staticmethod
    def _expand_metadata(value, num_segments, key):
        if isinstance(value, list):
            if len(value) > num_segments:
                raise ValueError(f'{key} has {len(value)} values for {num_segments} content segments.')
            return value + [None] * (num_segments - len(value))
        return [value] * num_segments

    def _merge_assistant_messages(self, pre_message, message):
        pre_segments = self._to_content_segments(pre_message['content'])
        cur_segments = self._to_content_segments(message['content'])
        pre_message['content'] = pre_segments + cur_segments
        for key in ['loss', 'loss_scale']:
            if key not in pre_message and key not in message:
                continue
            pre_values = self._expand_metadata(pre_message.get(key), len(pre_segments), key)
            cur_values = self._expand_metadata(message.get(key), len(cur_segments), key)
            pre_message[key] = pre_values + cur_values

    def _extend_assistant_metadata(self, message, old_num_segments):
        new_num_segments = len(self._to_content_segments(message['content']))
        if new_num_segments <= old_num_segments:
            return
        for key in ['loss', 'loss_scale']:
            if key not in message:
                continue
            values = message[key]
            if not isinstance(values, list):
                continue
            values = self._expand_metadata(values, old_num_segments, key)
            fill_value = values[-1] if values else None
            message[key] = values + [fill_value] * (new_num_segments - old_num_segments)

    def _merge_natural_messages(self, inputs):
        messages = inputs.messages
        i = 1
        while i < len(messages):
            pre_message, message = messages[i - 1], messages[i]
            role = message['role']
            if pre_message['role'] != role or role not in {'assistant', 'user'}:
                i += 1
                continue
            pre_content = pre_message.get('content') or ''
            content = message.get('content') or ''
            if role == 'assistant':
                for key in ['loss', 'loss_scale']:
                    if key not in pre_message and key not in message:
                        continue
                    if key not in pre_message or key not in message or pre_message[key] != message[key]:
                        raise ValueError(
                            f'TeleChat3 cannot merge consecutive assistant messages with different `{key}` values. '
                            'Merge the messages before encoding or use the same value.')
                pre_message['content'] = pre_content + content
                tool_calls = list(pre_message.get('tool_calls') or []) + list(message.get('tool_calls') or [])
                if tool_calls:
                    pre_message['tool_calls'] = tool_calls
                pre_reasoning = pre_message.get('reasoning_content')
                reasoning = message.get('reasoning_content')
                if isinstance(reasoning, str):
                    pre_message['reasoning_content'] = (pre_reasoning or '') + reasoning
            else:
                pre_message['content'] = pre_content + content
            messages.pop(i)

    def _prepare_assistant_thinking(self, inputs):
        for message in inputs.messages:
            if message['role'] != 'assistant':
                continue
            content = message['content']
            if isinstance(content, list) and (not content or not isinstance(content[0], int)):
                for i, value in enumerate(content):
                    if isinstance(value, str):
                        content[i] = value.split('</think>')[-1].lstrip('\n')
            elif isinstance(content, str):
                message['content'] = content.split('</think>')[-1].lstrip('\n')

    def _preprocess_tool_call_jinja(self, inputs):
        messages = inputs.messages
        i = 0
        while i < len(messages):
            if messages[i]['role'] != 'tool_call':
                i += 1
                continue
            i_start = i
            while i + 1 < len(messages) and messages[i + 1]['role'] == 'tool_call':
                i += 1
            tool_calls = []
            for message in messages[i_start:i + 1]:
                tool_call = self.agent_template._parse_tool_call(message['content'])
                tool_calls.append({'type': 'function', 'function': tool_call})
            if i_start > 0 and messages[i_start - 1]['role'] == 'assistant':
                assistant_message = messages[i_start - 1]
                if assistant_message.get('content') is None:
                    assistant_message['content'] = ''
                assistant_message['tool_calls'] = list(assistant_message.get('tool_calls') or []) + tool_calls
                messages[i_start:i + 1] = []
                i = i_start
            else:
                messages[i_start:i + 1] = [{'role': 'assistant', 'content': '', 'tool_calls': tool_calls}]
                i = i_start + 1

    def _preprocess_structured_tool_calls_swift(self, inputs):
        messages = inputs.messages
        i = 0
        while i < len(messages):
            message = messages[i]
            tool_calls = message.pop('tool_calls', None) if message['role'] == 'assistant' else None
            if not tool_calls:
                i += 1
                continue
            tool_call_messages = []
            for tool_call in tool_calls:
                function = tool_call.get('function', tool_call)
                arguments = function.get('arguments') or {}
                tool_call_message = {
                    'role': 'tool_call',
                    'content': json.dumps({
                        'name': function['name'],
                        'arguments': arguments
                    }, ensure_ascii=False)
                }
                for key in ['loss', 'loss_scale']:
                    value = message.get(key)
                    if key in message:
                        tool_call_message[key] = value[-1] if isinstance(value, list) and value else value
                tool_call_messages.append(tool_call_message)
            messages[i + 1:i + 1] = tool_call_messages
            i += len(tool_call_messages) + 1

    def _normalize_structured_tool_calls(self, inputs):
        for message in inputs.messages:
            if message['role'] != 'assistant':
                continue
            for tool_call in message.get('tool_calls') or []:
                function = tool_call.get('function', tool_call)
                arguments = function.get('arguments')
                if isinstance(arguments, str):
                    parsed_arguments = self.agent_template._parse_json(arguments)
                    if parsed_arguments is not None:
                        function['arguments'] = parsed_arguments

    def _swift_prepare_inputs(self, inputs):
        # Normalize natural assistant text before tool calls are expanded into strings.
        self._normalize_structured_tool_calls(inputs)
        self._merge_natural_messages(inputs)
        if self.template_backend == 'jinja':
            self._preprocess_tool_call_jinja(inputs)
            return
        self._prepare_assistant_thinking(inputs)
        self._preprocess_structured_tool_calls_swift(inputs)
        self._preprocess_tool_call(inputs)
        messages = inputs.messages
        if len(messages) < 2:
            return
        i = 1
        while i < len(messages):
            pre_message, message = messages[i - 1], messages[i]
            pre_role, pre_content = pre_message['role'], pre_message['content']
            role, content = message['role'], message['content']
            if pre_role == 'assistant' and role == 'tool' and self.template_backend == 'swift':
                i_start = i
                while i + 1 < len(messages) and messages[i + 1]['role'] == 'tool':
                    i += 1
                old_num_segments = len(self._to_content_segments(pre_content))
                pre_message['content'], tool_content = self.agent_template._format_tool_responses(
                    pre_content, messages[i_start:i + 1])
                self._extend_assistant_metadata(pre_message, old_num_segments)
                messages[i_start:i + 1] = [{'role': 'tool', 'content': tool_content}]
                i = i_start + 1
            elif pre_role == 'assistant' and role == 'assistant' or pre_role == 'user' and role == 'user':
                if self.template_backend == 'swift' and pre_role == 'assistant':
                    self._merge_assistant_messages(pre_message, message)
                else:
                    pre_message['content'] = pre_content + content
                messages.pop(i)
            else:
                i += 1

    def _remove_history_thinking(self, inputs) -> None:
        # Every assistant turn is already normalized against the model Jinja above.
        pass


@dataclass
class TeleChat3TemplateMeta(TemplateMeta):
    template_cls: type = TeleChat3Template
    prefix: Prompt = field(default_factory=lambda: ['<_system>'])
    prompt: Prompt = field(default_factory=lambda: ['<_user>{{QUERY}}<_bot>'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<_end>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<_end>\n'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<_system>{{SYSTEM}}\n'])
    is_thinking: bool = True
    thinking_prefix: str = '<think>\n'
    agent_template: Optional[str] = 'telechat3'


register_template(TeleChat3TemplateMeta(LLMTemplateType.telechat3))


class TeleChat3CoderTemplate(TeleChat3Template):

    def __init__(self, *args, **kwargs):
        if kwargs.get('enable_thinking') is None:
            kwargs['enable_thinking'] = True
        super().__init__(*args, **kwargs)

    def _prepare_assistant_thinking(self, inputs):
        messages = inputs.messages
        last_user_round = get_last_user_round(messages)
        clear_thinking_defined = ('clear_thinking' in self.chat_template_kwargs
                                  or 'clear_thinking' in inputs.chat_template_kwargs)
        clear_thinking = inputs.chat_template_kwargs.get('clear_thinking',
                                                         self.chat_template_kwargs.get('clear_thinking'))
        preserve_all = clear_thinking_defined and not clear_thinking
        for i, message in enumerate(messages):
            if message['role'] != 'assistant':
                continue
            preserve_reasoning = preserve_all or i > last_user_round
            content = message['content']
            reasoning_content = message.pop('reasoning_content', None)
            if isinstance(content, list) and (not content or not isinstance(content[0], int)):
                for j, value in enumerate(content):
                    if isinstance(value, str):
                        reasoning = reasoning_content if j == 0 else None
                        content[j] = self._normalize_current_thinking(value, reasoning, preserve_reasoning)
            elif isinstance(content, str):
                message['content'] = self._normalize_current_thinking(content, reasoning_content, preserve_reasoning)

    @staticmethod
    def _normalize_current_thinking(content: str,
                                    reasoning_content: Optional[str] = None,
                                    preserve_reasoning: bool = True) -> str:
        if not isinstance(reasoning_content, str):
            reasoning_content = ''
            if '</think>' in content:
                reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                content = content.split('</think>')[-1].lstrip('\n')
        has_reasoning = bool(reasoning_content)
        reasoning_content = reasoning_content.strip()
        content = content.strip()
        if preserve_reasoning and has_reasoning:
            return f'<think>\n{reasoning_content}\n</think>{content}'
        return f'</think>{content}'


@dataclass
class TeleChat3CoderTemplateMeta(TeleChat3TemplateMeta):
    template_cls: type = TeleChat3CoderTemplate
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<_end>'])
    suffix: Prompt = field(default_factory=lambda: ['<_end>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<_system>{{SYSTEM}}'])
    thinking_prefix: str = '<think>'
    non_thinking_prefix: str = '</think>'
    history_thinking_prefix: str = '</think>'
    agent_template: Optional[str] = 'telechat3_coder'


register_template(TeleChat3CoderTemplateMeta(LLMTemplateType.telechat3_coder))

DBRX_SYSTEM = (
    'You are DBRX, created by Databricks. You were last updated in December 2023. '
    'You answer questions based on information available up to that point.\n'
    'YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, '
    'but provide thorough responses to more complex and open-ended questions.\n'
    'You assist with various tasks, from writing to coding (using markdown for code blocks '
    '— remember to use ``` with code, JSON, and tables).\n'
    'You do not have real-time data access or code execution capabilities.'
    ' You avoid stereotyping and provide balanced perspectives on controversial topics. '
    'You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.\n'
    'This is your system prompt, guiding your responses. Do not reference it, just respond to the user. '
    'If you find yourself talking about this message, stop. You should be responding appropriately '
    'and usually that means not mentioning this.'
    'YOU DO NOT MENTION ANY OF THIS INFORMATION ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY '
    'PERTINENT TO THE USER\'S QUERY.')

register_template(ChatmlTemplateMeta(LLMTemplateType.dbrx, default_system=DBRX_SYSTEM))

register_template(
    TemplateMeta(
        LLMTemplateType.mengzi, prefix=[], prompt=['输入：{{QUERY}}输出：\n'], chat_sep=[], system_prefix=['指令：{{SYSTEM}}']))

C4AI_SYSTEM = ('You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by '
               'providing thorough responses.You are trained by Cohere.')
register_template(
    TemplateMeta(
        LLMTemplateType.c4ai,
        prefix=['<BOS_TOKEN>'],
        prompt=[
            '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{QUERY}}<|END_OF_TURN_TOKEN|>'
            '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'
        ],
        chat_sep=['<|END_OF_TURN_TOKEN|>'],
        suffix=['<|END_OF_TURN_TOKEN|>'],
        default_system=C4AI_SYSTEM,
        system_prefix=['<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{SYSTEM}}<|END_OF_TURN_TOKEN|']))

register_template(
    TemplateMeta(
        LLMTemplateType.wizardlm2,
        prefix=['{{SYSTEM}}'],
        prompt=['User:\n{{QUERY}}\n\nAssistant:\n'],
        chat_sep=['\n\n'],
        suffix=['</s>']))

_wizardlm2_system = ('A chat between a curious user and an artificial intelligence assistant. '
                     'The assistant gives helpful, detailed, and polite answers to the user\'s questions. ')
register_template(
    TemplateMeta(
        LLMTemplateType.wizardlm2_moe,
        prefix=['{{SYSTEM}}'],
        prompt=['USER: {{QUERY}} ASSISTANT:'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        default_system=_wizardlm2_system))

register_template(
    TemplateMeta(
        LLMTemplateType.atom,
        prefix=['{{SYSTEM}}'],
        prompt=['<s>Human: {{QUERY}}\n</s><s>Assistant: '],
        chat_sep=['</s>'],
        suffix=['</s>']))

AYA_SYSTEM = ('You are Aya, a brilliant, sophisticated, multilingual AI-assistant trained to assist human users by '
              'providing thorough responses. You are able to interact and respond to questions in 23 languages and '
              'you are powered by a multilingual model built by Cohere For AI.')
register_template(
    TemplateMeta(
        LLMTemplateType.aya,
        prefix=['<BOS_TOKEN>'],
        prompt=[
            '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{QUERY}}<|END_OF_TURN_TOKEN|>'
            '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'
        ],
        chat_sep=['<|END_OF_TURN_TOKEN|>'],
        suffix=['<|END_OF_TURN_TOKEN|>'],
        default_system=AYA_SYSTEM,
        system_prefix=['<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{SYSTEM}}<|END_OF_TURN_TOKEN|']))

register_template(
    TemplateMeta(
        LLMTemplateType.ling,
        prefix=[],
        system_prefix=['<role>SYSTEM</role>{{SYSTEM}}'],
        prompt=['<role>HUMAN</role>{{QUERY}}<role>ASSISTANT</role>'],
        chat_sep=[],
        suffix=['<|endoftext|>'],
    ))

register_template(
    QwenTemplateMeta(
        LLMTemplateType.mimo_rl,
        default_system='You are MiMo, an AI assistant developed by Xiaomi.',
    ))

register_template(
    TemplateMeta(
        LLMTemplateType.dots1,
        prefix=['<|system|>{{SYSTEM}}<|endofsystem|>'],
        prompt=['<|userprompt|>{{QUERY}}<|endofuserprompt|><|response|>'],
        chat_sep=['<|endofresponse|>'],
        suffix=['<|endofresponse|>'],
        default_system='You are a helpful assistant.',
    ))

register_template(
    TemplateMeta(
        LLMTemplateType.hunyuan_moe,
        prefix=['<|startoftext|>'],
        system_prefix=['<|startoftext|>{{SYSTEM}}<|extra_4|>'],
        prompt=['{{QUERY}}<|extra_0|>'],
        chat_sep=['<|eos|><|startoftext|>'],
        suffix=['<|eos|>'],
    ))


class HunyuanTemplate(Template):

    def _remove_thinking_content(self, content: str) -> str:
        content = content.split('<answer>')[-1].rstrip()
        if content.endswith('</answer>'):
            content = content[:-len('</answer>')]
        return self.template_meta.history_thinking_prefix + content.strip()


register_template(
    TemplateMeta(
        LLMTemplateType.hunyuan,
        prefix=['<｜hy_begin▁of▁sentence｜>'],
        system_prefix=['<｜hy_begin▁of▁sentence｜>{{SYSTEM}}<｜hy_place▁holder▁no▁3｜>'],
        prompt=['<｜hy_User｜>{{QUERY}}<｜hy_Assistant｜>'],
        chat_sep=['<｜hy_place▁holder▁no▁2｜>'],
        suffix=['<｜hy_place▁holder▁no▁2｜>'],
        template_cls=HunyuanTemplate,
        is_thinking=True,
        non_thinking_prefix='<think>\n\n</think>\n',
        agent_template='hunyuan_hermes'))


class HyV3PreviewTemplate(Template):
    HYTK = ''

    def init_env_args(self):
        super().init_env_args()
        # reasoning_effort: "no_think", "low", "high" (deep chain-of-thought)
        # TODO: sample level
        self.reasoning_effort = get_env_args('reasoning_effort', str, None)
        if self.reasoning_effort is None:
            self.reasoning_effort = 'high' if self.enable_thinking else 'no_think'
        self.enable_thinking = self.reasoning_effort != 'no_think'
        self.chat_template_kwargs['reasoning_effort'] = self.reasoning_effort

    def _get_enable_thinking(self, inputs=None):
        reasoning_effort = None if inputs is None else inputs.chat_template_kwargs.get('reasoning_effort')
        if reasoning_effort is not None:
            return reasoning_effort != 'no_think'
        return super()._get_enable_thinking(inputs)

    def _get_system(self, inputs):
        system = super()._get_system(inputs)
        reasoning_effort = inputs.chat_template_kwargs.get('reasoning_effort')
        if reasoning_effort is None:
            reasoning_effort = self.reasoning_effort
        if inputs.tools:
            # For tool calls, append reasoning_mode after </tool_calls> in the tool instruction
            system = system.replace(
                f'you should print </tool_calls{self.HYTK}>',
                f'you should print </tool_calls{self.HYTK}><｜reasoning_mode{self.HYTK}｜>'
                f'reasoning_effort:{reasoning_effort}')
        else:
            # For non-tool calls, append reasoning_mode to the system/prefix area
            mode_str = f'<｜reasoning_mode{self.HYTK}｜>reasoning_effort:{reasoning_effort}'
            system = (system or '') + mode_str
        return system


register_template(
    TemplateMeta(
        LLMTemplateType.hy_v3_preview,
        prefix=['<｜hy_begin▁of▁sentence｜>'],
        system_prefix=['<｜hy_begin▁of▁sentence｜>{{SYSTEM}}'],
        prompt=['<｜hy_User｜>{{QUERY}}<｜hy_Assistant｜>'],
        chat_sep=['<｜hy_eos｜>'],
        suffix=['<｜hy_eos｜>'],
        template_cls=HyV3PreviewTemplate,
        is_thinking=True,
        thinking_prefix='<think>',
        non_thinking_prefix='<think></think>',
        history_thinking_prefix='<think></think>',
        agent_template='hy_v3_preview'))


class HyV3Template(HyV3PreviewTemplate):
    HYTK = ':opensource'


register_template(
    TemplateMeta(
        LLMTemplateType.hy_v3,
        prefix=['<｜hy_begin_of_sentence:opensource｜>'],
        system_prefix=['<｜hy_begin_of_sentence:opensource｜>{{SYSTEM}}'],
        prompt=['<｜hy_User:opensource｜>{{QUERY}}<｜hy_Assistant:opensource｜>'],
        chat_sep=['<｜hy_eos:opensource｜>'],
        suffix=['<｜hy_eos:opensource｜>'],
        template_cls=HyV3Template,
        is_thinking=True,
        thinking_prefix='<think:opensource>',
        non_thinking_prefix='<think:opensource></think:opensource>',
        history_thinking_prefix='<think:opensource></think:opensource>',
        agent_template='hy_v3'))


class GptTemplate(Template):
    support_padding_free = False

    def _get_gpt_oss_prefix(self):
        today = datetime.now().strftime('%Y-%m-%d')
        return ('<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n'
                f'Knowledge cutoff: 2024-06\nCurrent date: {today}\n\nReasoning: medium\n\n'
                '# Valid channels: analysis, commentary, final. '
                'Channel must be included for every message.<|end|>')

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs):
        super()._swift_prepare_inputs(inputs)
        messages = inputs.messages
        if self.use_chat_template:
            if inputs.system is None:
                inputs.system = self._get_gpt_oss_prefix()
            elif not inputs.system.startswith('<|start|>'):
                inputs.system = self._get_gpt_oss_prefix() + (
                    f'<|start|>developer<|message|># Instructions\n\n{inputs.system}<|end|>')
            for i, message in enumerate(messages):
                if message['role'] == 'assistant' and isinstance(message['content'], str):
                    if not message['content'].startswith('<|channel|>'):
                        message['content'] = '<|channel|>final<|message|>' + message['content']


@dataclass
class GptOssTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['{{SYSTEM}}'])
    prompt: Prompt = field(default_factory=lambda: ['<|start|>user<|message|>{{QUERY}}<|end|><|start|>assistant'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|return|>'])


register_template(GptOssTemplateMeta(LLMTemplateType.gpt_oss, template_cls=GptTemplate))

register_template(
    TemplateMeta(
        LLMTemplateType.longchat,
        prefix=[],
        system_prefix=['SYSTEM:{{SYSTEM}}'],
        prompt=[' [Round {{ROUND0}}] USER:{{QUERY}} ASSISTANT:'],
        chat_sep=['</longcat_s>'],
        suffix=['</longcat_s>'],
    ))

register_template(
    TemplateMeta(
        LLMTemplateType.ling2,
        prefix=['<role>SYSTEM</role>detailed thinking off<|role_end|>'],
        system_prefix=['<role>SYSTEM</role>{{SYSTEM}}\ndetailed thinking off<|role_end|>'],
        prompt=['<role>HUMAN</role>{{QUERY}}<|role_end|><role>ASSISTANT</role>'],
        chat_sep=['<|role_end|>'],
        suffix=['<|role_end|>'],
    ))

register_template(
    TemplateMeta(
        LLMTemplateType.ring2,
        prefix=[],
        system_prefix=['<role>SYSTEM</role>{{SYSTEM}}'],
        prompt=['<role>HUMAN</role>{{QUERY}}<role>ASSISTANT</role>'],
        chat_sep=[],
        suffix=['<|endoftext|>'],
        is_thinking=True,
        thinking_prefix='<think>\n',
    ))

register_template(
    TemplateMeta(
        LLMTemplateType.ring2_5,
        prefix=[],
        system_prefix=['<role>SYSTEM</role>\n{{SYSTEM}}\n\n'],
        prompt=['<role>HUMAN</role>\n{{QUERY}}<|role_end|>\n\n<role>ASSISTANT</role>\n'],
        chat_sep=['<|role_end|>\n\n'],
        suffix=['<|role_end|>\n\n'],
        is_thinking=True,
    ))

register_template(
    QwenTemplateMeta(
        LLMTemplateType.iquestcoder,
        default_system='You are LoopCoder, a helpful assistant developed by IQuest.',
    ))


class YoutuLLMTemplate(Template):

    def _remove_thinking_content(self, content: str) -> str:
        if '</think>' in content:
            content = content.rsplit('</think>', 1)[-1].lstrip('\n')
        return self.template_meta.history_thinking_prefix + content.strip()

    def _add_non_thinking_prefix(self, inputs) -> None:
        messages = inputs.messages
        non_thinking_prefix = self.template_meta.non_thinking_prefix
        if non_thinking_prefix and messages:
            # Find the last assistant message
            for i in range(len(messages) - 1, -1, -1):
                message = messages[i]
                if message['role'] == 'assistant' and isinstance(message['content'], str):
                    if '<think>' not in message['content'] and '</think>' not in message['content']:
                        message['content'] = non_thinking_prefix + message['content']
                    break

    def _remove_history_thinking(self, inputs) -> None:
        messages = inputs.messages
        first_tool_index = len(messages)
        for i, message in enumerate(messages):
            if message['role'] == 'tool' or (message['role'] == 'user' and isinstance(message.get('content'), str)
                                             and message['content'].startswith('<tool_response>')
                                             and message['content'].endswith('</tool_response>')):
                first_tool_index = i
                break
        # Only remove thinking content for assistant messages before first_tool_index - 1
        for i, message in enumerate(messages):
            if message['role'] == 'assistant' and isinstance(message['content'], str):
                is_last = (i == len(messages) - 1)
                if not is_last and i < first_tool_index - 1:
                    message['content'] = self._remove_thinking_content(message['content'])


register_template(
    TemplateMeta(
        LLMTemplateType.youtu_llm,
        template_cls=YoutuLLMTemplate,
        prefix=[['bos_token_id']],
        system_prefix=[['bos_token_id'], '{{SYSTEM}}'],
        prompt=['<|User|>{{QUERY}}<|Assistant|>'],
        chat_sep=['<|end_of_text|>'],
        suffix=['<|end_of_text|>'],
        is_thinking=True,
        non_thinking_prefix='<think>\n\n</think>\n\n',
        agent_template='youtu',
    ))

register_template(
    TemplateMeta(
        LLMTemplateType.olmoe,
        prefix=['|||IP_ADDRESS|||'],
        system_prefix=['|||IP_ADDRESS|||<|system|>\n{{SYSTEM}}\n'],
        prompt=['<|user|>\n{{QUERY}}\n<|assistant|>\n'],
        chat_sep=['|||IP_ADDRESS|||\n'],
        suffix=['|||IP_ADDRESS|||'],
        stop_words=['<|endoftext|>'],
    ))

register_template(
    TemplateMeta(
        LLMTemplateType.olmoe_0924,
        prefix=['<|endoftext|>'],
        system_prefix=['<|endoftext|><|system|>\n{{SYSTEM}}\n'],
        prompt=['<|user|>\n{{QUERY}}\n<|assistant|>\n'],
        chat_sep=['<|endoftext|>\n'],
        suffix=['<|endoftext|>'],
        stop_words=['<|endoftext|>'],
    ))
