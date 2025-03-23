# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Prompt
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

register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_gme, template_cls=Qwen2VLTemplate, suffix=['<|endoftext|>']))

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

today = datetime.now().strftime('%Y-%m-%d')

mistral_2501_system = (
    'You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup '
    'headquartered in Paris.\n'
    f'Your knowledge base was last updated on 2023-10-01. The current date is {today}.\n\n'
    "When you're not sure about some information, you say that you don't have the information and don't "
    'make up anything.\n'
    "If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer "
    'the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. '
    '"What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "'
    'Where do you travel from?")')

register_template(
    TemplateMeta(
        LLMTemplateType.mistral_2501,
        prefix=['<s>'],
        prompt=['[INST]{{QUERY}}[/INST]'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        system_prefix=['<s>[SYSTEM_PROMPT]{{SYSTEM}}[/SYSTEM_PROMPT]'],
        default_system=mistral_2501_system))

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
        LLMTemplateType.moonlight,
        prefix=[],
        system_prefix=['<|im_system|>system<|im_middle|>{{SYSTEM}}<|im_end|>'],
        prompt=['<|im_user|>user<|im_middle|>{{QUERY}}<|im_end|><|im_assistant|>assistant<|im_middle|>'],
        chat_sep=['<|im_end|>'],
        suffix=['<|im_end|>'],
        default_system='You are a helpful assistant',
    ))

register_template(
    TemplateMeta(
        LLMTemplateType.ling,
        prefix=[],
        system_prefix=['<role>SYSTEM</role>{{SYSTEM}}'],
        prompt=['<role>HUMAN</role>{{QUERY}}<role>ASSISTANT</role>'],
        chat_sep=[],
        suffix=['<|endoftext|>'],
    ))
