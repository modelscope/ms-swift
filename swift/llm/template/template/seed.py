import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Type

from transformers.utils import strtobool

from swift.llm.template.constant import LLMTemplateType
from swift.llm.template.template_inputs import StdTemplateInputs
from ..register import Template, TemplateMeta, register_template
from ..utils import Prompt, Word


class SeedTemplate(Template):

    def get_thinking_budget(self, inputs: StdTemplateInputs):
        thinking_budget = os.environ.get('THINKING_BUDGET')
        if thinking_budget is not None:
            max_length = int(thinking_budget)
        else:
            max_length = 0
            for m in inputs.messages:
                if m['role'] == 'assistant' and m['content']:
                    if '<think>' in m['content'] and '</think>' in m['content']:
                        _, think = m['content'].split('<think>', maxsplit=1)
                        think, _ = think.split('</think>', maxsplit=1)
                        thinking_token_len = len(self.tokenizer(think)['input_ids'])
                        if thinking_token_len > max_length:
                            max_length = thinking_token_len

        def convert_integer_v2(n):
            if n is None:
                return None
            elif n <= 0:
                return 0
            elif n <= 512:
                return 512
            elif n <= 1024:
                return 1024
            elif n <= 2048:
                return 2048
            elif n <= 4096:
                return 4096
            elif n <= 8192:
                return 8192
            elif n <= 16384:
                return 16384
            else:
                return n

        return convert_integer_v2(max_length)

    def get_reflect_interval(self, inputs: StdTemplateInputs):
        interval_mapping = {0: 0, 512: 128, 1024: 256, 2048: 512, 4096: 512, 8192: 1024, 16384: 1024}
        budget = self.get_thinking_budget(inputs)
        if budget is None:
            return None
        elif budget <= 0:
            return 0
        elif budget > 16384:
            return 1024
        else:
            assert budget in interval_mapping.keys(
            ), f'Supported thinking budget is {interval_mapping.keys()} or bigger.'
            return interval_mapping[budget]

    @staticmethod
    def insert_budget_markers(text: str, tokenizer, interval: int, total_budget: int) -> str:
        if total_budget > 0:
            sentences = re.split(r'(?<=[.!?。！？])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            result = []
            current_tokens = 0
            insertion_count = 0

            for sentence in sentences:
                sentence_tokens = len(tokenizer.encode(sentence))
                if current_tokens + sentence_tokens >= (insertion_count + 1) * interval:
                    remaining_budget = total_budget - (current_tokens + sentence_tokens)
                    marker = (f'<seed:cot_budget_reflect>I have used {current_tokens + sentence_tokens} tokens, '
                              f'and there are {remaining_budget} tokens remaining for use.</seed:cot_budget_reflect>')
                    result.append(marker)
                    insertion_count += 1

                result.append(sentence)
                current_tokens += sentence_tokens

            return '\n'.join(result)
        else:
            return ('<seed:cot_budget_reflect>The current thinking budget is 0, so I will '
                    'directly start answering the question.</seed:cot_budget_reflect>\n\n')

    def _prepare_system(self, inputs):
        budget = self.get_thinking_budget(inputs)
        interval = self.get_reflect_interval(inputs)
        if budget is None:
            default_system = ''
        elif budget > 0:
            default_system = (
                'You are an intelligent assistant with reflective ability. '
                'In the process of thinking and reasoning, you need to strictly follow the thinking budget, '
                f'which is {budget}. That is, you need to complete your thinking within {budget} tokens and start '
                f'answering the user\'s questions. You will reflect on your thinking process every {interval} tokens, '
                'stating how many tokens have been used and how many are left.\n')
        else:
            default_system = ('You are an intelligent assistant that can answer questions in one step without the need '
                              'for reasoning and thinking, that is, your thinking budget is 0. Next, please skip the '
                              'thinking process and directly start answering the user\'s questions.\n')

        if default_system:
            if inputs.system:
                inputs.system = inputs.system + '<seed:eos><seed:bos>system\n' + default_system
            else:
                inputs.system = default_system

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs):
        super()._swift_prepare_inputs(inputs)
        if strtobool(os.environ.get('SEED_USE_THINKING', 'true')):
            budget = self.get_thinking_budget(inputs)
            interval = self.get_reflect_interval(inputs)
            self._prepare_system(inputs)
            if budget is not None:
                for message in inputs.messages:
                    if message['role'] == 'assistant':
                        if '<think>' in message['content'] and '</think>' in message['content']:
                            pre_text, post_text = message['content'].split('<think>', maxsplit=1)
                            think, post_text = post_text.split('</think>', maxsplit=1)
                            if '<seed:cot_budget_reflect>' not in message['content'] and strtobool(
                                    os.environ.get('SEED_USE_BUDGET_INTERVAL', 'false')):
                                think = self.insert_budget_markers(think, self.tokenizer, interval, budget)
                            message['content'] = pre_text + '<seed:think>' + think + '</seed:think>' + post_text
                        elif budget > 0:
                            message['content'] = message['content'].replace('<think>', '').replace('</think>', '')
                            message['content'] = '<seed:think></seed:think>' + message['content']
                        elif budget <= 0:
                            message['content'] = message['content'].replace('<think>', '').replace('</think>', '')
                            message['content'] = (
                                '<seed:think><seed:cot_budget_reflect>The current thinking budget is 0, '
                                'so I will directly start answering the question.'
                                '</seed:cot_budget_reflect>\n\n</seed:think>') + message['content']

    def _simplify_context_list(self, context_list, loss_scale_list, inputs):
        res, res_loss_scale = super()._simplify_context_list(context_list, loss_scale_list, inputs)
        budget = self.get_thinking_budget(inputs)
        if res[-1].endswith('assistant\n') and budget == 0:
            res.append('<seed:think><seed:cot_budget_reflect>')
            res_loss_scale.append(res_loss_scale[-1])
        return res, res_loss_scale

    def _jinja_encode(self, inputs: StdTemplateInputs):
        self._prepare_system(inputs)
        return super()._jinja_encode(inputs)


@dataclass
class SeedTemplateMeta(TemplateMeta):
    template_type: str = 'seed'
    prefix: Prompt = field(default_factory=lambda: ['<seed:bos>'])
    prompt: Prompt = field(default_factory=lambda: ['<seed:bos>user\n{{QUERY}}<seed:eos><seed:bos>assistant\n'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<seed:bos>system\n{{SYSTEM}}<seed:eos>'])
    auto_add_bos: bool = True
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<seed:eos>'])
    suffix: Prompt = field(default_factory=lambda: ['<seed:eos>'])
    template_cls: Type[Template] = SeedTemplate
    default_system: Optional[str] = None
    response_prefix: str = ''
    stop_words: List[Word] = field(default_factory=lambda: ['<seed:eos>'])
    agent_template: str = 'react_en'


register_template(SeedTemplateMeta(LLMTemplateType.seed_oss, default_system=None, template_cls=SeedTemplate))
