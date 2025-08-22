import os
import re
from dataclasses import field, dataclass
from typing import Optional, Type, List

from transformers.utils import strtobool

from swift.llm import register_template, TemplateMeta, Template, Word
from swift.llm.template import Prompt
from swift.llm.template.constant import LLMTemplateType
from swift.llm.template.template.utils import ThinkingTemplate
from swift.llm.template.template_inputs import StdTemplateInputs


class SeedTemplate(ThinkingTemplate):

    @staticmethod
    def get_thinking_budget(inputs: StdTemplateInputs):
        budget = None
        if inputs.messages[0]['role'] == 'system':
            budget = inputs.messages[0].get('thinking_budget')
        return int(budget) if budget is not None else None

    @staticmethod
    def get_reflect_interval(inputs: StdTemplateInputs):
        interval_mapping = {
            0:      0,
             512:    128,
             1024:   256,
             2048:   512,
             4096:   512,
             8192:   1024,
             16384:  1024}
        budget = SeedTemplate.get_thinking_budget(inputs)
        if budget is None:
            return None
        elif budget <= 0:
            return 0
        elif budget > 16384:
            return 1024
        else:
            assert budget in interval_mapping.keys(), f'Supported thinking budget is {interval_mapping.keys()} or bigger.'
            return interval_mapping[budget]

    @staticmethod
    def insert_budget_markers(text: str, tokenizer, interval: int, total_budget: int) -> str:
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        result = []
        current_tokens = 0
        insertion_count = 0

        for sentence in sentences:
            sentence_tokens = len(tokenizer.encode(sentence))
            if current_tokens > 0 and current_tokens + sentence_tokens >= (insertion_count + 1) * interval:
                # TODO this value may not be accurate
                remaining_budget = total_budget - (insertion_count + 1) * interval
                marker = f"<seed:cot_budget_reflect>I have used {(insertion_count + 1) * interval} tokens, and there are {remaining_budget} tokens remaining for use.</seed:cot_budget_reflect>"
                result.append(marker)
                insertion_count += 1

            result.append(sentence)
            current_tokens += sentence_tokens

        return '\n'.join(result)

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs):
        super()._swift_prepare_inputs(inputs)
        budget = self.get_thinking_budget(inputs)
        interval = self.get_reflect_interval(inputs)
        if budget is None:
            default_system = ''
        elif budget > 0:
            default_system = ('You are an intelligent assistant with reflective ability. '
                              'In the process of thinking and reasoning, you need to strictly follow the thinking budget, '
                              f'which is {budget}. That is, you need to complete your thinking within {budget} tokens and start '
                              f'answering the user\')s questions. You will reflect on your thinking process every {interval} tokens, '
                              'stating how many tokens have been used and how many are left.')
        else:
            default_system = ('You are an intelligent assistant that can answer questions in one step without the need '
                              'for reasoning and thinking, that is, your thinking budget is 0. Next, please skip the '
                              'thinking process and directly start answering the user\'s questions.')

        if inputs.messages[0]['role'] == 'system':
            if not default_system:
                pass
            elif not inputs.messages[0]['content']:
                inputs.messages[0]['content'] = default_system
            else:
                inputs.messages[0]['content'] = (inputs.messages[0]['content'] +
                                                 '<seed:eos><seed:bos>system\n' + default_system)

            if not inputs.messages[0]['content']:
                inputs.messages.pop(0)
        else:
            if not default_system:
                pass
            else:
                inputs.messages.insert(0, {'content': default_system, 'role': 'system'})

        if budget is not None and budget > 0 and interval is not None and interval > 0:
            for message in inputs.messages:
                if message['role'] == 'assistant' and '<think>' in message['content'] and '</think>' in message['content']:
                    pre_text, post_text = message['content'].split('<think>')
                    think, post_text = post_text.split('</think>')
                    if '<seed:cot_budget_reflect>' not in message['content'] and strtobool(os.environ.get('SEED_USE_BUDGET_INTERVAL', 'false')):
                        think = self.insert_budget_markers(think, self.tokenizer, interval, budget)
                    message['content'] = pre_text + '<seed:think>' + think + '</seed:think>' + post_text

    def _simplify_context_list(self, context_list, loss_scale_list, inputs):
        res, res_loss_scale = super()._simplify_context_list(context_list, loss_scale_list, inputs)
        budget = self.get_thinking_budget(inputs)
        if res[-1].endswith('assistant\n') and budget == 0:
            res.append('<seed:think><seed:cot_budget_reflect>')
            res_loss_scale.append(res_loss_scale[-1])
        return res, res_loss_scale


@dataclass
class SeedTemplateMeta(TemplateMeta):
    template_type: str = 'seed'
    prefix: Prompt = '<seed:bos>'
    prompt: Prompt = field(default_factory=lambda: ['<seed:bos>user\n{{QUERY}}<seed:eos><seed:bos>assistant\n'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<seed:bos>system\n{{system}}<seed:eos>'])
    auto_add_bos: bool = True
    chat_sep: Optional[Prompt] = field(default_factory=lambda: [['<seed:eos>']])
    suffix: Prompt = field(default_factory=lambda: [['<seed:eos>']])
    template_cls: Type[Template] = SeedTemplate
    default_system: Optional[str] = None
    response_prefix: str = ''
    stop_words: List[Word] = field(default_factory=lambda: ['<seed:eos>'])
    agent_template: str = 'react_en'


register_template(SeedTemplateMeta(LLMTemplateType.seed_oss, default_system=None, template_cls=SeedTemplate))