import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Type

import torch
from torch import nn
from transformers.utils import strtobool

from swift.llm.template.constant import LLMTemplateType, MLLMTemplateType
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.utils import is_deepspeed_enabled
from ..register import Template, TemplateMeta, register_template
from ..utils import Context, Prompt, Word, findall
from .utils import ChatmlTemplateMeta


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
                        if think.strip():
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

SAIL_VL_DEFAULT_SYSTEM = '你是由抖音内容理解组开发的多模态大模型，英文名叫UniVL, 是一个有用无害的人工智能助手。'


class SailVLTemplate(Template):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_prompt = False
        self.num_image_token = self.processor.num_image_token
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image', 'This model only supports image input'
        if self.mode == 'vllm':
            raise NotImplementedError('vLLM not support this model now')
        else:
            image_context = ['<img>', [-100], '</img>\n']
        return image_context

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        idx_list = findall(input_ids, -100)
        pixel_values = None
        loss_scale = encoded.get('loss_scale', None)
        images = inputs.images
        processor = self.processor
        if images:
            labels = encoded.get('labels')
            image_inputs = processor.image_processor(images)
            num_patches = image_inputs['num_patches_list']
            pixel_values = image_inputs['pixel_values']
        else:
            pixel_values = None
            num_patches = []
        assert len(num_patches) == len(
            idx_list), f'len(num_patches): {len(num_patches)}, len(idx_list): {len(idx_list)}'

        def _get_new_tokens(i):
            img_tokens: List[int] = self.processor.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * num_patches[i]
            return img_tokens

        encoded['input_ids'], encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
            input_ids, labels, loss_scale, idx_list, _get_new_tokens)
        encoded['pixel_values'] = pixel_values
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embedding = model.language_model.get_input_embeddings()
        device = embedding.weight.device
        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values')
        if pixel_values is not None:
            vit_embeds = model.extract_feature(pixel_values)
            inputs_embeds = embedding(input_ids)
            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[selected] = vit_embeds.reshape(-1, C).to(inputs_embeds.device)

            inputs_embeds = inputs_embeds.reshape(B, N, C)
        elif is_deepspeed_enabled():
            inputs_embeds = embedding(input_ids).to(device=device)
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), device=device, dtype=inputs_embeds.dtype)
            vit_embeds = model.extract_feature(dummy_pixel_values).to(device=device)
            inputs_embeds = inputs_embeds + vit_embeds.mean() * 0.

        return {'inputs_embeds': inputs_embeds.to(input_ids.device)}


@dataclass
class SailVLTemplateMeta(ChatmlTemplateMeta):
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|im_start|>system\n{{SYSTEM}}<|im_end|>'])
    prompt: Prompt = field(default_factory=lambda: ['<|im_start|>user\n{{QUERY}}<|im_end|><|im_start|>assistant\n'])


register_template(
    SailVLTemplateMeta(MLLMTemplateType.sail_vl2, default_system=SAIL_VL_DEFAULT_SYSTEM, template_cls=SailVLTemplate))
