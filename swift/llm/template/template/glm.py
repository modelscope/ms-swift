from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import torch
from transformers import PreTrainedTokenizerBase

from swift.llm import history_to_messages
from ..base import Template
from ..constant import TemplateType
from ..register import TemplateMeta, register_template
from ..utils import Context, Prompt, Word, findall


class GLMTemplate(Template):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        token_list = tokenizer.encode('')
        template_meta = self.template_meta
        template_meta.prefix.insert(0, token_list)
        if template_meta.system_prefix is not None:
            template_meta.system_prefix.insert(0, token_list)


register_template(
    TemplateMeta(
        TemplateType.chatglm2, ['{{SYSTEM}}'], ['[Round {{ROUND1}}]\n\n问：{{QUERY}}\n\n答：'], ['\n\n'],
        [['eos_token_id']],
        template_cls=GLMTemplate))


@dataclass
class GLM3TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: [])
    prompt: Prompt = field(default_factory=lambda: ['<|user|>\n{{QUERY}}<|assistant|>\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: [])
    suffix: Prompt = field(default_factory=lambda: ['<|user|>'])
    template_cls: Type[Template] = GLMTemplate
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|system|>\n{{SYSTEM}}'])

    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>', '<|user|>', '<|observation|>'])


class GLM4VTemplate(GLMTemplate):

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) <= 1

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-100]]

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = findall(input_ids, -100)
        if idx_list:
            idx = idx_list[0]
            image = example.get('images')[0]
            placeholder = '<|begin_of_image|><|endoftext|><|end_of_image|>'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            # TODO: history_to_messages
            messages = history_to_messages(example.get('history') or [], example['query'], example.get('system'))
            messages[0]['image'] = image
            inputs2: Dict[str, Any] = self.tokenizer.apply_chat_template(messages, return_dict=True)
            inputs['images'] = inputs2['images']
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs, {}

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      *,
                      padding_side: Optional[str] = None,
                      padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_side=padding_side, padding_to=padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        return res


# not '<|assistant|>\n'
register_template(
    GLM3TemplateMeta(
        TemplateType.glm4v,
        prompt=['<|user|>\n{{QUERY}}<|assistant|>'],
        suffix=['<|endoftext|>'],
        template_cls=GLM4VTemplate))

register_template(GLM3TemplateMeta(TemplateType.chatglm3))

register_template(
    GLM3TemplateMeta(
        TemplateType.chatglm4, default_tools_prompt='glm4', tool_prompt=['<|observation|>\n{{QUERY}}<|assistant|>\n']))

codegeex4_system = '你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。'

register_template(GLM3TemplateMeta(TemplateType.codegeex4, suffix=['<|endoftext|>'], default_system=codegeex4_system))

register_template(
    TemplateMeta(
        TemplateType.longwriter_llama3, ['[INST]'], ['{{QUERY}}[/INST]'], ['[INST]'], ['<|end_of_text|>'],
        system_prefix=['<<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))
