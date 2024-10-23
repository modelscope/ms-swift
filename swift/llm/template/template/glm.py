from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from swift.llm import history_to_messages
from ..base import Template
from ..constant import TemplateType
from ..register import register_template
from ..utils import Context, findall


class GLMTemplate(Template):

    def _init_template(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs) -> None:
        res = super()._init_template(tokenizer, *args, **kwargs)
        token_list = tokenizer.encode('')
        self.prefix.insert(0, token_list)
        if self.system_prefix is not None:
            self.system_prefix.insert(0, token_list)
        return res


class GLM4VTemplate(GLMTemplate):

    def __init__(self):
        super().__init__([], ['<|user|>\n{{QUERY}}<|assistant|>'], [], ['<|endoftext|>'], None,
                         ['<|system|>\n{{SYSTEM}}'])

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

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        return res


register_template(TemplateType.glm4v, GLM4VTemplate(), infer_media_type='dialogue', lazy_tokenize=True, use_model=True)

register_template(
    TemplateType.chatglm2,
    GLMTemplate(['{{SYSTEM}}'], ['[Round {{ROUND1}}]\n\n问：{{QUERY}}\n\n答：'], ['\n\n'], [['eos_token_id']]))

register_template(
    TemplateType.chatglm_generation, GLMTemplate([], ['{{QUERY}}'], None, [['eos_token_id']]), is_generation=True)

register_template(
    TemplateType.chatglm3,
    GLMTemplate([], ['<|user|>\n{{QUERY}}<|assistant|>\n'], [], ['<|user|>'], None, ['<|system|>\n{{SYSTEM}}']))

register_template(
    TemplateType.chatglm4,
    GLMTemplate([], ['<|user|>\n{{QUERY}}<|assistant|>\n'], [], ['<|user|>'],
                None, ['<|system|>\n{{SYSTEM}}'],
                tools_prompt='glm4',
                tool_prompt=['<|observation|>\n{{QUERY}}<|assistant|>\n']))

codegeex4_system = '你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。'

register_template(
    TemplateType.codegeex4,
    GLMTemplate([], ['<|user|>\n{{QUERY}}<|assistant|>\n'], [], ['<|endoftext|>'], codegeex4_system,
                ['<|system|>\n{{SYSTEM}}']))

register_template(
    TemplateType.longwriter_llama3,
    Template(['[INST]'], ['{{QUERY}}[/INST]'], ['[INST]'], ['<|end_of_text|>'], None,
             ['<<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))
