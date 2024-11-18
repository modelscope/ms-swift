# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, Word, findall
from ..vision_utils import load_batch, load_video_cogvlm2


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
        LLMTemplateType.chatglm2,
        prefix=['{{SYSTEM}}'],
        prompt=['[Round {{ROUND1}}]\n\n问：{{QUERY}}\n\n答：'],
        chat_sep=['\n\n'],
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

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-100]]

    def _encode(self,
                inputs: StdTemplateInputs,
                *,
                model: Optional[nn.Module] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(inputs)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = findall(input_ids, -100)
        if idx_list:
            idx = idx_list[0]
            image = inputs.images[0]
            placeholder = '<|begin_of_image|><|endoftext|><|end_of_image|>'
            placeholder_id = self.processor.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            messages = inputs.messages
            messages[0]['image'] = image
            inputs2: Dict[str, Any] = self.processor.apply_chat_template(messages, return_dict=True)
            inputs['images'] = inputs2['images']
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs, {}

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      *,
                      padding_side: Optional[str] = None,
                      padding_to: Optional[int] = None,
                      model: Optional[nn.Module] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_side=padding_side, padding_to=padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        return res


# not '<|assistant|>\n'
register_template(
    GLM3TemplateMeta(
        MLLMTemplateType.glm4v,
        prompt=['<|user|>\n{{QUERY}}<|assistant|>'],
        suffix=['<|endoftext|>'],
        template_cls=GLM4VTemplate))

register_template(GLM3TemplateMeta(LLMTemplateType.chatglm3))

register_template(
    GLM3TemplateMeta(
        LLMTemplateType.chatglm4,
        default_tools_prompt='glm4',
        tool_prompt=['<|observation|>\n{{QUERY}}<|assistant|>\n']))

codegeex4_system = '你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。'

register_template(
    GLM3TemplateMeta(LLMTemplateType.codegeex4, suffix=['<|endoftext|>'], default_system=codegeex4_system))

register_template(
    TemplateMeta(
        LLMTemplateType.longwriter_llama3, ['[INST]'], ['{{QUERY}}[/INST]'], ['[INST]'], ['<|end_of_text|>'],
        system_prefix=['<<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))


class CogTemplate(Template):

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) <= 1

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        return []

    def _encode(self,
                inputs: StdTemplateInputs,
                *,
                model: Optional[nn.Module] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(inputs)
        if len(inputs) == 0:
            return inputs, {}
        image = inputs.images or []
        inputs.pop('loss_scale', None)
        inputs2 = model.build_conversation_input_ids(
            self.processor, query=inputs.query, history=inputs.history, images=image)
        image_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        inputs['token_type_ids'] = [0] + [1] * image_token_len + [0] * len(input_ids[1:])
        inputs['input_ids'] = input_ids[:1] + [self.processor.pad_token_id] * image_token_len + input_ids[1:]
        if labels is not None:
            inputs['labels'] = labels[:1] + [-100] * image_token_len + labels[1:]
        if len(image) > 0:
            dtype = model.dtype
            inputs['images'] = [[img.to(dtype=dtype)] for img in inputs2['images']]
            if 'cross_images' in inputs2:
                # is cogagent
                inputs['cross_images'] = [[cross_img.to(dtype=dtype)] for cross_img in inputs2['cross_images']]
        return inputs, {}

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      *,
                      padding_side: Optional[str] = None,
                      padding_to: Optional[int] = None,
                      model: Optional[nn.Module] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_side=padding_side, padding_to=padding_to)
        keys = ['images', 'cross_images']
        for key in keys:
            if key in batch[0]:
                res[key] = [b[key][0] for b in batch]
        token_type_ids = [torch.tensor(b['token_type_ids']) for b in batch]
        token_type_ids = self._pad_sequence(token_type_ids, 0, padding_side=padding_side)
        res['token_type_ids'] = token_type_ids
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.cogagent_chat,
        prefix=['<s>'],
        prompt=[' [INST] {{QUERY}} [/INST] '],
        chat_sep=[],
        suffix=['</s>'],
        template_cls=CogTemplate,
    ))

register_template(
    TemplateMeta(
        MLLMTemplateType.cogagent_vqa,
        prefix=['<s>'],
        prompt=['<EOI>Question: {{QUERY}} Answer:'],
        chat_sep=None,
        suffix=['</s>'],
        template_cls=CogTemplate))


@dataclass
class CogVLMTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: [['bos_token_id']])
    prompt: Prompt = field(default_factory=lambda: ['Question: {{QUERY}} Answer:'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['\n'])


register_template(CogVLMTemplateMeta(MLLMTemplateType.cogvlm, template_cls=CogTemplate))


class Cog2VideoTemplate(CogTemplate):

    def check_example(self, example):
        videos = example.get('videos') or []
        assert len(videos) <= 1

    def _encode(self,
                inputs: StdTemplateInputs,
                *,
                model: Optional[nn.Module] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super(CogTemplate, self)._encode(inputs)
        if len(inputs) == 0:
            return inputs, {}
        videos_path = inputs.videos or []
        video = load_batch(videos_path, load_video_cogvlm2)
        inputs.pop('loss_scale', None)
        inputs2 = model.build_conversation_input_ids(
            self.processor, query=inputs.query, history=inputs.history, images=video, template_version='chat')
        video_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        inputs['token_type_ids'] = [0] + [1] * video_token_len + [0] * len(input_ids[1:])
        inputs['input_ids'] = input_ids[:1] + [self.processor.pad_token_id] * video_token_len + input_ids[1:]
        if labels is not None:
            inputs['labels'] = labels[:1] + [-100] * video_token_len + labels[1:]
        if len(video) > 0:
            dtype = model.dtype
            inputs['images'] = [[img.to(dtype=dtype)] for img in inputs2['images']]
        return inputs, {}


register_template(CogVLMTemplateMeta(
    MLLMTemplateType.cogvlm2_video,
    template_cls=Cog2VideoTemplate,
))
