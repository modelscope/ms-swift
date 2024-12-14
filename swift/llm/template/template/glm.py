# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, Word, findall
from ..vision_utils import load_batch, load_video_cogvlm2


@dataclass
class GLMTemplateMeta(TemplateMeta):
    auto_add_bos: bool = True


register_template(
    GLMTemplateMeta(
        LLMTemplateType.chatglm2,
        prefix=['{{SYSTEM}}'],
        prompt=['[Round {{ROUND1}}]\n\n问：{{QUERY}}\n\n答：'],
        chat_sep=['\n\n']))


@dataclass
class GLM4TemplateMeta(GLMTemplateMeta):
    prefix: Prompt = field(default_factory=lambda: [])
    prompt: Prompt = field(default_factory=lambda: ['<|user|>\n{{QUERY}}<|assistant|>\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: [])
    suffix: Prompt = field(default_factory=lambda: ['<|user|>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|system|>\n{{SYSTEM}}'])

    default_tools_prompt: str = 'glm4'
    tool_prompt: Optional[Prompt] = field(default_factory=lambda: ['<|observation|>\n{{QUERY}}<|assistant|>\n'])
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>', '<|user|>', '<|observation|>'])


class GLM4VTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-100]]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
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
            encoded['images'] = inputs2['images']
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        return res


# not '<|assistant|>\n'
register_template(GLM4TemplateMeta(MLLMTemplateType.glm4v, template_cls=GLM4VTemplate))

register_template(GLM4TemplateMeta(LLMTemplateType.glm4))

codegeex4_system = '你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。'

register_template(GLM4TemplateMeta(LLMTemplateType.codegeex4, default_system=codegeex4_system))

register_template(
    TemplateMeta(
        LLMTemplateType.longwriter_llama, ['[INST]'], ['{{QUERY}}[/INST]'], ['[INST]'], ['<|end_of_text|>'],
        system_prefix=['<<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))


class CogTemplate(Template):

    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        return []

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        model = self.model
        image = inputs.images or []
        history_inputs = inputs.to_history()
        inputs2 = model.build_conversation_input_ids(
            self.processor, query=history_inputs['query'], history=history_inputs['history'], images=image)
        image_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        encoded['token_type_ids'] = [0] + [1] * image_token_len + [0] * len(input_ids[1:])
        encoded['input_ids'] = input_ids[:1] + [self.processor.pad_token_id] * image_token_len + input_ids[1:]
        if labels is not None:
            encoded['labels'] = labels[:1] + [-100] * image_token_len + labels[1:]
        if len(image) > 0:
            encoded['images'] = [[img.to(dtype=self.config.torch_dtype)] for img in inputs2['images']]
            if 'cross_images' in inputs2:
                # is cogagent
                encoded['cross_images'] = [[cross_img.to(dtype=self.config.torch_dtype)]
                                           for cross_img in inputs2['cross_images']]
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        keys = ['images', 'cross_images']
        for key in keys:
            if key in batch[0]:
                res[key] = [b[key][0] for b in batch]
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

register_template(
    CogVLMTemplateMeta(
        MLLMTemplateType.cogvlm2, template_cls=CogTemplate, placeholder_tokens=['<|reserved_special_token_0|>']))


class Cog2VideoTemplate(CogTemplate):
    use_model = True

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        model = self.model
        encoded = super(CogTemplate, self)._encode(inputs)
        videos_path = inputs.videos or []
        video = load_batch(videos_path, load_video_cogvlm2)
        history_inputs = inputs.to_history()
        inputs2 = model.build_conversation_input_ids(
            self.processor,
            query=history_inputs['query'],
            history=history_inputs['history'],
            images=video,
            template_version='chat')
        video_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        encoded['token_type_ids'] = [0] + [1] * video_token_len + [0] * len(input_ids[1:])
        encoded['input_ids'] = input_ids[:1] + [self.processor.pad_token_id] * video_token_len + input_ids[1:]
        if labels is not None:
            encoded['labels'] = labels[:1] + [-100] * video_token_len + labels[1:]
        if len(video) > 0:
            dtype = model.dtype
            encoded['images'] = [[img.to(dtype=dtype)] for img in inputs2['images']]
        return encoded


register_template(
    CogVLMTemplateMeta(
        MLLMTemplateType.cogvlm2_video,
        template_cls=Cog2VideoTemplate,
        placeholder_tokens=['<|reserved_special_token_0|>'],
    ))


class GLMEdgeVTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['<|begin_of_image|>' * 578]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        if images:
            encoded['pixel_values'] = torch.tensor(self.processor(images).pixel_values)
        return encoded


register_template(
    GLM4TemplateMeta(
        MLLMTemplateType.glm_edge_v,
        prompt=['<|user|>\\n{{QUERY}}\\n<|assistant|>\\n'],
        chat_sep=['\\n'],
        system_prefix=['<|system|>\\n{{SYSTEM}}\\n'],
        template_cls=GLMEdgeVTemplate,
        placeholder_tokens=['<|begin_of_image|>'],
    ))
