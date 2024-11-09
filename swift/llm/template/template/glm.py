# Copyright (c) Alibaba, Inc. and its affiliates.
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


class CogTemplate(Template):

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) <= 1

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        return []

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        image = example.get('images') or []
        inputs.pop('loss_scale', None)
        model = self.model
        inputs2 = model.build_conversation_input_ids(
            self.tokenizer, query=example['query'], history=example.get('history'), images=image)
        image_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        inputs['token_type_ids'] = [0] + [1] * image_token_len + [0] * len(input_ids[1:])
        inputs['input_ids'] = input_ids[:1] + [self.tokenizer.pad_token_id] * image_token_len + input_ids[1:]
        if labels is not None:
            inputs['labels'] = labels[:1] + [-100] * image_token_len + labels[1:]
        if len(image) > 0:
            dtype = model.dtype
            inputs['images'] = [[img.to(dtype=dtype)] for img in inputs2['images']]
            if 'cross_images' in inputs2:
                # is cogagent
                inputs['cross_images'] = [[cross_img.to(dtype=dtype)] for cross_img in inputs2['cross_images']]
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        keys = ['images', 'cross_images']
        for key in keys:
            if key in batch[0]:
                res[key] = [b[key][0] for b in batch]
        token_type_ids = [torch.tensor(b['token_type_ids']) for b in batch]
        token_type_ids = self.pad_sequence(token_type_ids, 0, self.padding_side)
        res['token_type_ids'] = token_type_ids
        return res


register_template(
    TemplateType.cogagent_chat,
    CogTemplate(['<s>'], [' [INST] {{QUERY}} [/INST] '], [], ['</s>']),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)

register_template(
    TemplateType.cogagent_instruct,
    CogTemplate(['<s>'], ['<EOI>Question: {{QUERY}} Answer:'], None, ['</s>']),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)

register_template(
    TemplateType.cogvlm,
    CogTemplate([['bos_token_id']], ['Question: {{QUERY}} Answer:'], ['\n'], [['eos_token_id']]),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)


class Cog2VideoTemplate(CogTemplate):

    def check_example(self, example):
        videos = example.get('videos') or []
        assert len(videos) <= 1

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super(CogTemplate, self)._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        videos_path = example.get('videos') or []
        video = load_batch(videos_path, load_video_cogvlm2)
        inputs.pop('loss_scale', None)
        model = self.model
        inputs2 = model.build_conversation_input_ids(
            self.tokenizer,
            query=example['query'],
            history=example.get('history'),
            images=video,
            template_version='chat')
        video_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        inputs['token_type_ids'] = [0] + [1] * video_token_len + [0] * len(input_ids[1:])
        inputs['input_ids'] = input_ids[:1] + [self.tokenizer.pad_token_id] * video_token_len + input_ids[1:]
        if labels is not None:
            inputs['labels'] = labels[:1] + [-100] * video_token_len + labels[1:]
        if len(video) > 0:
            dtype = model.dtype
            inputs['images'] = [[img.to(dtype=dtype)] for img in inputs2['images']]
        return inputs, {}


register_template(
    TemplateType.cogvlm2_video,
    Cog2VideoTemplate([['bos_token_id']], ['Question: {{QUERY}} Answer:'], ['\n'], [['eos_token_id']]),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True,
    media_type='video')
