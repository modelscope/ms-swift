# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import json
import torch
from torch import nn

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from ..vision_utils import load_file


class FlorenceTemplate(Template):
    # If it's an encoder-decoder architecture, the default settings are
    # loss_scale: 'last_round' and skip_prompt: False.
    is_encoder_decoder = True

    @staticmethod
    def _add_default_tags(inputs: StdTemplateInputs) -> None:
        return

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        return []

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [''.join(f'<loc_{box}>' for box in bbox)]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        processor = self.processor
        inputs.query = inputs.to_history()['query']
        new_query = processor._construct_prompts([inputs.query])[0]
        for i in reversed(range(len(inputs.messages))):
            if inputs.messages[i]['role'] == 'user':
                inputs.messages[i]['content'] = new_query
                break
        encoded = super()._encode(inputs)
        input_ids = encoded['prompt_input_ids']
        images = inputs.images or []
        labels = encoded['answer_labels']
        if labels is not None:
            labels = [0] + labels
        if images:
            pixel_values = processor.image_processor(
                images, return_tensors='pt')['pixel_values'].to(self.model_info.torch_dtype)
            encoded['pixel_values'] = pixel_values
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
        pixel_values = inputs.get('pixel_values')
        if pixel_values is not None:
            image_features = model._encode_image(pixel_values)
            inputs_embeds, inputs['attention_mask'] = model._merge_input_ids_with_image_features(
                image_features, inputs_embeds)
        return {'inputs_embeds': inputs_embeds}

    def decode(self, generate_ids: List[int], **kwargs) -> Any:
        response = super().decode(generate_ids, **kwargs)
        template_inputs = kwargs.get('template_inputs')
        images = template_inputs.images
        image_size = None
        if images:
            image_size = (images[0].width, images[0].height)
        return json.dumps(
            self.processor.post_process_generation(response, task=template_inputs.query, image_size=image_size))


register_template(
    TemplateMeta(
        MLLMTemplateType.florence,
        prefix=['<s>'],
        prompt=['{{QUERY}}</s>'],
        chat_sep=None,
        suffix=['</s>'],
        template_cls=FlorenceTemplate,
    ))


@dataclass
class Phi3TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['<|user|>\n{{QUERY}}<|end|>\n<|assistant|>\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|end|>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|system|>\n{{SYSTEM}}<|end|>\n'])
    auto_add_bos: bool = True


register_template(Phi3TemplateMeta(LLMTemplateType.phi3))


@dataclass
class Phi4TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(
        default_factory=lambda: ['<|im_start|>user<|im_sep|>{{QUERY}}<|im_end|><|im_start|>assistant<|im_sep|>'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|im_start|>system<|im_sep|>{{SYSTEM}}<|im_end|>'])
    auto_add_bos: bool = True


register_template(Phi4TemplateMeta(LLMTemplateType.phi4))


class Phi3VisionTemplate(Template):
    image_placeholder = ['<|image|><s>\n']  # <|image|>\n

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            return [f'<|image_{index + 1}|>\n']  # <|image_1|>\n
        else:
            return super().replace_tag(media_type, index, inputs)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        images = inputs.images or []
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, 32044)  # '<|image|>'

        if len(images) > 0:
            processor = self.processor
            encoded.update(processor.image_processor(images, return_tensors='pt'))
            assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
            res_input_ids = []
            res_labels = []
            num_img_tokens = encoded.pop('num_img_tokens').tolist()
            idx_list.insert(0, -1)
            for i in range(len(idx_list) - 1):
                image_token_id = -i - 1
                res_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]] + [image_token_id] * num_img_tokens[i]
                if labels is not None:
                    res_labels += labels[idx_list[i] + 1:idx_list[i + 1]] + [-100] * num_img_tokens[i]
            res_input_ids += input_ids[idx_list[-1] + 1:]
            input_ids = res_input_ids
            if labels is not None:
                res_labels += labels[idx_list[-1] + 1:]
                labels = res_labels

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded


class Phi4MMTemplate(Template):
    placeholder_tokens = ['<|endoftext10|>', '<|endoftext11|>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return [[-100]]
        elif media_type == 'audio':
            import soundfile as sf
            inputs.audios[index] = sf.read(load_file(inputs.audios[index]))
            return [[-200]]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        images_idx = findall(input_ids, -100)
        audios_idx = findall(input_ids, -200)
        text = '\n'.join(['<|image_1|>'] * len(inputs.images) + ['<|audio_1|>'] * len(inputs.audios))
        new_encoded = self.processor(
            text=text, images=inputs.images or None, audios=inputs.audios or None, return_tensors='pt')
        placeholders = self._split_list(new_encoded.pop('input_ids')[0].tolist(), 198)

        def _get_new_tokens(i):
            return placeholders[i]

        encoded['input_ids'], encoded['labels'] = self._extend_tokens(input_ids, labels, images_idx + audios_idx,
                                                                      _get_new_tokens)
        new_encoded.pop('attention_mask')
        encoded.update(new_encoded)
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        keys = [
            'input_image_embeds', 'image_sizes', 'image_attention_mask', 'input_audio_embeds', 'audio_embed_sizes',
            'input_mode'
        ]
        inputs = self.fetch_inputs(batch, keys)
        for k, v in inputs.items():
            inputs[k] = torch.concat(v)
        res.update(inputs)
        return res


register_template(Phi3TemplateMeta(MLLMTemplateType.phi3_vision, template_cls=Phi3VisionTemplate))

register_template(Phi3TemplateMeta(
    MLLMTemplateType.phi4_multimodal,
    template_cls=Phi4MMTemplate,
))
