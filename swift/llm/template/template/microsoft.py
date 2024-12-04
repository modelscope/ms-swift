# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import json
import torch
from torch import nn

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall, gather_list
from ..vision_utils import load_image


class FlorenceTemplate(Template):
    # loss_scale = 'last_round'
    is_encoder_decoder = True

    def __init__(self, *args, **kwargs):
        self.task_prompts_without_inputs = {
            '<OCR>': 'What is the text in the image?',
            '<OCR_WITH_REGION>': 'What is the text in the image, with regions?',
            '<CAPTION>': 'What does the image describe?',
            '<DETAILED_CAPTION>': 'Describe in detail what is shown in the image.',
            '<MORE_DETAILED_CAPTION>': 'Describe with a paragraph what is shown in the image.',
            '<OD>': 'Locate the objects with category name in the image.',
            '<DENSE_REGION_CAPTION>': 'Locate the objects in the image, with their descriptions.',
            '<REGION_PROPOSAL>': 'Locate the region proposals in the image.'
        }
        self.task_prompts_with_input = {
            '<CAPTION_TO_PHRASE_GROUNDING>': 'Locate the phrases in the caption: {input}',
            '<REFERRING_EXPRESSION_SEGMENTATION>': 'Locate {input} in the image with mask',
            '<REGION_TO_SEGMENTATION>': 'What is the polygon mask of region {input}',
            '<OPEN_VOCABULARY_DETECTION>': 'Locate {input} in the image.',
            '<REGION_TO_CATEGORY>': 'What is the region {input}?',
            '<REGION_TO_DESCRIPTION>': 'What does the region {input} describe?',
            '<REGION_TO_OCR>': 'What text is in the region {input}?',
        }
        super().__init__(*args, **kwargs)

    @staticmethod
    def _add_default_tags(inputs: StdTemplateInputs) -> None:
        return

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        return ['']

    def replace_box(self, object_: Dict[str, Any], index: int, inputs: StdTemplateInputs) -> List[Context]:
        object_ = inputs.objects[index]
        if isinstance(object_['bbox'][0], list):
            all_objects = ''
            for sub_object in object_['bbox']:
                x1, y1, x2, y2 = sub_object
                all_objects += f'<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>,'
            return [all_objects[:-1]]
        else:
            x1, y1, x2, y2 = object_['bbox']
            return [f'<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        processor = self.processor
        new_query = processor._construct_prompts([inputs.to_history()['query']])[0]
        for i in reversed(range(len(inputs.messages))):
            if inputs.messages[i]['role'] == 'user':
                inputs.messages[i]['content'] = new_query
                break
        encoded = super()._encode(inputs)
        input_ids = encoded['prompt_input_ids']
        if len(encoded) == 0:
            return encoded
        images = inputs.images or []
        labels = encoded['labels']
        if labels is not None:
            labels = [0] + labels
        pixel_values = processor.image_processor(
            images, return_tensors='pt')['pixel_values'].to(self.config.torch_dtype)
        encoded = {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values,
        }
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
        image_features = model._encode_image(inputs['pixel_values'])
        inputs_embeds, _ = model._merge_input_ids_with_image_features(image_features, inputs_embeds)
        return {'inputs_embeds': inputs_embeds[0]}

    def post_process_generate_response(self, response, example):
        if isinstance(example['images'], list):
            example['images'] = example['images'][0]
        image = load_image(example['images'])
        return json.dumps(
            self.processor.post_process_generation(
                response, task=example['query'], image_size=(image.width, image.height)))


register_template(
    TemplateMeta(
        MLLMTemplateType.florence,
        prefix=['<s>'],
        prompt=['{{QUERY}}</s>'],
        chat_sep=None,
        suffix=['</s>'],
        template_cls=FlorenceTemplate,
        support_stream=False))


@dataclass
class Phi3TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['<|user|>\n{{QUERY}}<|end|>\n<|assistant|>\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|end|>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|system|>\n{{SYSTEM}}<|end|>\n'])
    auto_add_bos: bool = True


register_template(Phi3TemplateMeta(LLMTemplateType.phi3))


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
        if len(encoded) == 0:
            return encoded
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


register_template(Phi3TemplateMeta(MLLMTemplateType.phi3_vision, template_cls=Phi3VisionTemplate))
