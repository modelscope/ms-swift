# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..template_meta import TemplateMeta
from ..utils import GenerationProperty, findall
from .utils import DEFAULT_SYSTEM, EmptyTemplateMeta


class Emu3GenTemplate(Template):

    NULL_PROMPT_PROB = 0.1

    COOKBOOK_SIZE = 32768

    APPLY_LOSS_ON_ONLY_VISION = True
    NEGATIVE_PROMPT = os.environ.get('NEGATIVE_PROMPT')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bov = self.processor.tokenizer.encode(self.processor.visual_template[0].format(token_id=0))[0]
        self.eov = self.processor.tokenizer.encode(self.processor.visual_template[0].format(token_id=self.COOKBOOK_SIZE
                                                                                            - 1))[0]
        self.config = kwargs.get('config')

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        query = inputs.to_history()['query']

        kwargs = dict(
            mode='U' if self.mode == 'train' else 'G',
            ratio='1:1',
            image_area=model.config.image_area,
            return_tensors='pt',
            padding='longest',
        )

        # image
        raw_image = inputs.images
        encoded = self.processor(query, raw_image, **kwargs)
        labels = encoded['input_ids']
        if self.APPLY_LOSS_ON_ONLY_VISION:
            labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, -100)

        encoded['labels'] = labels
        for k, v in encoded.items():
            encoded[k] = v.squeeze(0)
        return encoded

    def prepare_for_output(self, output: str) -> str:
        return output

    def prepare_for_generation(self,
                               generation_config,
                               inputs: Optional[Dict[str, Any]] = None,
                               model=None) -> GenerationProperty:
        from transformers import UnbatchedClassifierFreeGuidanceLogitsProcessor
        from transformers import PrefixConstrainedLogitsProcessor
        from transformers import LogitsProcessorList

        kwargs = dict(
            mode='G',
            ratio='1:1',
            image_area=self.config.image_area,
            return_tensors='pt',
            padding='longest',
        )
        negative_prompt = self.NEGATIVE_PROMPT
        if 'negative_prompt' in inputs:
            negative_prompt = inputs['negative_prompt']

        classifier_free_guidance = 3.0
        h, w = self.processor.calculate_generate_size('1:1', self.config.image_area,
                                                      self.processor.vision_tokenizer.spatial_scale_factor)
        # h = pos_inputs.image_size[:, 0]
        # w = pos_inputs.image_size[:, 1]
        neg_inputs = self.processor(text=negative_prompt, **kwargs)
        constrained_fn = self.processor.build_prefix_constrained_fn(h, w)
        logits_processor = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                classifier_free_guidance,
                model,
                unconditional_ids=neg_inputs.input_ids.to('cuda:0'),
            ),
            PrefixConstrainedLogitsProcessor(
                constrained_fn,
                num_beams=1,
            ),
        ])
        res = super().prepare_for_generation(generation_config, inputs, model)
        res.logits_processor += logits_processor
        return res

    def decode(self, input_ids: List[int], **tokenizer_kwargs) -> Image.Image:
        mm_list = self.processor.decode(input_ids)
        for idx, im in enumerate(mm_list):
            if not isinstance(im, Image.Image):
                continue
            return im

    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        imgstr = self.processor.to_imgstr(image_tokens)

        image_prompt = (
            self.processor.boi_token + f'{h}*{w}' + self.processor.img_token + imgstr + self.processor.eol_token
            + self.processor.eof_token + self.processor.eoi_token)

        return image_prompt

    def smart_resize(self, image):
        w, h = image.size
        current_area = h * w
        target_ratio = (self.processor.config.image_area / current_area)**0.5

        th = int(round(h * target_ratio))
        tw = int(round(w * target_ratio))

        image = image.resize((tw, th))
        return image


register_template(EmptyTemplateMeta(
    MLLMTemplateType.emu3_gen,
    template_cls=Emu3GenTemplate,
))


class Emu3ChatTemplate(Template):
    system = 'You are a helpful assistant.'
    image_placeholder = ['<|image token|>']

    def _encode(self, template_inputs: StdTemplateInputs) -> Dict[str, Any]:
        inputs, _ = super()._encode(template_inputs)
        if len(inputs) == 0:
            return inputs
        # image
        raw_image = template_inputs.images
        if raw_image:
            inputs['_data'] = {'raw_image': raw_image, 'input_ids': inputs['input_ids'], 'labels': inputs['labels']}

        return inputs

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        raw_images = data['raw_image']
        input_ids = data['input_ids']
        labels = data['labels']
        image_tokens = self.tokenizer.processor.tokenize_image(raw_images)
        image_prompts = []
        idxs = findall(input_ids, self.tokenizer.encode(self.image_placeholder))
        # Create image prompts
        for i in range(len(raw_images)):
            h, w = image_tokens[i].shape
            imgstr = self.tokenizer.processor.to_imgstr(image_tokens[i])
            image_prompt = (
                self.tokenizer.boi_token + self.tokenizer.processor.prefix_template.format(H=h, W=w)
                + self.tokenizer.img_token + imgstr + self.tokenizer.eol_token + self.tokenizer.eof_token
                + self.tokenizer.eoi_token)
            image_prompts.append(self.tokenizer.encode(image_prompt))
        added_tokens_len = 0
        # Insert image tokens into input_ids
        for idx, img_tokens in zip(idxs, image_prompts):
            input_ids = input_ids[:idx + added_tokens_len] + img_tokens + input_ids[idx + added_tokens_len + 1:]
            if labels is not None:
                labels = labels[:idx + added_tokens_len] + [-100] * len(img_tokens) + labels[idx + added_tokens_len
                                                                                             + 1:]
            added_tokens_len += len(img_tokens) - 1

        return {'input_ids': input_ids, 'labels': labels}


register_template(
    TemplateMeta(
        MLLMTemplateType.emu3_chat,
        prefix=[['bos_token_id'], '{{SYSTEM}}'],
        prompt=[' User: {{QUERY}}. Assistant:'],
        chat_sep=[['eos_token_id']],
        suffix=[['eos_token_id']],
        default_system=DEFAULT_SYSTEM,
        template_cls=Emu3ChatTemplate))
