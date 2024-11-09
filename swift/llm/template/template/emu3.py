# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from PIL import Image

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Context, GenerationProperty, findall, gather_list
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta, EmptyTemplateMeta


class Emu3GenTemplate(Template):

    NULL_PROMPT_PROB = 0.1

    COOKBOOK_SIZE = 32768

    APPLY_LOSS_ON_ONLY_VISION = True
    NEGATIVE_PROMPT = os.environ.get('NEGATIVE_PROMPT')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bov = self.tokenizer.encode(self.tokenizer.processor.visual_template[0].format(token_id=0))[0]
        self.eov = self.tokenizer.encode(self.tokenizer.processor.visual_template[0].format(token_id=self.COOKBOOK_SIZE
                                                                                            - 1))[0]
        self.config = kwargs.get('config')

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        query = example['query']

        kwargs = dict(
            mode='U' if self._is_training else 'G',
            ratio='1:1',
            image_area=self.config.image_area,
            return_tensors='pt',
            padding='longest',
        )

        # image
        raw_image = example.get('images', None)
        inputs = self.tokenizer.processor(query, raw_image, **kwargs)
        labels = inputs['input_ids']
        if self.APPLY_LOSS_ON_ONLY_VISION:
            labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, -100)

        inputs['labels'] = labels
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        return inputs, {}

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
        h, w = self.tokenizer.processor.calculate_generate_size(
            '1:1', self.config.image_area, self.tokenizer.processor.vision_tokenizer.spatial_scale_factor)
        # h = pos_inputs.image_size[:, 0]
        # w = pos_inputs.image_size[:, 1]
        neg_inputs = self.tokenizer.processor(text=negative_prompt, **kwargs)
        constrained_fn = self.tokenizer.processor.build_prefix_constrained_fn(h, w)
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

    def safe_decode(self, generate_ids: List[int], is_finished: bool, **decode_kwargs) -> Image.Image:
        mm_list = self.tokenizer.processor.decode(generate_ids)
        for idx, im in enumerate(mm_list):
            if not isinstance(im, Image.Image):
                continue
            return im

    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        imgstr = self.tokenizer.processor.to_imgstr(image_tokens)

        image_prompt = (
            self.tokenizer.boi_token + f'{h}*{w}' + self.tokenizer.img_token + imgstr + self.tokenizer.eol_token
            + self.tokenizer.eof_token + self.tokenizer.eoi_token)

        return image_prompt

    def smart_resize(self, image):
        w, h = image.size
        current_area = h * w
        target_ratio = (self.tokenizer.config.image_area / current_area)**0.5

        th = int(round(h * target_ratio))
        tw = int(round(w * target_ratio))

        image = image.resize((tw, th))
        return image


register_template(EmptyTemplateMeta(
    MLLMTemplateType.emu3_gen,
    template_cls=Emu3GenTemplate,
))
