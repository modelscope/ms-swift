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
from ..utils import findall
from .utils import DEFAULT_SYSTEM, EmptyTemplateMeta


class Emu3GenTemplate(Template):

    NULL_PROMPT_PROB = 0.1

    COOKBOOK_SIZE = 32768

    APPLY_LOSS_ON_ONLY_VISION = True
    NEGATIVE_PROMPT = os.environ.get(
        'NEGATIVE_PROMPT',
        'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, '
        'low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.')
    CFG_SCALE = os.environ.get('CFG_SCALE', 3.0)
    GENERATION_RATIO = '1:1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bov = self.processor.tokenizer.encode(self.processor.visual_template[0].format(token_id=0))[0]
        self.eov = self.processor.tokenizer.encode(self.processor.visual_template[0].format(token_id=self.COOKBOOK_SIZE
                                                                                            - 1))[0]
        self.h, self.w = self.processor.calculate_generate_size(self.GENERATION_RATIO, self.processor.image_area,
                                                                self.processor.vision_tokenizer.spatial_scale_factor)
        self.skip_prompt = False

    def _process_prompt(self, raw_prompt: Optional[str | List[str]]):
        if isinstance(raw_prompt, str):
            raw_prompt = [raw_prompt]
        prompt_list = []
        size_list = []
        for text_prompt in raw_prompt:
            prompt = self.processor.tokenizer.bos_token
            image_prompt = (
                self.processor.tokenizer.boi_token + self.processor.prefix_template.format(H=self.h, W=self.w)
                + self.processor.tokenizer.img_token)
            prompt += (text_prompt + image_prompt)
            prompt_list.append(prompt)
            size_list.append([self.h, self.w])
        prompt_list = self.processor.tokenizer(prompt_list, padding='longest')
        return prompt_list

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        prompt = inputs.to_history()['query']
        encoded = self._process_prompt(prompt)
        encoded = {key: encoded[key][0] for key in encoded.keys()}  # [1, L] -> [L]
        encoded.pop('token_type_ids')

        # query = inputs.to_history()['query']
        # labels = encoded['input_ids']
        # if self.APPLY_LOSS_ON_ONLY_VISION:
        #     labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, -100)

        # encoded['labels'] = labels
        # for k, v in encoded.items():
        #     encoded[k] = v.squeeze(0)
        return encoded

    def prepare_for_output(self, output: str) -> str:
        return output

    def prepare_generate_kwargs(self, generate_kwargs: Dict[str, Any], *, model=None) -> Dict[str, Any]:
        from transformers import UnbatchedClassifierFreeGuidanceLogitsProcessor
        from transformers import PrefixConstrainedLogitsProcessor
        from transformers import LogitsProcessorList

        negative_prompt = self.NEGATIVE_PROMPT
        neg_inputs = self._process_prompt(negative_prompt)
        neg_inputs = {key: torch.tensor(val) for key, val in neg_inputs.items()}
        batch_size = generate_kwargs['input_ids'].shape[0]
        h = torch.tensor([self.h] * batch_size)
        w = torch.tensor([self.w] * batch_size)

        constrained_fn = self.processor.build_prefix_constrained_fn(h, w)
        logits_processor = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                self.CFG_SCALE,
                model,
                unconditional_ids=neg_inputs['input_ids'].to('cuda:0'),
            ),
            PrefixConstrainedLogitsProcessor(
                constrained_fn,
                num_beams=1,
            ),
        ])
        res = super().prepare_generate_kwargs(generate_kwargs, model=model)
        res['logits_processor'] = logits_processor
        return res

    def decode(self, generate_ids: List[int], is_finished: bool = True, **decode_kwargs) -> Any:
        mm_list = self.processor.decode(generate_ids)
        for im in mm_list:
            if not isinstance(im, Image.Image):
                continue
            im.save('test1.png')
            return [{'type': 'image', 'image': im}]

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

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if len(encoded) == 0:
            return encoded
        # image
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        image_tokens = self.processor.tokenize_image(images)
        image_prompts = []
        idxs = findall(input_ids, self.tokenizer.encode(self.image_placeholder))
        # Create image prompts
        for i in range(len(images)):
            h, w = image_tokens[i].shape
            imgstr = self.processor.to_imgstr(image_tokens[i])
            image_prompt = (
                self.tokenizer.boi_token + self.processor.prefix_template.format(H=h, W=w) + self.tokenizer.img_token
                + imgstr + self.tokenizer.eol_token + self.tokenizer.eof_token + self.tokenizer.eoi_token)
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
