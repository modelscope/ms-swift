# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
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
    CFG_SCALE = os.environ.get('CFG_SCALE', 3.0)
    GENERATION_RATIO = os.environ.get('GENERATION_RATIO', '1:1')
    NEGATIVE_PROMPT = os.environ.get(
        'NEGATIVE_PROMPT',
        'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, '
        'worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bov = self.processor.tokenizer.encode(self.processor.visual_template[0].format(token_id=0))[0]
        self.eov = self.processor.tokenizer.encode(self.processor.visual_template[0].format(token_id=self.COOKBOOK_SIZE
                                                                                            - 1))[0]
        self.h, self.w = self.processor.calculate_generate_size(self.GENERATION_RATIO, self.processor.image_area,
                                                                self.processor.vision_tokenizer.spatial_scale_factor)
        self.skip_prompt = False
        self.apply_loss_on_only_vision = True

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        if self.is_training:
            p_prob = random.random()
            if p_prob < self.NULL_PROMPT_PROB:
                prompt = ''
            else:
                prompt = inputs.to_history()['response']
            image = self.smart_resize(inputs.images[0].convert('RGB'))
            with torch.no_grad():
                image = self.processor.image_processor(
                    image, return_tensors='pt')['pixel_values'].to(device=self.processor.vision_tokenizer.device)
                image_token_ids = self.processor.vision_tokenizer.encode(image).squeeze(0)
            encoded = self._process_prompt_train(prompt, image_token_ids)
        else:
            prompt = inputs.to_history()['query']
            encoded = self._process_prompt_test(prompt)
            encoded = {key: encoded[key][0] for key in encoded.keys()}  # [1, L] -> [L]

        return encoded

    def _process_prompt_train(self, raw_prompt, image_token_ids):
        image_prompt = self.format_image_prompt(image_token_ids)
        prompt = self.tokenizer.bos_token + raw_prompt + image_prompt
        sample = self.tokenizer(prompt, padding='max_length', return_token_type_ids=False)
        labels = torch.tensor(sample['input_ids'])
        if self.apply_loss_on_only_vision:
            labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, -100)
        sample['labels'] = labels.tolist()
        return sample

    def _process_prompt_test(self, raw_prompt):
        # for supporting multi inputs, use list instead of single string
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
        prompt_list = self.tokenizer(prompt_list, padding='longest', return_token_type_ids=False)
        return prompt_list

    def prepare_for_output(self, output: str) -> str:
        return output

    def prepare_generate_kwargs(self, generate_kwargs: Dict[str, Any], *, model=None) -> Dict[str, Any]:
        from transformers import UnbatchedClassifierFreeGuidanceLogitsProcessor
        from transformers import PrefixConstrainedLogitsProcessor
        from transformers import LogitsProcessorList

        negative_prompt = self.NEGATIVE_PROMPT
        neg_inputs = self._process_prompt_test(negative_prompt)
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
            return [{'type': 'image', 'image': im}]

    def to_imgstr(self, image_tokens):
        image_token_str = [[self.processor.visual_template[0].format(token_id=token_id) for token_id in token_row]
                           for token_row in image_tokens]
        image_row_str = [''.join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr

    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        imgstr = self.to_imgstr(image_tokens)
        image_prompt = (
            self.tokenizer.boi_token + f'{h}*{w}' + self.tokenizer.img_token + imgstr + self.tokenizer.eol_token
            + self.tokenizer.eof_token + self.tokenizer.eoi_token)
        return image_prompt

    def smart_resize(self, image):
        w, h = image.size
        current_area = h * w
        target_ratio = (self.processor.image_area / current_area)**0.5
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
