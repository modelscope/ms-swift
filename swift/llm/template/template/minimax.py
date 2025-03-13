# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from swift.utils import get_logger
from ..base import Template
from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt

logger = get_logger()


@dataclass
class MinimaxTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: [
        '<beginning_of_sentence>user name=user\n{{QUERY}}<end_of_sentence>\n'
        '<beginning_of_sentence>ai name=assistant\n'
    ])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<end_of_sentence>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<end_of_sentence>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<beginning_of_sentence>system ai_setting=assistant\n{{SYSTEM}}<end_of_sentence>\n'])


register_template(MinimaxTemplateMeta(LLMTemplateType.minimax))


class MinimaxVLTemplate(Template):
    image_placeholder = ['<image>']
    skip_prompt = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return self.image_placeholder * inputs.all_image_tokens[index]

    def calc_num_image_tokens(self, image_inputs):
        from transformers.image_utils import get_image_size, to_numpy_array
        pixel_values = image_inputs['pixel_values']
        image_sizes = image_inputs['image_sizes']
        all_image_tokens = []
        if not image_inputs:
            return all_image_tokens

        if self.processor.process_image_mode == 'anyres':
            for pixel_value, image_size in zip(pixel_values, image_sizes):
                height, width = image_size
                num_image_tokens = self.processor.get_num_token(height, width, self.processor.grid_pinpoints,
                                                                self.processor.patch_size)
                all_image_tokens.append(num_image_tokens)
        elif self.processor.process_image_mode == 'resize':
            pixel_values = image_inputs['pixel_values']
            all_image_tokens = []
            for pixel_value in pixel_values:
                height, width = get_image_size(to_numpy_array(pixel_value))
                all_image_tokens.append(int(height * width / self.processor.patch_size**2))
        else:
            if self.processor.patch_size is not None:
                pixel_values = image_inputs['pixel_values']
                all_image_tokens = []
                for pixel_value in pixel_values:
                    height, width = get_image_size(to_numpy_array(pixel_value))
                    new_width, new_height = self.processor.get_hw_multiple_of(
                        (width, height), self.processor.patch_size, self.processor.max_size)
                    num_image_tokens = ((new_height // self.processor.patch_size) *
                                        (new_width // self.processor.patch_size))  # + 1
                    all_image_tokens.append(num_image_tokens)
            else:
                logger.warning_once(
                    'Expanding inputs for image tokens in MiniMaxVL01 should be done in processing. '
                    "Please add `patch_size` and `vision_feature_select_strategy` to the model's "
                    'processing config or set directly '
                    'with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = '
                    '{{vision_feature_select_strategy}}`. '
                    'Using processors without these attributes in the config is deprecated '
                    'and will throw an error in v4.47.')
                raise ValueError(
                    "You need to provide `patch_size` and `vision_feature_select_strategy` in the model's processing "
                    'config to expand inputs for image tokens.')
        return all_image_tokens

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        output_kwargs = self.processor._merge_kwargs(
            self.processor.MiniMaxVL01ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        )
        if inputs.images:
            image_inputs = self.processor.image_processor(
                inputs.images, **output_kwargs['images_kwargs'], return_tensors='pt')
            inputs.all_image_tokens = self.calc_num_image_tokens(image_inputs)
        else:
            image_inputs = {}
        encoded = super()._encode(inputs)
        for key in image_inputs:
            encoded[key] = image_inputs[key]
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        pixel_values = self.gather_list(batch, 'pixel_values')
        image_sizes = self.gather_list(batch, 'image_sizes')
        res = super()._data_collator(batch, padding_to=padding_to)
        if pixel_values:
            res['pixel_values'] = pixel_values
        if image_sizes:
            res['image_sizes'] = image_sizes
        return res


register_template(MinimaxTemplateMeta(LLMTemplateType.minimax_vl, template_cls=MinimaxVLTemplate))
