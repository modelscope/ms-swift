from typing import Any, Dict, List, Literal, Optional, Tuple

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Context, align_image_inputs, findall, gather_list


class Idefics3Template(Template):

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []
        processor = self.tokenizer.processor
        prompt = self.tokenizer.decode(inputs['input_ids'])
        if images:
            image_inputs = processor(text=prompt, images=images, return_tensors='pt', add_special_tokens=False)
            image_token = 128257  # <image>
            inputs['input_ids'], inputs['labels'] = align_image_inputs(inputs['input_ids'], inputs['labels'],
                                                                       image_inputs['input_ids'][0], image_token)
            inputs['pixel_values'] = image_inputs['pixel_values']
        return inputs, {}


register_template(
    TemplateMeta(
        MLLMTemplateType.idefics3,
        prefix=['<|begin_of_text|>'],
        prompt=['User:{{QUERY}}<end_of_utterance>\nAssistant:'],
        chat_sep=['<end_of_utterance>\n'],
        suffix=['<end_of_utterance>'],
        system_prefix=['System:{{SYSTEM}}<end_of_utterance>\n'],
        template_cls=Idefics3Template,
    ))
