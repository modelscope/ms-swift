# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict

from ..base import Template
from ..constant import MLLMTemplateType
from ..inputs import StdTemplateInputs
from ..register import TemplateMeta, register_template
from ..utils import align_image_inputs


class Idefics3Template(Template):
    placeholder_tokens = ["<image>"]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images or []
        processor = self.processor
        prompt = self.processor.decode(encoded["input_ids"])
        if images:
            image_inputs = processor(text=prompt, images=images, return_tensors="pt", add_special_tokens=False)
            image_token = 128257  # <image>
            encoded["input_ids"], encoded["labels"] = align_image_inputs(
                encoded["input_ids"], encoded["labels"], image_inputs["input_ids"][0], image_token
            )
            encoded["pixel_values"] = image_inputs["pixel_values"]
        return encoded


register_template(
    # Idefics3's oracle is the processor's own template (Idefics3Processor ships it as a class
    # attribute), not the generic Llama-3 one in tokenizer_config: the processor is what the model
    # card calls, and it is the only side that renders the '<image>' placeholder.
    TemplateMeta(
        MLLMTemplateType.idefics3,
        prefix=["<|begin_of_text|>"],
        prompt=["User: {{QUERY}}<end_of_utterance>\n"],
        response_opener=["Assistant: "],
        chat_sep=["<end_of_utterance>\n"],
        suffix=["<end_of_utterance>\n"],
        system_prefix=["<|begin_of_text|>System: {{SYSTEM}}<end_of_utterance>\n"],
        # every message reads 'Assistant: {text}', the generation prompt just 'Assistant:'
        strip_opener_whitespace=True,
        template_cls=Idefics3Template,
    )
)
