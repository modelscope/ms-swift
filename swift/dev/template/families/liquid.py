# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import List, Optional

from ..base import Template
from ..inputs import StdTemplateInputs
from ..register import register_template
from ..utils import Word, findall
from .utils import ChatmlTemplateMeta


@dataclass
class LFM3TemplateMeta(ChatmlTemplateMeta):
    prefix: List = field(default_factory=lambda: ["<|startoftext|>"])
    system_prefix: Optional[List] = field(
        default_factory=lambda: ["<|startoftext|><|im_start|>system\n{{SYSTEM}}<|im_end|>\n"]
    )
    is_thinking: bool = True
    thinking_prefix: str = "<think>"
    default_enable_thinking: Optional[bool] = True
    non_thinking_prefix: str = "<think></think>"
    history_thinking_prefix: str = "<think></think>"
    train_final_think_block: Optional[str] = "<think></think>"
    agent_template: Optional[str] = "lfm3"
    stop_words: List[Word] = field(default_factory=lambda: ["<|im_end|>"])


class LFM3Template(Template):
    support_padding_free = False

    def _validate_declared_capabilities(self) -> None:
        super()._validate_declared_capabilities()
        if self.padding_free:
            raise ValueError("LFM3 short-convolution layers do not isolate packed sample boundaries.")

    def _encode(self, inputs):
        if inputs.messages and inputs.messages[-1]["role"] == "tool":
            raise ValueError("LFM3 requires an assistant or user turn after the final tool result.")
        return super()._encode(inputs)

    def _non_thinking_prefix_start(self, messages) -> int:
        return -1


class LFM3VLTemplate(LFM3Template):
    placeholder_tokens = ["<|image|>"]

    def replace_tag(self, media_type, index, inputs: StdTemplateInputs):
        if media_type != "image":
            raise NotImplementedError("LFM3-VL supports image inputs only.")
        inputs.images[index] = self._load_image(inputs.images[index], True)
        return ["<|vision_start|><|image|><|vision_end|>"]

    def _encode(self, inputs: StdTemplateInputs):
        encoded = super()._encode(inputs)
        if not inputs.images:
            return encoded
        media = self.processor.image_processor(images=inputs.images, return_tensors="pt")
        grids = media["image_grid_thw"]
        image_token_id = self.processor.image_token_id
        merge_length = self.processor.image_processor.merge_size**2
        indices = findall(encoded["input_ids"], image_token_id)
        encoded["input_ids"], encoded["labels"], encoded["loss_scale"] = self._extend_tokens(
            encoded["input_ids"], encoded["labels"], encoded.get("loss_scale"), indices,
            lambda i: [image_token_id] * int(grids[i].prod() // merge_length),
        )
        encoded.update(media)
        return encoded


register_template(LFM3TemplateMeta("lfm3", template_cls=LFM3Template))
register_template(LFM3TemplateMeta("lfm3_vl", template_cls=LFM3VLTemplate))
