# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers.utils import is_torch_npu_available

from swift.template import TemplateType
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


class Olmo3Loader(ModelLoader):

    def _update_attn_impl(self, config):
        if is_torch_npu_available():
            # HF's NPU FlashAttention wrapper does not implement window_size. Using it
            # would silently turn Olmo3's sliding-attention layers into full attention.
            # SDPA/eager construct both masks in the native Olmo3 forward instead.
            self.attn_impl = self.attn_impl or 'sdpa'
            if self.attn_impl not in {'sdpa', 'eager'}:
                raise ValueError(
                    'Olmo3 on NPU requires --attn_impl sdpa (recommended) or eager to preserve '
                    'sliding-window attention. FlashAttention, padding_free and packing are not supported.')
        super()._update_attn_impl(config)


register_model(
    ModelMeta(
        LLMModelType.olmo3,
        [ModelGroup([Model('allenai/Olmo-3-7B-Think', 'allenai/Olmo-3-7B-Think')])],
        Olmo3Loader,
        template=TemplateType.olmo3_think,
        model_arch=ModelArch.llama,
        architectures=['Olmo3ForCausalLM'],
        requires=['transformers>=4.57'],
        additional_saved_files=['chat_template.jinja'],
    ))
