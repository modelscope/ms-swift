import torch

from swift.llm import ModelType
from ..constant import MegatronModelType
from ..gpt.hf2mcore import convert_hf2mcore
from ..gpt.mcore2hf import convert_mcore2hf
from ..register import register_megatron_model
from .utils import HuggingFaceModule, MMGPTMegatronModelMeta


def convert_hf2mcore_internvl3(hf_model, mg_model):

    convert_hf2mcore(hf_model.language_model, mg_model.language_model)
    mg_model.visual.vision_model.load_state_dict(hf_model.vision_model.state_dict())
    mg_model.visual.mlp1.load_state_dict(hf_model.mlp1.state_dict())


def convert_mcore2hf_internvl3(hf_model, mg_model):
    convert_mcore2hf(hf_model.language_model, mg_model.language_model)
    hf_model.vision_model.load_state_dict(mg_model.visual.vision_model.state_dict())
    hf_model.mlp1.load_state_dict(mg_model.visual.mlp1.state_dict())


class Internvl3Vit(HuggingFaceModule):
    module_mapping = {'vision_model': 'vision_model', 'mlp1': 'mlp1'}
    vision_tower = ['vision_model']
    aligner = ['mlp1']

    def __init__(self, config):
        model_cls = []
        from transformers.models.qwen2 import Qwen2ForCausalLM
        model_cls.append(Qwen2ForCausalLM)
        try:
            from transformers.models import Qwen3ForCausalLM
            model_cls.append(Qwen3ForCausalLM)
        except ImportError:
            pass
        try:
            from transformers.models import Qwen3MoeForCausalLM
            model_cls.append(Qwen3MoeForCausalLM)
        except ImportError:
            pass
        super().__init__(config, model_cls)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        model = self._hf_model[0]
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        if pixel_values is None:
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), dtype=self.vision_model.dtype, device=inputs_embeds.device)
            vit_embeds = model.extract_feature(dummy_pixel_values)
            inputs_embeds = inputs_embeds + vit_embeds.mean() * 0.
        else:
            vit_embeds = model.extract_feature(pixel_values)
            selected = (input_ids == self.processor.encode('<IMG_CONTEXT>', add_special_tokens=False)[0])
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[selected] = vit_embeds.reshape(-1, vit_embeds.shape[-1]).to(dtype=inputs_embeds.dtype)
        return inputs_embeds


register_megatron_model(
    MMGPTMegatronModelMeta(
        MegatronModelType.internvl3, [
            ModelType.internvl3,
            ModelType.internvl3_5,
        ],
        convert_hf2mcore=convert_hf2mcore_internvl3,
        convert_mcore2hf=convert_mcore2hf_internvl3,
        visual_cls=Internvl3Vit))
