from typing import Any, Dict

from transformers import AutoConfig

from swift.llm import TemplateType
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo


def get_model_tokenizer_minimind(model_dir: str,
                                 model_info: ModelInfo,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    kwargs['model_config'] = model_config
    return get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.minimind,
        [
            # MiniMind2
            ModelGroup(
                [
                    # MiniMind2
                    Model('gongjy/MiniMind2', 'jingyaogong/MiniMind2'),
                    # MiniMind2-Small
                    Model(None, 'jingyaogong/MiniMind2-Small'),
                ],
                requires=['transformers>=4.57.1']),
        ],
        TemplateType.minimind,
        get_model_tokenizer_minimind,
        architectures=['LlamaForCausalLM'],
        model_arch=ModelArch.minimind,
    ))
