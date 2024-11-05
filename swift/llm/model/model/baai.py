# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from transformers import PretrainedConfig

from swift.llm import TemplateType
from ..constant import MLLMModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import safe_snapshot_download


def get_model_tokenizer_emu3_gen(model_dir: str,
                                 config: PretrainedConfig,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    import sys
    sys.path.append(model_dir)
    from processing_emu3 import Emu3Processor
    vq_hub = safe_snapshot_download('BAAI/Emu3-VisionTokenizer')
    from transformers import AutoModel, AutoImageProcessor
    image_processor = AutoImageProcessor.from_pretrained(vq_hub, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(vq_hub, trust_remote_code=True).eval()
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, config, model_kwargs, load_model, **kwargs)
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
    tokenizer.processor = processor
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.emu3_gen,
        [
            ModelGroup([
                Model('BAAI/Emu3-Gen', 'BAAI/Emu3-Gen'),
            ], tags=['multi-modal', 't2i']),
        ],
        TemplateType.emu3_gen,
        get_model_tokenizer_emu3_gen,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=False,
        support_lmdeploy=False,
    ))
