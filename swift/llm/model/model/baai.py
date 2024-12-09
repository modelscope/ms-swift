# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from typing import Any, Dict

from transformers import AutoModel

from swift.llm import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo, git_clone_github, safe_snapshot_download


def get_model_tokenizer_emu3_gen(model_dir: str,
                                 model_info: ModelInfo,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    import sys
    sys.path.append(model_dir)
    from processing_emu3 import Emu3Processor
    vq_hub = safe_snapshot_download('BAAI/Emu3-VisionTokenizer')
    from transformers import AutoModel, AutoImageProcessor
    image_processor = AutoImageProcessor.from_pretrained(vq_hub, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(vq_hub, trust_remote_code=True).eval().to('cuda:0')
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
    model_info.max_model_len = model_info.max_model_len + 40960
    if model:
        model.config.image_area = int(os.environ.get('image_area', model.config.image_area))
        model.config.max_position_embeddings = int(
            os.environ.get('max_position_embeddings', model.config.max_position_embeddings))
        processor.image_area = model.config.image_area
        model.generation_config.do_sample = True
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.emu3_gen,
        [
            ModelGroup([
                Model('BAAI/Emu3-Gen', 'BAAI/Emu3-Gen'),
            ]),
        ],
        TemplateType.emu3_gen,
        get_model_tokenizer_emu3_gen,
        architectures=['Emu3ForCausalLM'],
        model_arch=ModelArch.emu3_chat,
        tags=['t2i'],
    ))


def get_model_tokenizer_emu3_chat(model_dir: str,
                                  model_info: ModelInfo,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)

    # download and load vision tokenizer
    from transformers import AutoImageProcessor
    vq_model = safe_snapshot_download('BAAI/Emu3-VisionTokenizer')
    image_processor = AutoImageProcessor.from_pretrained(vq_model, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(vq_model, device_map=model_kwargs['device_map'], trust_remote_code=True)
    image_tokenizer.requires_grad_(False)
    image_tokenizer.to('cuda:0')  # TODO: check npu

    # load processor
    local_repo_path = kwargs.get('local_repo_path')
    if not local_repo_path:
        local_repo_path = git_clone_github('https://github.com/baaivision/Emu3.git')
    sys.path.append(os.path.join(local_repo_path))
    from emu3.mllm.processing_emu3 import Emu3Processor
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.emu3_chat,
        [
            ModelGroup([
                Model('BAAI/Emu3-Chat', 'BAAI/Emu3-Chat'),
            ]),
        ],
        TemplateType.emu3_chat,
        get_model_tokenizer_emu3_chat,
        architectures=['Emu3ForCausalLM'],
        model_arch=ModelArch.emu3_chat,
        tags=['vision'],
        requires=['transformers>=4.44.0'],
    ))
