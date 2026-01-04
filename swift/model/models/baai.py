# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from typing import Any, Dict

from transformers import AutoModel, AutoModelForSequenceClassification

from swift.template import TemplateType
from swift.utils import get_device
from ..constant import MLLMModelType, RerankerModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model
from ..utils import git_clone_github, safe_snapshot_download


class Emu3GenLoader(ModelLoader):

    def get_model(self, model_dir: str, config, model_kwargs):
        model = super().get_model(model_dir, config, model_kwargs)
        model.config.image_area = int(os.environ.get('image_area', model.config.image_area))
        model.config.max_position_embeddings = int(
            os.environ.get('max_position_embeddings', model.config.max_position_embeddings))
        model.generation_config.do_sample = True


def get_model_tokenizer_emu3_gen(model_dir: str,
                                 model_info,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    import sys
    sys.path.append(model_dir)
    from processing_emu3 import Emu3Processor
    vq_hub = safe_snapshot_download('BAAI/Emu3-VisionTokenizer', check_local=True)
    from transformers import AutoModel, AutoImageProcessor
    image_processor = AutoImageProcessor.from_pretrained(vq_hub, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(vq_hub, trust_remote_code=True).eval().to(get_device())
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
    model_info.max_model_len = model_info.max_model_len + 40960
    processor.image_area = model.config.image_area
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.emu3_gen,
        [
            ModelGroup([
                Model('BAAI/Emu3-Gen', 'BAAI/Emu3-Gen'),
            ]),
        ],
        Emu3GenLoader,
        template=TemplateType.emu3_gen,
        architectures=['Emu3ForCausalLM'],
        model_arch=ModelArch.emu3_chat,
        tags=['t2i'],
    ))


def get_model_tokenizer_emu3_chat(model_dir: str,
                                  model_info,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    # download and load vision tokenizer
    from transformers import AutoImageProcessor
    vq_model = safe_snapshot_download('BAAI/Emu3-VisionTokenizer', check_local=True)
    image_processor = AutoImageProcessor.from_pretrained(vq_model, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(vq_model, device_map=model_kwargs['device_map'], trust_remote_code=True)
    image_tokenizer.requires_grad_(False)
    image_tokenizer.to(get_device())

    # load processor
    local_repo_path = kwargs.get('local_repo_path')
    if not local_repo_path:
        local_repo_path = git_clone_github('https://github.com/baaivision/Emu3.git')
    sys.path.append(local_repo_path)
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
        get_model_tokenizer_emu3_chat,
        template=TemplateType.emu3_chat,
        architectures=['Emu3ForCausalLM'],
        model_arch=ModelArch.emu3_chat,
        tags=['vision'],
        requires=['transformers>=4.44.0'],
    ))


def get_model_tokenizer_bge_reranker(*args, **kwargs):
    kwargs['automodel_class'] = AutoModelForSequenceClassification
    return get_model_tokenizer_with_flash_attn(*args, **kwargs)


register_model(
    ModelMeta(
        RerankerModelType.bge_reranker,
        [
            ModelGroup([
                Model('BAAI/bge-reranker-base', 'BAAI/bge-reranker-base'),
                Model('BAAI/bge-reranker-v2-m3', 'BAAI/bge-reranker-v2-m3'),
                Model('BAAI/bge-reranker-large', 'BAAI/bge-reranker-large'),
            ]),
        ],
        get_model_tokenizer_bge_reranker,
        template=TemplateType.bge_reranker,
        architectures=['XLMRobertaForSequenceClassification'],
    ))
