# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys
from typing import Any, Dict

from transformers import AutoModel, AutoModelForSequenceClassification, PretrainedConfig, PreTrainedModel

from swift.template import TemplateType
from swift.utils import Processor, get_device, git_clone_github, safe_snapshot_download
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


class Emu3GenLoader(ModelLoader):

    def get_processor(self, model_dir, config) -> Processor:
        self.model_info.max_model_len = self.model_info.max_model_len + 40960
        config.image_area = int(os.environ.get('image_area', config.image_area))
        config.max_position_embeddings = int(os.environ.get('max_position_embeddings', config.max_position_embeddings))
        tokenizer = super().get_processor(model_dir, config)
        import sys
        sys.path.append(model_dir)
        from processing_emu3 import Emu3Processor
        vq_hub = safe_snapshot_download('BAAI/Emu3-VisionTokenizer', check_local=True)
        from transformers import AutoModel, AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(vq_hub, trust_remote_code=True)
        image_tokenizer = AutoModel.from_pretrained(vq_hub, trust_remote_code=True).eval().to(get_device())
        processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
        processor.image_area = config.image_area
        return processor

    def get_model(self, model_dir: str, config, processor, model_kwargs):
        model = super().get_model(model_dir, config, processor, model_kwargs)
        model.generation_config.do_sample = True


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


class Emu3ChatLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        tokenizer = super().get_processor(model_dir, config)
        # download and load vision tokenizer
        from transformers import AutoImageProcessor
        vq_model = safe_snapshot_download('BAAI/Emu3-VisionTokenizer', check_local=True)
        image_processor = AutoImageProcessor.from_pretrained(vq_model, trust_remote_code=True)
        image_tokenizer = AutoModel.from_pretrained(
            vq_model, device_map=self.model_kwargs['device_map'], trust_remote_code=True)
        image_tokenizer.requires_grad_(False)
        image_tokenizer.to(get_device())
        # load processor
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            local_repo_path = git_clone_github('https://github.com/baaivision/Emu3.git')
        sys.path.append(local_repo_path)
        from emu3.mllm.processing_emu3 import Emu3Processor
        return Emu3Processor(image_processor, image_tokenizer, tokenizer)


register_model(
    ModelMeta(
        MLLMModelType.emu3_chat,
        [
            ModelGroup([
                Model('BAAI/Emu3-Chat', 'BAAI/Emu3-Chat'),
            ]),
        ],
        Emu3ChatLoader,
        template=TemplateType.emu3_chat,
        architectures=['Emu3ForCausalLM'],
        model_arch=ModelArch.emu3_chat,
        tags=['vision'],
        requires=['transformers>=4.44.0'],
    ))


class BgeRerankerLoader(ModelLoader):

    def get_model(self, *args, **kwargs) -> PreTrainedModel:
        self.auto_model_cls = self.auto_model_cls or AutoModelForSequenceClassification
        return super().get_model(*args, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.bge_reranker,
        [
            ModelGroup([
                Model('BAAI/bge-reranker-base', 'BAAI/bge-reranker-base'),
                Model('BAAI/bge-reranker-v2-m3', 'BAAI/bge-reranker-v2-m3'),
                Model('BAAI/bge-reranker-large', 'BAAI/bge-reranker-large'),
            ]),
        ],
        BgeRerankerLoader,
        template=TemplateType.bge_reranker,
        task_type='reranker',
        architectures=['XLMRobertaForSequenceClassification'],
    ))
