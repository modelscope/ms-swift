# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from functools import wraps
from typing import Any, Dict

from transformers import PreTrainedModel

from swift.template import TemplateType
from swift.utils import git_clone_github, safe_snapshot_download
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


class ValleyLoader(ModelLoader):

    def get_config(self, model_dir: str):
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            repo_path = 'https://github.com/bytedance/Valley.git'
            local_repo_path = git_clone_github(repo_path)
        sys.path.append(local_repo_path)
        from valley_eagle.model.language_model.valley_qwen2 import ValleyConfig
        self.autoconfig_class = ValleyConfig
        return super().get_config(model_dir)

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from valley_eagle.model.language_model.valley_qwen2 import ValleyQwen2ForCausalLM
        config.mm_vision_tower = safe_snapshot_download('AI-ModelScope/siglip-so400m-patch14-384', check_local=True)
        config.eagle_vision_tower = safe_snapshot_download('Qwen/Qwen2-VL-7B-Instruct', check_local=True)
        auto_model_cls = ValleyQwen2ForCausalLM

        if not hasattr(ValleyQwen2ForCausalLM, '_origin_forward'):
            forward = ValleyQwen2ForCausalLM.forward
            ValleyQwen2ForCausalLM._origin_forward = forward

            @wraps(forward)
            def new_forward(*args, **kwargs):
                import torch
                outputs = forward(*args, **kwargs)
                loss = outputs.loss
                if loss is not None and loss.shape[-1] > 0:
                    loss = torch.mean(loss, dim=-1)
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=outputs.logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

            ValleyQwen2ForCausalLM.forward = new_forward
        self.auto_model_cls = auto_model_cls
        model = super().get_model(model_dir, config, processor, model_kwargs)
        model.generation_config.repetition_penalty = 1.0  # Otherwise, Error. Same for original code.

        from transformers import AutoProcessor, SiglipImageProcessor
        processor.image_processor = SiglipImageProcessor.from_pretrained(model.config.mm_vision_tower)
        processor.qwen2vl_processor = AutoProcessor.from_pretrained(
            model.config.eagle_vision_tower, max_pixels=1280 * 28 * 28)
        processor.image_processor.crop_size = processor.image_processor.size['height']
        return model


register_model(
    ModelMeta(
        MLLMModelType.valley,
        [
            ModelGroup([
                Model('bytedance-research/Valley-Eagle-7B'),
            ], ),
        ],
        ValleyLoader,
        template=TemplateType.valley,
        architectures=['ValleyQwen2ForCausalLM'],
        model_arch=ModelArch.valley,
        requires=['transformers>=4.42', 'av'],
        tags=['vision'],
    ))
