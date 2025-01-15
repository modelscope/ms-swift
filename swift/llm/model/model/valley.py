# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from cProfile import label
from functools import partial, wraps
from typing import Any, Dict

from transformers import AutoConfig

from swift.llm import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import ModelInfo, git_clone_github, safe_snapshot_download


def get_model_tokenizer_valley(model_dir: str,
                               model_info: ModelInfo,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    llm_model_type = kwargs.pop('llm_model_type')
    local_repo_path = kwargs.get('local_repo_path')
    if not local_repo_path:
        repo_path = 'https://github.com/bytedance/Valley.git'
        local_repo_path = git_clone_github(repo_path)
    sys.path.append(os.path.join(local_repo_path))

    if llm_model_type == 'valley':
        from torch.nn import CrossEntropyLoss
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from valley_eagle.model.language_model.valley_qwen2 import ValleyQwen2ForCausalLM, ValleyConfig
        model_config = ValleyConfig.from_pretrained(model_dir)
        model_config.mm_vision_tower = safe_snapshot_download('AI-ModelScope/siglip-so400m-patch14-384')
        model_config.eagle_vision_tower = safe_snapshot_download('Qwen/Qwen2-VL-7B-Instruct')
        automodel_class = ValleyQwen2ForCausalLM

        if not hasattr(ValleyQwen2ForCausalLM, '_origin_forward'):
            forward = ValleyQwen2ForCausalLM.forward
            ValleyQwen2ForCausalLM._origin_forward = forward

            @wraps(forward)
            def new_forward(*args, **kwargs):
                prepare_kwargs = {
                    'input_ids': kwargs.get('input_ids'),
                    'position_ids': kwargs.get('position_ids'),
                    'attention_mask': kwargs.get('attention_mask'),
                    'past_key_values': kwargs.get('past_key_values'),
                    'labels': kwargs.get('labels'),
                    'images': kwargs.get('images'),
                    'image_sizes': kwargs.get('image_sizes'),
                    'pixel_values': kwargs.get('pixel_values'),
                    'pixel_values_videos': kwargs.get('pixel_values_videos'),
                    'image_grid_thw': kwargs.get('image_grid_thw'),
                    'video_grid_thw': kwargs.get('video_grid_thw'),
                    'pack_ids': kwargs.get('pack_ids'),
                }
                _, _, _, _, _, new_labels = args[0].prepare_inputs_labels_for_multimodal(**prepare_kwargs)
                outputs = forward(*args, **kwargs)
                loss = outputs.loss
                logits = outputs.logits
                if 'labels' in kwargs:
                    # Upcast to float if we need to compute the loss to avoid potential precision issues
                    logits = logits.float()
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = new_labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, args[0].config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

            ValleyQwen2ForCausalLM.forward = new_forward
    kwargs['model_config'] = model_config
    kwargs['automodel_class'] = automodel_class
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    model.generation_config.repetition_penalty = 1.0  # Otherwise, Error. Same for original code.
    if model is not None:
        from transformers import AutoProcessor, SiglipImageProcessor
        tokenizer.image_processor = SiglipImageProcessor.from_pretrained(model.config.mm_vision_tower)
        tokenizer.qwen2vl_processor = AutoProcessor.from_pretrained(
            model.config.eagle_vision_tower, max_pixels=1280 * 28 * 28)
        tokenizer.image_processor.crop_size = tokenizer.image_processor.size['height']
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.valley,
        [
            ModelGroup([
                Model('bytedance-research/Valley-Eagle-7B'),
            ], ),
        ],
        TemplateType.valley,
        partial(get_model_tokenizer_valley, llm_model_type='valley'),
        architectures=['ValleyQwen2ForCausalLM'],
        model_arch=ModelArch.valley,
        requires=['transformers>=4.42', 'av'],
        tags=['vision'],
    ))
