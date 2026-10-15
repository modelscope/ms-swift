# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from types import MethodType

from swift.template import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


def _nemotron_omni_forward(self,
                           input_ids=None,
                           pixel_values=None,
                           pixel_values_videos=None,
                           image_token_mask=None,
                           video_token_mask=None,
                           inputs_embeds=None,
                           attention_mask=None,
                           position_ids=None,
                           image_flags=None,
                           past_key_values=None,
                           labels=None,
                           sound_clips=None,
                           use_cache=None,
                           output_attentions=None,
                           output_hidden_states=None,
                           return_dict=None,
                           **kwargs):
    # Keep the official named arguments visible to HF Trainer's remove_unused_columns.
    kwargs.update(
        attention_mask=attention_mask,
        position_ids=position_ids,
        image_flags=image_flags,
        past_key_values=past_key_values,
        labels=labels,
        sound_clips=sound_clips,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict)
    # Cache decode has no media; never re-use the prefill masks on the shortened sequence.
    if pixel_values is None and pixel_values_videos is None:
        return self._nemotron_omni_original_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
    if input_ids is None:
        raise ValueError('Nemotron multimodal prefill requires input_ids and explicit media masks.')
    if pixel_values_videos is not None and self.video_pruning_rate > 0:
        if labels is not None:
            raise ValueError('Nemotron HF training requires video_pruning_rate=0 because EVS does not '
                             'realign labels.')
        has_images = pixel_values is not None and (not isinstance(pixel_values, (list, tuple)) or len(pixel_values) > 0)
        if has_images:
            raise NotImplementedError('Official HF EVS does not support mixed image/video inputs.')
        if isinstance(pixel_values_videos, (list, tuple)):
            if len(pixel_values_videos) != 1:
                raise NotImplementedError('Official HF EVS supports one video clip per request.')
            pixel_values_videos = pixel_values_videos[0]
        return self._nemotron_omni_original_forward(
            input_ids=input_ids, pixel_values_videos=pixel_values_videos, inputs_embeds=inputs_embeds, **kwargs)
    if kwargs.get('image_flags') is not None:
        raise ValueError('Nemotron media lists contain only real images; image_flags is not supported.')
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
    masks = []
    for pixels, mask, kind in [(pixel_values, image_token_mask, 'image'),
                               (pixel_values_videos, video_token_mask, 'video')]:
        if mask is None:
            if pixels is not None:
                raise ValueError(f'Nemotron {kind} input requires {kind}_token_mask.')
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if mask.dtype != torch.bool or mask.shape != input_ids.shape:
            raise ValueError(f'{kind}_token_mask must be bool with the same shape as input_ids.')
        mask = mask.to(input_ids.device)
        if (mask & (input_ids != self.img_context_token_id)).any():
            raise ValueError(f'{kind}_token_mask includes a non-image placeholder token.')
        if masks and (masks[0] & mask).any():
            raise ValueError('Image and video token masks must be disjoint.')
        masks.append(mask)
        if pixels is None:
            if mask.any():
                raise ValueError(f'{kind}_token_mask has no corresponding pixel values.')
            continue
        if not isinstance(pixels, (list, tuple)) or not pixels:
            raise ValueError(f'Nemotron {kind} pixels must be a non-empty list preserving media boundaries.')
        features = []
        for value in pixels:
            if value.ndim != 4 or value.shape[1] != 3 or value.shape[0] == 0:
                raise ValueError(f'Expected non-empty [frames, 3, H, W] {kind} tensor.')
            # Never concatenate heterogeneous images on dim 0, or frames across clip boundaries.
            if kind == 'image':
                projected = self.vision_projector(value, self.vision_model)
            else:
                projected = self.vision_projector.forward_video(value, self.vision_model)
            features.append(projected.reshape(-1, projected.shape[-1]))
        features = torch.cat(features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        if features.shape[0] != int(mask.sum()):
            raise ValueError(f'Nemotron {kind} features ({features.shape[0]}) != mask tokens ({int(mask.sum())}).')
        expanded_mask = mask.to(inputs_embeds.device).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, features)
    # Reuse the official language_model call, sound path, cache handling, output type and CE loss.
    return self._nemotron_omni_original_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)


def _nemotron_omni_expand_inputs_for_generation(self,
                                                expand_size=1,
                                                is_encoder_decoder=False,
                                                input_ids=None,
                                                **model_kwargs):
    # HF repeats tensors but not the variable-length media lists owned by each sample.
    if expand_size > 1 and any(model_kwargs.get(key) is not None for key in ('pixel_values', 'pixel_values_videos')):
        raise NotImplementedError('Nemotron Omni media generation requires num_beams=1 and num_return_sequences=1; '
                                  'media list expansion is not supported.')
    return self._nemotron_omni_original_expand_inputs_for_generation(
        expand_size=expand_size, is_encoder_decoder=is_encoder_decoder, input_ids=input_ids, **model_kwargs)


def patch_nemotron_omni_media(model):
    """Patch this loaded Omni instance only; do not modify the checkpoint or other model classes."""
    if not hasattr(model, '_nemotron_omni_original_forward'):
        model._nemotron_omni_original_forward = model.forward
        model.forward = MethodType(_nemotron_omni_forward, model)
    if not hasattr(model, '_nemotron_omni_original_expand_inputs_for_generation'):
        model._nemotron_omni_original_expand_inputs_for_generation = model._expand_inputs_for_generation
        model._expand_inputs_for_generation = MethodType(_nemotron_omni_expand_inputs_for_generation, model)
    return model


class NemotronHLoader(ModelLoader):
    default_trust_remote_code = False


class NemotronHOmniLoader(ModelLoader):
    default_trust_remote_code = True

    def get_model(self, model_dir, config, processor, model_kwargs):
        model = super().get_model(model_dir, config, processor, model_kwargs)
        return patch_nemotron_omni_media(model)


register_model(
    ModelMeta(
        LLMModelType.nemotron_h,
        [
            ModelGroup([
                Model('nv-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
                      'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16'),
                Model('nv-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4',
                      'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4'),
            ]),
        ],
        NemotronHLoader,
        template=TemplateType.nemotron_h,
        architectures=['NemotronHForCausalLM'],
        model_arch=None,
        requires=['transformers>=5.0', 'mamba-ssm', 'causal-conv1d>=1.2.0'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.nemotron_h_omni,
        [
            ModelGroup([
                Model('nv-community/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16',
                      'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16'),
                Model('nv-community/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4',
                      'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4'),
            ]),
        ],
        NemotronHOmniLoader,
        template=TemplateType.nemotron_h_omni,
        model_arch=ModelArch.nemotron_h_omni,
        architectures=['NemotronH_Omni_Reasoning_V3'],
        requires=['transformers>=5.16.1', 'mamba-ssm', 'causal-conv1d>=1.2.0'],
        tags=['vision', 'video'],
    ))
