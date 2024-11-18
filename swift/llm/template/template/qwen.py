# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn

from swift.utils import get_env_args, is_deepspeed_enabled
from ..base import Template
from ..constant import TemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, fetch_one, findall
from ..vision_utils import load_audio_qwen, load_batch, load_video_qwen2
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta


@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False


@dataclass
class Qwen2_5TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


register_template(ChatmlTemplateMeta(TemplateType.chatml))
register_template(QwenTemplateMeta(TemplateType.qwen))
register_template(Qwen2_5TemplateMeta(TemplateType.qwen2_5))


class QwenVLTemplate(Template):
    load_medias = False

    def _check_inputs(self, inputs: StdTemplateInputs):
        if self.mode in {'lmdeploy', 'vllm'}:
            return
        images = inputs.images
        assert not images or isinstance(fetch_one(images), str), 'QwenVL only supports datasets with images paths!'

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        if self.mode == 'lmdeploy':
            return [f'Picture {index + 1}: ', [-100], '\n']
        else:
            image = inputs.images[index]
            if self.mode == 'vllm':
                return [f'Picture {index + 1}: <img></img>\n']
            else:
                assert isinstance(image, str)
                return [f'Picture {index + 1}: <img>{image}</img>\n']

    def replace_object(self, object_: Dict[str, Any], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<ref>{object_["caption"]}</ref>']

    def replace_box(self, object_: Dict[str, Any], index: int, inputs: StdTemplateInputs) -> List[Context]:
        if isinstance(object_['bbox'][0], list):
            all_objects = ''
            for sub_object in object_['bbox']:
                all_objects += f'<box>({sub_object[0]},{sub_object[1]}),({sub_object[2]},{sub_object[3]})</box>'
            return [all_objects]
        else:
            return [
                f'<box>({object_["bbox"][0]},{object_["bbox"][1]}),'
                f'({object_["bbox"][2]},{object_["bbox"][3]})</box>'
            ]


register_template(QwenTemplateMeta(TemplateType.qwen_vl, template_cls=QwenVLTemplate))


class QwenAudioTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        audios = inputs.audios
        audio = audios[index]
        assert isinstance(audio, str)
        return [f'Audio {index + 1}:<audio>{audio}</audio>\n']

    def _get_tokenizer_kwargs(self, context: str) -> Dict[str, Any]:
        return {'audio_info': self.processor.process_audio(context)}

    def _concat_tokenizer_kwargs(self, tokenizer_kwargs: Dict[str, Any], curr_tokenizer_kwargs: Dict[str, Any]) -> None:
        audio_info = curr_tokenizer_kwargs.get('audio_info')
        old_audio_info = tokenizer_kwargs.get('audio_info')
        if old_audio_info is None:
            tokenizer_kwargs['audio_info'] = audio_info
        elif audio_info is not None:
            for k in ['input_audios', 'input_audio_lengths']:
                old_audio_info[k] = torch.concat([old_audio_info[k], audio_info[k]], dim=0)
            for k in ['audio_span_tokens', 'audio_urls']:
                old_audio_info[k] = old_audio_info[k] + audio_info[k]

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      *,
                      padding_side: Optional[str] = None,
                      padding_to: Optional[int] = None,
                      model: Optional[nn.Module] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_side=padding_side, padding_to=padding_to, model=model)
        if batch[0].get('audio_info') is not None:
            res['audio_info'] = [b['audio_info'] for b in batch]
        return res


register_template(QwenTemplateMeta(TemplateType.qwen_audio, template_cls=QwenAudioTemplate))


class Qwen2AudioTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        if self.use_generate_template:
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>\n']
        else:
            return [f'Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n']

    def _encode(self, template_inputs: StdTemplateInputs) -> Dict[str, Any]:
        inputs = super()._encode(template_inputs, model=model)
        if len(inputs) == 0:
            return inputs
        processor = self.processor
        sampling_rate = processor.feature_extractor.sampling_rate
        audios = load_batch(template_inputs.audios, load_func=partial(load_audio_qwen, sampling_rate=sampling_rate))
        if audios:
            audio_inputs = processor.feature_extractor(
                audios, sampling_rate=sampling_rate, return_attention_mask=True, return_tensors='pt')
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
            inputs.update(audio_inputs)
        return inputs

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      *,
                      padding_side: Optional[str] = None,
                      padding_to: Optional[int] = None,
                      model: Optional[nn.Module] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_side=padding_side, padding_to=padding_to, model=model)
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res


register_template(QwenTemplateMeta(TemplateType.qwen2_audio, template_cls=Qwen2AudioTemplate))


def _process_image_qwen(image):
    from qwen_vl_utils.vision_process import IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS, smart_resize
    size_factor = get_env_args('size_factor', int, IMAGE_FACTOR)
    # resize
    resized_height = get_env_args('resized_height', int, None)
    resized_width = get_env_args('resized_width', int, None)
    if resized_height and resized_width:
        resized_height, resized_width = smart_resize(
            resized_height,
            resized_width,
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = get_env_args('min_pixels', int, MIN_PIXELS)
        max_pixels = get_env_args('max_pixels', int, MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))
    return image


class Qwen2VLTemplate(Template):
    image_token_id = 151655
    video_token_id = 151656

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = _process_image_qwen(inputs.images[index])
            return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            inputs.videos[index] = load_video_qwen2(inputs.videos[index])
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def replace_object(self, object_: Dict[str, Any], index: int, inputs: StdTemplateInputs) -> List[Context]:
        if object_:
            return ['<|object_ref_start|>', object_['caption'], '<|object_ref_end|>']
        else:
            return ['<ref-object>']

    def replace_box(self, object_: Dict[str, Any], index: int, inputs: StdTemplateInputs) -> List[Context]:
        if object_:
            if isinstance(object_['bbox'][0], list):
                all_objects = ''
                for sub_object in object_['bbox']:
                    all_objects += (f'<|box_start|>({sub_object[0]},{sub_object[1]}),'
                                    f'({sub_object[2]},{sub_object[3]})<|box_end|>')
                return [all_objects]
            else:
                return [
                    f'<|box_start|>({object_["bbox"][0]},{object_["bbox"][1]}),'
                    f'({object_["bbox"][2]},{object_["bbox"][3]})<|box_end|>'
                ]
        else:
            return ['<bbox>']

    def _encode(self, template_inputs: StdTemplateInputs) -> Dict[str, Any]:
        inputs = super()._encode(template_inputs, model=model)
        if len(inputs) == 0:
            return inputs
        processor = self.processor
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        images = template_inputs.images
        videos = template_inputs.videos
        for media_type in ['images', 'videos']:
            if locals()[media_type]:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=images, videos=None, return_tensors='pt')
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    media_inputs = processor.image_processor(images=None, videos=videos, return_tensors='pt')
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                idx_list = findall(input_ids, media_token)
                added_tokens_len = 0
                for i, idx in enumerate(idx_list):
                    merge_length = processor.image_processor.merge_size**2
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    input_ids = input_ids[:idx
                                          + added_tokens_len] + [media_token] * token_len + input_ids[added_tokens_len
                                                                                                      + idx + 1:]
                    if labels:
                        labels = labels[:idx + added_tokens_len] + [-100] * token_len + labels[added_tokens_len + idx
                                                                                               + 1:]
                    added_tokens_len += token_len - 1
                inputs.update(media_inputs)

        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        if self.mode != 'train':
            return data
        input_ids = data['input_ids'][None]
        _model = model.model
        if not hasattr(_model, 'embed_tokens'):
            _model = _model.model  # LoRA
        pixel_values = data.get('pixel_values')
        pixel_values_videos = data.get('pixel_values_videos')
        inputs_embeds = _model.embed_tokens(input_ids)
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            if is_deepspeed_enabled():
                from PIL import Image
                images = [Image.new('RGB', (32, 32), (0, 0, 0))]
                processor = self.processor
                media_inputs = processor.image_processor(images=images, videos=None, return_tensors='pt')
                device = input_ids.device
                pixel_values = media_inputs['pixel_values'].to(device)

                pixel_values = pixel_values.type(model.visual.get_dtype())
                image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
                inputs_embeds += image_embeds.mean() * 0.
        else:
            if pixel_values is not None:
                image_grid_thw = data['image_grid_thw']
                pixel_values = pixel_values.type(model.visual.get_dtype())
                image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_grid_thw = data['video_grid_thw']
                pixel_values_videos = pixel_values_videos.type(model.visual.get_dtype())
                video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        res = {'inputs_embeds': inputs_embeds[0]}
        for key in ['input_ids', 'image_grid_thw', 'video_grid_thw']:
            value = data.get(key, None)
            if value is not None:
                res[key] = value
        return res

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      *,
                      padding_side: Optional[str] = None,
                      padding_to: Optional[int] = None,
                      model: Optional[nn.Module] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_side=padding_side, padding_to=padding_to, model=model)
        for media_type in ['image', 'video']:
            grid_thw = [b[f'{media_type}_grid_thw'] for b in batch if b.get(f'{media_type}_grid_thw') is not None]
            if grid_thw:
                res[f'{media_type}_grid_thw'] = torch.concat(grid_thw)
        if 'input_ids' in res and self.mode == 'train':
            image_grid_thw = res.pop('image_grid_thw', None)
            video_grid_thw = res.pop('video_grid_thw', None)
            # fix https://github.com/huggingface/transformers/pull/33487
            position_ids, _ = model.get_rope_index(res['input_ids'], image_grid_thw, video_grid_thw,
                                                   res['attention_mask'])
            res['position_ids'] = position_ids.contiguous()
        return res


register_template(
    QwenTemplateMeta(
        TemplateType.qwen2_vl, template_cls=Qwen2VLTemplate, placeholder_tokens=['<|image_pad|>', '<|video_pad|>']))
