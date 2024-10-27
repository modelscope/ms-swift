# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from swift.utils import get_env_args, is_deepspeed_enabled
from ..base import Template
from ..constant import TemplateType
from ..register import register_template
from ..utils import Context, findall
from ..vision_utils import load_audio_qwen, load_batch, load_video_qwen2

DEFAULT_SYSTEM = 'You are a helpful assistant.'


# You can set the query as '' to serve as a template for pre-training.
class DefaultGenerationTemplate(Template):

    def __init__(self, template_type: str):
        super().__init__(template_type, [], ['{{QUERY}}'], None, [['eos_token_id']], auto_add_bos=True)


register_template(DefaultGenerationTemplate(TemplateType.default_generation), is_generation=True)


class ChatmlTemplateMixin:
    system = None

    def __init__(self, template_type: str, auto_add_bos: bool = True):
        Template.__init__(
            self,
            template_type, [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'], ['<|im_end|>\n'],
            ['<|im_end|>'],
            self.system, ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            auto_add_bos=auto_add_bos)


class ChatmlTemplate(ChatmlTemplateMixin, Template):
    pass


class QwenTemplateMixin(ChatmlTemplateMixin):
    system = DEFAULT_SYSTEM

    def __init__(self, template_type: str):
        super().__init__(template_type, auto_add_bos=False)


class QwenTemplate(QwenTemplateMixin, Template):
    pass


class Qwen2_5Template(QwenTemplate):
    system = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


register_template(ChatmlTemplate(TemplateType.chatml))
register_template(QwenTemplate(TemplateType.qwen))
register_template(Qwen2_5Template(TemplateType.qwen2_5))


class _QwenVLTemplateMixin:
    load_medias = False

    def check_example(self, example):
        if self._is_lmdeploy or self._is_vllm:
            return
        images = example.get('images') or []
        from ..utils import fetch_one
        assert not images or isinstance(fetch_one(images), str), 'QwenVL only supports datasets with images paths!'

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type == 'image'
        if self._is_lmdeploy:
            return [f'Picture {index + 1}: ', [-100], '\n']
        else:
            images = example.get('images') or []
            image = images[index]
            if self._is_vllm:
                return [f'Picture {index + 1}: <img></img>\n']
            else:
                assert isinstance(image, str)
                return [f'Picture {index + 1}: <img>{image}</img>\n']

    def replace_object(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example['objects']
        object_ = objects[index]
        return [f'<ref>{object_["caption"]}</ref>']

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example['objects']
        object_ = objects[index]
        if isinstance(object_['bbox'][0], list):
            all_objects = ''
            for sub_object in object_['bbox']:
                all_objects += (f'<box>({sub_object[0]},{sub_object[1]}),' f'({sub_object[2]},{sub_object[3]})</box>')
            return [all_objects]
        else:
            return [
                f'<box>({object_["bbox"][0]},{object_["bbox"][1]}),'
                f'({object_["bbox"][2]},{object_["bbox"][3]})</box>'
            ]


class QwenVLTemplate(_QwenVLTemplateMixin, QwenTemplate):
    pass


class QwenVLGenerationTemplate(_QwenVLTemplateMixin, DefaultGenerationTemplate):
    pass


register_template(QwenVLTemplate(TemplateType.qwen_vl))
register_template(QwenVLGenerationTemplate(TemplateType.qwen_vl_generation))


class _QwenAudioTemplateMixin:

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type == 'audio'
        audios = example.get('audios') or []
        audio = audios[index]
        assert isinstance(audio, str)
        return [f'Audio {index + 1}:<audio>{audio}</audio>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = Template._encode(self, example)
        if len(inputs) == 0:
            return inputs, tokenizer_kwargs
        inputs.pop('loss_scale', None)
        inputs.update(tokenizer_kwargs)
        return inputs, tokenizer_kwargs

    def _get_tokenizer_kwargs(self, context: str) -> Dict[str, Any]:
        return {'audio_info': self.tokenizer.process_audio(context)}

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

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = Template.data_collator(self, batch, padding_to)
        if batch[0].get('audio_info') is not None:
            res['audio_info'] = [b['audio_info'] for b in batch]
        return res


class QwenAudioTemplate(_QwenAudioTemplateMixin, QwenTemplate):
    pass


class QwenAudioGenerationTemplate(_QwenAudioTemplateMixin, DefaultGenerationTemplate):
    pass


register_template(QwenAudioTemplate(TemplateType.qwen_audio), lazy_tokenize=True)
register_template(
    QwenAudioGenerationTemplate(TemplateType.qwen_audio_generation), lazy_tokenize=True, is_generation=True)


class _Qwen2AudioTemplateMixin:

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = Template._encode(self, example)
        if len(inputs) == 0:
            return inputs, {}
        processor = self.tokenizer.processor
        sampling_rate = processor.feature_extractor.sampling_rate
        audios = load_batch(
            example.get('audios') or [], load_func=partial(load_audio_qwen, sampling_rate=sampling_rate))
        if audios:
            audio_inputs = processor.feature_extractor(
                audios, sampling_rate=sampling_rate, return_attention_mask=True, return_tensors='pt')
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
            inputs.update(audio_inputs)
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = Template.data_collator(self, batch, padding_to)
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res


class Qwen2AudioTemplate(_Qwen2AudioTemplateMixin, QwenTemplate):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type == 'audio'
        return [f'Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n']


class Qwen2AudioGenerationTemplate(_Qwen2AudioTemplateMixin, DefaultGenerationTemplate):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type == 'audio'
        return ['<|audio_bos|><|AUDIO|><|audio_eos|>\n']


register_template(Qwen2AudioTemplate(TemplateType.qwen2_audio), lazy_tokenize=True)

register_template(
    Qwen2AudioGenerationTemplate(TemplateType.qwen2_audio_generation), lazy_tokenize=True, is_generation=True)


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


class _Qwen2VLTemplateMixin:

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            example['images'][index] = _process_image_qwen(example['images'][index])
            return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            example['videos'][index] = load_video_qwen2(example['videos'][index])
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def replace_object(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            return ['<|object_ref_start|>', object_['caption'], '<|object_ref_end|>']
        else:
            return ['<ref-object>']

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
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

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        processor = self.tokenizer.processor
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        images = example.get('images') or []
        videos = example.get('videos') or []
        for media_type in ['images', 'videos']:
            if locals()[media_type]:
                if media_type == 'images':
                    media_token = 151655
                    media_inputs = processor.image_processor(images=images, videos=None, return_tensors='pt')
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    media_inputs = processor.image_processor(images=None, videos=videos, return_tensors='pt')
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = 151656
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
        inputs['_data'] = {'plain_text': not images and not videos, 'input_ids': torch.tensor(input_ids)[None]}
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        plain_text = data.pop('plain_text', False)
        if is_deepspeed_enabled() and plain_text:
            from PIL import Image
            images = [Image.new('RGB', (32, 32), (0, 0, 0))]
            processor = self.tokenizer.processor
            media_inputs = processor.image_processor(images=images, videos=None, return_tensors='pt')
            input_ids = data['input_ids']
            device = input_ids.device
            pixel_values = media_inputs['pixel_values'].to(device)
            _model = model.model
            if not hasattr(_model, 'embed_tokens'):
                _model = _model.model  # LoRA
            inputs_embeds = _model.embed_tokens(input_ids)
            pixel_values = pixel_values.type(model.visual.get_dtype())
            image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
            inputs_embeds += image_embeds.mean() * 0.
            return {'inputs_embeds': inputs_embeds[0]}
        return {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        for media_type in ['image', 'video']:
            grid_thw = [b[f'{media_type}_grid_thw'] for b in batch if b.get(f'{media_type}_grid_thw') is not None]
            if grid_thw:
                res[f'{media_type}_grid_thw'] = torch.concat(grid_thw)
        if 'input_ids' in res:
            # fix https://github.com/huggingface/transformers/pull/33487
            position_ids, _ = self.model.get_rope_index(res['input_ids'], res.get('image_grid_thw'),
                                                        res.get('video_grid_thw'), res['attention_mask'])
            res['position_ids'] = position_ids.contiguous()
        return res


class Qwen2VLTemplate(_Qwen2VLTemplateMixin, QwenTemplate):
    pass


class Qwen2VLGenerationTemplate(_Qwen2VLTemplateMixin, DefaultGenerationTemplate):
    pass


register_template(Qwen2VLTemplate(TemplateType.qwen2_vl), lazy_tokenize=True)

register_template(Qwen2VLGenerationTemplate(TemplateType.qwen2_vl_generation), lazy_tokenize=True, is_generation=True)
