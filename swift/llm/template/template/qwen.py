# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch

from swift.utils import get_env_args, is_deepspeed_enabled
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..template_meta import TemplateMeta
from ..utils import Context, Word, findall
from ..vision_utils import load_audio_qwen, load_batch
from .llama import Llama3TemplateMeta
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta


@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])


@dataclass
class Qwen2_5TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


@dataclass
class Qwen2_5MathTemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'Please reason step by step, and put your final answer within \\boxed{}.'


@dataclass
class QwqTemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = ('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                                     'You should think step-by-step.')


register_template(QwenTemplateMeta(LLMTemplateType.qwen))
register_template(Qwen2_5TemplateMeta(LLMTemplateType.qwen2_5))
register_template(QwqTemplateMeta(LLMTemplateType.qwq))

register_template(Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math))


class QwenVLTemplate(Template):
    load_images = False

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


register_template(QwenTemplateMeta(MLLMTemplateType.qwen_vl, template_cls=QwenVLTemplate))


class QwenAudioTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        audios = inputs.audios
        audio = audios[index]
        assert isinstance(audio, str)
        return [f'Audio {index + 1}:<audio>{audio}</audio>\n']

    def _tokenize(self, context, **tokenizer_kwargs):
        audio_info = self.processor.process_audio(context)
        return super()._tokenize(context, audio_info=audio_info)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        text = ''.join([f'<audio>{audio}</audio>' for audio in inputs.audios])
        audio_info = self.processor.process_audio(text)
        if audio_info:
            tokenizer_kwargs = {'audio_info': audio_info}
            encoded.update(tokenizer_kwargs)
            encoded['tokenizer_kwargs'] = tokenizer_kwargs
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if batch[0].get('audio_info') is not None:
            res['audio_info'] = [b['audio_info'] for b in batch]
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen_audio, template_cls=QwenAudioTemplate))


class Qwen2AudioTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        if not self.use_chat_template:
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>\n']
        else:
            return [f'Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        sampling_rate = processor.feature_extractor.sampling_rate
        audios = load_batch(inputs.audios, load_func=partial(load_audio_qwen, sampling_rate=sampling_rate))
        if audios:
            audio_inputs = processor.feature_extractor(
                audios, sampling_rate=sampling_rate, return_attention_mask=True, return_tensors='pt')
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
            encoded.update(audio_inputs)
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_audio, template_cls=Qwen2AudioTemplate))


class Qwen2VLTemplate(Template):
    image_token_id = 151655
    video_token_id = 151656

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            inputs.videos[index] = fetch_video({'video': inputs.videos[index]}).to(torch.uint8)
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

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        images = inputs.images
        videos = inputs.videos
        for media_type in ['images', 'videos']:
            if locals()[media_type]:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(
                        images=images, videos=None, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    media_inputs = processor.image_processor(
                        images=None, videos=videos, return_tensors='pt', do_resize=False)
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
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        _model = model.model
        if not hasattr(_model, 'embed_tokens'):
            _model = _model.model  # LoRA
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')

        inputs_embeds = _model.embed_tokens(input_ids)
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            if is_deepspeed_enabled():
                from PIL import Image
                images = [Image.new('RGB', (32, 32), (0, 0, 0))]
                media_inputs = self.processor.image_processor(images=images, videos=None, return_tensors='pt')
                device = input_ids.device
                pixel_values = media_inputs['pixel_values'].to(device)

                pixel_values = pixel_values.type(model.visual.get_dtype())
                image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
                inputs_embeds += image_embeds.mean() * 0.
        else:
            if pixel_values is not None:
                pixel_values = pixel_values.type(model.visual.get_dtype())
                image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(model.visual.get_dtype())
                video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # fix https://github.com/huggingface/transformers/pull/33487
        position_ids, _ = model.get_rope_index(input_ids, image_grid_thw, video_grid_thw, inputs['attention_mask'])
        return {'inputs_embeds': inputs_embeds, 'position_ids': position_ids.contiguous()}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        for media_type in ['image', 'video']:
            grid_thw = [b[f'{media_type}_grid_thw'] for b in batch if b.get(f'{media_type}_grid_thw') is not None]
            if grid_thw:
                res[f'{media_type}_grid_thw'] = torch.concat(grid_thw)
        return res


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qwen2_vl, template_cls=Qwen2VLTemplate, placeholder_tokens=['<|image_pad|>', '<|video_pad|>']))

register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qvq,
        default_system=('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                        'Answer in the language of the question. You should think step-by-step.'),
        template_cls=Qwen2VLTemplate,
        placeholder_tokens=['<|image_pad|>', '<|video_pad|>']))


class Ovis1_6Template(Template):
    skip_prompt = False
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, [-200])
        added_tokens_len = 0
        pixel_values = []
        for i, idx in enumerate(idx_list):
            max_partition = get_env_args('max_partition', int, 9)
            raw_pixel_values, image_placeholders = self.model.visual_tokenizer.preprocess_image(
                images[i], max_partition=max_partition)
            input_ids = input_ids[:idx] + image_placeholders + input_ids[idx + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * len(image_placeholders) + labels[idx + 1:]
            pixel_values.append(raw_pixel_values)
            added_tokens_len += len(image_placeholders) - 1
        dtype = self.model.visual_tokenizer.dtype
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0).to(dtype)
        else:
            pixel_values = torch.zeros((1, 3, 384, 384), dtype=dtype)  # dummpy
        encoded.update({'input_ids': input_ids, 'labels': labels})
        encoded['pixel_values'] = [pixel_values]
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        padding_side = self.padding_side if self.is_training else 'left'
        _, inputs_embeds, labels, attention_mask = self.model.merge_multimodal(
            text_input_ids=inputs['input_ids'],
            text_attention_masks=torch.ones_like(inputs['input_ids']),  # not use, only compat
            text_labels=inputs.get('labels'),
            pixel_values=inputs['pixel_values'],
            left_padding=padding_side == 'left')

        return {'inputs_embeds': inputs_embeds, 'labels': labels, 'attention_mask': attention_mask}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        pixel_values = self.gather_list(batch, 'pixel_values')
        res = super()._data_collator(batch, padding_to=padding_to)
        res['pixel_values'] = pixel_values
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.ovis1_6,
        prefix=['<bos>'],
        prompt=['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
        chat_sep=['<end_of_turn>\n'],
        suffix=['<end_of_turn>'],
        system_prefix=['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'],
        template_cls=Ovis1_6Template,
    ))

register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.ovis1_6_llama3,
        default_system='You are a helpful and honest multimodal assistant.',
        template_cls=Ovis1_6Template,
    ))


@dataclass
class MarcoO1TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = """
你是一个经过良好训练的AI助手，你的名字是Marco-o1.由阿里国际数字商业集团的AI Business创造.
        \n## 重要！！！！！
当你回答问题时，你的思考应该在<Thought>内完成，<Output>内输出你的结果。
<Thought>应该尽可能是英文，但是有2个特例，一个是对原文中的引用，另一个是是数学应该使用markdown格式，<Output>内的输出需要遵循用户输入的语言。
        """


register_template(MarcoO1TemplateMeta(LLMTemplateType.marco_o1))
