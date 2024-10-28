from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import transformers
from packaging import version

from ..base import Template
from ..constant import TemplateType
from ..register import register_template
from ..utils import Context, findall
from ..vision_utils import load_video_llava
from .llama import Llama3TemplateMeta
from .qwen import QwenTemplateMeta


class LlavaHfTemplate(Template):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if version.parse(transformers.__version__) < version.parse('4.43.0'):
            self.padding_side = 'left'

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return ['<image>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        if images:
            image_processor = self.tokenizer.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model.dtype)
            inputs['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                inputs['image_sizes'] = image_inputs['image_sizes']
        return inputs, {}


class Llava1_6Llama3Template(LlavaHfTemplate):
    default_system = 'You are a helpful language and vision assistant. ' \
                     'You are able to understand the visual content that the user provides, ' \
                     'and assist the user with a variety of tasks using natural language.'

    def __init__(self, template_type: str):
        super().__init__(template_type, ['<|begin_of_text|>'], [
            '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n\n'
        ], ['<|eot_id|>'], ['<|eot_id|>'], None,
                         ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>'])

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs['pixel_values'].shape) == 5:  # (1, num_patch, 3, H/W, W/H)
            inputs['pixel_values'] = torch.squeeze(inputs['pixel_values'], dim=0)  # (num_patch, 3, H/W, W/H)
        return inputs, {}


register_template(Llava1_6Llama3Template(TemplateType.llava_next_llama3), use_model=True, lazy_tokenize=True)


class LlavaVideoTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:

        if media_type == 'image':
            return ['<image>\n']
        assert media_type == 'video'
        media_file = example['videos'][index]
        if media_file.rsplit('.', 1)[-1] in {'jpg', 'png'}:
            return ['<image>\n']
        else:
            example['videos'][index] = load_video_llava(example['videos'][index])
            return ['<video>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []
        videos_path = example.get('videos') or []
        if len(videos_path) > 0:
            videos = load_batch(videos_path, load_video_llava)
            video_processor = self.tokenizer.processor.video_processor
            video_inputs = video_processor(videos, return_tensors='pt').to(self.model.dtype)
            inputs['pixel_values_videos'] = video_inputs['pixel_values_videos']
        if len(images) > 0:
            image_processor = self.tokenizer.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model.dtype)
            inputs['pixel_values'] = image_inputs['pixel_values']
            inputs['image_sizes'] = image_inputs['image_sizes']
        return inputs, {}


register_template(
    LlavaVideoTemplate(TemplateType.llava_next_video, ['<s>{{SYSTEM}} '], ['USER: {{QUERY}} ASSISTANT:'], [' '],
                       ['</s>']),
    use_model=True,
    lazy_tokenize=True)

register_template(
    LlavaVideoTemplate(TemplateType.llava_next_video_yi, ['{{SYSTEM}} '], ['USER: {{QUERY}} ASSISTANT:'], [' '],
                       ['<|im_end|>']),
    use_model=True,
    infer_media_type='round',
    lazy_tokenize=True)


class Llava1_5Template(LlavaHfTemplate):

    def __init__(self, template_type: str):
        super().__init__(template_type, ['<s>'], ['USER: {{QUERY}}\nASSISTANT:'], ['</s>'], ['</s>'])


register_template(Llava1_5Template(TemplateType.llava1_5), use_model=True, lazy_tokenize=True)


class LLavaTemplate(Template):

    def __init__(self):
        # This template follows: https://github.com/haotian-liu/LLaVA/blob/main/llava/conversation.py#L350
        super().__init__(['<s>[INST] '], ['{{QUERY}} [/INST]'],
                         None, ['</s>'],
                         system_prefix=['<<SYS>>\n{{system}}\n<</SYS>>\n\n'])

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []
        image_sizes = [x.size for x in images]
        from llava.mm_utils import process_images
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        if images:
            images_tensor = process_images(images, image_processor, self.model.config)
            inputs['images'] = images_tensor.to(model.dtype).squeeze(0)
            inputs['image_sizes'] = image_sizes
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = images
            res['image_sizes'] = sum([b['image_sizes'] for b in batch if 'image_sizes' in b], start=[])
        has_images = [(b == -200).sum() for b in res['input_ids']]
        assert all([
            h > 0 for h in has_images
        ]) or not any([h > 0
                       for h in has_images]), 'Llava does not support mix-batch nlp dataset and multi-modal dataset'
        return res

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


class Llava1_6Template(LlavaHfTemplate):

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        for b in batch:
            pixel_values = b.get('pixel_values')
            if pixel_values is not None:
                b['pixel_values'] = pixel_values.squeeze(0)  # 5d -> 4d
        res = super().data_collator(batch, padding_to)
        return res


class Llava1_6MistralTemplate(Llava1_6Template):

    def __init__(self, template_type: str):
        super().__init__(
            template_type, ['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s>'], ['</s>'],
            system_prefix=['<<SYS>>\n{{system}}\n<</SYS>>\n\n'])


class Llava1_6VicunaTemplate(Llava1_6Template):
    system = ('A chat between a curious human and an artificial intelligence assistant. '
              "The assistant gives helpful, detailed, and polite answers to the human's questions.")

    def __init__(self, template_type: str):
        super().__init__(
            template_type, ['<s>'], ['USER: {{QUERY}} ASSISTANT:'], ['</s>'], ['</s>'],
            self.system,
            system_prefix=['<s>{{SYSTEM}} '])


register_template(Llava1_6MistralTemplate(TemplateType.llava_mistral), use_model=True, lazy_tokenize=True)

register_template(Llava1_6VicunaTemplate(TemplateType.llava_vicuna), use_model=True, lazy_tokenize=True)


class LLava1_6YiTemplate(Llava1_6Template):

    def __init__(self, template_type: str):
        super().__init__(
            template_type, [], ['<|im_start|>user\n{{QUERY}}<|im_end|><|im_start|>assistant\n'], ['<|im_end|>'],
            ['<|im_end|>'],
            system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>'])

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        if self._is_vllm:
            return [[64000], '\n']
        else:
            return super().replace_tag(media_type, index, example)


register_template(LLava1_6YiTemplate(TemplateType.llava_yi), use_model=True, lazy_tokenize=True)


class Llama3LlavaNextHfTemplate(Llama3TemplateMixin, Llava1_6Template):
    pass


register_template(Llama3LlavaNextHfTemplate(TemplateType.llama3_llava_next_hf), use_model=True, lazy_tokenize=True)


class LlavaQwenHfTemplate(QwenTemplateMixin, Llava1_6Template):
    pass


register_template(LlavaQwenHfTemplate(TemplateType.llava_qwen_hf), use_model=True, lazy_tokenize=True)


class LlavaOneVisonTemplate(QwenTemplateMixin, Llava1_6Template):
    system = None

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = Template._encode(self, example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = findall(input_ids, 151646)  # <image>
        processor = self.tokenizer.processor
        if images:
            image_processor = processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model.dtype)
            height, width = image_inputs['pixel_values'][0].shape[-2:]
            added_tokens_len = 0
            for idx, pixel_v, image_size in zip(idx_list, image_inputs['pixel_values'], image_inputs['image_sizes']):
                orig_height, orig_width = image_size
                num_image_tokens = processor._get_number_of_features(orig_height, orig_width, height, width)
                input_ids = input_ids[:added_tokens_len
                                      + idx] + [151646] * num_image_tokens + input_ids[added_tokens_len + idx + 1:]
                if labels is not None:
                    labels = labels[:added_tokens_len + idx] + [-100] * num_image_tokens + labels[added_tokens_len + idx
                                                                                                  + 1:]
                added_tokens_len += num_image_tokens - 1
            inputs['input_ids'] = input_ids
            inputs['labels'] = labels
            inputs['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                inputs['image_sizes'] = image_inputs['image_sizes']
        return inputs, {}


register_template(LlavaOneVisonTemplate(TemplateType.llava_onevision_qwen), use_model=True, lazy_tokenize=True)


class LLavaLlamaTemplate(Llama3Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example):
        return ['<image>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        raw_image = example.get('images')
        if raw_image:
            pixel_values = self.tokenizer.processor.image_processor(raw_image, return_tensors='pt')['pixel_values']
            inputs['pixel_values'] = pixel_values.to(self.model.dtype)
        return inputs, {}


register_template(LLavaLlamaTemplate(TemplateType.llava_llama_instruct), use_model=True, lazy_tokenize=True)
