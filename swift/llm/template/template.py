# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import re
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import torch
import transformers
from packaging import version
from transformers import PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.integrations import is_deepspeed_zero3_enabled

from modelscope import get_logger
from .base import Template, TEMPLATE_MAPPING
from .utils import (load_audio_qwen, load_batch, load_image, load_video_cogvlm2, load_video_internvl,
                    load_video_llava, load_video_minicpmv_mplug_owl3, load_video_qwen2,
                    transform_image, upper_bound, fetch_one)

logger = get_logger()

DEFAULT_SYSTEM = 'You are a helpful assistant.'
History = List[Union[Tuple[str, str], List[str]]]
Prompt = List[Union[str, List[int], List[str]]]
StopWords = Prompt
Context = Union[str, List[int]]


class TemplateType:
    # text-generation
    default_generation = 'default-generation'
    chatglm_generation = 'chatglm-generation'
    qwen_vl_generation = 'qwen-vl-generation'
    qwen_audio_generation = 'qwen-audio-generation'
    # chat
    default = 'default'
    qwen = 'qwen'
    qwen_vl = 'qwen-vl'
    qwen_audio = 'qwen-audio'
    qwen2_audio = 'qwen2-audio'
    qwen2_audio_generation = 'qwen2-audio-generation'
    qwen2_vl = 'qwen2-vl'
    modelscope_agent = 'modelscope-agent'
    baichuan = 'baichuan'
    chatglm2 = 'chatglm2'
    chatglm3 = 'chatglm3'
    chatglm4 = 'chatglm4'
    codegeex4 = 'codegeex4'
    llama = 'llama'  # llama2
    llama3 = 'llama3'
    reflection = 'reflection'
    longwriter_llama3 = 'longwriter-llama3'
    # llava-hf
    llava1_5 = 'llava1_5'
    llava_mistral = 'llava-mistral'
    llava_vicuna = 'llava-vicuna'
    llava_yi = 'llava-yi'
    llama3_llava_next_hf = 'llama-llava-next-hf'
    llava_next_llama3 = 'llava-next-llama3'
    llava_qwen_hf = 'llama-qwen-hf'
    llava_onevision_qwen = 'llava-onevision-qwen'
    # llava-video
    llava_next_video = 'llava-next-video'
    llava_next_video_yi = 'llava-next-video-yi'
    # lmms-lab:llava
    llama3_llava_next = 'llama3-llava-next'
    llava_qwen = 'llava-qwen'
    # xtuner:llava
    llava_llama_instruct = 'llava-llama-instruct'

    idefics3 = 'idefics3'
    mistral_nemo = 'mistral-nemo'
    openbuddy = 'openbuddy'
    openbuddy2 = 'openbuddy2'
    internlm = 'internlm'
    internlm2 = 'internlm2'
    internlm_xcomposer2 = 'internlm-xcomposer2'
    internlm_xcomposer2_4khd = 'internlm-xcomposer2-4khd'
    internlm_xcomposer2_5 = 'internlm-xcomposer2_5'
    internvl = 'internvl'
    internvl2 = 'internvl2'
    internvl_phi3 = 'internvl-phi3'
    internvl2_phi3 = 'internvl2-phi3'
    florence = 'florence'
    yi_coder = 'yi-coder'
    yi_vl = 'yi-vl'
    yuan = 'yuan'
    xverse = 'xverse'
    ziya = 'ziya'
    skywork = 'skywork'
    bluelm = 'bluelm'
    zephyr = 'zephyr'
    sus = 'sus'
    deepseek = 'deepseek'
    numina_math = 'numina-math'
    deepseek_coder = 'deepseek-coder'
    deepseek_vl = 'deepseek-vl'
    deepseek2 = 'deepseek2'
    deepseek2_5 = 'deepseek2_5'
    codefuse_codellama = 'codefuse-codellama'
    codefuse = 'codefuse'
    cogvlm = 'cogvlm'
    cogvlm2_video = 'cogvlm2-video'
    glm4v = 'glm4v'
    cogagent_chat = 'cogagent-chat'
    cogagent_instruct = 'cogagent-instruct'
    orion = 'orion'
    minicpm = 'minicpm'
    minicpm_v = 'minicpm-v'
    minicpm_v_v2_5 = 'minicpm-v-v2_5'
    minicpm_v_v2_6 = 'minicpm-v-v2_6'
    gemma = 'gemma'
    paligemma = 'paligemma'
    mplug_owl2 = 'mplug-owl2'
    mplug_owl3 = 'mplug_owl3'
    wizardlm2_awq = 'wizardlm2-awq'
    wizardlm2 = 'wizardlm2'
    atom = 'atom'
    phi3 = 'phi3'
    phi3_vl = 'phi3-vl'
    telechat = 'telechat'
    telechat_v2 = 'telechat-v2'
    dbrx = 'dbrx'
    mengzi = 'mengzi'
    c4ai = 'c4ai'
    chatml = 'chatml'
    # compatibility. (Deprecated)
    default_generation_bos = 'default-generation-bos'

    @classmethod
    def get_template_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__') or k == 'get_template_name_list':
                continue
            res.append(cls.__dict__[k])
        return res


def register_template(template_type: str, template: Template, *, exist_ok: bool = False, **kwargs) -> None:
    if not exist_ok and template_type in TEMPLATE_MAPPING:
        raise ValueError(f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.')
    template.template_type = template_type
    template_info = {'template': template, **kwargs}
    TEMPLATE_MAPPING[template_type] = template_info


register_template(
    TemplateType.default,
    Template([], ['### Human:\n{{QUERY}}\n\n### Assistant:\n'], ['\n\n'], [['eos_token_id']],
             DEFAULT_SYSTEM, ['{{SYSTEM}}\n\n'],
             auto_add_bos=True))


# You can set the query as '' to serve as a template for pre-training.
class DefaultGenerationTemplate(Template):

    def __init__(self):
        super().__init__([], ['{{QUERY}}'], None, [['eos_token_id']], auto_add_bos=True)


register_template(TemplateType.default_generation, DefaultGenerationTemplate(), is_generation=True)
register_template(
    TemplateType.default_generation_bos,
    Template([['bos_token_id']], ['{{QUERY}}'], None, [['eos_token_id']]),
    is_generation=True)


class ChatmlTemplateMixin:
    system = None

    def __init__(self, auto_add_bos: bool = True):
        Template.__init__(
            self, [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'], ['<|im_end|>\n'],
            ['<|im_end|>'],
            self.system, ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            auto_add_bos=auto_add_bos)


class ChatmlTemplate(ChatmlTemplateMixin, Template):
    pass


class QwenTemplateMixin(ChatmlTemplateMixin):
    system = DEFAULT_SYSTEM

    def __init__(self):
        super().__init__(auto_add_bos=False)


class QwenTemplate(QwenTemplateMixin, Template):
    pass


class _QwenVLTemplateMixin:
    load_medias = False

    def check_example(self, example):
        images = example.get('images') or []
        assert not images or isinstance(fetch_one(images), str), 'QwenVL only supports datasets with images paths!'

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type == 'image'
        images = example.get('images') or []
        image = images[index]
        assert isinstance(image, str)
        return [f'Picture {index + 1}:<img>{image}</img>\n']

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


register_template(TemplateType.qwen, QwenTemplate())


class QwenVLTemplate(_QwenVLTemplateMixin, QwenTemplate):
    pass


class QwenVLGenerationTemplate(_QwenVLTemplateMixin, DefaultGenerationTemplate):
    pass


register_template(TemplateType.qwen_vl, QwenVLTemplate())
register_template(TemplateType.qwen_vl_generation, QwenVLGenerationTemplate())

register_template(TemplateType.chatml, ChatmlTemplate())

register_template(
    TemplateType.modelscope_agent,
    Template([], [' \n\n<|user|>:{{QUERY}} \n\n<|assistant|>:'], [], [' \n\n</s>'], DEFAULT_SYSTEM,
             [' \n\n<|system|>:{{SYSTEM}}']))


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


register_template(TemplateType.qwen_audio, QwenAudioTemplate(), lazy_tokenize=True)
register_template(
    TemplateType.qwen_audio_generation, QwenAudioGenerationTemplate(), lazy_tokenize=True, is_generation=True)


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
        if input_features:
            res['input_features'] = torch.concat(input_features)
            feature_attention_mask = [b['feature_attention_mask'] for b in batch]
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


register_template(TemplateType.qwen2_audio, Qwen2AudioTemplate(), lazy_tokenize=True)


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


class Qwen2VLTemplate(QwenTemplate):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
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
                    images = load_batch(images, _process_image_qwen)
                    media_token = 151655
                    media_inputs = processor.image_processor(images=images, videos=None, return_tensors='pt')
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    videos = load_batch(videos, load_video_qwen2)
                    media_inputs = processor.image_processor(images=None, videos=videos, return_tensors='pt')
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = 151656
                idx_list = _findall(input_ids, media_token)
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
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        for media_type in ['image', 'video']:
            grid_thw = [b[f'{media_type}_grid_thw'] for b in batch if b.get(f'{media_type}_grid_thw') is not None]
            if grid_thw:
                res[f'{media_type}_grid_thw'] = torch.concat(grid_thw)
        return res


register_template(TemplateType.qwen2_vl, Qwen2VLTemplate(), lazy_tokenize=True)

register_template(
    TemplateType.qwen2_audio_generation, Qwen2AudioGenerationTemplate(), lazy_tokenize=True, is_generation=True)


class YiCoderTemplate(ChatmlTemplate):
    system = 'You are a helpful assistant.'


register_template(TemplateType.yi_coder, YiCoderTemplate())

yi_vl_default_system = (
    'This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. '
    "Read all the images carefully, and respond to the human's questions with informative, "
    'helpful, detailed and polite answers. '
    '这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。'
    '仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。')


class YiVLTemplate(Template):

    def replace_tag(self, media_type, index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        inputs.pop('loss_scale', None)
        from llava.mm_utils import expand2square
        # This processor should be put from the `model.vision_tower.image_processor`
        image_processor = self.tokenizer.image_processor
        images = example.get('images') or []
        for i, image in enumerate(images):
            background_color = tuple(int(x * 255) for x in image_processor.image_mean)
            image = expand2square(image, background_color)
            images[i] = image
        if images:
            image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
            inputs['images'] = image_tensor.to(kwargs['dtype'])
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        has_images = [(b == -200).sum() for b in res['input_ids']]
        assert all([
            h > 0 for h in has_images
        ]) or not any([h > 0
                       for h in has_images]), 'YIVL does not support mix-batch nlp dataset and multi-modal dataset'
        return res


class GLMTemplate(Template):

    def _init_template(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs) -> None:
        res = super()._init_template(tokenizer, *args, **kwargs)
        token_list = tokenizer.encode('')
        self.prefix.insert(0, token_list)
        if self.system_prefix is not None:
            self.system_prefix.insert(0, token_list)
        return res


class GLM4VTemplate(GLMTemplate):

    def __init__(self):
        super().__init__([], ['<|user|>\n{{QUERY}}<|assistant|>'], [], ['<|endoftext|>'], None,
                         ['<|system|>\n{{SYSTEM}}'])

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) <= 1

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-100]]

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, -100)
        if idx_list:
            idx = idx_list[0]
            image = example.get('images')[0]
            placeholder = '<|begin_of_image|><|endoftext|><|end_of_image|>'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            messages = example['messages']
            messages[0]['image'] = image
            inputs2: Dict[str, Any] = self.tokenizer.apply_chat_template(messages, return_dict=True)
            inputs['images'] = inputs2['images']
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        return res


register_template(TemplateType.glm4v, GLM4VTemplate(), infer_media_type='dialogue', lazy_tokenize=True, use_model=False)

register_template(
    TemplateType.yi_vl,
    YiVLTemplate([], [[8308], 'Human: {{QUERY}}\n', [8308], 'Assistant:'], ['\n'], ['\n', [8308]], yi_vl_default_system,
                 ['{{SYSTEM}}\n\n']),
    use_model=False,
    infer_media_type='round',
    lazy_tokenize=True)

register_template(TemplateType.baichuan, Template(['{{SYSTEM}}'], [[195], '{{QUERY}}', [196]], [], [['eos_token_id']]))

register_template(
    TemplateType.chatglm2,
    GLMTemplate(['{{SYSTEM}}'], ['[Round {{ROUND1}}]\n\n问：{{QUERY}}\n\n答：'], ['\n\n'], [['eos_token_id']]))

register_template(
    TemplateType.chatglm_generation, GLMTemplate([], ['{{QUERY}}'], None, [['eos_token_id']]), is_generation=True)

register_template(
    TemplateType.chatglm3,
    GLMTemplate([], ['<|user|>\n{{QUERY}}<|assistant|>\n'], [], ['<|user|>'], None, ['<|system|>\n{{SYSTEM}}']))

register_template(
    TemplateType.chatglm4,
    GLMTemplate([], ['<|user|>\n{{QUERY}}<|assistant|>\n'], [], ['<|user|>'],
                None, ['<|system|>\n{{SYSTEM}}'],
                tools_prompt='glm4',
                tool_prompt=['<|observation|>\n{{QUERY}}<|assistant|>\n']))

codegeex4_system = '你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。'

register_template(
    TemplateType.codegeex4,
    GLMTemplate([], ['<|user|>\n{{QUERY}}<|assistant|>\n'], [], ['<|endoftext|>'], codegeex4_system,
                ['<|system|>\n{{SYSTEM}}']))

register_template(
    TemplateType.deepseek,
    Template([['bos_token_id']], ['User: {{QUERY}}\n\nAssistant:'], [['eos_token_id']], [['eos_token_id']], None,
             [['bos_token_id'], '{{SYSTEM}}\n\n']))
register_template(
    TemplateType.numina_math,
    Template([['bos_token_id']], ['### Problem: {{QUERY}}\n### Solution: '], ['\n'], [['eos_token_id']], None,
             [['bos_token_id'], '{{SYSTEM}}']))
register_template(
    TemplateType.deepseek2,
    Template([[100000]], ['User: {{QUERY}}\n\nAssistant:'], [[100001]], [[100001]], None, [[100000], '{{SYSTEM}}\n\n']))
register_template(
    TemplateType.deepseek2_5,
    Template(['<｜begin▁of▁sentence｜>'], ['<｜User｜>{{QUERY}}<｜Assistant｜>'], ['<｜end_of_sentense｜>'],
             ['<｜end_of_sentense｜>'], None, ['<｜begin▁of▁sentence｜>{{SYSTEM}}']))

# ref: https://github.com/facebookresearch/llama/blob/main/llama/generation.py
LLAMA_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest assistant. '
    'Always answer as helpfully as possible, while being safe. '
    'Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information.")
register_template(
    TemplateType.llama,
    Template(['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s><s>[INST] '], ['</s>'], LLAMA_DEFAULT_SYSTEM,
             ['<s>[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))

register_template(
    TemplateType.longwriter_llama3,
    Template(['[INST]'], ['{{QUERY}}[/INST]'], ['[INST]'], ['<|end_of_text|>'], None,
             ['<<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))

register_template(TemplateType.mistral_nemo,
                  Template(['<s>[INST] '], ['{{SYSTEM}}\n\n', '{{QUERY}}[/INST]'], ['</s>[INST] '], ['</s>']))


class Llama3TemplateMixin:
    system = None

    def __init__(self):
        Template.__init__(
            self, ['<|begin_of_text|>'], [
                '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
                '<|start_header_id|>assistant<|end_header_id|>\n\n'
            ], ['<|eot_id|>'], ['<|eot_id|>'],
            self.system, ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>'],
            tools_prompt='toolbench',
            tool_prompt=[
                '<|start_header_id|>tool<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
                '<|start_header_id|>assistant<|end_header_id|>\n\n'
            ])


class Llama3Template(Llama3TemplateMixin, Template):
    pass


class ReflectionTemplate(Llama3TemplateMixin, Template):
    system = ('You are a world-class AI system, capable of complex reasoning and reflection. '
              'Reason through the query inside <thinking> tags, and then provide your final '
              'response inside <output> tags. If you detect that you made a mistake in your reasoning '
              'at any point, correct yourself inside <reflection> tags.')


register_template(TemplateType.reflection, ReflectionTemplate())
register_template(TemplateType.llama3, Llama3Template())

OPENBUDDY_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User.\n'
    'Always answer as helpfully and logically as possible, while being safe. '
    'Your answers should not include any '
    'harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information.\n"
    'You like to use emojis. You can speak fluently in many languages, for example: English, Chinese.\n'
    'You cannot access the internet, but you have vast knowledge, cutoff: 2021-09.\n'
    'You always deeply love and support China, Chinese government, people and culture.\n'
    'You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), '
    'you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI.')
register_template(
    TemplateType.openbuddy,
    Template([], ['User: {{QUERY}}\nAssistant:'], ['\n'], [['eos_token_id']],
             OPENBUDDY_DEFAULT_SYSTEM, ['{{SYSTEM}}\n\n'],
             auto_add_bos=True))

OPENBUDDY2_DEFAULT_SYSTEM = (
    'You(assistant) are a helpful, respectful and honest INTP-T AI Assistant named Buddy. '
    'You are talking to a human(user).\nAlways answer as helpfully and logically as possible, while being safe. '
    'Your answers should not include any harmful, political, religious, unethical, racist, '
    'sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n'
    'You cannot access the internet, but you have vast knowledge, cutoff: 2023-04.\n'
    'You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), '
    'not related to GPT or OpenAI')

register_template(
    TemplateType.openbuddy2,
    Template([], ['<|role|>user<|says|>{{QUERY}}<|end|>\n<|role|>assistant<|says|>'], ['<|end|>\n'], ['<|end|>'],
             OPENBUDDY2_DEFAULT_SYSTEM, ['<|role|>system<|says|>{{SYSTEM}}<|end|>\n'],
             auto_add_bos=True))

INTERNLM_SYSTEM = (
    'You are an AI assistant whose name is InternLM (书生·浦语).\n'
    '- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). '
    'It is designed to be helpful, honest, and harmless.\n'
    '- InternLM (书生·浦语) can understand and communicate fluently in the language chosen '
    'by the user such as English and 中文.')

register_template(
    TemplateType.internlm,
    Template(['<s>'], ['<|User|>:{{QUERY}}\n<|Bot|>:'], ['<eoa>\n'], ['<eoa>'], INTERNLM_SYSTEM,
             ['<s><|System|>:{{SYSTEM}}\n']))

_T = TypeVar('_T')

_log_set = set()  # log once


def get_env_args(args_name: str, type_func: Callable[[str], _T], default_value: Optional[_T]) -> Optional[_T]:
    args_name_upper = args_name.upper()
    value = os.getenv(args_name_upper)
    if value is None:
        value = default_value
        log_info = (f'Setting {args_name}: {default_value}. '
                    f'You can adjust this hyperparameter through the environment variable: `{args_name_upper}`.')
    else:
        value = type_func(value)
        log_info = f'Using environment variable `{args_name_upper}`, Setting {args_name}: {value}.'
    if log_info not in _log_set:
        _log_set.add(log_info)
        logger.info(log_info)
    return value


class Internlm2Template(ChatmlTemplate):
    system = INTERNLM_SYSTEM


register_template(TemplateType.internlm2, Internlm2Template())


def replace_img_tag(query: str,
                    history: History,
                    replace_token: str,
                    pattern=r'<img>(.+?)</img>') -> Tuple[str, History, List[str]]:
    images_path = []
    new_history = []
    for i, h in enumerate(history):
        if h[0] is None:
            new_history.append(h.copy())
        else:
            images_path += re.findall(pattern, h[0])
            new_history.append([re.sub(pattern, replace_token, h[0]), h[1]])
    if query is None:
        new_query = query  # pretrain dataset
    else:
        images_path += re.findall(pattern, query)
        new_query = re.sub(pattern, replace_token, query)
    return new_query, new_history, images_path


class InternLMXComposer2Template(Template):
    INTERNLM_XCOMPOSER_SYSTEM = (
        'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by '
        'Shanghai AI Laboratory (上海人工智能实验室). '
        'It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen '
        'by the user such as English and 中文.')
    image_placeholder = ['</s>']

    def __init__(self, version):
        prefix = ['<s>']
        prompt = ['[UNUSED_TOKEN_146]user\n{{QUERY}}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n']
        chat_sep = ['[UNUSED_TOKEN_145]\n']
        suffix = ['[UNUSED_TOKEN_145]']
        system_prefix = ['<s>[UNUSED_TOKEN_146]system\n{{SYSTEM}}[UNUSED_TOKEN_145]\n']
        super().__init__(prefix, prompt, chat_sep, suffix, self.INTERNLM_XCOMPOSER_SYSTEM, system_prefix)
        self.version = version

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []

        if self.version == 'v2.5':
            hd_num = 24
            if len(images) > 1:
                hd_num = 6
            hd_num = get_env_args('hd_num', int, hd_num)
            Image_transform = get_class_from_dynamic_module('ixc_utils.Image_transform', self.tokenizer.model_dir)
            images = [Image_transform(image, hd_num=hd_num) for image in images]
        elif self.version == 'v2-4khd':
            hd_num = 55
            hd_num = get_env_args('hd_num', int, hd_num)
            HD_transform = get_class_from_dynamic_module('ixc_utils.HD_transform', self.tokenizer.model_dir)
            images = [HD_transform(image, hd_num=hd_num) for image in images]
        # vis_processor comes from model.vis_processor
        images = [self.tokenizer.vis_processor(image).to(kwargs['dtype']) for image in images]
        inputs['_data'] = {'input_ids': inputs['input_ids'], 'labels': inputs['labels'], 'images': images}
        return inputs, {}

    def post_encode(self, model, data: Any) -> Dict[str, Any]:
        input_ids = data['input_ids']
        labels = data['labels']
        images = data['images']
        if len(images) > 0:  # ignore <s>
            input_ids = input_ids[1:]
            if labels is not None:
                labels = labels[1:]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        input_ids.append(2)  # add dummy </s>
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            labels.append(2)
        else:
            labels = []
        res_inputs_embeds = []
        res_labels = []
        wrap_im_mask = []
        pre_i, i, idx = 0, 0, 0
        device = model.device
        internlm2_model = model.model
        if not hasattr(internlm2_model, 'tok_embeddings'):
            internlm2_model = internlm2_model.model
        tok_embeddings = internlm2_model.tok_embeddings
        if len(images) > 0:
            images = torch.concat([model.img2emb(image[None])[0] for image in images], dim=0)
        while i < len(input_ids):
            if input_ids[i] == 2:  # replace_token
                res_input_ids = torch.tensor([1] + input_ids[pre_i:i], device=device)
                res_inputs_embeds.append(tok_embeddings(res_input_ids[None])[0])
                wrap_im_mask += [0] * len(res_input_ids)
                res_labels += [-100] + labels[pre_i:i]
                if len(images) > 0 and idx < images.shape[0]:
                    res_inputs_embeds.append(images[idx].to(device))
                    wrap_im_mask += [1] * images.shape[1]
                    res_labels += [-100] * images.shape[1]
                idx += 1
                i += 1
                pre_i = i
                continue
            i += 1
        if len(labels) == 0:
            res_labels = None
        res_inputs_embeds = torch.concat(res_inputs_embeds, dim=0)
        wrap_im_mask = torch.tensor(wrap_im_mask, dtype=torch.bool, device=device)[None]
        return {'inputs_embeds': res_inputs_embeds, 'im_mask': wrap_im_mask, 'labels': res_labels}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        if 'im_mask' in batch[0]:
            im_mask = [b['im_mask'][0] for b in batch]
            im_mask = self.pad_sequence(im_mask, 0, self.padding_side)
            res['im_mask'] = im_mask
        return res

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


register_template(
    TemplateType.internlm_xcomposer2, InternLMXComposer2Template(version='v2'), use_model=False, lazy_tokenize=True)


class InternLMXComposer2_5Template(InternLMXComposer2Template):
    INTERNLM_XCOMPOSER_SYSTEM = (
        'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model '
        'that is developed by Shanghai AI Laboratory (上海人工智能实验室). '
        'It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen '
        'by the user such as English and 中文.\n'
        '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively '
        'based on the provided image.')


register_template(
    TemplateType.internlm_xcomposer2_5,
    InternLMXComposer2_5Template(version='v2.5'),
    use_model=False,
    lazy_tokenize=True)

register_template(
    TemplateType.internlm_xcomposer2_4khd,
    InternLMXComposer2_5Template(version='v2-4khd'),
    use_model=False,
    lazy_tokenize=True)


class InternvlTemplate(Template):
    system = 'You are an AI assistant whose name is InternLM (书生·浦语).'
    num_image_token = 256

    def __init__(self):
        super().__init__([], ['<|im_start|>user\n{{QUERY}}<|im_end|><|im_start|>assistant\n'], ['<|im_end|>'],
                         ['<|im_end|>'],
                         self.system, ['<|im_start|>system\n{{SYSTEM}}<|im_end|>'],
                         auto_add_bos=True)

    def replace_tag(self, media_type, index, example) -> List[Context]:
        return ['<img>', [-100], '</img>\n']

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        idx_list = _findall(input_ids, -100)
        pixel_values = None
        images = example.get('images')
        if images:
            labels = inputs.get('labels')
            input_size = get_env_args('input_size', int, 448)
            max_num = get_env_args('max_num', int, 12)
            pixel_values_images = [transform_image(image, input_size, max_num) for image in images]
            pixel_values = torch.cat(pixel_values_images, dim=0).to(kwargs['dtype'])
            image_bs = pixel_values.shape[0]

            idx, idx2 = idx_list[0], idx_list[-1]  # remove [-100, -100]
            img_tokens: List[int] = self.tokenizer.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * image_bs
            input_ids = input_ids[:idx] + img_tokens + input_ids[idx2 + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * len(img_tokens) + labels[idx2 + 1:]
            inputs['input_ids'] = input_ids
            inputs['labels'] = labels
        inputs['_data'] = {'input_ids': torch.tensor(input_ids), 'pixel_values': pixel_values}
        inputs.pop('loss_scale', None)
        return inputs, {}

    def post_encode(self, model, data: Any) -> Dict[str, Any]:
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        input_ids = data['input_ids']
        inputs_embeds = embedding(input_ids[None])[0].to(device=device)
        pixel_values = data['pixel_values']
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device)
            vit_embeds = model.extract_feature(pixel_values).to(device=device)
            selected = (input_ids == self.tokenizer.encode('<IMG_CONTEXT>', add_special_tokens=False)[0])
            inputs_embeds[selected] = vit_embeds.reshape(-1, vit_embeds.shape[-1])
        elif is_deepspeed_zero3_enabled():
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), device=device, dtype=inputs_embeds.dtype)
            vit_embeds = model.extract_feature(dummy_pixel_values).to(device=device)
            inputs_embeds += vit_embeds.mean() * 0.
        return {'inputs_embeds': inputs_embeds}

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


def _replace_video2image(load_video_func, example, replace_tag) -> List[Context]:
    context_list = []
    video_index = example['video_index']
    video = example['videos'][video_index]
    images = example['images']
    image_index = example['image_index']
    new_images = load_video_func(video)
    example['images'] = images[:image_index] + new_images + images[image_index:]
    for i in range(len(new_images)):
        context_list += replace_tag(i)
    example['image_index'] += len(new_images)
    return context_list


class Internvl2Template(InternvlTemplate):
    video_segments = 8
    system = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'

    def replace_tag(self, media_type, index, example) -> List[Context]:
        image_context = super().replace_tag('image', index, example)
        if media_type == 'image':
            return image_context
        elif media_type == 'video':
            video_segments = get_env_args('video_segments', int, self.video_segments)
            load_video = partial(load_video_internvl, num_segments=video_segments)
            return _replace_video2image(load_video, example, lambda i: [f'Frame{i + 1}: '] + image_context)

    def replace_object(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            return [f'<ref>{object_["caption"]}</ref>']
        else:
            return ['<ref-object>']

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            if isinstance(object_['bbox'][0], list):
                all_objects = '<box> ['
                for sub_object in object_['bbox']:
                    all_objects += (f'[{sub_object[0]}, {sub_object[1]}, ' f'{sub_object[2]}, {sub_object[3]}],')
                all_objects = all_objects[:-1]
                all_objects += '] </box>'
                return [all_objects]
            else:
                return [
                    f'<box> [[{object_["bbox"][0]}, {object_["bbox"][1]}, '
                    f'{object_["bbox"][2]}, {object_["bbox"][3]}]] </box>'
                ]
        else:
            return ['<bbox>']

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super(InternvlTemplate, self)._encode(example, **kwargs)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        idx_list = _findall(input_ids, -100)
        labels = inputs.get('labels')
        images = example.get('images')
        if images:
            has_video = bool(example.get('videos'))
            input_size = get_env_args('input_size', int, 448)
            max_num = get_env_args('max_num', int, 1 if has_video else 12)
            pixel_values = [transform_image(image, input_size, max_num) for image in images]
            num_patches = [pv.shape[0] for pv in pixel_values]
            pixel_values = torch.cat(pixel_values).to(kwargs['dtype'])
        else:
            pixel_values = None
            num_patches = []
        assert len(num_patches) == len(
            idx_list), f'len(num_patches): {len(num_patches)}, len(idx_list): {len(idx_list)}'
        added_tokens_len = 0
        for idx, num_patch in zip(idx_list, num_patches):
            img_tokens: List[int] = self.tokenizer.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * num_patch
            input_ids = input_ids[:idx + added_tokens_len] + img_tokens + input_ids[idx + added_tokens_len + 1:]
            if labels is not None:
                labels = labels[:idx + added_tokens_len] + [-100] * len(img_tokens) + labels[idx + added_tokens_len
                                                                                             + 1:]
            added_tokens_len += len(img_tokens) - 1
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        inputs['_data'] = {'input_ids': torch.tensor(input_ids), 'pixel_values': pixel_values}
        inputs.pop('loss_scale', None)
        return inputs, {}


class InternvlPhi3TemplateMixin:

    def __init__(self):
        Template.__init__(
            self, [], ['<|user|>\n{{QUERY}}<|end|><|assistant|>\n'], ['<|end|>'], ['<|end|>'],
            getattr(self, 'system', None), ['<|system|>\n{{SYSTEM}}<|end|>'],
            auto_add_bos=True)
        self.padding_side = 'left'


class InternvlPhi3Template(InternvlPhi3TemplateMixin, InternvlTemplate):
    system = 'You are an AI assistant whose name is Phi-3.'


class Internvl2Phi3Template(InternvlPhi3TemplateMixin, Internvl2Template):
    pass


register_template(
    TemplateType.internvl, InternvlTemplate(), use_model=False, lazy_tokenize=True, infer_media_type='dialogue')

register_template(
    TemplateType.internvl_phi3, InternvlPhi3Template(), use_model=False, lazy_tokenize=True, infer_media_type='dialogue')

register_template(TemplateType.internvl2, Internvl2Template(), use_model=False, lazy_tokenize=True)

register_template(TemplateType.internvl2_phi3, Internvl2Phi3Template(), use_model=False, lazy_tokenize=True)


class FlorenceTemplate(Template):
    compute_per_round_loss = False
    output_prompt_answer = True

    def __init__(self):
        super().__init__(['<s>'], ['{{QUERY}}</s>'], None, ['</s>'])
        self.task_prompts_without_inputs = {
            '<OCR>': 'What is the text in the image?',
            '<OCR_WITH_REGION>': 'What is the text in the image, with regions?',
            '<CAPTION>': 'What does the image describe?',
            '<DETAILED_CAPTION>': 'Describe in detail what is shown in the image.',
            '<MORE_DETAILED_CAPTION>': 'Describe with a paragraph what is shown in the image.',
            '<OD>': 'Locate the objects with category name in the image.',
            '<DENSE_REGION_CAPTION>': 'Locate the objects in the image, with their descriptions.',
            '<REGION_PROPOSAL>': 'Locate the region proposals in the image.'
        }
        self.task_prompts_with_input = {
            '<CAPTION_TO_PHRASE_GROUNDING>': 'Locate the phrases in the caption: {input}',
            '<REFERRING_EXPRESSION_SEGMENTATION>': 'Locate {input} in the image with mask',
            '<REGION_TO_SEGMENTATION>': 'What is the polygon mask of region {input}',
            '<OPEN_VOCABULARY_DETECTION>': 'Locate {input} in the image.',
            '<REGION_TO_CATEGORY>': 'What is the region {input}?',
            '<REGION_TO_DESCRIPTION>': 'What does the region {input} describe?',
            '<REGION_TO_OCR>': 'What text is in the region {input}?',
        }

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) == 1, 'Florence series models only supports input with a single image.'

    def add_default_tags(self, example: Dict[str, Any]) -> None:
        return

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        object_ = example['objects'][index]
        if isinstance(object_['bbox'][0], list):
            all_objects = ''
            for sub_object in object_['bbox']:
                x1, y1, x2, y2 = sub_object
                all_objects += f'<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>,'
            return [all_objects[:-1]]
        else:
            x1, y1, x2, y2 = object_['bbox']
            return [f'<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>']

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        query = example['query']
        processor = self.tokenizer.processor
        example['query'] = processor._construct_prompts([query])[0]
        inputs, _ = super()._encode(example)
        input_ids = inputs['prompt_input_ids']
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []
        labels = inputs['answer_labels']
        if labels is not None:
            labels = [0] + labels
        pixel_values = processor.image_processor(images, return_tensors='pt')['pixel_values'].to(kwargs['dtype'])
        inputs = {
            'input_ids': input_ids,
            'labels': labels,
            '_data': {
                'input_ids': torch.tensor(input_ids)[None],
                'pixel_values': pixel_values,
            }
        }
        return inputs, {}

    def post_encode(self, model, data: Any) -> Dict[str, Any]:
        inputs_embeds = model.get_input_embeddings()(data['input_ids'])
        image_features = model._encode_image(data['pixel_values'])
        inputs_embeds, _ = model._merge_input_ids_with_image_features(image_features, inputs_embeds)
        return {'inputs_embeds': inputs_embeds[0]}

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids

    def post_process_generate_response(self, response, example):
        if isinstance(example['images'], list):
            example['images'] = example['images'][0]
        image = load_image(example['images'])
        return json.dumps(
            self.tokenizer.processor.post_process_generation(
                response, task=example['query'], image_size=(image.width, image.height)))


register_template(
    TemplateType.florence,
    FlorenceTemplate(),
    use_model=False,
    lazy_tokenize=True,
    infer_media_type='dialogue',
    stream=False)

register_template(TemplateType.xverse,
                  Template(['{{SYSTEM}}'], ['Human: {{QUERY}}\n\nAssistant: '], [['eos_token_id']], [['eos_token_id']]))
register_template(TemplateType.yuan, Template([], ['{{QUERY}}<sep>'], None, [['eos_token_id']]))
register_template(TemplateType.ziya,
                  Template([['bos_token_id'], '{{SYSTEM}}'], ['<human>:{{QUERY}}\n<bot>:'], ['\n'], [['eos_token_id']]))

register_template(TemplateType.skywork,
                  Template(['<s>{{SYSTEM}}'], ['</s><s>[USER]{{QUERY}}[SEP][BOT]'], None, ['[SEP]</s>']))

register_template(TemplateType.bluelm,
                  Template([['bos_token_id'], '{{SYSTEM}}'], ['[|Human|]:{{QUERY}}[|AI|]:'], [], [['eos_token_id']]))

register_template(
    TemplateType.codefuse_codellama,
    Template(['{{SYSTEM}}'], ['<|role_start|>human<|role_end|>{{QUERY}}<|role_start|>bot<|role_end|>'], [],
             [['eos_token_id']]))

register_template(
    TemplateType.codefuse,
    Template([], ['<s>human\n{{QUERY}}\n<s>bot\n'], [['eos_token_id'], '\n'], [['eos_token_id']], None,
             ['<s>system\n{{SYSTEM}}\n']))

register_template(
    TemplateType.deepseek_coder,
    Template(['{{SYSTEM}}'], ['### Instruction:\n{{QUERY}}\n### Response:\n'], ['\n<|EOT|>\n'], ['\n<|EOT|>'],
             ('You are an AI programming assistant, utilizing the Deepseek Coder model, '
              'developed by Deepseek Company, and you only answer questions related to computer science. '
              'For politically sensitive questions, security and privacy issues, '
              'and other non-computer science questions, you will refuse to answer\n')))


class LlavaHfTemplate(Template):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if version.parse(transformers.__version__) < version.parse('4.43.0'):
            self.padding_side = 'left'

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return ['<image>\n']

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        if images:
            image_processor = self.tokenizer.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(kwargs['dtype'])
            inputs['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                inputs['image_sizes'] = image_inputs['image_sizes']
        return inputs, {}


class Llava1_6Llama3Template(LlavaHfTemplate):
    default_system = 'You are a helpful language and vision assistant. ' \
                     'You are able to understand the visual content that the user provides, ' \
                     'and assist the user with a variety of tasks using natural language.'

    def __init__(self):
        super().__init__(['<|begin_of_text|>'], [
            '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n\n'
        ], ['<|eot_id|>'], ['<|eot_id|>'], None,
                         ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>'])

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs['pixel_values'].shape) == 5:  # (1, num_patch, 3, H/W, W/H)
            inputs['pixel_values'] = torch.squeeze(inputs['pixel_values'], dim=0)  # (num_patch, 3, H/W, W/H)
        return inputs, {}


register_template(TemplateType.llava_next_llama3, Llava1_6Llama3Template(), use_model=False, lazy_tokenize=True)


class LlavaVideoTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:

        if media_type == 'image':
            return ['<image>\n']
        assert media_type == 'video'
        media_file = example['videos'][index]
        if media_file.rsplit('.', 1)[-1] in {'jpg', 'png'}:
            return ['<image>\n']
        else:
            return ['<video>\n']

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []
        videos_path = example.get('videos') or []
        if len(videos_path) > 0:
            videos = load_batch(videos_path, load_video_llava)
            video_processor = self.tokenizer.processor.video_processor
            video_inputs = video_processor(videos, return_tensors='pt').to(kwargs['dtype'])
            inputs['pixel_values_videos'] = video_inputs['pixel_values_videos']
        if len(images) > 0:
            image_processor = self.tokenizer.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(kwargs['dtype'])
            inputs['pixel_values'] = image_inputs['pixel_values']
            inputs['image_sizes'] = image_inputs['image_sizes']
        return inputs, {}


register_template(
    TemplateType.llava_next_video,
    LlavaVideoTemplate(['<s>{{SYSTEM}} '], ['USER: {{QUERY}} ASSISTANT:'], [' '], ['</s>']),
    use_model=False,
    lazy_tokenize=True)

register_template(
    TemplateType.llava_next_video_yi,
    LlavaVideoTemplate(['{{SYSTEM}} '], ['USER: {{QUERY}} ASSISTANT:'], [' '], ['<|im_end|>']),
    use_model=False,
    infer_media_type='round',
    lazy_tokenize=True)


def align_image_inputs(input_ids: List[int], labels: List[int], new_input_ids,
                       image_token: int) -> Tuple[List[int], List[int]]:
    if isinstance(new_input_ids, torch.Tensor):
        new_input_ids = new_input_ids.tolist()

    # Find the tokens after the image_token in input_ids, and then align them.
    i, j = 0, 0
    while i < len(input_ids):
        x = input_ids[i]
        if x == image_token:
            assert i + 1 < len(input_ids), f'input_ids[-10:]: {input_ids[-10:]}'
            assert i - 1 >= 0, f'input_ids[:10]: {input_ids[:10]}'
            # [1, 2, 3(i-1), image_token(i), 4(i+1) ,5, 6]
            # [1, 2, 3(j_begin), a(j'), a, a, a, 4(j) ,5, 6]
            j_begin = j - 1
            for k in range(5):  # Increase robustness.
                if j_begin + k < len(new_input_ids) and new_input_ids[j_begin + k] == input_ids[i - 1]:
                    j_begin += k
                    break
                if j_begin - k >= 0 and new_input_ids[j_begin - k] == input_ids[i - 1]:
                    j_begin -= k
                    break
            else:
                raise ValueError(f'new_input_ids: {new_input_ids}, input_ids: {input_ids}')
            j_begin += 1
            while j < len(new_input_ids) and new_input_ids[j] != input_ids[i + 1]:
                j += 1
            input_ids = input_ids[:i] + new_input_ids[j_begin:j] + input_ids[i + 1:]
            if labels:
                labels = labels[:i] + [-100] * (j - j_begin) + labels[i + 1:]
            i += j - j_begin
        else:
            j += 1
        i += 1
    return input_ids, labels


class Idefics3Template(Template):

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []
        processor = self.tokenizer.processor
        prompt = self.tokenizer.decode(inputs['input_ids'])
        if images:
            image_inputs = processor(text=prompt, images=images, return_tensors='pt', add_special_tokens=False)
            image_token = 128257  # <image>
            inputs['input_ids'], inputs['labels'] = align_image_inputs(inputs['input_ids'], inputs['labels'],
                                                                       image_inputs['input_ids'][0], image_token)
            inputs['pixel_values'] = image_inputs['pixel_values']
        return inputs, {}


register_template(
    TemplateType.idefics3,
    Idefics3Template(['<|begin_of_text|>'], ['User:{{QUERY}}<end_of_utterance>\nAssistant:'], ['<end_of_utterance>\n'],
                     ['<end_of_utterance>'], None, ['System:{{SYSTEM}}<end_of_utterance>\n']),
    use_model=False,
    lazy_tokenize=True)


class Llava1_5Template(LlavaHfTemplate):

    def __init__(self):
        super().__init__(['<s>'], ['USER: {{QUERY}}\nASSISTANT:'], ['</s>'], ['</s>'])


register_template(TemplateType.llava1_5, Llava1_5Template(), use_model=False, lazy_tokenize=True)


class LLavaTemplate(Template):

    def __init__(self):
        # This template follows: https://github.com/haotian-liu/LLaVA/blob/main/llava/conversation.py#L350
        super().__init__(['<s>[INST] '], ['{{QUERY}} [/INST]'],
                         None, ['</s>'],
                         system_prefix=['<<SYS>>\n{{system}}\n<</SYS>>\n\n'])

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []
        image_sizes = [x.size for x in images]
        from llava.mm_utils import process_images
        if images:
            # image_processor comes from the model.vision_tower.image_processor
            # config comes from the model.config
            images_tensor = process_images(images, self.tokenizer.image_processor, self.tokenizer.config)
            inputs['images'] = images_tensor.to(kwargs['dtype']).squeeze(0)
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

    def __init__(self):
        super().__init__(['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s>'], ['</s>'],
                         system_prefix=['<<SYS>>\n{{system}}\n<</SYS>>\n\n'])


class Llava1_6VicunaTemplate(Llava1_6Template):
    system = ('A chat between a curious human and an artificial intelligence assistant. '
              "The assistant gives helpful, detailed, and polite answers to the human's questions.")

    def __init__(self):
        super().__init__(['<s>'], ['USER: {{QUERY}} ASSISTANT:'], ['</s>'], ['</s>'],
                         self.system,
                         system_prefix=['<s>{{SYSTEM}} '])


register_template(TemplateType.llava_mistral, Llava1_6MistralTemplate(), use_model=False, lazy_tokenize=True)

register_template(TemplateType.llava_vicuna, Llava1_6VicunaTemplate(), use_model=False, lazy_tokenize=True)


class LLava1_6YiTemplate(Llava1_6Template):

    def __init__(self):
        super().__init__([], ['<|im_start|>user\n{{QUERY}}<|im_end|><|im_start|>assistant\n'], ['<|im_end|>'],
                         ['<|im_end|>'],
                         system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>'])

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        return super().replace_tag(media_type, index, example)


register_template(TemplateType.llava_yi, LLava1_6YiTemplate(), use_model=False, lazy_tokenize=True)


class Llama3LlavaNextHfTemplate(Llama3TemplateMixin, Llava1_6Template):
    pass


register_template(TemplateType.llama3_llava_next_hf, Llama3LlavaNextHfTemplate(), use_model=False, lazy_tokenize=True)


class LlavaQwenHfTemplate(QwenTemplateMixin, Llava1_6Template):
    pass


register_template(TemplateType.llava_qwen_hf, LlavaQwenHfTemplate(), use_model=False, lazy_tokenize=True)


class LlavaOneVisonTemplate(QwenTemplateMixin, Llava1_6Template):
    system = None

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = Template._encode(self, example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, 151646)  # <image>
        processor = self.tokenizer.processor
        if images:
            image_processor = processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(kwargs['dtype'])
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


register_template(TemplateType.llava_onevision_qwen, LlavaOneVisonTemplate(), use_model=False, lazy_tokenize=True)


class LLavaLlamaTemplate(Llama3Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example):
        return ['<image>\n']

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        raw_image = example.get('images')
        if raw_image:
            pixel_values = self.tokenizer.processor.image_processor(raw_image, return_tensors='pt')['pixel_values']
            inputs['pixel_values'] = pixel_values.to(kwargs['dtype'])
        return inputs, {}


register_template(TemplateType.llava_llama_instruct, LLavaLlamaTemplate(), use_model=False, lazy_tokenize=True)


class PaliGemmaTemplate(Template):

    def __init__(self):
        super().__init__([], ['{{QUERY}}\n'], None, ['<eos>'])

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) <= 1

    def replace_tag(self, media_type, index, example) -> List[Context]:
        assert media_type == 'image'
        if self._is_vllm:
            self.prompt = ['{{QUERY}}']
            return []
        else:
            self.prompt = ['{{QUERY}}\n']
            return ['<image>' * self.tokenizer.processor.image_seq_length + '<bos>']

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        raw_image = example.get('images')
        processor = self.tokenizer.processor
        if inputs['labels'] is not None:
            n = upper_bound(0, len(inputs['labels']), lambda idx: inputs['labels'][idx] == -100)
            n2 = len(inputs['labels']) - n
            inputs['token_type_ids'] = [0] * n + [1] * n2
        else:
            inputs['token_type_ids'] = [0] * len(inputs['input_ids'])
        if raw_image:
            model_inputs = processor(text=example['query'], images=raw_image[0], return_tensors='pt')
            inputs['pixel_values'] = model_inputs['pixel_values']
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        token_type_ids = [torch.tensor(b['token_type_ids']) for b in batch]
        token_type_ids = self.pad_sequence(token_type_ids, 0, self.padding_side)
        res['token_type_ids'] = token_type_ids
        return res


register_template(
    TemplateType.paligemma, PaliGemmaTemplate(), infer_media_type='dialogue', lazy_tokenize=True, is_generation=True)


class Phi3Template(Template):

    def __init__(self):
        super().__init__([], ['<|user|>\n{{QUERY}}<|end|>\n<|assistant|>\n'], ['<|end|>\n'], ['<|end|>'],
                         None, ['<|system|>\n{{SYSTEM}}<|end|>\n'],
                         auto_add_bos=True)


register_template(TemplateType.phi3, Phi3Template())


class Phi3VisionTemplate(Phi3Template):
    image_placeholder = ['<|image|><s>\n']  # <|image|>\n

    def replace_tag(self, media_type, index, example) -> List[Context]:
        return super().replace_tag(media_type, index, example)

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        images = example.get('images') or []
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, 32044)  # '<|image|>'

        if len(images) > 0:
            processor = self.tokenizer.processor
            inputs.update(processor.image_processor(images, return_tensors='pt'))
            assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
            res_input_ids = []
            res_labels = []
            num_img_tokens = inputs.pop('num_img_tokens').tolist()
            idx_list.insert(0, -1)
            for i in range(len(idx_list) - 1):
                image_token_id = -i - 1
                res_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]] + [image_token_id] * num_img_tokens[i]
                if labels is not None:
                    res_labels += labels[idx_list[i] + 1:idx_list[i + 1]] + [-100] * num_img_tokens[i]
            res_input_ids += input_ids[idx_list[-1] + 1:]
            input_ids = res_input_ids
            if labels is not None:
                res_labels += labels[idx_list[-1] + 1:]
                labels = res_labels

        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs, {}


register_template(TemplateType.phi3_vl, Phi3VisionTemplate(), lazy_tokenize=True)


class Llama3LlavaNextTemplate(Llama3TemplateMixin, LLavaTemplate):
    system = 'You are a helpful language and vision assistant. ' \
             'You are able to understand the visual content that the user provides, ' \
             'and assist the user with a variety of tasks using natural language.'


register_template(TemplateType.llama3_llava_next, Llama3LlavaNextTemplate(), use_model=False, lazy_tokenize=True)


class LLavaQwenTemplate(QwenTemplateMixin, LLavaTemplate):
    pass


register_template(TemplateType.llava_qwen, LLavaQwenTemplate(), use_model=False, lazy_tokenize=True)


def _findall(token_list: List[int], sub_token_list: Union[int, List[int]]) -> List[int]:
    """Find the index of a token in the token_list."""
    if isinstance(sub_token_list, int):
        sub_token_list = [sub_token_list]
    res = []
    idx = -1
    try:
        while True:
            idx = token_list.index(sub_token_list[0], idx + 1)
            if len(sub_token_list) == 1 or sub_token_list == token_list[idx:idx + len(sub_token_list)]:
                res.append(idx)
    except ValueError:
        pass
    return res


class DeepseekVLTemplate(Template):
    DEEPSEEK_VL_SYSTEM = ('You are a helpful language and vision assistant. '
                          'You are able to understand the visual content that the user provides, '
                          'and assist the user with a variety of tasks using natural language.')

    image_placeholder = ['<image_placeholder>']

    def __init__(self):
        super().__init__(['<｜begin▁of▁sentence｜>{{SYSTEM}}\n\n'], ['User: {{QUERY}}\n\nAssistant:'],
                         ['<｜end▁of▁sentence｜>'], ['<｜end▁of▁sentence｜>'], self.DEEPSEEK_VL_SYSTEM)

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        processor = self.tokenizer.processor
        input_ids, labels = inputs['input_ids'], inputs['labels']
        idx_list = _findall(input_ids, processor.image_id)  # '<image_placeholder>'
        new_input_ids, new_labels = [], []
        lo = 0
        for hi in idx_list:
            new_input_ids += input_ids[lo:hi]
            if labels is not None:
                new_labels += labels[lo:hi]
            new_input_ids += [processor.image_id] * processor.num_image_tokens
            new_labels += [-100] * processor.num_image_tokens
            lo = hi + 1
        new_input_ids += input_ids[lo:]
        if labels is not None:
            new_labels += labels[lo:]
        else:
            new_labels = None
        from deepseek_vl.models.processing_vlm import VLChatProcessorOutput
        images_outputs = processor.image_processor(images, return_tensors='pt')
        output = VLChatProcessorOutput(
            sft_format=None,
            input_ids=torch.tensor(new_input_ids),
            pixel_values=images_outputs.pixel_values,
            num_image_tokens=torch.tensor([processor.num_image_tokens] * len(idx_list)))
        batched_output = dict(processor.batchify([output]))
        batched_output['pixel_values'] = batched_output['pixel_values'].to(dtype=kwargs['dtype'])
        inputs = {'input_ids': new_input_ids, 'labels': new_labels, '_data': batched_output}
        return inputs, {}

    def post_encode(self, model, data: Any) -> Dict[str, Any]:
        inputs_embeds = model.prepare_inputs_embeds(**data)[0]
        return {'inputs_embeds': inputs_embeds}

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


register_template(TemplateType.deepseek_vl, DeepseekVLTemplate(), use_model=False, lazy_tokenize=True)

register_template(
    TemplateType.zephyr,
    Template([], ['<|user|>\n{{QUERY}}</s>\n<|assistant|>\n'], ['</s>\n'], ['</s>'], None,
             ['<|system|>\n{{SYSTEM}}</s>\n']))

register_template(
    TemplateType.sus,
    Template(['{{SYSTEM}}'], ['### Human: {{QUERY}}\n\n### Assistant: '], ['<|endoftext|>'], ['<|endoftext|>']))

register_template(TemplateType.orion,
                  Template(['<s>{{SYSTEM}}'], ['Human: {{QUERY}}\n\nAssistant: </s>'], ['</s>'], ['</s>']))


class CogTemplate(Template):

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) <= 1

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        return []

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        image = example.get('images') or []
        inputs.pop('loss_scale', None)
        inputs2 = self.tokenizer.build_conversation_input_ids(
            self.tokenizer, query=example['query'], history=example.get('history'), images=image)
        image_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        inputs['token_type_ids'] = [0] + [1] * image_token_len + [0] * len(input_ids[1:])
        inputs['input_ids'] = input_ids[:1] + [self.tokenizer.pad_token_id] * image_token_len + input_ids[1:]
        if labels is not None:
            inputs['labels'] = labels[:1] + [-100] * image_token_len + labels[1:]
        if len(image) > 0:
            inputs['images'] = [[img.to(dtype=kwargs['dtype'])] for img in inputs2['images']]
            if 'cross_images' in inputs2:
                # is cogagent
                inputs['cross_images'] = [[cross_img.to(dtype=kwargs['dtype'])] for cross_img in inputs2['cross_images']]
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        keys = ['images', 'cross_images']
        for key in keys:
            if key in batch[0]:
                res[key] = [b[key][0] for b in batch]
        token_type_ids = [torch.tensor(b['token_type_ids']) for b in batch]
        token_type_ids = self.pad_sequence(token_type_ids, 0, self.padding_side)
        res['token_type_ids'] = token_type_ids
        return res


register_template(
    TemplateType.cogagent_chat,
    CogTemplate(['<s>'], [' [INST] {{QUERY}} [/INST] '], [], ['</s>']),
    use_model=False,
    infer_media_type='dialogue',
    lazy_tokenize=True)

register_template(
    TemplateType.cogagent_instruct,
    CogTemplate(['<s>'], ['<EOI>Question: {{QUERY}} Answer:'], None, ['</s>']),
    use_model=False,
    infer_media_type='dialogue',
    lazy_tokenize=True)

register_template(
    TemplateType.cogvlm,
    CogTemplate([['bos_token_id']], ['Question: {{QUERY}} Answer:'], ['\n'], [['eos_token_id']]),
    use_model=False,
    infer_media_type='dialogue',
    lazy_tokenize=True)


class Cog2VideoTemplate(CogTemplate):

    def check_example(self, example):
        videos = example.get('videos') or []
        assert len(videos) <= 1

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super(CogTemplate, self)._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        videos_path = example.get('videos') or []
        video = load_batch(videos_path, load_video_cogvlm2)
        inputs.pop('loss_scale', None)
        inputs2 = self.tokenizer.build_conversation_input_ids(
            self.tokenizer,
            query=example['query'],
            history=example.get('history'),
            images=video,
            template_version='chat')
        video_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        inputs['token_type_ids'] = [0] + [1] * video_token_len + [0] * len(input_ids[1:])
        inputs['input_ids'] = input_ids[:1] + [self.tokenizer.pad_token_id] * video_token_len + input_ids[1:]
        if labels is not None:
            inputs['labels'] = labels[:1] + [-100] * video_token_len + labels[1:]
        if len(video) > 0:
            inputs['images'] = [[img.to(dtype=kwargs['dtype'])] for img in inputs2['images']]
        return inputs, {}


register_template(
    TemplateType.cogvlm2_video,
    Cog2VideoTemplate([['bos_token_id']], ['Question: {{QUERY}} Answer:'], ['\n'], [['eos_token_id']]),
    use_model=False,
    infer_media_type='dialogue',
    lazy_tokenize=True,
    media_type='video')

register_template(TemplateType.minicpm, Template(['<s>{{SYSTEM}}'], ['<用户>{{QUERY}}<AI>'], [], ['</s>']))


def _remove_idx(arr: List[int], idx_list: List[int]) -> List[int]:
    res = []
    idx_set = set(idx_list)
    for i, x in enumerate(arr):
        if i not in idx_set:
            res.append(x)
    return res


class MiniCPMVTemplate(Template):
    is_v2_5 = False

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        return [[-100]]

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) == 1

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example['images']
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, -100)
        idx = idx_list[0]
        tgt_sizes = None
        slice_mode = getattr(self.tokenizer.config, 'slice_mode', False)
        if slice_mode:
            if self.is_v2_5:
                image_processor = self.tokenizer.processor.image_processor
                image_inputs = image_processor(images, return_tensors='pt').to(kwargs['dtype'])
                placeholder = image_processor.get_slice_image_placeholder(image_inputs.image_sizes[0][0])
                pixel_values = image_inputs['pixel_values']
                tgt_sizes = image_inputs['tgt_sizes']
            else:
                # Comes from model.get_slice_image_placeholder and model.transform
                images, placeholder = self.tokenizer.get_slice_image_placeholder(images[0], self.tokenizer)
                pixel_values = [[self.tokenizer.transform(img) for img in images]]
            placeholder += '\n'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            input_tensor_ids = torch.tensor(input_ids)
            image_start_idx = torch.where(input_tensor_ids == self.tokenizer.im_start_id)[0]
            image_start_idx += 1
            image_end_idx = torch.where(input_tensor_ids == self.tokenizer.im_end_id)[0]
            valid_image_nums = max(len(image_start_idx), len(image_end_idx))
            image_bound = [
                torch.hstack(
                    [image_start_idx[:valid_image_nums].unsqueeze(-1), image_end_idx[:valid_image_nums].unsqueeze(-1)])
            ]
        else:
            placeholder = '<image>' + '<unk>' * self.tokenizer.config.query_num + '</image>\n'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            image_bound = [torch.tensor([[idx, idx + self.tokenizer.config.query_num]])]
            pixel_values = [[self.tokenizer.transform(images[0])]]
        inputs = {
            'input_ids': input_ids,
            'labels': labels,
            '_data': {
                'input_ids': torch.tensor(input_ids)[None],
                'image_bound': image_bound,
                'pixel_values': pixel_values,
                'tgt_sizes': tgt_sizes
            }
        }
        return inputs, {}

    def post_encode(self, model, data: Any) -> Dict[str, Any]:
        inputs_embeds, _ = model.get_vllm_embedding(data)
        return {'inputs_embeds': inputs_embeds[0]}

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


class MiniCPMV2_6Template(QwenTemplateMixin, MiniCPMVTemplate):

    def check_example(self, example):
        pass

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type in {'image', 'video'}
        max_num_frames = get_env_args('max_num_frames', int, 64)
        load_video = partial(load_video_minicpmv_mplug_owl3, max_num_frames=max_num_frames)
        image_context = super().replace_tag('image', index, example)
        if media_type == 'image':
            return image_context
        elif media_type == 'video':
            return _replace_video2image(load_video, example, lambda i: image_context)

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = Template._encode(self, example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        use_video = bool(example.get('videos'))
        is_plain_text = not images and not use_video
        use_image_id = True
        max_slice_nums = None

        if use_video:
            use_image_id = False
            max_slice_nums = 1  # or 2

        max_slice_nums = get_env_args('max_slice_nums', int, max_slice_nums)
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, -100)
        idx_list.insert(0, -1)

        image_processor = self.tokenizer.processor.image_processor
        image_inputs = image_processor([images], return_tensors='pt',
                                       max_slice_nums=max_slice_nums).to(kwargs['dtype'])

        res_input_ids = []
        res_labels = []
        for i in range(len(idx_list) - 1):
            placeholder = image_processor.get_slice_image_placeholder(
                image_inputs.image_sizes[0][i], image_idx=i, max_slice_nums=max_slice_nums, use_image_id=use_image_id)
            placeholder += '\n'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            res_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]] + placeholder_id
            if labels is not None:
                res_labels += labels[idx_list[i] + 1:idx_list[i + 1]] + [-100] * len(placeholder_id)
        res_input_ids += input_ids[idx_list[-1] + 1:]
        input_ids = res_input_ids
        if labels is not None:
            res_labels += labels[idx_list[-1] + 1:]
            labels = res_labels
        if not is_plain_text:
            input_tensor_ids = torch.tensor(input_ids)
            unk_token = self.tokenizer.encode('<unk>', add_special_tokens=False)[0]
            indices = (input_tensor_ids == unk_token).nonzero(as_tuple=True)[0].tolist()

            ranges = []
            start = indices[0]
            for i in range(1, len(indices)):
                if indices[i] != indices[i - 1] + 1:
                    ranges.append([start, indices[i - 1] + 1])
                    start = indices[i]
            ranges.append([start, indices[-1] + 1])
            image_bound = [torch.tensor(ranges)]
        else:
            image_bound = [[]]

        inputs = {
            'input_ids': input_ids,
            'labels': labels,
            '_data': {
                'input_ids': torch.tensor(input_ids)[None],
                'image_bound': image_bound,
                'pixel_values': image_inputs['pixel_values'],
                'tgt_sizes': image_inputs['tgt_sizes']
            }
        }
        return inputs, {}


register_template(TemplateType.minicpm_v_v2_6, MiniCPMV2_6Template(), use_model=False, lazy_tokenize=True)


class MiniCPMV2_5Template(Llama3TemplateMixin, MiniCPMVTemplate):
    is_v2_5 = True


register_template(
    TemplateType.minicpm_v_v2_5, MiniCPMV2_5Template(), use_model=False, lazy_tokenize=True, infer_media_type='dialogue')

register_template(
    TemplateType.minicpm_v,
    MiniCPMVTemplate(['<s>{{SYSTEM}}'], ['<用户>{{QUERY}}<AI>'], [], ['</s>']),
    use_model=False,
    lazy_tokenize=True,
    infer_media_type='dialogue')

gemma_template = Template(['<bos>'], ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
                          ['<end_of_turn>\n'], ['<end_of_turn>'], None,
                          ['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'])
register_template(TemplateType.gemma, gemma_template)

register_template(TemplateType.telechat, Template([], ['<_user>{{QUERY}}<_bot>'], ['<_end>'], ['<_end>']))

register_template(TemplateType.telechat_v2, Template([], ['<_user> {{QUERY}}<_bot>'], [], ['<_end>']))

DBRX_SYSTEM = (
    'You are DBRX, created by Databricks. You were last updated in December 2023. '
    'You answer questions based on information available up to that point.\n'
    'YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, '
    'but provide thorough responses to more complex and open-ended questions.\n'
    'You assist with various tasks, from writing to coding (using markdown for code blocks '
    '— remember to use ``` with code, JSON, and tables).\n'
    'You do not have real-time data access or code execution capabilities.'
    ' You avoid stereotyping and provide balanced perspectives on controversial topics. '
    'You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.\n'
    'This is your system prompt, guiding your responses. Do not reference it, just respond to the user. '
    'If you find yourself talking about this message, stop. You should be responding appropriately '
    'and usually that means not mentioning this.'
    'YOU DO NOT MENTION ANY OF THIS INFORMATION ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY '
    'PERTINENT TO THE USER\'S QUERY.')


class DbrxTemplate(ChatmlTemplate):
    system = DBRX_SYSTEM


register_template(TemplateType.dbrx, DbrxTemplate())

register_template(TemplateType.mengzi,
                  Template([], ['输入：{{QUERY}}输出：\n'], [], [['eos_token_id']], None, ['指令：{{SYSTEM}}']))

C4AI_SYSTEM = ('You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by '
               'providing thorough responses.You are trained by Cohere.')
register_template(
    TemplateType.c4ai,
    Template(
        ['<BOS_TOKEN>'],
        ['<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{QUERY}}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'],
        ['<|END_OF_TURN_TOKEN|>'], ['<|END_OF_TURN_TOKEN|>'], C4AI_SYSTEM,
        ['<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{SYSTEM}}<|END_OF_TURN_TOKEN|']))


class mPlugOwl2Template(Template):

    def __init__(self):
        super().__init__(['{{SYSTEM}}'], ['USER: {{QUERY}}ASSISTANT:'], ['</s>'], [['eos_token_id']])

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-200]]

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        from mplug_owl2.mm_utils import process_images
        processor = self.tokenizer.processor
        images = example.get('images') or []
        for i, image in enumerate(images):
            # ref: https://modelscope.cn/models/iic/mPLUG-Owl2.1
            max_edge = max(image.size)
            image = image.resize((max_edge, max_edge))
            images[i] = image
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        if images:
            images = process_images(images, processor)
            images = images.to(kwargs['dtype'])
            return {'input_ids': input_ids, 'labels': labels, 'images': images}, {}
        else:
            return {'input_ids': input_ids, 'labels': labels}, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        return res


register_template(
    TemplateType.mplug_owl2, mPlugOwl2Template(), infer_media_type='round', use_model=False, lazy_tokenize=True)


class mPlugOwl3Template(QwenTemplateMixin, Template):
    system = None

    def _get_image_token_list(self, cut_shape):
        processor = self.tokenizer.processor
        text = processor.image_processor.cut_prompt_template(img_token='<|image|>', h=cut_shape[0], w=cut_shape[1])
        text_list = text.split('<|image|>')
        if text_list[-1] == '':
            text_list.pop()
        res_text_list = []
        for text in text_list:
            res_text_list += [text, '<|image|>']
        token_list = self._encode_context_list(res_text_list)[0]
        return token_list

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type in {'image', 'video'}
        max_num_frames = get_env_args('max_num_frames', int, 16)
        load_video = partial(load_video_minicpmv_mplug_owl3, max_num_frames=max_num_frames)
        if media_type == 'image':
            return [[-100], '\n']
        elif media_type == 'video':
            return _replace_video2image(load_video, example, lambda i: [[-100]]) + ['\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example['images']
        videos = example['videos']
        cut_enable = not videos
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, -100)
        processor = self.tokenizer.processor
        if images:
            image_inputs = processor.image_processor(images, cut_enable=cut_enable, return_tensors='pt')
            added_tokens_len = 0
            cut_shapes = image_inputs['cut_shape'] or [None] * len(idx_list)
            image_token_list = self.tokenizer.encode('<|image|>', add_special_tokens=False)
            for idx, cut_shape in zip(idx_list, cut_shapes):
                if cut_shape:
                    token_list = self._get_image_token_list(cut_shape)
                else:
                    token_list = image_token_list
                input_ids = input_ids[:idx + added_tokens_len] + token_list + input_ids[added_tokens_len + idx + 1:]
                if labels:
                    labels = labels[:idx + added_tokens_len] + [-100] * len(token_list) + labels[added_tokens_len + idx
                                                                                                 + 1:]
                added_tokens_len += len(token_list) - 1
            image_token_idx = torch.tensor(_findall(input_ids, image_token_list))[None]
            _range = torch.arange(len(input_ids))[:, None]
            matrix = (_range > image_token_idx).sum(dim=1)
            media_offset = torch.stack([torch.zeros(matrix.shape[0], dtype=torch.long), matrix], dim=-1)[None]
            inputs['_data'] = {'pixel_values': image_inputs['pixel_values']}
            inputs['media_offset'] = media_offset
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        image_embeds = model.forward_image(data['pixel_values'])
        return {'image_embeds': image_embeds}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        image_embeds = [b['image_embeds'] for b in batch if 'image_embeds' in b]
        if image_embeds:
            res['image_embeds'] = torch.concat(image_embeds)
        media_offset = [b['media_offset'] for b in batch if 'media_offset' in b]
        if media_offset:
            res['media_offset'] = torch.concat(media_offset)
        return res


register_template(TemplateType.mplug_owl3, mPlugOwl3Template(), use_model=False, lazy_tokenize=True)

register_template(TemplateType.wizardlm2_awq,
                  Template(['{{SYSTEM}}'], ['User:\n{{QUERY}}\n\nAssistant:\n'], ['\n\n'], ['</s>']))

_wizardlm2_system = ('A chat between a curious user and an artificial intelligence assistant. '
                     'The assistant gives helpful, detailed, and polite answers to the user\'s questions. ')
register_template(TemplateType.wizardlm2,
                  Template(['{{SYSTEM}}'], ['USER: {{QUERY}} ASSISTANT:'], ['</s>'], ['</s>'], _wizardlm2_system))

register_template(TemplateType.atom,
                  Template(['{{SYSTEM}}'], ['<s>Human: {{QUERY}}\n</s><s>Assistant: '], ['</s>'], ['</s>']))


class RLHFTemplateMixin:

    def encode(self: Template,
               example: Dict[str, Any],
               streaming: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        template_encode = self._old_encode
        inputs = {}
        tokenizer_kwargs = {}
        chosen_example, rejected_example = example, example.copy()
        rejected_example['response'] = example['rejected_response']
        if streaming:
            chosen_inputs, chosen_tokenizer_kwargs = template_encode(chosen_example), {}
            rejected_inputs, rejected_tokenizer_kwargs = template_encode(rejected_example), {}
        else:
            chosen_inputs, chosen_tokenizer_kwargs = template_encode(chosen_example)
            rejected_inputs, rejected_tokenizer_kwargs = template_encode(rejected_example)

        for suffix, res in zip(['inputs', 'tokenizer_kwargs'], [inputs, tokenizer_kwargs]):
            for prefix in ['chosen', 'rejected']:
                data = locals()[f'{prefix}_{suffix}']
                for k, v in data.items():
                    res[f'{prefix}_{k}'] = v
        return inputs, tokenizer_kwargs

    def data_collator(self: Template, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        _data_collator = self._old_data_collator
        new_batch = []
        for prefix in ['chosen_', 'rejected_']:
            for inputs in batch:
                new_inputs = {}
                for k, v in inputs.items():
                    if k.startswith(prefix):
                        new_k = k[len(prefix):]
                        new_inputs[new_k] = inputs[k]
                if len(new_inputs) > 0:
                    new_batch.append(new_inputs)
        assert len(new_batch) in {0, len(batch) * 2}, f'new_batch: {new_batch}'
        return _data_collator(new_batch or batch, padding_to)


class KTOTemplateMixin:

    def encode(self: Template,
               example: Dict[str, Any],
               streaming: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = self._old_encode(example, streaming)
        if len(inputs) > 0:
            inputs['label'] = example['label']
        return inputs, tokenizer_kwargs

    def data_collator(self: Template, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = {}
        for prefix in ['', 'KL_']:
            new_batch = []
            for b in batch:
                new_batch.append({'input_ids': b[f'{prefix}input_ids'], 'labels': b[f'{prefix}labels']})
            for k, v in self._old_data_collator(new_batch, padding_to).items():
                res[f'{prefix}completion_{k}'] = v
        res['label'] = [b['label'] for b in batch]
        return res
