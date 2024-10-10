# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import json
import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.utils import get_logger, upper_bound
from .base import Template
from .constant import TemplateType
from .llama import Llama3Template, Llama3TemplateMixin
from .qwen import DEFAULT_SYSTEM, ChatmlTemplate, QwenTemplate, QwenTemplateMixin
from .register import register_template
from .utils import Context, align_image_inputs, findall
from .vision_utils import (load_batch, load_image, load_video_cogvlm2, load_video_internvl,
                           load_video_minicpmv_mplug_owl3, transform_image)

logger = get_logger()

register_template(
    TemplateType.default,
    Template([], ['### Human:\n{{QUERY}}\n\n### Assistant:\n'], ['\n\n'], [['eos_token_id']],
             DEFAULT_SYSTEM, ['{{SYSTEM}}\n\n'],
             auto_add_bos=True))


class GOTImageEvalProcessor:

    def __init__(self, image_size=384, mean=None, std=None):
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, item):
        return self.transform(item)


class GOT_OCR2Template(QwenTemplate):
    system = '        You should follow the instructions carefully and explain your answers in detail.'

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        # OCR:
        # OCR with format:
        assert media_type == 'image'
        return ['<img>' + '<imgpad>' * 256 + '</img>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example['images']
        image_processor_high = GOTImageEvalProcessor(image_size=1024)
        for i, image in enumerate(images):
            images[i] = image_processor_high(image)[None].to(self.model.dtype)
        if images:
            inputs['images'] = images
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = _gather_list(batch, 'images')
        if images:
            res['images'] = images
        return res


register_template(TemplateType.got_ocr2, GOT_OCR2Template(), lazy_tokenize=True, use_model=True)

register_template(
    TemplateType.modelscope_agent,
    Template([], [' \n\n<|user|>:{{QUERY}} \n\n<|assistant|>:'], [], [' \n\n</s>'], DEFAULT_SYSTEM,
             [' \n\n<|system|>:{{SYSTEM}}']))


def _gather_list(batch: List[Dict[str, Any]], attr_name: str) -> Optional[List[Any]]:
    # List[Tensor] ->  List[Tensor]
    res = []
    for b in batch:
        if b.get(attr_name) is not None:
            res += b.pop(attr_name)
    return res


class PixtralTemplate(Template):

    def __init__(self):
        super().__init__(['<s>{{SYSTEM}}'], ['[INST]{{QUERY}}[/INST]'], ['</s>'], ['</s>'], None)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        return ['[IMG]']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        processor = self.tokenizer.processor
        images = example['images']
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = findall(input_ids, 10)
        if idx_list:
            image_inputs = processor.image_processor(images, patch_size=processor.patch_size, return_tensors='pt')
            inputs['pixel_values'] = image_inputs['pixel_values'][0]
            image_sizes = image_inputs['image_sizes'][0]
            added_tokens_len = 0
            for idx, image_size in zip(idx_list, image_sizes):
                height, width = image_size
                num_height_tokens = height // processor.patch_size
                num_width_tokens = width // processor.patch_size
                replace_tokens = [processor.image_token * num_width_tokens + processor.image_break_token] * (
                    num_height_tokens - 1)
                replace_tokens += [processor.image_token * num_width_tokens + processor.image_end_token]
                # Flatten list
                replace_str = ''.join(replace_tokens)
                img_tokens: List[int] = self.tokenizer.encode(replace_str, add_special_tokens=False)
                input_ids = input_ids[:idx + added_tokens_len] + img_tokens + input_ids[idx + added_tokens_len + 1:]
                if labels is not None:
                    labels = labels[:idx + added_tokens_len] + [-100] * len(img_tokens) + labels[idx + added_tokens_len
                                                                                                 + 1:]
                added_tokens_len += len(img_tokens) - 1
            inputs['input_ids'] = input_ids
            inputs['labels'] = labels

        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        pixel_values = _gather_list(batch, 'pixel_values')
        res = super().data_collator(batch, padding_to)
        if pixel_values:
            res['pixel_values'] = pixel_values
        return res


register_template(TemplateType.pixtral, PixtralTemplate(), lazy_tokenize=True)


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

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        inputs.pop('loss_scale', None)
        from llava.mm_utils import expand2square
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        images = example.get('images') or []
        for i, image in enumerate(images):
            background_color = tuple(int(x * 255) for x in image_processor.image_mean)
            image = expand2square(image, background_color)
            images[i] = image
        if images:
            image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
            inputs['images'] = image_tensor.to(model.dtype)
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


register_template(
    TemplateType.yi_vl,
    YiVLTemplate([], [[8308], 'Human: {{QUERY}}\n', [8308], 'Assistant:'], ['\n'], ['\n', [8308]], yi_vl_default_system,
                 ['{{SYSTEM}}\n\n']),
    use_model=True,
    infer_media_type='round',
    lazy_tokenize=True)

register_template(TemplateType.baichuan, Template(['{{SYSTEM}}'], [[195], '{{QUERY}}', [196]], [], [['eos_token_id']]))

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

register_template(TemplateType.mistral_nemo,
                  Template(['<s>[INST] '], ['{{SYSTEM}}\n\n', '{{QUERY}}[/INST]'], ['</s>[INST] '], ['</s>']))


class ReflectionTemplate(Llama3TemplateMixin, Template):
    system = ('You are a world-class AI system, capable of complex reasoning and reflection. '
              'Reason through the query inside <thinking> tags, and then provide your final '
              'response inside <output> tags. If you detect that you made a mistake in your reasoning '
              'at any point, correct yourself inside <reflection> tags.')


register_template(TemplateType.reflection, ReflectionTemplate())


class Llama3_1OmniTemplate(Llama3Template):
    system = ('You are a helpful language and speech assistant. '
              'You are able to understand the speech content that the user provides, '
              'and assist the user with a variety of tasks using natural language.')

    def replace_tag(self, media_type, index, example) -> List[Context]:
        assert media_type == 'audio'
        return [[-200]]

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import whisper
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        audios = example['audios']
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        inputs['_data'] = {'input_ids': torch.tensor(input_ids)[None]}
        if labels is not None:
            inputs['_data']['labels'] = torch.tensor(labels)[None]
        if audios:
            audios = load_batch(audios, whisper.load_audio)
            n_mels = get_env_args('n_mels', int, 128)
            for i, audio in enumerate(audios):
                audio = whisper.pad_or_trim(audio)
                audios[i] = whisper.log_mel_spectrogram(audio, n_mels=n_mels).permute(1, 0)
            audios = torch.stack(audios)
            inputs['_data'].update({'speech': audios, 'speech_lengths': torch.tensor([[audios.shape[1]]])})

        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        speech = data.get('speech')
        input_ids = data['input_ids']
        labels = data.get('labels')
        if speech is not None:
            speech_lengths = data['speech_lengths']
            speech = speech.to(model.dtype)
            inputs_embeds, labels = model.prepare_inputs_labels_for_speech_and_text(input_ids, None, None, None, labels,
                                                                                    speech, speech_lengths)[4:]
        else:
            inputs_embeds = model.get_model().embed_tokens(input_ids)
        res = {'inputs_embeds': inputs_embeds[0]}
        if labels is not None:
            res['labels'] = labels[0]
        return res


register_template(TemplateType.llama3_1_omni, Llama3_1OmniTemplate(), lazy_tokenize=True)

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


class Internlm2Template(ChatmlTemplate):
    system = INTERNLM_SYSTEM


register_template(TemplateType.internlm2, Internlm2Template())


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

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        dtype = self.model.dtype
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
        images = [self.model.vis_processor(image).to(dtype) for image in images]
        inputs['_data'] = {'input_ids': inputs['input_ids'], 'labels': inputs['labels'], 'images': images}
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
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
    TemplateType.internlm_xcomposer2, InternLMXComposer2Template(version='v2'), use_model=True, lazy_tokenize=True)


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
    use_model=True,
    lazy_tokenize=True)

register_template(
    TemplateType.internlm_xcomposer2_4khd,
    InternLMXComposer2_5Template(version='v2-4khd'),
    use_model=True,
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
        if self._is_vllm:
            image_context = ['<img><image></img>\n']
        else:
            image_context = ['<img>', [-100], '</img>\n']
        return image_context

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        idx_list = findall(input_ids, -100)
        pixel_values = None
        images = example.get('images')
        if images:
            labels = inputs.get('labels')
            input_size = get_env_args('input_size', int, 448)
            max_num = get_env_args('max_num', int, 12)
            pixel_values_images = [transform_image(image, input_size, max_num) for image in images]
            pixel_values = torch.cat(pixel_values_images, dim=0).to(self.model.dtype)
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

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
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
        elif is_deepspeed_enabled():
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), device=device, dtype=inputs_embeds.dtype)
            vit_embeds = model.extract_feature(dummy_pixel_values).to(device=device)
            inputs_embeds += vit_embeds.mean() * 0.
        return {'inputs_embeds': inputs_embeds}

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


def _replace_video2image(load_video_func, example, replace_tag: Callable) -> List[Context]:
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

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super(InternvlTemplate, self)._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        idx_list = findall(input_ids, -100)
        labels = inputs.get('labels')
        images = example.get('images')
        if images:
            has_video = bool(example.get('videos'))
            input_size = get_env_args('input_size', int, 448)
            max_num = get_env_args('max_num', int, 1 if has_video else 12)
            pixel_values = [transform_image(image, input_size, max_num) for image in images]
            num_patches = [pv.shape[0] for pv in pixel_values]
            pixel_values = torch.cat(pixel_values).to(self.model.dtype)
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
    TemplateType.internvl, InternvlTemplate(), use_model=True, lazy_tokenize=True, infer_media_type='dialogue')

register_template(
    TemplateType.internvl_phi3, InternvlPhi3Template(), use_model=True, lazy_tokenize=True, infer_media_type='dialogue')

register_template(TemplateType.internvl2, Internvl2Template(), use_model=True, lazy_tokenize=True)

register_template(TemplateType.internvl2_phi3, Internvl2Phi3Template(), use_model=True, lazy_tokenize=True)


class FlorenceTemplate(Template):
    loss_scale = 'last_round'
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

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        pixel_values = processor.image_processor(images, return_tensors='pt')['pixel_values'].to(self.model.dtype)
        inputs = {
            'input_ids': input_ids,
            'labels': labels,
            '_data': {
                'input_ids': torch.tensor(input_ids)[None],
                'pixel_values': pixel_values,
            }
        }
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
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
    use_model=True,
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


class Idefics3Template(Template):

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    use_model=True,
    lazy_tokenize=True)


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

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        if self._is_vllm:
            return [f'<|image_{index + 1}|>\n']  # <|image_1|>\n
        else:
            return super().replace_tag(media_type, index, example)

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        images = example.get('images') or []
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = findall(input_ids, 32044)  # '<|image|>'

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


register_template(TemplateType.llama3_llava_next, Llama3LlavaNextTemplate(), use_model=True, lazy_tokenize=True)


class LLavaQwenTemplate(QwenTemplateMixin, LLavaTemplate):
    pass


register_template(TemplateType.llava_qwen, LLavaQwenTemplate(), use_model=True, lazy_tokenize=True)


class DeepseekVLTemplate(Template):
    DEEPSEEK_VL_SYSTEM = ('You are a helpful language and vision assistant. '
                          'You are able to understand the visual content that the user provides, '
                          'and assist the user with a variety of tasks using natural language.')

    image_placeholder = ['<image_placeholder>']

    def __init__(self):
        super().__init__(['<｜begin▁of▁sentence｜>{{SYSTEM}}\n\n'], ['User: {{QUERY}}\n\nAssistant:'],
                         ['<｜end▁of▁sentence｜>'], ['<｜end▁of▁sentence｜>'], self.DEEPSEEK_VL_SYSTEM)

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        processor = self.tokenizer.processor
        input_ids, labels = inputs['input_ids'], inputs['labels']
        idx_list = findall(input_ids, processor.image_id)  # '<image_placeholder>'
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
        batched_output['pixel_values'] = batched_output['pixel_values'].to(dtype=self.model.dtype)
        inputs = {'input_ids': new_input_ids, 'labels': new_labels, '_data': batched_output}
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        inputs_embeds = model.prepare_inputs_embeds(**data)[0]
        return {'inputs_embeds': inputs_embeds}

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


register_template(TemplateType.deepseek_vl, DeepseekVLTemplate(), use_model=True, lazy_tokenize=True)

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

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        image = example.get('images') or []
        inputs.pop('loss_scale', None)
        model = self.model
        inputs2 = model.build_conversation_input_ids(
            self.tokenizer, query=example['query'], history=example.get('history'), images=image)
        image_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        inputs['token_type_ids'] = [0] + [1] * image_token_len + [0] * len(input_ids[1:])
        inputs['input_ids'] = input_ids[:1] + [self.tokenizer.pad_token_id] * image_token_len + input_ids[1:]
        if labels is not None:
            inputs['labels'] = labels[:1] + [-100] * image_token_len + labels[1:]
        if len(image) > 0:
            dtype = model.dtype
            inputs['images'] = [[img.to(dtype=dtype)] for img in inputs2['images']]
            if 'cross_images' in inputs2:
                # is cogagent
                inputs['cross_images'] = [[cross_img.to(dtype=dtype)] for cross_img in inputs2['cross_images']]
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
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)

register_template(
    TemplateType.cogagent_instruct,
    CogTemplate(['<s>'], ['<EOI>Question: {{QUERY}} Answer:'], None, ['</s>']),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)

register_template(
    TemplateType.cogvlm,
    CogTemplate([['bos_token_id']], ['Question: {{QUERY}} Answer:'], ['\n'], [['eos_token_id']]),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)


class Cog2VideoTemplate(CogTemplate):

    def check_example(self, example):
        videos = example.get('videos') or []
        assert len(videos) <= 1

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super(CogTemplate, self)._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        videos_path = example.get('videos') or []
        video = load_batch(videos_path, load_video_cogvlm2)
        inputs.pop('loss_scale', None)
        model = self.model
        inputs2 = model.build_conversation_input_ids(
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
            dtype = model.dtype
            inputs['images'] = [[img.to(dtype=dtype)] for img in inputs2['images']]
        return inputs, {}


register_template(
    TemplateType.cogvlm2_video,
    Cog2VideoTemplate([['bos_token_id']], ['Question: {{QUERY}} Answer:'], ['\n'], [['eos_token_id']]),
    use_model=True,
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
        if self._is_vllm:
            return ['(<image>./</image>)\n']
        else:
            return [[-100]]

    def check_example(self, example):
        images = example.get('images') or []
        if not self._is_vllm and not self._is_lmdeploy:
            assert len(images) == 1

    async def prepare_lmdeploy_inputs(self, inputs: Dict[str, Any]) -> None:
        images = inputs.pop('images', None) or []
        if len(images) == 0:
            return
        input_ids = inputs['input_ids']
        idx_list = findall(input_ids, -100)
        idx_list.insert(0, -1)
        new_input_ids = []
        features = []
        for i in range(len(idx_list) - 1):
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            context_list = ['<image>', [-100], '</image>']
            feat = [x.squeeze() for x in images[i]['embeddings'].split(1)]
            grid = images[i].get('grid')
            if len(feat) > 1 and grid is not None:
                context_list.append('<slice>')
                for j in range(grid[1]):
                    if j > 0:
                        context_list.append('\n')
                    for _ in range(grid[0]):
                        context_list += ['<image>', [-100], '</image>']
                context_list.append('</slice>\n')
            new_input_ids += self._encode_context_list(context_list)[0]
            features += feat
        new_input_ids += input_ids[idx_list[-1] + 1:]
        inputs['input_ids'] = new_input_ids
        inputs['images'] = features
        await super().prepare_lmdeploy_inputs(inputs)

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example['images']
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = findall(input_ids, -100)
        idx = idx_list[0]
        config = self.model.config
        tgt_sizes = None
        slice_mode = getattr(config, 'slice_mode', False)
        if slice_mode:
            if self.is_v2_5:
                image_processor = self.tokenizer.processor.image_processor
                image_inputs = image_processor(images, return_tensors='pt').to(self.model.dtype)
                placeholder = image_processor.get_slice_image_placeholder(image_inputs.image_sizes[0][0])
                pixel_values = image_inputs['pixel_values']
                tgt_sizes = image_inputs['tgt_sizes']
            else:
                images, placeholder = self.model.get_slice_image_placeholder(images[0], self.tokenizer)
                pixel_values = [[self.model.transform(img) for img in images]]
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
            placeholder = '<image>' + '<unk>' * config.query_num + '</image>\n'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            image_bound = [torch.tensor([[idx, idx + config.query_num]])]
            pixel_values = [[self.model.transform(images[0])]]
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

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
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

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        idx_list = findall(input_ids, -100)
        idx_list.insert(0, -1)

        image_processor = self.tokenizer.processor.image_processor
        image_inputs = image_processor([images], return_tensors='pt',
                                       max_slice_nums=max_slice_nums).to(self.model.dtype)

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


register_template(TemplateType.minicpm_v_v2_6, MiniCPMV2_6Template(), use_model=True, lazy_tokenize=True)


class MiniCPMV2_5Template(Llama3TemplateMixin, MiniCPMVTemplate):
    is_v2_5 = True


register_template(
    TemplateType.minicpm_v_v2_5, MiniCPMV2_5Template(), use_model=True, lazy_tokenize=True, infer_media_type='dialogue')

register_template(
    TemplateType.minicpm_v,
    MiniCPMVTemplate(['<s>{{SYSTEM}}'], ['<用户>{{QUERY}}<AI>'], [], ['</s>']),
    use_model=True,
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

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
            images = images.to(self.model.dtype)
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
    TemplateType.mplug_owl2, mPlugOwl2Template(), infer_media_type='round', use_model=True, lazy_tokenize=True)


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
        idx_list = findall(input_ids, -100)
        processor = self.tokenizer.processor
        inputs = {'_data': {}}
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
            image_token_idx = torch.tensor(findall(input_ids, image_token_list))[None]
            _range = torch.arange(len(input_ids))[:, None]
            matrix = (_range > image_token_idx).sum(dim=1)
            media_offset = torch.stack([torch.zeros(matrix.shape[0], dtype=torch.long), matrix], dim=-1)[None]
            inputs['_data'].update({
                'pixel_values': image_inputs['pixel_values'],
                'media_offset': media_offset,
            })
        inputs['_data']['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        if 'pixel_values' in data:
            pixel_values = data.pop('pixel_values')
            data['image_embeds'] = model.forward_image(pixel_values)
        return data

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        image_embeds = [b['image_embeds'] for b in batch if 'image_embeds' in b]
        if image_embeds:
            res['image_embeds'] = torch.concat(image_embeds)
        media_offset = []
        cusum_offset = 0

        for bi, b in enumerate(batch):
            if 'media_offset' in b:
                max_sequence_length = res['input_ids'].shape[1]
                curr_media_offset = b['media_offset']
                if curr_media_offset.shape[1] < max_sequence_length:
                    padding = curr_media_offset[:, -1:, :].expand(curr_media_offset.shape[0],
                                                                  max_sequence_length - curr_media_offset.shape[1],
                                                                  curr_media_offset.shape[2])
                    curr_media_offset = torch.concat([curr_media_offset, padding], dim=1)
                media_offset.append(curr_media_offset + cusum_offset)
                cusum_offset += image_embeds[bi].shape[0]

        # media_offset = [b['media_offset'] for b in batch if 'media_offset' in b]

        if media_offset:
            res['media_offset'] = torch.concat(media_offset)
        return res


register_template(TemplateType.mplug_owl3, mPlugOwl3Template(), use_model=True, lazy_tokenize=True)

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

        if len(chosen_inputs) == 0 or len(rejected_inputs) == 0:
            return {}, {}
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
