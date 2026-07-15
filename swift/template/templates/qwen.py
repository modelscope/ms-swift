# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import json
import numpy as np
import os
import shutil
import torch
import torch.nn.functional as F
import transformers
from dataclasses import dataclass, field
from functools import partial
from packaging import version
from PIL import Image
from torch import nn
from transformers.integrations import is_deepspeed_zero3_enabled
from typing import Any, Dict, List, Literal, Optional

from swift.utils import get_env_args, get_packed_seq_params, is_deepspeed_enabled, to_float_dtype
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..template_meta import TemplateMeta
from ..utils import Context, Word, findall
from ..vision_utils import load_audio, load_batch, load_video_ovis2, load_video_ovis2_5
from .llama import Llama3TemplateMeta
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta


@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    agent_template: str = 'hermes'


@dataclass
class Qwen2_5TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


@dataclass
class Qwen2_5MathTemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'Please reason step by step, and put your final answer within \\boxed{}.'


qwq_preview_system = ('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                      'You should think step-by-step.')

register_template(QwenTemplateMeta(LLMTemplateType.qwen))
register_template(Qwen2_5TemplateMeta(LLMTemplateType.qwen2_5))
register_template(QwenTemplateMeta(LLMTemplateType.qwq_preview, default_system=qwq_preview_system))

register_template(
    QwenTemplateMeta(LLMTemplateType.qwq, default_system=None, is_thinking=True, thinking_prefix='<think>\n'))


@dataclass
class Qwen3MixedTemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = None
    non_thinking_prefix: str = '<think>\n\n</think>\n\n'


register_template(Qwen3MixedTemplateMeta(LLMTemplateType.qwen3, is_thinking=True))

QWEN3_GUARD_TEMPLATE = (
    '<|im_start|>user\n'
    "# Task:\n1) Evaluate whether THE LAST USER's query belongs to safe or unsafe content based on our "
    'safety policy.\n2) Identify the category of unsafe content found.\n\n<BEGIN SAFETY POLICY>\n'
    "**Safety: Safe**: The USER's query does not contain any prohibited content or content that could "
    "lead to prohibited content.\n**Safety: Unsafe**: The USER's query contains prohibited content.\n"
    "**Safety: Controversial**: The USER's query does not explicitly contain prohibited content, but "
    'its intent, context, or potential responses could be misinterpreted, misused, or exploited to '
    'generate unsafe content under certain conditions.\n<END SAFETY POLICY>\n\n<BEGIN UNSAFE CONTENT '
    'CATEGORIES>\nViolent.\nNon-violent Illegal Acts.\nSexual Content or Sexual Acts.\nPII.\nSuicide '
    '& Self-Harm.\nUnethical Acts.\nPolitically Sensitive Topics.\nCopyright Violation.\nJailbreak.\n'
    '<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\nUSER: {{QUERY}}\n\n<END CONVERSATION>'
    "\n\nProvide your safety assessment for ONLY THE LAST **USER's query** in the above conversation:\n"
    " - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.\n - "
    "The second line should start with 'Categories:' followed by a list of any unsafe content "
    "categories, separated by commas. If the content is safe, use 'Categories: None'."
    '<|im_end|>\n<|im_start|>assistant\n')

register_template(Qwen3MixedTemplateMeta(
    LLMTemplateType.qwen3_guard,
    prompt=[QWEN3_GUARD_TEMPLATE],
))

register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwen3_thinking, default_system=None, is_thinking=True, thinking_prefix='<think>\n'))

register_template(QwenTemplateMeta(LLMTemplateType.qwen3_nothinking, default_system=None))

register_template(QwenTemplateMeta(LLMTemplateType.qwen3_coder, default_system=None, agent_template='qwen3_coder'))


class Qwen3EmbTemplate(Template):

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        if inputs.system is not None:
            inputs.messages[0]['content'] = inputs.system + ' ' + inputs.messages[0]['content']
            inputs.system = None
        if len(inputs.messages) % 2 == 1 and inputs.messages[-1]['role'] != 'assistant':
            inputs.messages.append({'role': 'assistant', 'content': ''})
        return inputs


register_template(
    TemplateMeta(
        LLMTemplateType.qwen3_emb,
        template_cls=Qwen3EmbTemplate,
        suffix=['<|endoftext|>'],
        prefix=[],
        chat_sep=[],
        prompt=['{{QUERY}}']))


class Qwen3RerankerTemplate(Template):
    instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        if inputs.system is not None:
            instruction = inputs.system
            inputs.system = None
        else:
            instruction = self.instruction
        query = inputs.messages[0]['content']
        document = inputs.messages[1]['content']
        user_message = '<Instruct>: ' + instruction + '\n' + '<Query>: ' + query + '\n' + '<Document>: ' + document
        inputs.messages = [{'role': 'user', 'content': user_message}]
        return inputs

    def prepare_engine_kwargs(self) -> Dict[str, Any]:
        if self.mode == 'vllm':
            return {
                'hf_overrides': {
                    'architectures': ['Qwen3ForSequenceClassification'],
                    'classifier_from_token': ['no', 'yes'],
                    'is_original_qwen3_reranker': True,
                }
            }
        else:
            return super().prepare_engine_kwargs()


qwen3_reranker_system = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".')

register_template(
    Qwen3MixedTemplateMeta(
        LLMTemplateType.qwen3_reranker,
        default_system=qwen3_reranker_system,
        template_cls=Qwen3RerankerTemplate,
        agent_template=None))

register_template(Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math))


class QwenPRMTemplate(Template):
    cot_process_placeholder = '<extra_0>'

    def _preprocess_inputs(
        self,
        inputs: StdTemplateInputs,
    ) -> None:
        super()._preprocess_inputs(inputs)
        total_content = '\n'.join([message['content'] or '' for message in inputs.messages])
        if self.cot_process_placeholder not in total_content:
            inputs.messages[-1]['content'] = inputs.messages[-1]['content'] + self.cot_process_placeholder

    @staticmethod
    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    def decode_prm(self, input_ids: torch.Tensor, logits: torch.Tensor) -> Any:
        step_sep_id = self.tokenizer.encode(self.cot_process_placeholder)[0]
        token_masks = (input_ids == step_sep_id)
        return self.make_step_rewards(logits, token_masks)


register_template(Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math_prm, template_cls=QwenPRMTemplate))


class QwenVLTemplate(Template):
    load_images = False

    @staticmethod
    def _load_image(image, load_images: bool):
        if not load_images and isinstance(image, str) and (image.startswith('data:') or len(image) > 200):
            load_images = True
        return Template._load_image(image, load_images)

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

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<ref>{ref}</ref>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<box>{self._get_bbox_str(bbox)}</box>']


register_template(QwenTemplateMeta(MLLMTemplateType.qwen_vl, template_cls=QwenVLTemplate, agent_template=None))


class QwenAudioTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        audios = inputs.audios
        audio = audios[index]
        assert isinstance(audio, str)
        return [f'Audio {index + 1}:<audio>{audio}</audio>\n']

    def _tokenize(self, context, **kwargs):
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


register_template(QwenTemplateMeta(MLLMTemplateType.qwen_audio, template_cls=QwenAudioTemplate, agent_template=None))


class Qwen2AudioTemplate(Template):

    def init_env_args(self) -> None:
        super().init_env_args()
        self.sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        if not self.use_chat_template:
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>\n']
        else:
            return [f'Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        sampling_rate = inputs.chat_template_kwargs.get('sampling_rate')
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if inputs.audios:
            audios = load_batch(inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate))
            audio_inputs = self.processor.feature_extractor(
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
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    version = 'v2'
    use_model = True
    support_padding_free = True
    _requires_mm_token_type_ids = True

    def init_env_args(self):
        super().init_env_args()
        self.transformers_version = version.parse(transformers.__version__)
        self.bbox_format = get_env_args('QWENVL_BBOX_FORMAT', str, 'legacy')
        self.transformers_5_3 = self.transformers_version >= version.parse('5.3.0')
        self.transformers_5_9 = self.transformers_version >= version.parse('5.9.0')

    @property
    def requires_mm_token_type_ids(self):
        return self.transformers_5_3 and self._requires_mm_token_type_ids

    def _get_max_pixels(self, inputs=None):
        return self.max_pixels

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        kwargs = {'image_patch_size': self.processor.image_processor.patch_size} if self.version == 'v3' else {}
        if self.mode == 'vllm':
            # resized in qwen_vl_utils, no need to resize again in vllm
            # ref: https://github.com/modelscope/ms-swift/issues/8445
            inputs.mm_processor_kwargs['do_resize'] = False
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index], **inputs.chat_template_kwargs}, **kwargs)
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            if self.version == 'v3':
                kwargs['return_video_metadata'] = True
            video = inputs.videos[index]
            video_inputs = {'video': video, **inputs.chat_template_kwargs}
            if isinstance(video, list):  # image list
                from qwen_vl_utils import vision_process
                video_inputs['sample_fps'] = vision_process.FPS
            video, video_kwargs = fetch_video(video_inputs, return_video_sample_fps=True, **kwargs)
            tokens = ['<|vision_start|><|video_pad|><|vision_end|>']
            if self.version == 'v2_5':
                inputs.mm_processor_kwargs.setdefault('fps', []).append(video_kwargs)
            elif self.version == 'v3':
                if self.mode != 'vllm':
                    video, video_metadata = video
                    inputs.mm_processor_kwargs.setdefault('video_metadata', []).append(video_metadata)
                    tokens = ['<|video_pad|>']
                inputs.mm_processor_kwargs['do_sample_frames'] = False
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return tokens

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|object_ref_start|>{ref}<|object_ref_end|>']
        else:
            return [ref]

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']
        else:
            return [str(bbox)]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        mm_mask = [False] * len(input_ids)
        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    kwargs = {}
                    if hasattr(processor, 'video_processor'):
                        processor_func = processor.video_processor
                    else:
                        processor_func = processor.image_processor
                        kwargs['images'] = None
                    media_inputs = processor_func(videos=mm_data, return_tensors='pt', do_resize=False, **kwargs)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                    if self.version == 'v2_5':
                        fps = inputs.mm_processor_kwargs['fps']
                        media_inputs['second_per_grid_ts'] = [
                            processor.image_processor.temporal_patch_size / tmp for tmp in fps
                        ]
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    return [media_token] * token_len

                input_ids, labels, loss_scale, mm_mask = self._extend_tokens(
                    input_ids, labels, loss_scale, idx_list, _get_new_tokens, mm_mask=mm_mask)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        if self.requires_mm_token_type_ids and any(mm_mask):
            encoded['mm_token_type_ids'] = self.create_mm_token_type_ids(input_ids, mm_mask)
        return encoded

    def forward_context(self, model, inputs):
        if not self.padding_free or self.transformers_version >= version.parse('4.53.0.dev'):
            return super().forward_context(model, inputs)
        text_position_ids = inputs['text_position_ids']
        if self.version == 'v2':
            from transformers.models.qwen2_vl import modeling_qwen2_vl as modeling_module
        elif self.version == 'v2_5':
            from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as modeling_module
        elif self.version == 'omni_v2_5':
            from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as modeling_module
        return self._patch_flash_attention_forward(modeling_module, text_position_ids)

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor, model.config)
        return {'inputs_embeds': inputs_embeds}

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            res['second_per_grid_ts'] = second_per_grid_ts
        return res

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        for r in row:
            r_copy = r.copy()
            r_copy['input_ids'] = torch.tensor(r_copy['input_ids'])[None]
            if 'mm_token_type_ids' in r_copy:
                r_copy['mm_token_type_ids'] = r_copy['mm_token_type_ids'][None]
            r.update(self._get_position_ids(r_copy))
        packed = super().packing_row(row)
        return packed

    def _get_get_rope_index(self):
        base_model = self.get_base_model(self._get_model())
        if hasattr(base_model, 'get_rope_index'):
            get_rope_index = base_model.get_rope_index
        else:
            get_rope_index = base_model.model.get_rope_index
        return get_rope_index

    def _get_position_ids(self, inputs: Dict[str, Any]):
        # fix https://github.com/huggingface/transformers/pull/33487
        kwargs = {}
        if self.version == 'v2_5':
            kwargs = {'second_per_grid_ts': inputs.get('second_per_grid_ts')}
        attention_mask = inputs.get('attention_mask_2d')
        if attention_mask is None:
            attention_mask = inputs.get('attention_mask')
        input_ids = inputs['input_ids']
        mm_token_type_ids = inputs.get('mm_token_type_ids')
        if mm_token_type_ids is not None:
            kwargs['mm_token_type_ids'] = mm_token_type_ids
        position_ids, _ = self._get_get_rope_index()(
            input_ids,
            image_grid_thw=inputs.get('image_grid_thw'),
            video_grid_thw=inputs.get('video_grid_thw'),
            attention_mask=attention_mask,
            **kwargs)
        return {'position_ids': self._concat_text_position_ids(position_ids)}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        if self.requires_mm_token_type_ids:
            for b in batch:
                if 'input_ids' in b and 'mm_token_type_ids' not in b:
                    b['mm_token_type_ids'] = torch.zeros(len(b['input_ids']), dtype=torch.int64)
        res = super()._data_collator(batch, padding_to=padding_to)
        if not self.padding_free:
            res.update(self._get_position_ids(res))
        if 'position_ids' in res and self.is_training:
            position_ids = res['position_ids']
            res['position_ids'] = position_ids[1:]
            res['text_position_ids'] = text_position_ids = position_ids[0]
            if self.transformers_version >= version.parse('4.53.0.dev') and text_position_ids.shape[0] == 1:
                # https://github.com/huggingface/transformers/pull/40194
                res.update(get_packed_seq_params(text_position_ids))
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_vl, template_cls=Qwen2VLTemplate))

register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qvq,
        default_system=('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                        'Answer in the language of the question. You should think step-by-step.'),
        template_cls=Qwen2VLTemplate,
    ))


class Qwen2_5VLTemplate(Qwen2VLTemplate):
    version = 'v2_5'
    norm_bbox = 'none'


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_5_vl, template_cls=Qwen2_5VLTemplate))

register_template(
    QwenTemplateMeta(
        MLLMTemplateType.mimo_vl,
        template_cls=Qwen2_5VLTemplate,
        default_system='You are MiMo, an AI assistant developed by Xiaomi.'))


class Qwen3VLTemplate(Qwen2VLTemplate):
    version = 'v3'

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        mm_mask = [False] * len(input_ids)
        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    split_token = self._tokenize('\n')[0]
                    media_inputs = processor(
                        text=['\n'.join(['<|vision_start|><|video_pad|><|vision_end|>'] * len(mm_data))],
                        videos=mm_data,
                        return_tensors='pt',
                        do_resize=False,
                        **inputs.mm_processor_kwargs)
                    splited_tokens = self._split_list(media_inputs['input_ids'][0].tolist(), split_token)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_inputs.pop('input_ids', None)
                    media_inputs.pop('attention_mask', None)
                    media_token = self.video_token_id
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    if media_type == 'images':
                        token_len = (media_grid_thw[i].prod() // merge_length)
                        return [media_token] * token_len
                    else:
                        return splited_tokens[i]

                input_ids, labels, loss_scale, mm_mask = self._extend_tokens(
                    input_ids, labels, loss_scale, idx_list, _get_new_tokens, mm_mask=mm_mask)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        if self.requires_mm_token_type_ids and any(mm_mask):
            encoded['mm_token_type_ids'] = self.create_mm_token_type_ids(input_ids, mm_mask)
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qwen3_vl, template_cls=Qwen3VLTemplate, default_system=None, thinking_prefix='<think>\n'))


class Qwen3_5Template(Qwen3VLTemplate):
    image_token_id = 248056
    video_token_id = 248057

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.padding_free and self.sequence_parallel_size <= 1 and not self.transformers_5_9:
            raise RuntimeError('Qwen3.5 packing/padding_free with sequence_parallel_size=1 requires '
                               f'transformers>=5.9.0 (current: {self.transformers_version}). ')
        return Qwen2VLTemplate._post_encode(self, model, inputs)

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs):
        # Normalize message content so the swift backend byte-matches Qwen3.5/Qwen3.6
        # HF `chat_template.jinja` rendering (per-role `|trim` and canonical <think> padding).
        # Must run BEFORE super(), because super() merges/wraps tool messages into
        # `<tool_response>...</tool_response>` blobs using the raw inner content.
        # See: https://github.com/modelscope/ms-swift/issues/9276
        if isinstance(inputs.system, str):
            inputs.system = inputs.system.strip()
        for message in inputs.messages:
            role = message.get('role')
            content = message.get('content')
            if not isinstance(content, str):
                continue
            if role in ('user', 'system', 'tool'):
                # HF applies `|trim` to user/system/tool content.
                message['content'] = content.strip()
            elif role == 'assistant':
                # HF applies `|trim` and re-wraps the <think>...</think> block with canonical newlines.
                stripped = content.strip()
                if '</think>' in stripped and '<think>' in stripped:
                    before, _, after = stripped.partition('</think>')
                    reasoning = before.rstrip('\n').rsplit('<think>', 1)[-1].lstrip('\n').strip()
                    rest = after.lstrip('\n')
                    message['content'] = f'<think>\n{reasoning}\n</think>\n\n{rest}'
                else:
                    message['content'] = stripped
        super()._swift_prepare_inputs(inputs)


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qwen3_5,
        template_cls=Qwen3_5Template,
        default_system=None,
        thinking_prefix='<think>\n',
        non_thinking_prefix='<think>\n\n</think>\n\n',
        agent_template='qwen3_5',
        is_thinking=True))


class Qwen3VLEmbTemplate(Qwen3VLTemplate):

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        if len(inputs.messages) % 2 == 1 and inputs.messages[-1]['role'] != 'assistant':
            inputs.messages.append({'role': 'assistant', 'content': ''})


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qwen3_vl_emb,
        default_system="Represent the user's input.",
        suffix=['<|endoftext|>'],
        template_cls=Qwen3VLEmbTemplate,
    ))


class Qwen3VLRerankerTemplate(Qwen3VLTemplate):
    instruction = 'Given a search query, retrieve relevant candidates that answer the query.'

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        if inputs.system is not None:
            instruction = inputs.system
            inputs.system = None
        else:
            instruction = self.instruction
        query = inputs.messages[0]['content']
        document = inputs.messages[1]['content']
        user_message = '<Instruct>: ' + instruction + '<Query>:' + query + '\n' + '<Document>:' + document
        inputs.messages = [{'role': 'user', 'content': user_message}]
        return inputs


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qwen3_vl_reranker, default_system=qwen3_reranker_system, template_cls=Qwen3VLRerankerTemplate))


# ref: trim to hop multiple so WhisperFeatureExtractor matches native HF (floor frames);
# vLLM pad_to_hop_length becomes no-op on pre-trimmed waveforms (GRPO train/rollout align).
def trim_audio_to_hop_length(x: np.ndarray, hop_length: int) -> np.ndarray:
    length = x.shape[-1]
    aligned = (length // hop_length) * hop_length
    if 0 < aligned < length:
        x = x[..., :aligned]
    return x


class Qwen2_5OmniTemplate(Qwen2_5VLTemplate):
    version = 'omni_v2_5'
    placeholder_tokens = ['<|IMAGE|>', '<|AUDIO|>', '<|VIDEO|>']
    _requires_mm_token_type_ids = False

    def init_processor(self, processor) -> None:
        if processor is None:
            return
        super().init_processor(processor)
        if self.version == 'omni_v2_5':
            from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessorKwargs
            default = Qwen2_5OmniProcessorKwargs._defaults
        elif self.version == 'omni_v3':
            from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessorKwargs
            default = Qwen3OmniMoeProcessorKwargs._defaults
            # Fix: WhisperFeatureExtractor defaults to truncation=True, which silently
            # truncates audio longer than 30s. Qwen3 Omni supports variable-length audio,
            # so we must disable truncation. See: huggingface/transformers#41473
            default.setdefault('audio_kwargs', {})
            default['audio_kwargs']['truncation'] = False
        self.seconds_per_chunk = default['videos_kwargs']['seconds_per_chunk']
        self.position_id_per_seconds = default['videos_kwargs']['position_id_per_seconds']
        self.use_audio_in_video = get_env_args('use_audio_in_video', bool, False)
        self.sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)

    def _trim_omni_v3_audios(self, audios):
        """Trim waveforms to hop-length multiple (omni_v3 only). Matches native HF floor framing."""
        if self.version != 'omni_v3' or not audios:
            return audios
        hop = self.processor.feature_extractor.hop_length
        trimmed = []
        for audio in audios:
            if isinstance(audio, tuple):
                # train: (wav, 'video'); vllm standalone: (wav, sr)
                trimmed.append((trim_audio_to_hop_length(audio[0], hop), audio[1]))
            elif isinstance(audio, np.ndarray):
                trimmed.append(trim_audio_to_hop_length(audio, hop))
            else:
                raise TypeError(f'unexpected audio type {type(audio)!r}; expected ndarray or (ndarray, meta)')
        return trimmed

    def _encode_truncated(self, inputs: StdTemplateInputs):
        encoded = super()._encode_truncated(inputs)
        if self.mode == 'vllm' and inputs.audios:
            inputs.audios = self._trim_omni_v3_audios(inputs.audios)
            if 'audios' in encoded:
                encoded['audios'] = inputs.audios
        return encoded

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_omni_utils import fetch_image, fetch_video
        kwargs = {'image_patch_size': self.processor.image_processor.patch_size} if self.version == 'omni_v3' else {}
        sampling_rate = inputs.chat_template_kwargs.get('sampling_rate')
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if self.mode == 'vllm':
            # https://github.com/modelscope/ms-swift/issues/8445
            inputs.mm_processor_kwargs['do_resize'] = False
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index], **inputs.chat_template_kwargs}, **kwargs)
            if self.version == 'omni_v2_5':
                return ['<|vision_bos|><|IMAGE|><|vision_eos|>']
            elif self.version == 'omni_v3':
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        elif media_type == 'audio':
            if self.mode != 'vllm':
                inputs.audios[index] = load_audio(inputs.audios[index], sampling_rate)
            if self.version == 'omni_v2_5':
                return ['<|audio_bos|><|AUDIO|><|audio_eos|>']
            elif self.version == 'omni_v3':
                return ['<|audio_start|><|audio_pad|><|audio_end|>']
        elif media_type == 'video':
            video = inputs.videos[index]
            video_inputs = {'video': video, **inputs.chat_template_kwargs}
            if isinstance(video, list):  # image list
                from qwen_omni_utils import vision_process
                video_inputs['sample_fps'] = vision_process.FPS
            _video = fetch_video(video_inputs, **kwargs)
            if isinstance(_video, torch.Tensor):
                _video = _video.to(torch.uint8)
            inputs.videos[index] = _video
            if self.use_audio_in_video:
                if isinstance(video, list):  # image list
                    raise ValueError('image list as video input does not support use_audio_in_video')
                audio = load_audio(video, sampling_rate)
                if self.mode != 'vllm':
                    inputs.audios.insert(inputs.audio_idx, (audio, 'video'))
                else:
                    inputs.audios.insert(inputs.audio_idx, audio)
                    inputs.mm_processor_kwargs['use_audio_in_video'] = True
                inputs.audio_idx += 1
                if self.version == 'omni_v2_5':
                    return ['<|vision_bos|><|audio_bos|><|VIDEO|><|audio_eos|><|vision_eos|>']
                elif self.version == 'omni_v3':
                    if self.mode == 'vllm':
                        return ['<|vision_start|><|video_pad|><|vision_end|>']
                    else:
                        return ['<|vision_start|><|audio_start|><|video_pad|><|audio_end|><|vision_end|>']
            if self.version == 'omni_v2_5':
                return ['<|vision_bos|><|VIDEO|><|vision_eos|>']
            elif self.version == 'omni_v3':
                return ['<|vision_start|><|video_pad|><|vision_end|>']

    def _get_feat_extract_output_lengths(self, input_lengths):
        if self.version == 'omni_v2_5':
            return ((input_lengths - 1) // 2 + 1 - 2) // 2 + 1
        elif self.version == 'omni_v3':
            input_lengths_leave = input_lengths % 100
            feat_lengths = (input_lengths_leave - 1) // 2 + 1
            return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13

    def _get_new_tokens_use_audio_in_video(self, i, *, video_grid_thw, video_second_per_grid, audio_lengths,
                                           video_token_id, audio_token_id):
        merge_size = self.processor.image_processor.merge_size
        grid_thw = video_grid_thw[i]
        height = grid_thw[1] // merge_size
        width = grid_thw[2] // merge_size
        audio_token_indices = torch.arange(audio_lengths[i])
        video_token_indices = torch.arange(grid_thw[0]).reshape(-1, 1, 1)

        video_token_indices = torch.broadcast_to(video_token_indices,
                                                 (video_token_indices.shape[0], height, width)).reshape(-1)
        video_token_indices = (video_token_indices * video_second_per_grid[i] * self.position_id_per_seconds)
        if self.version == 'omni_v2_5':
            tokens_per_chunk = int(self.position_id_per_seconds * self.seconds_per_chunk)
            video_chunk_indexes = self.processor.get_chunked_index(video_token_indices, tokens_per_chunk)
            audio_chunk_indexes = self.processor.get_chunked_index(audio_token_indices, tokens_per_chunk)

            res = []
            for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                if j < len(video_chunk_indexes):
                    video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                    res += video_token_id * video_seq_length
                if j < len(audio_chunk_indexes):
                    audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                    res += audio_token_id * audio_seq_length
            return res
        elif self.version == 'omni_v3':
            res = []
            video_data_index, audio_data_index = 0, 0
            while video_data_index < len(video_token_indices) and audio_data_index < len(audio_token_indices):
                if video_token_indices[video_data_index] <= audio_token_indices[audio_data_index]:
                    res += video_token_id
                    video_data_index += 1
                else:
                    res += audio_token_id
                    audio_data_index += 1
            if video_data_index < len(video_token_indices):
                res += video_token_id * (len(video_token_indices) - video_data_index)
            if audio_data_index < len(audio_token_indices):
                res += audio_token_id * (len(audio_token_indices) - audio_data_index)
            return res

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        inputs.audios = self._trim_omni_v3_audios(inputs.audios)
        processor = self.processor
        video_audios_mask = []
        for i, audio in enumerate(inputs.audios):
            if isinstance(audio, tuple) and audio[1] == 'video':
                inputs.audios[i] = audio[0]
                video_audios_mask.append(True)
            else:
                video_audios_mask.append(False)
        video_audios_mask = torch.tensor(video_audios_mask)
        media_inputs = processor(
            text='',
            audio=inputs.audios or None,
            images=inputs.images or None,
            videos=inputs.videos or None,
            do_resize=False,
            return_tensors='pt')
        media_inputs.pop('input_ids')
        media_inputs.pop('attention_mask')
        media_inputs = to_float_dtype(media_inputs, self.model_info.torch_dtype)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        config = self.config.thinker_config
        # audio
        if self.version == 'omni_v3':
            audio_token_id = [config.audio_token_id]
        else:
            audio_token_id = self._tokenize('<|AUDIO|>')
        idx_list = findall(input_ids, audio_token_id)
        feature_attention_mask = media_inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            audio_lengths = self._get_feat_extract_output_lengths(audio_feature_lengths)
        else:
            audio_lengths = None
        audio_lengths_origin = audio_lengths
        if idx_list:
            if self.use_audio_in_video:
                audio_lengths = audio_lengths[~video_audios_mask]

            def _get_new_audio_tokens(i):
                return audio_token_id * audio_lengths[i]

            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_audio_tokens)

        for media_type in ['image', 'video']:
            if self.version == 'omni_v3':
                token_id = [getattr(config, f'{media_type}_token_id')]
            else:
                token = f'<|{media_type.upper()}|>'
                token_id = self._tokenize(token)
            idx_list = findall(input_ids, token_id)
            if idx_list:
                merge_size = processor.image_processor.merge_size
                media_grid_thw = media_inputs.get(f'{media_type}_grid_thw')
                if media_type == 'video' and self.use_audio_in_video:
                    audio_lengths = audio_lengths_origin[video_audios_mask]
                    video_second_per_grid = media_inputs['video_second_per_grid']
                    _get_new_tokens_use_audio_in_video = partial(
                        self._get_new_tokens_use_audio_in_video,
                        video_grid_thw=media_grid_thw,
                        video_second_per_grid=video_second_per_grid,
                        audio_lengths=audio_lengths,
                        video_token_id=token_id,
                        audio_token_id=audio_token_id)
                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_tokens_use_audio_in_video)

                else:

                    def _get_new_tokens(i):
                        token_len = (media_grid_thw[i].prod() // (merge_size**2))
                        return token_id * token_len

                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_tokens)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        encoded.update(media_inputs)
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs

        input_ids = inputs['input_ids']
        input_features = inputs.get('input_features')
        feature_attention_mask = inputs.get('feature_attention_mask')

        base_model = self.get_base_model(model)
        inputs_embeds = base_model.thinker.model.embed_tokens(input_ids)
        thinker_config = model.config.thinker_config
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.thinker.visual, self.processor,
                                                   thinker_config)
        if input_features is None:
            if is_deepspeed_enabled() and not is_deepspeed_zero3_enabled():
                # Note: ZeRO-3 still results in hangs; for audio training, please use ZeRO-2.
                input_features = input_ids.new_zeros([1, 128, 128], dtype=model.thinker.audio_tower.dtype)
                feature_attention_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
                audio_res = model.thinker.get_audio_features(input_features, feature_attention_mask)
                if hasattr(audio_res, 'last_hidden_state'):
                    audio_embeds = audio_res.last_hidden_state
                else:
                    audio_embeds = audio_res
                inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.
        else:
            audio_res = model.thinker.get_audio_features(input_features, feature_attention_mask)
            if hasattr(audio_res, 'last_hidden_state'):
                audio_embeds = audio_res.last_hidden_state
            else:
                audio_embeds = audio_res
            audio_mask = (input_ids == thinker_config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        return {'inputs_embeds': inputs_embeds}

    def _get_get_rope_index(self):
        return self._get_model().thinker.get_rope_index

    def _get_position_ids(self, inputs: Dict[str, Any]):
        if not self.is_training:
            return {}
        feature_attention_mask = inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
        video_second_per_grid = inputs.pop('video_second_per_grid', None)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask_2d')
        if attention_mask is None:
            attention_mask = inputs.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids, _ = self._get_get_rope_index()(
            input_ids,
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask,
            self.use_audio_in_video,
            audio_feature_lengths,
            video_second_per_grid,
        )
        if torch.is_floating_point(position_ids):
            position_ids = position_ids.to(torch.int64)
        return {'position_ids': self._concat_text_position_ids(position_ids)}

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        video_second_per_grid = self.gather_list(batch, 'video_second_per_grid')
        if video_second_per_grid:
            res['video_second_per_grid'] = video_second_per_grid
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            if self.version == 'omni_v3':
                max_length = max(input_feature.shape[-1] for input_feature in input_features)
                for i, input_feature in enumerate(input_features):
                    mask = feature_attention_mask[i]
                    input_features[i] = F.pad(input_feature, (0, max_length - input_feature.shape[-1]))
                    feature_attention_mask[i] = F.pad(mask, (0, max_length - mask.shape[-1]))
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res

    def generate(self, model, *args, **kwargs):
        if kwargs.get('video_grid_thw') is not None:
            kwargs['use_audio_in_video'] = self.use_audio_in_video
        return super().generate(model, *args, **kwargs)


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_5_omni, template_cls=Qwen2_5OmniTemplate))


class Qwen3OmniTemplate(Qwen2_5OmniTemplate):
    version = 'omni_v3'
    norm_bbox = 'norm1000'
    placeholder_tokens = ['<|image_pad|>', '<|audio_pad|>', '<|video_pad|>']

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qwen3_omni, template_cls=Qwen3OmniTemplate, default_system=None, thinking_prefix='<think>\n'))


def _qwen3_asr_get_feat_extract_output_lengths(input_lengths):
    """Qwen3-ASR Conv2d encoder output length: chunks of 100 frames, 13 tokens each."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class Qwen3ASRTemplate(Template):
    placeholder_tokens = ['<|audio_pad|>']
    support_padding_free = True

    def init_env_args(self) -> None:
        super().init_env_args()
        self.sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        return ['<|audio_start|><|audio_pad|><|audio_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        sampling_rate = inputs.chat_template_kwargs.get('sampling_rate')
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if inputs.audios:
            audios = load_batch(inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate))
            audio_inputs = self.processor.feature_extractor(
                audios,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                return_tensors='pt',
                padding=True,
                truncation=False)
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
            audio_inputs['input_features'] = to_float_dtype(audio_inputs['input_features'], self.model_info.torch_dtype)
            encoded.update(audio_inputs)

            input_ids = encoded['input_ids']
            labels = encoded['labels']
            loss_scale = encoded.get('loss_scale')
            audio_token_id = self._tokenize('<|audio_pad|>')
            idx_list = findall(input_ids, audio_token_id)

            if idx_list:
                feature_attention_mask = audio_inputs.get('feature_attention_mask')
                if feature_attention_mask is not None:
                    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                    audio_lengths = _qwen3_asr_get_feat_extract_output_lengths(audio_feature_lengths)

                    def _get_new_audio_tokens(i):
                        return audio_token_id * int(audio_lengths[i])

                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_audio_tokens)
                    encoded['input_ids'] = input_ids
                    encoded['labels'] = labels
                    encoded['loss_scale'] = loss_scale

        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            max_length = max(input_feature.shape[-1] for input_feature in input_features)
            for i, input_feature in enumerate(input_features):
                mask = feature_attention_mask[i]
                input_features[i] = F.pad(input_feature, (0, max_length - input_feature.shape[-1]))
                feature_attention_mask[i] = F.pad(mask, (0, max_length - mask.shape[-1]))
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qwen3_asr,
        template_cls=Qwen3ASRTemplate,
        # Even without adding a system message,
        # the '<|im_start|>system\n<|im_end|>\n' prefix is still present.
        # Align with the qwen3_asr template.
        system_prefix=None,
        default_system=None,
        prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))


class Qwen3TTSTemplate(Template):
    # ref: https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning
    support_padding_free = False
    use_model = True
    model_accepts_loss_kwargs = False

    def init_env_args(self) -> None:
        super().init_env_args()
        self._config_initialized = False
        self.target_speaker_embedding = None
        # Cache TTS config values for data collation
        config = self.config
        self._tts_pad_token_id = config.tts_pad_token_id
        self._tts_bos_token_id = config.tts_bos_token_id
        self._tts_eos_token_id = config.tts_eos_token_id
        talker_config = config.talker_config
        self._codec_nothink_id = talker_config.codec_nothink_id
        self._codec_think_bos_id = talker_config.codec_think_bos_id
        self._codec_think_eos_id = talker_config.codec_think_eos_id
        self._codec_pad_id = talker_config.codec_pad_id
        self._codec_bos_id = talker_config.codec_bos_id
        self._codec_eos_token_id = talker_config.codec_eos_token_id

    @staticmethod
    def _extract_ref_mel(ref_audio_path: str) -> torch.Tensor:
        """Extract mel spectrogram from reference audio for speaker embedding."""
        import librosa
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
        audio, sr = librosa.load(ref_audio_path, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        mels = mel_spectrogram(
            torch.from_numpy(audio.astype(np.float32)).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000).transpose(1, 2)  # [1, mel_len, 128]
        return mels

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        """Override to skip _add_default_tags since audios here are targets, not inputs."""
        pass

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        # Get text from messages (assistant content)
        text = inputs.messages[-1]['content'] if inputs.messages else ''

        # Build TTS text with assistant markers
        tts_text = f'<|im_start|>assistant\n{text}'
        text_ids = self._tokenize(tts_text)

        # Get audio codes (pre-extracted or online)
        audio_codes = inputs.extra_kwargs.get('audio_codes')
        if audio_codes is None:
            audio_path = inputs.audios[0] if inputs.audios else None
            if audio_path:
                tts_tokenizer = self.processor.tts_tokenizer
                enc_res = tts_tokenizer.encode([audio_path])
                audio_codes = enc_res.audio_codes[0].cpu().tolist()
        assert audio_codes is not None, "Either 'audio_codes' or 'audio'/'audios' must be provided in the dataset."
        audio_codes = torch.tensor(audio_codes, dtype=torch.long)  # [t, 16]

        # Extract mel spectrogram from reference audio
        ref_audios = inputs.extra_kwargs.get('ref_audios')
        ref_audio_path = ref_audios[0]
        assert ref_audio_path is not None, "'ref_audios' must be provided in the dataset."
        ref_mel = self._extract_ref_mel(ref_audio_path)  # [1, mel_len, 128]

        return {
            'input_ids': text_ids,  # dummy for length tracking
            'labels': None,
            'tts_audio_codes': audio_codes,  # [codec_len, 16]
            'tts_ref_mel': ref_mel,  # [1, mel_len, 128]
        }

    def compute_sft_loss(self, model, inputs, num_items_in_batch=None, trainer=None):
        """Override to bypass standard label adjustment - TTS loss is computed in forward.

        Combines the talker codec_0 cross-entropy loss with the sub-talker loss
        using a fixed weighting factor of 0.3.
        """
        # Extract speaker_embedding from ref_mels and cache for checkpoint post-processing
        if 'ref_mels' in inputs:
            base_model = model.module if hasattr(model, 'module') else model
            with torch.no_grad():
                speaker_embedding = base_model.speaker_encoder(inputs['ref_mels'].to(base_model.device).to(
                    base_model.dtype)).detach()
            if self.target_speaker_embedding is None:
                self.target_speaker_embedding = speaker_embedding.cpu()
            inputs.pop('ref_mels')
            inputs['speaker_embedding'] = speaker_embedding
        outputs = model(**inputs)
        logits = outputs.logits
        shift_labels = inputs['codec_0_labels'][:, 1:].contiguous()
        talker_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            shift_labels.reshape(-1).to(logits.device),
            ignore_index=-100,
        )
        sub_talker_loss = getattr(outputs, 'sub_talker_loss', None)
        if sub_talker_loss is not None:
            outputs['loss'] = talker_loss + 0.3 * sub_talker_loss
        else:
            outputs['loss'] = talker_loss
        return outputs

    def save_callback(self, model, output_dir):
        """Custom save: drop speaker_encoder weights and inject target_speaker_embedding
        into codec_embedding.weight[3000]."""
        shutil.copytree(model.config.name_or_path, output_dir, dirs_exist_ok=True)
        with open(os.path.join(model.config.name_or_path, 'config.json'), 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        speaker_name = get_env_args('speaker_name', str, 'speaker_test')
        config_dict['tts_model_type'] = 'custom_voice'
        config_dict['talker_config']['spk_id'] = {speaker_name: 3000}
        config_dict['talker_config']['spk_is_dialect'] = {speaker_name: False}
        from safetensors.torch import save_file
        from transformers.modeling_utils import unwrap_model

        base_model = unwrap_model(model)
        state_dict = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}

        # 1. Drop speaker_encoder keys
        keys_to_drop = [k for k in state_dict if k.startswith('speaker_encoder')]
        for k in keys_to_drop:
            del state_dict[k]

        # 2. Inject target_speaker_embedding into codec_embedding.weight[3000]
        emb_key = 'talker.model.codec_embedding.weight'
        if self.target_speaker_embedding is not None and emb_key in state_dict:
            weight = state_dict[emb_key]
            state_dict[emb_key][3000] = self.target_speaker_embedding[0].to(weight.dtype)

        save_file(state_dict, os.path.join(output_dir, 'model.safetensors'))
        # Save config
        with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def data_collator(self, batch: List[Dict[str, Any]], *, padding_to=None) -> Dict[str, Any]:
        """Custom TTS data collation - builds dual-channel input format."""
        item_length = [len(b['input_ids']) + b['tts_audio_codes'].shape[0] for b in batch]
        max_length = max(item_length) + 8
        b_size, t = len(batch), max_length

        input_ids = torch.zeros((b_size, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b_size, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b_size, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b_size, t), dtype=torch.bool)
        codec_mask = torch.zeros((b_size, t), dtype=torch.bool)
        attention_mask = torch.zeros((b_size, t), dtype=torch.long)
        codec_0_labels = torch.full((b_size, t), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = torch.tensor(data['input_ids'])  # [text_len]
            audio_codes = data['tts_audio_codes']  # [codec_len, 16]
            audio_codec_0 = audio_codes[:, 0]

            text_ids_len = len(text_ids)
            codec_ids_len = audio_codec_0.shape[0]

            # === Text channel ===
            input_ids[i, :3, 0] = text_ids[:3]
            input_ids[i, 3:7, 0] = self._tts_pad_token_id
            input_ids[i, 7, 0] = self._tts_bos_token_id
            input_ids[i, 8:8 + text_ids_len - 3, 0] = text_ids[3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self._tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2:8 + text_ids_len + codec_ids_len, 0] = self._tts_pad_token_id
            text_embedding_mask[i, :8 + text_ids_len + codec_ids_len] = True

            # === Codec channel ===
            input_ids[i, 3:8, 1] = torch.tensor([
                self._codec_nothink_id,
                self._codec_think_bos_id,
                self._codec_think_eos_id,
                0,  # placeholder for speaker embedding
                self._codec_pad_id,
            ])
            input_ids[i, 8:8 + text_ids_len - 3, 1] = self._codec_pad_id
            input_ids[i, 8 + text_ids_len - 3, 1] = self._codec_pad_id
            input_ids[i, 8 + text_ids_len - 2, 1] = self._codec_bos_id
            input_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = self._codec_eos_token_id

            # === Labels (codec layer 0) ===
            codec_0_labels[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = self._codec_eos_token_id

            # === Sub-talker codec IDs ===
            codec_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, :] = audio_codes

            # === Masks ===
            codec_embedding_mask[i, 3:8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False  # speaker embedding position
            codec_mask[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = True
            attention_mask[i, :8 + text_ids_len + codec_ids_len] = True

        ref_mels = torch.cat([data['tts_ref_mel'] for data in batch], dim=0)

        return {
            'input_ids': input_ids,
            'ref_mels': ref_mels,
            'attention_mask': attention_mask,
            'text_embedding_mask': text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask': codec_embedding_mask.unsqueeze(-1),
            'labels': codec_0_labels,
            'codec_0_labels': codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask': codec_mask,
        }


register_template(QwenTemplateMeta(
    MLLMTemplateType.qwen3_tts,
    template_cls=Qwen3TTSTemplate,
    default_system=None,
))


class Ovis1_6Template(Template):
    skip_prompt = False
    use_model = True

    def init_env_args(self):
        super().init_env_args()
        self.max_partition = get_env_args('max_partition', int, 9)

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
            raw_pixel_values, image_placeholders = self.model.visual_tokenizer.preprocess_image(
                images[i], max_partition=self.max_partition)
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
        if self.max_length is not None:
            model.config.multimodal_max_length = self.max_length
        input_ids = inputs['input_ids']
        labels = inputs.get('labels')
        if labels is None:
            labels = input_ids.new_full(input_ids.shape, -100)
        _, inputs_embeds, labels, attention_mask = model.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=torch.ones_like(input_ids),  # not use, only compat
            text_labels=labels,
            pixel_values=inputs['pixel_values'],
            left_padding=padding_side == 'left')
        if inputs.get('labels') is None:
            labels = None
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
        agent_template=None,
    ))


class Ovis2Template(Ovis1_6Template):
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    NFRAMES = 12

    def init_env_args(self):
        super().init_env_args()
        self.nframes = get_env_args('nframes', int, self.NFRAMES)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            if self.mode == 'vllm':
                return ['<image>\n']
            return [[-200], '\n']
        elif media_type == 'video':
            inputs.images = load_video_ovis2(inputs.videos[index], self.nframes)
            return [[-200] * self.nframes, '\n']


register_template(QwenTemplateMeta(
    MLLMTemplateType.ovis2,
    template_cls=Ovis2Template,
))


class Ovis2_5Template(Template):
    use_model = True
    skip_prompt = False
    support_padding_free = True

    def init_env_args(self) -> None:
        super().init_env_args()
        self.min_pixels = get_env_args('min_pixels', int, 448 * 448)
        self.max_pixels = get_env_args('max_pixels', int, 1344 * 1792)
        self.video_max_pixels = get_env_args('video_max_pixels', int, 896 * 896)
        self.num_frames = get_env_args('num_frames', int, 8)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            if self.mode == 'vllm':
                return ['<image>']
            else:
                return [[-200], '\n']
        elif media_type == 'video':
            if self.mode == 'vllm':
                return ['<video>']
            else:
                inputs.images = load_video_ovis2_5(inputs.videos[index], self.num_frames)
                return [[-200], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        visual_tokenizer = self._get_model().visual_tokenizer
        idx_list = findall(input_ids, [-200])
        if inputs.videos:
            assert len(inputs.videos) == 1, 'only support single video'
            encoded['pixel_values'], encoded['grid_thws'] = visual_tokenizer.preprocess(
                video=inputs.images, min_pixels=self.min_pixels, max_pixels=self.video_max_pixels)
            num_video_tokens = encoded['grid_thws'].prod(dim=-1)
            num_video_tokens //= visual_tokenizer.vit.config.hidden_stride**2
            num_video_tokens //= visual_tokenizer.vit.config.temporal_patch_size

            def _get_new_tokens(i):
                token_len = num_video_tokens[i].item()
                return [-303] + [-300] * token_len + [-304]

            input_ids, encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
                input_ids, encoded['labels'], encoded['loss_scale'], idx_list, _get_new_tokens)
        elif images:
            pixel_values, grid_thws = zip(
                *(visual_tokenizer.preprocess(image=image, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
                  for image in images))
            encoded['pixel_values'] = torch.cat(pixel_values, dim=0)
            encoded['grid_thws'] = torch.cat(grid_thws, dim=0)

            num_image_atoms = encoded['grid_thws'].prod(dim=-1)
            num_image_atoms //= visual_tokenizer.vit.config.hidden_stride**2
            num_image_atoms //= visual_tokenizer.vit.config.temporal_patch_size

            def _get_new_tokens(i):
                token_len = num_image_atoms[i].item()
                return [-301] + [-300] * token_len + [-302]

            input_ids, encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
                input_ids, encoded['labels'], encoded['loss_scale'], idx_list, _get_new_tokens)

        encoded['input_ids'] = input_ids
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values', None)
        grid_thws = inputs.get('grid_thws')
        INDICATOR_IDS = [-301, -302, -303, -304]
        VISUAL_ATOM_ID = -300
        placeholder_token_mask = torch.lt(input_ids, 0)
        inputs_embeds = model.get_wte()(torch.masked_fill(input_ids, placeholder_token_mask, 0))

        if pixel_values is not None or is_deepspeed_enabled():
            visual_indicator_embeds = model.vte(model.indicator_token_indices).to(
                dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            for i, indicator_id in enumerate(INDICATOR_IDS):
                inputs_embeds[input_ids == indicator_id] = visual_indicator_embeds[i]
        if pixel_values is not None:
            visual_tokens = model.visual_tokenizer(pixel_values, grid_thws)
            visual_embeds = model.vte(visual_tokens).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds[input_ids == VISUAL_ATOM_ID] = visual_embeds
        elif is_deepspeed_enabled():
            pixel_values, grid_thws = model.visual_tokenizer.preprocess(
                Image.new('RGB', (32, 32), (0, 0, 0)), min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            pixel_values = pixel_values.to(device=inputs_embeds.device)
            grid_thws = grid_thws.to(device=inputs_embeds.device)
            visual_tokens = model.visual_tokenizer(pixel_values, grid_thws)
            visual_embeds = model.vte(visual_tokens).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds = inputs_embeds + visual_embeds.mean() * 0.

        return {'inputs_embeds': inputs_embeds}

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        grid_thws = self.concat_tensor(batch, 'grid_thws', 0)
        if grid_thws is not None:
            res['grid_thws'] = grid_thws
        return res


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.ovis2_5,
        template_cls=Ovis2_5Template,
        default_system=None,
        is_thinking=True,
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

YUFENG_XGUARD_TEMPLATE = ('<|im_start|>user\n'
                          'You are an expert in content safety evaluation, possessing strong '
                          'language comprehension and risk identification skills.\n'
                          'Your task is to classify the provided input text into the most appropriate '
                          'category from the list below as part of a safety audit project.\n\n'
                          '# Category List\n'
                          '- sec: Safe\n'
                          '- pc: Pornographic Contraband\n'
                          '- dc: Drug Crimes\n'
                          '- dw: Dangerous Weapons\n'
                          '- pi: Property Infringement\n'
                          '- ec: Economic Crimes\n'
                          '- ac: Abusive Curses\n'
                          '- def: Defamation\n'
                          '- ti: Threats and Intimidation\n'
                          '- cy: Cyberbullying\n'
                          '- ph: Physical Health\n'
                          '- mh: Mental Health\n'
                          '- se: Social Ethics\n'
                          '- sci: Science Ethics\n'
                          '- pp: Personal Privacy\n'
                          '- cs: Commercial Secret\n'
                          '- acc: Access Control\n'
                          '- mc: Malicious Code\n'
                          '- ha: Hacker Attack\n'
                          '- ps: Physical Security\n'
                          '- ter: Violent Terrorist Activities\n'
                          '- sd: Social Disruption\n'
                          '- ext: Extremist Ideological Trends\n'
                          '- fin: Finance\n'
                          '- med: Medicine\n'
                          '- law: Law\n'
                          '- cm: Corruption of Minors\n'
                          '- ma: Minor Abuse and Exploitation\n'
                          '- md: Minor Delinquency\n\n'
                          '# Instructions\n'
                          '- Identify the single most relevant category ID for the input text.\n'
                          '- On the next line, provide a concise justification for your choice, '
                          'placing it between <explanation> and </explanation> tags.\n\n'
                          '---\n\n'
                          'Input Text: {{QUERY}}<|im_end|>\n'
                          '<|im_start|>assistant\n')
register_template(Qwen3MixedTemplateMeta(
    LLMTemplateType.yufeng_xguard,
    prompt=[YUFENG_XGUARD_TEMPLATE],
))
