# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from swift.utils import upper_bound
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from ..vision_utils import load_audio


@dataclass
class GemmaTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<bos>'])
    prompt: Prompt = field(
        default_factory=lambda: ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<end_of_turn>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<end_of_turn>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'])


register_template(GemmaTemplateMeta(LLMTemplateType.gemma))


class PaliGemmaTemplate(Template):
    placeholder_tokens = ['<image>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        if self.mode == 'vllm':
            self.prompt = ['{{QUERY}}']
            return []
        else:
            self.prompt = ['{{QUERY}}\n']
            return ['<image>' * self.processor.image_seq_length + '<bos>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        raw_image = inputs.images
        processor = self.processor
        if encoded['labels'] is not None:
            n = upper_bound(0, len(encoded['labels']), lambda idx: encoded['labels'][idx] == -100)
            n2 = len(encoded['labels']) - n
            encoded['token_type_ids'] = [0] * n + [1] * n2
        else:
            encoded['token_type_ids'] = [0] * len(encoded['input_ids'])
        if raw_image:
            model_inputs = processor(text='<image>' * len(raw_image), images=raw_image, return_tensors='pt')
            encoded['pixel_values'] = model_inputs['pixel_values'].to(self.model_info.torch_dtype)
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.paligemma,
        prefix=[],
        prompt=['{{QUERY}}\n'],
        chat_sep=None,
        suffix=['<eos>'],
        template_cls=PaliGemmaTemplate,
    ))


@dataclass
class Gemma3TextTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<bos>'])
    prompt: Prompt = field(
        default_factory=lambda: ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<end_of_turn>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<end_of_turn>'])


class Gemma3Template(Template):

    def _swift_encode(self, inputs: StdTemplateInputs):
        if inputs.system is not None:
            system = inputs.system
            inputs.system = None
            inputs.messages[0]['content'] = system + '\n\n' + inputs.messages[0]['content']
        for message in inputs.messages:
            if message['role'] == 'assistant' and isinstance(message['content'], str):
                message['content'] = message['content'].strip('\n')
        return super()._swift_encode(inputs)


register_template(Gemma3TextTemplateMeta(LLMTemplateType.gemma3_text, template_cls=Gemma3Template))


class Gemma3VisionTemplate(Gemma3Template):
    boi_token_id = 255999
    placeholder_tokens = ['<start_of_image>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['<start_of_image>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from transformers.models.gemma3.processing_gemma3 import Gemma3ProcessorKwargs

        encoded = super()._encode(inputs)
        if inputs.images:
            input_ids = encoded['input_ids']
            labels = encoded['labels']
            loss_scale = encoded.get('loss_scale', None)
            idx_list = findall(input_ids, self.boi_token_id)
            img_tokens = self._tokenize(self.processor.full_image_sequence)
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                lambda _: img_tokens)

            # TODO: customize
            processor_kwargs = Gemma3ProcessorKwargs._defaults['images_kwargs']
            image_inputs = self.processor.image_processor(inputs.images, **processor_kwargs)
            image_inputs['pixel_values'] = torch.as_tensor(np.array(image_inputs['pixel_values']))
            image_inputs.pop('num_crops')

            array_ids = np.array(input_ids)
            mm_token_type_ids = np.zeros_like(input_ids)
            mm_token_type_ids[array_ids == self.processor.image_token_id] = 1
            encoded['token_type_ids'] = mm_token_type_ids.tolist()
            encoded['input_ids'] = input_ids
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['labels'] = labels
            encoded['loss_scale'] = loss_scale
        return encoded


register_template(GemmaTemplateMeta(MLLMTemplateType.gemma3_vision, template_cls=Gemma3VisionTemplate))


class Gemma3nTemplate(Gemma3Template):
    boi_token_id = 255999
    boa_token_id = 256000
    placeholder_tokens = ['<start_of_image>', '<start_of_audio>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            if self.mode == 'vllm':
                return ['<image_soft_token>']
            else:
                return ['\n\n<start_of_image>']
        elif media_type == 'audio':
            if self.mode == 'vllm':
                raise ValueError('Audio is not supported in vLLM')
            inputs.audios[index] = load_audio(inputs.audios[index], self.processor.feature_extractor.sampling_rate)
            return ['<start_of_audio>']
        else:
            raise ValueError(f'Unsupported media type: {media_type}. Supported types are: image, audio')

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from transformers.models.gemma3n.processing_gemma3n import Gemma3nProcessorKwargs

        # Input validation
        if not inputs.images and not inputs.audios and not inputs.messages:
            raise ValueError('Provide at least one of `images`, `audios`, or `messages`.')

        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        # Initialize token_type_ids and other outputs
        array_ids = np.array(input_ids)
        mm_token_type_ids = np.zeros_like(input_ids)

        # Handle images
        if inputs.images:
            idx_list = findall(input_ids, self.boi_token_id)
            img_tokens = self._tokenize(processor.full_image_sequence[2:])
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                lambda _: img_tokens)

            # Process images
            processor_kwargs = Gemma3nProcessorKwargs._defaults.get('images_kwargs', {})
            image_inputs = processor.image_processor(inputs.images, **processor_kwargs)
            image_inputs['pixel_values'] = torch.as_tensor(
                np.array(image_inputs['pixel_values']), dtype=self.model_info.torch_dtype)
            if 'num_crops' in image_inputs:
                image_inputs.pop('num_crops')
            encoded.update(image_inputs)

        # Handle audios
        if inputs.audios:
            audio_idx_list = findall(input_ids, self.boa_token_id)
            if audio_idx_list:
                # Get audio token sequence from processor
                audio_tokens = self._tokenize(processor.full_audio_sequence)
                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, audio_idx_list,
                                                                    lambda _: audio_tokens)

                # Process audios
                processor_kwargs = Gemma3nProcessorKwargs._defaults.get('audio_kwargs', {})
                audio_inputs = processor.feature_extractor(inputs.audios, **processor_kwargs)

                if 'input_features' in audio_inputs:
                    audio_inputs['input_features'] = torch.tensor(audio_inputs['input_features']).to(
                        self.model_info.torch_dtype)
                if 'input_features_mask' in audio_inputs:
                    audio_inputs['input_features_mask'] = torch.tensor(audio_inputs['input_features_mask'])
                encoded.update(audio_inputs)

        # Update array_ids after token extension
        array_ids = np.array(input_ids)
        mm_token_type_ids = np.zeros_like(input_ids)

        if hasattr(processor, 'image_token_id') and processor.image_token_id is not None:
            mm_token_type_ids[array_ids == processor.image_token_id] = 1

        if hasattr(processor, 'audio_token_id') and processor.audio_token_id is not None:
            mm_token_type_ids[array_ids == processor.audio_token_id] = 3

        encoded['token_type_ids'] = mm_token_type_ids.tolist()
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle multimodal data collation for Gemma3n, including audio features"""
        res = super()._data_collator_mm_data(batch)

        # Handle audio features like other templates do
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        input_features_mask = [b['input_features_mask'] for b in batch if b.get('input_features_mask') is not None]

        if input_features:
            res['input_features'] = torch.concat(input_features)
        if input_features_mask:
            res['input_features_mask'] = torch.concat(input_features_mask)

        return res


register_template(GemmaTemplateMeta(MLLMTemplateType.gemma3n, template_cls=Gemma3nTemplate))


class Gemma4Template(Template):
    placeholder_tokens = ['<|image|>', '<|audio|>', '<|video|>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<|image|>']
        elif media_type == 'audio':
            if self.mode != 'vllm':
                inputs.audios[index] = load_audio(inputs.audios[index], self.processor.feature_extractor.sampling_rate)
            return ['<|audio|>']
        elif media_type == 'video':
            if self.mode == 'vllm':
                from vllm.assets.video import video_get_metadata, video_to_ndarrays
                num_frames = self.processor.video_processor.num_frames
                video_data = video_to_ndarrays(inputs.videos[index], num_frames)
                video_metadatas = video_get_metadata(inputs.videos[index], num_frames)
                inputs.videos[index] = [(video_data, video_metadatas)]
            return ['<|video|>']

    def _get_system(self, inputs: StdTemplateInputs) -> Optional[str]:
        system = super()._get_system(inputs)
        if self._get_enable_thinking(inputs):
            system = '<|think|>\n' + (system or '')
        return system

    def _add_non_thinking_prefix(self, inputs: StdTemplateInputs, thinking_prefix: str = '<|channel>thought'):
        return super()._add_non_thinking_prefix(inputs, thinking_prefix=thinking_prefix)

    def _remove_thinking_content(self, content: str, thinking_suffix: str = '<channel|>') -> str:
        return super()._remove_thinking_content(content, thinking_suffix=thinking_suffix)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        split_token = self._tokenize('\n')
        media_inputs = self.processor(
            text='\n'.join(['<|image|>'] * len(inputs.images) + ['<|video|>'] * len(inputs.videos)
                           + ['<|audio|>'] * len(inputs.audios)),
            audio=inputs.audios or None,
            images=inputs.images or None,
            videos=inputs.videos or None,
            return_tensors='pt',
            add_special_tokens=False,
        )
        splited_tokens = self._split_list(media_inputs['input_ids'][0].tolist(), split_token)
        media_inputs.pop('input_ids')
        media_inputs.pop('attention_mask')
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        mm_mask = [False] * len(input_ids)

        idx_list = []
        for key in ['image', 'video', 'audio']:
            idx_list += findall(input_ids, getattr(self.config, f'{key}_token_id'))
        sorted_order = sorted(range(len(idx_list)), key=lambda i: idx_list[i])
        idx_list = [idx_list[i] for i in sorted_order]
        splited_tokens = [splited_tokens[i] for i in sorted_order]

        def _get_new_tokens(i):
            return splited_tokens[i]

        if idx_list:
            input_ids, labels, loss_scale, mm_mask = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_tokens, mm_mask=mm_mask)
        for key in [
                'pixel_values', 'image_position_ids', 'pixel_values_videos', 'video_position_ids', 'input_features',
                'input_features_mask'
        ]:
            if key in media_inputs:
                encoded[key] = media_inputs[key]
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        encoded['mm_token_type_ids'] = self.create_mm_token_type_ids(input_ids, mm_mask)
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        for key in ['image_position_ids', 'video_position_ids']:
            value = [b[key] for b in batch if b.get(key) is not None]
            if value:
                res[key] = torch.concat(value)
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        if input_features:
            input_features_mask = [b['input_features_mask'] for b in batch if b.get('input_features_mask') is not None]
            max_len = max([x.shape[1] for x in input_features_mask])
            res['input_features'] = torch.concat([F.pad(x, (0, 0, 0, max_len - x.shape[1])) for x in input_features])
            res['input_features_mask'] = torch.concat(
                [F.pad(x, (0, max_len - x.shape[1])) for x in input_features_mask])
        return res


@dataclass
class Gemma4TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<bos>'])
    prompt: Prompt = field(default_factory=lambda: ['<|turn>user\n{{QUERY}}<turn|>\n<|turn>model\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<turn|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<turn|>\n'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<bos><|turn>system\n{{SYSTEM}}<turn|>\n'])


register_template(Gemma4TemplateMeta(MLLMTemplateType.gemma4_nothinking, template_cls=Gemma4Template))

register_template(
    Gemma4TemplateMeta(
        MLLMTemplateType.gemma4,
        template_cls=Gemma4Template,
        is_thinking=True,
        non_thinking_prefix='<|channel>thought\n<channel|>'))
