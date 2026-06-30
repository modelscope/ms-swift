# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from swift.utils import get_logger, upper_bound
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, Word, findall
from ..vision_utils import load_audio, load_vllm_video

logger = get_logger()


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
    non_thinking_prefix_only_after_user = True

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
                num_frames = self.processor.video_processor.num_frames
                video_data, video_metadatas = load_vllm_video(inputs.videos[index], num_frames)
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
            token_id = getattr(self.config, f'{key}_token_id', None)
            if token_id is None:
                continue
            idx_list += findall(input_ids, token_id)
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
        # unpad input_features
        # https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/model_executor/models/gemma4_mm.py#L747-L758
        if 'input_features' in encoded and 'input_features_mask' in encoded:
            masks = encoded['input_features_mask']
            features = encoded['input_features']
            if isinstance(masks, torch.Tensor) and masks.ndim >= 2:
                bool_masks = masks.bool()
                encoded['input_features'] = [f[m] for f, m in zip(features, bool_masks)]
                encoded['input_features_mask'] = [m[m] for m in bool_masks]
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
        input_features = []
        input_features_mask = []
        for b in batch:
            feats = b.get('input_features')
            masks = b.get('input_features_mask')
            if feats is None:
                continue
            if isinstance(feats, list):
                input_features.extend(feats)
                input_features_mask.extend(masks)
            else:
                input_features.append(feats)
                input_features_mask.append(masks)
        if input_features:
            max_len = max(x.shape[0] for x in input_features_mask)
            res['input_features'] = torch.concat([F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in input_features])
            res['input_features_mask'] = torch.concat(
                [F.pad(x, (0, max_len - x.shape[0])) for x in input_features_mask])
        return res


@dataclass
class Gemma4TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<bos>'])
    prompt: Prompt = field(default_factory=lambda: ['<|turn>user\n{{QUERY}}<turn|>\n<|turn>model\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<turn|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<turn|>\n'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<bos><|turn>system\n{{SYSTEM}}<turn|>\n'])
    stop_words: List[Word] = field(default_factory=lambda: ['<eos>', '<turn|>', '<|tool_response>'])


register_template(
    Gemma4TemplateMeta(MLLMTemplateType.gemma4_nothinking, template_cls=Gemma4Template, agent_template='gemma4'))

register_template(
    Gemma4TemplateMeta(
        MLLMTemplateType.gemma4,
        template_cls=Gemma4Template,
        agent_template='gemma4',
        is_thinking=True,
        non_thinking_prefix='<|channel>thought\n<channel|>'))


class DiffusionGemmaTemplate(Gemma4Template):
    is_encoder_decoder = True
    skip_prompt = True

    @property
    def loss_scale(self):
        loss_scale = super().loss_scale
        if self.is_training and loss_scale.base_strategy != 'last_round':
            logger.warning_once('DiffusionGemmaTemplate only supports the `last_round` base strategy for loss scaling. '
                                'Setting loss_scale.base_strategy to `last_round`.')
        loss_scale.base_strategy = 'last_round'
        return loss_scale

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        inputs = super()._data_collator(batch, padding_to=padding_to)
        if self.is_training:
            inputs = self._update_inputs(inputs)
        return inputs

    # Code reference: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/DiffusionGemma_(26B-A4B)-Sudoku.ipynb  # noqa
    def _update_inputs(self, inputs):
        canvas_length = self.config.canvas_length
        if inputs['labels'].shape[0] > 1:
            raise ValueError('per_device_train_batch_size must be 1 for diffusion gemma')
        first_idx = (inputs['labels'] != -100).int().argmax().item()
        prompt_ids = inputs['input_ids'][:, :first_idx]
        # reserve one slot at the end of the canvas for the explicit eos token expected by
        # the diffusion sampler as the termination signal.
        response_length = inputs['input_ids'].shape[1] - first_idx
        if response_length > canvas_length - 1:
            raise ValueError(f'response length ({response_length}) exceeds canvas_length-1 ({canvas_length - 1}); '
                             'please use a shorter response or increase canvas_length.')
        canvas_content = inputs['input_ids'][:, first_idx:first_idx + canvas_length - 1]
        # x0: clean canvas padded to canvas_length; loss is only computed on response + eos.
        device = prompt_ids.device
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        x0 = torch.full((prompt_ids.shape[0], canvas_length), pad_token_id, dtype=torch.long, device=device)
        n = canvas_content.shape[1]
        x0[:, :n] = canvas_content
        # explicitly append eos as the canvas-end signal expected by the diffusion sampler.
        # without it, sampler keeps denoising the trailing positions during inference and emits garbage.
        x0[:, n] = eos_token_id
        labels = x0.clone()
        labels[:, n + 1:] = -100

        # forward diffusion: per-sample noise level t ∈ [min, max], replace tokens with random vocab ids
        t = torch.empty((), device=device).uniform_(0.1, 1.)
        noise_mask = torch.rand(canvas_length, device=device) < t
        random_tokens = torch.randint(0, self.config.text_config.vocab_size, (canvas_length, ), device=device)
        decoder_input_ids = torch.where(noise_mask, random_tokens, x0)
        return {'input_ids': prompt_ids, 'decoder_input_ids': decoder_input_ids, 'labels': labels}

    def compute_sft_loss(self, model, inputs: Dict[str, Any], num_items_in_batch: Optional[int] = None, trainer=None):
        if trainer.args.gradient_checkpointing:
            raise ValueError('Gradient checkpointing is not supported for diffusion gemma')
        outputs = model(**inputs)
        logits = outputs.logits.view(-1, outputs.logits.shape[-1])
        labels = inputs['labels'].view(-1)
        outputs.loss = F.cross_entropy(logits, labels, reduction='sum')
        outputs.loss = outputs.loss / num_items_in_batch
        return outputs


register_template(
    Gemma4TemplateMeta(
        MLLMTemplateType.diffusion_gemma,
        template_cls=DiffusionGemmaTemplate,
        agent_template='gemma4',
        is_thinking=True,
        non_thinking_prefix='<|channel>thought\n<channel|>'))
