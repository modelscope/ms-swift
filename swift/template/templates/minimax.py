# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from swift.utils import get_env_args, get_logger
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall

logger = get_logger()


@dataclass
class MinimaxTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: [
        '<beginning_of_sentence>user name=user\n{{QUERY}}<end_of_sentence>\n'
        '<beginning_of_sentence>ai name=assistant\n'
    ])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<end_of_sentence>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<end_of_sentence>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<beginning_of_sentence>system ai_setting=assistant\n{{SYSTEM}}<end_of_sentence>\n'])


register_template(MinimaxTemplateMeta(LLMTemplateType.minimax))

register_template(
    MinimaxTemplateMeta(
        LLMTemplateType.minimax_m1,
        prefix=['<begin_of_document>'],
        system_prefix=[
            '<begin_of_document><beginning_of_sentence>system ai_setting=assistant\n{{SYSTEM}}<end_of_sentence>\n'
        ],
    ))


class MinimaxVLTemplate(Template):
    image_placeholder = ['<image>']
    skip_prompt = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return self.image_placeholder * inputs.all_image_tokens[index]

    def calc_num_image_tokens(self, image_inputs):
        from transformers.image_utils import get_image_size, to_numpy_array
        pixel_values = image_inputs['pixel_values']
        image_sizes = image_inputs['image_sizes']
        all_image_tokens = []
        if not image_inputs:
            return all_image_tokens

        if self.processor.process_image_mode == 'anyres':
            for pixel_value, image_size in zip(pixel_values, image_sizes):
                height, width = image_size
                num_image_tokens = self.processor.get_num_token(height, width, self.processor.grid_pinpoints,
                                                                self.processor.patch_size)
                all_image_tokens.append(num_image_tokens)
        elif self.processor.process_image_mode == 'resize':
            pixel_values = image_inputs['pixel_values']
            all_image_tokens = []
            for pixel_value in pixel_values:
                height, width = get_image_size(to_numpy_array(pixel_value))
                all_image_tokens.append(int(height * width / self.processor.patch_size**2))
        else:
            if self.processor.patch_size is not None:
                pixel_values = image_inputs['pixel_values']
                all_image_tokens = []
                for pixel_value in pixel_values:
                    height, width = get_image_size(to_numpy_array(pixel_value))
                    new_width, new_height = self.processor.get_hw_multiple_of(
                        (width, height), self.processor.patch_size, self.processor.max_size)
                    num_image_tokens = ((new_height // self.processor.patch_size) *
                                        (new_width // self.processor.patch_size))  # + 1
                    all_image_tokens.append(num_image_tokens)
            else:
                logger.warning_once(
                    'Expanding inputs for image tokens in MiniMaxVL01 should be done in processing. '
                    "Please add `patch_size` and `vision_feature_select_strategy` to the model's "
                    'processing config or set directly '
                    'with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = '
                    '{{vision_feature_select_strategy}}`. '
                    'Using processors without these attributes in the config is deprecated '
                    'and will throw an error in v4.47.')
                raise ValueError(
                    "You need to provide `patch_size` and `vision_feature_select_strategy` in the model's processing "
                    'config to expand inputs for image tokens.')
        return all_image_tokens

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        output_kwargs = self.processor._merge_kwargs(
            self.processor.MiniMaxVL01ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        )
        if inputs.images:
            image_inputs = self.processor.image_processor(
                inputs.images, **output_kwargs['images_kwargs'], return_tensors='pt')
            inputs.all_image_tokens = self.calc_num_image_tokens(image_inputs)
        else:
            image_inputs = {}
        encoded = super()._encode(inputs)
        for key in image_inputs:
            encoded[key] = image_inputs[key]
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        pixel_values = self.gather_list(batch, 'pixel_values')
        image_sizes = self.gather_list(batch, 'image_sizes')
        res = super()._data_collator(batch, padding_to=padding_to)
        if pixel_values:
            res['pixel_values'] = pixel_values
        if image_sizes:
            res['image_sizes'] = image_sizes
        return res


register_template(MinimaxTemplateMeta(LLMTemplateType.minimax_vl, template_cls=MinimaxVLTemplate))


@dataclass
class MinimaxM2TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: [']~!b[]~b]system\n{{SYSTEM}}[e~[\n'])
    prompt: Prompt = field(default_factory=lambda: [']~b]user\n{{QUERY}}[e~[\n]~b]ai\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['[e~[\n'])
    suffix: Prompt = field(default_factory=lambda: ['[e~[\n'])
    agent_template: Optional[str] = 'minimax_m2'
    is_thinking: bool = True
    thinking_prefix: str = '<think>\n'


register_template(
    MinimaxM2TemplateMeta(
        LLMTemplateType.minimax_m2,
        default_system='You are MiniMax-M2, a helpful AI assistant built by MiniMax. Knowledge cutoff: 2025-06.',
    ))

register_template(
    MinimaxM2TemplateMeta(
        LLMTemplateType.minimax_m2_1,
        default_system='You are a helpful assistant. Your name is MiniMax-M2.1 and is built by MiniMax.',
    ))

register_template(
    MinimaxM2TemplateMeta(
        LLMTemplateType.minimax_m2_5,
        default_system='You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax.',
    ))

register_template(
    MinimaxM2TemplateMeta(
        LLMTemplateType.minimax_m2_7,
        default_system='You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.',
    ))

# ===================== MiniMax-M3 VL =====================
# Reference: tokenizer_config.jinja shipped with MiniMax/MiniMax-M3.
# The chat template renders two system blocks:
# 1) high-priority `system` block (MiniMax model identity + thinking instructions)
# 2) low-priority `developer` block (user-provided system text + tools)
# In SWIFT we keep the high-priority block inside prefix/system_prefix,
# and route user-provided system / tools into the developer slot.
# The thinking_mode is dynamic and controlled via chat_template_kwargs.

_MINIMAX_M3_IDENTITY = ('Your model version is MiniMax-M3, developed by MiniMax. Knowledge cutoff: January 2026. '
                        'Founded in early 2022, MiniMax is a global AI foundation model company committed to '
                        'advancing the frontiers of AI towards AGI.')

_MINIMAX_M3_THINKING_BASE = (
    'You have a thinking capability that allows you to reason step by step before responding. '
    'When thinking is enabled, wrap your reasoning in <mm:think></mm:think> tags before your '
    'response. When thinking is disabled, begin your response directly after the </mm:think> '
    'prefix. When thinking is adaptive, decide on your own whether to think for the current turn.')

_MINIMAX_M3_THINKING_MODE_TEXT = {
    'enabled': ('Current thinking mode: enabled. You MUST think step by step before every response, '
                'including after receiving function/tool results.'),
    'disabled':
    'Current thinking mode: disabled. Do not output any thinking process.',
    'adaptive': ('Current thinking mode: adaptive. You are encouraged to think for complex '
                 'decision-making, multi-step reasoning, or when analyzing function/tool results.'),
}

_MINIMAX_M3_DEFAULT_DEVELOPER = 'You are a helpful assistant.'


def _build_m3_system_block(thinking_mode: str = 'adaptive') -> str:
    mode_text = _MINIMAX_M3_THINKING_MODE_TEXT.get(thinking_mode, _MINIMAX_M3_THINKING_MODE_TEXT['adaptive'])
    return (f'{_MINIMAX_M3_IDENTITY}'
            f'\n\n<thinking_instructions>\n{_MINIMAX_M3_THINKING_BASE}\n{mode_text}\n</thinking_instructions>')


def _build_m3_system_prefix(thinking_mode: str = 'adaptive') -> str:
    return f']~!b[]~b]system\n{_build_m3_system_block(thinking_mode)}[e~[\n]~b]developer\n'


class MinimaxM3VLTemplate(Template):
    image_token = ']<]image[>['
    video_token = ']<]video[>['
    placeholder_tokens = [']<]image[>[', ']<]video[>[']
    use_model = True

    def init_env_args(self):
        super().init_env_args()
        # thinking_mode: "enabled" / "disabled" / "adaptive"
        self.thinking_mode = get_env_args('thinking_mode', str, 'adaptive')
        self.chat_template_kwargs['thinking_mode'] = self.thinking_mode
        # Map thinking_mode to enable_thinking for the broader framework
        if self.thinking_mode == 'disabled':
            self.enable_thinking = False
        else:
            self.enable_thinking = True

    def _get_thinking_mode(self, inputs=None) -> str:
        thinking_mode = None if inputs is None else inputs.chat_template_kwargs.get('thinking_mode')
        if thinking_mode is None:
            thinking_mode = self.chat_template_kwargs.get('thinking_mode', 'adaptive')
        return thinking_mode

    def _get_enable_thinking(self, inputs=None):
        thinking_mode = self._get_thinking_mode(inputs)
        return thinking_mode != 'disabled'

    def _get_response_prefix(self, inputs=None):
        # Check explicit override first
        response_prefix = None if inputs is None else inputs.chat_template_kwargs.get('response_prefix')
        if response_prefix is not None:
            return response_prefix
        if self.response_prefix is not None:
            return self.response_prefix
        thinking_mode = self._get_thinking_mode(inputs)
        if thinking_mode == 'enabled':
            return self.template_meta.thinking_prefix  # '<mm:think>'
        elif thinking_mode == 'disabled':
            return self.template_meta.non_thinking_prefix  # '</mm:think>'
        else:  # adaptive
            return ''  # No prefix, let model decide

    def _swift_encode(self, inputs: StdTemplateInputs):
        # Dynamically build prefix with the correct thinking_mode text
        thinking_mode = self._get_thinking_mode(inputs)
        system_prefix_str = _build_m3_system_prefix(thinking_mode)
        # Temporarily patch template_meta prefix/system_prefix
        orig_prefix = self.template_meta.prefix
        orig_system_prefix = self.template_meta.system_prefix
        self.template_meta.prefix = [system_prefix_str + _MINIMAX_M3_DEFAULT_DEVELOPER + '[e~[\n']
        self.template_meta.system_prefix = [system_prefix_str + '{{SYSTEM}}[e~[\n']
        try:
            return super()._swift_encode(inputs)
        finally:
            self.template_meta.prefix = orig_prefix
            self.template_meta.system_prefix = orig_system_prefix

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return [self.image_token]
        elif media_type == 'video':
            return [self.video_token]
        else:
            raise ValueError(f'Unsupported media type for MiniMax-M3 VL: {media_type}')

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if not inputs.images and not inputs.videos:
            return encoded
        # Render media placeholders through the official processor so that the
        # vision_start / vision_end wrapping tokens (and per-frame timestamps for
        # videos) are produced by the same code path used at inference time.
        media_text_parts = ([self.image_token] * len(inputs.images) + [self.video_token] * len(inputs.videos))
        media_inputs = self.processor(
            text='\n'.join(media_text_parts),
            images=inputs.images or None,
            videos=inputs.videos or None,
            return_tensors='pt',
        )
        split_token = self._tokenize('\n')
        splited_tokens = self._split_list(media_inputs['input_ids'][0].tolist(), split_token)
        media_inputs.pop('input_ids', None)
        media_inputs.pop('attention_mask', None)

        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_token)
        idx_list = findall(input_ids, [image_token_id, video_token_id])

        def _get_new_tokens(i):
            return splited_tokens[i]

        if idx_list:
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_tokens)
        for key in ['pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw']:
            if key in media_inputs:
                value = media_inputs[key]
                if key == 'pixel_values' or key == 'pixel_values_videos':
                    value = value.to(self.model_info.torch_dtype)
                encoded[key] = value
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        for key in ['image_grid_thw', 'video_grid_thw']:
            value = [b[key] for b in batch if b.get(key) is not None]
            if value:
                res[key] = torch.concat(value)
        return res


@dataclass
class MinimaxM3VLTemplateMeta(TemplateMeta):
    prefix: Prompt = field(
        default_factory=lambda: [_build_m3_system_prefix('adaptive') + _MINIMAX_M3_DEFAULT_DEVELOPER + '[e~[\n'])
    prompt: Prompt = field(default_factory=lambda: [']~b]user\n{{QUERY}}[e~[\n]~b]ai\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['[e~[\n'])
    suffix: Prompt = field(default_factory=lambda: ['[e~[\n'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: [_build_m3_system_prefix('adaptive') + '{{SYSTEM}}[e~[\n'])
    agent_template: Optional[str] = 'minimax_m3'
    is_thinking: bool = True
    thinking_prefix: str = '<mm:think>'
    non_thinking_prefix: str = '</mm:think>'
    history_thinking_prefix: str = '</mm:think>'


register_template(MinimaxM3VLTemplateMeta(MLLMTemplateType.minimax_m3_vl, template_cls=MinimaxM3VLTemplate))
