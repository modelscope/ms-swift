# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from typing import Any, Dict, List, Literal

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context
from ..vision_utils import load_video_hf


class Idefics3Template(Template):
    """Idefics3, SmolVLM and SmolVLM2. Each `<image>`/`<video>` is expanded by the HF processor, in order."""
    placeholder_tokens = ['<image>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode in {'vllm', 'lmdeploy'} or media_type not in {'image', 'video'}:
            return super().replace_tag(media_type, index, inputs)
        processor = self.processor
        if media_type == 'image':
            media_inputs = processor(
                text='<image>', images=[[inputs.images[index]]], return_tensors='pt', add_special_tokens=False)
        else:
            videos, video_metadata = load_video_hf([inputs.videos[index]])
            video_processor = processor.video_processor
            # Same frame sampling as the processor's chat template.
            media_inputs = processor(
                text='<video>',
                videos=[videos],
                video_metadata=[video_metadata],
                do_sample_frames=True,
                num_frames=video_processor.num_frames,
                fps=video_processor.fps,
                return_tensors='pt',
                add_special_tokens=False)
        if not hasattr(inputs, '_idefics3_media'):
            inputs._idefics3_media = []
        inputs._idefics3_media.append(media_inputs)
        input_ids = media_inputs['input_ids'][0].tolist()
        if media_type == 'video':
            # The video prompt starts and ends with plain text ("You are provided ...", "\n\n"), which the
            # processor tokenizes together with the surrounding text, so return it as text to be merged.
            return [processor.tokenizer.decode(input_ids)]
        return [input_ids]

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs):
        super()._swift_prepare_inputs(inputs)
        if self.template_backend != 'swift':
            return
        # The chat template writes `Role: text`, but `Role:<image>` when the turn starts with an image.
        # The generation prompt ends with `Assistant:`, so the space belongs to the response.
        for message in inputs.messages:
            content = message['content']
            if message['role'] not in {'user', 'assistant', 'tool'}:
                continue
            # Consecutive assistant messages are merged into a list.
            first = content[0] if isinstance(content, list) and content else content
            if not isinstance(first, str):
                continue
            if message['role'] == 'assistant' or not first.startswith('<image>'):
                first = ' ' + first
                if isinstance(content, list):
                    content[0] = first
                else:
                    message['content'] = first

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        inputs._idefics3_media = []
        encoded = super()._encode(inputs)
        media_inputs = inputs._idefics3_media
        if media_inputs:
            # Shape (1, num_patches, 3, H, W), in order of appearance.
            encoded['pixel_values'] = torch.concat([x['pixel_values'] for x in media_inputs], dim=1)
            encoded['pixel_attention_mask'] = torch.concat([x['pixel_attention_mask'] for x in media_inputs], dim=1)
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = {'pixel_values', 'pixel_attention_mask'}
        res = super()._data_collator_mm_data([{k: v for k, v in b.items() if k not in keys} for b in batch])
        pixel_values = [b['pixel_values'] for b in batch if b.get('pixel_values') is not None]
        if pixel_values:
            # Samples have different numbers of patches; pad with all-zero patches, which the model drops.
            max_num = max(x.shape[1] for x in pixel_values)
            res['pixel_values'] = torch.concat(
                [torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 0, 0, max_num - x.shape[1])) for x in pixel_values])
            masks = [b['pixel_attention_mask'] for b in batch if b.get('pixel_values') is not None]
            res['pixel_attention_mask'] = torch.concat(
                [torch.nn.functional.pad(x, (0, 0, 0, 0, 0, max_num - x.shape[1])) for x in masks])
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.idefics3,
        prefix=[['bos_token_id']],
        prompt=['User:{{QUERY}}<end_of_utterance>\nAssistant:'],
        chat_sep=['<end_of_utterance>\n'],
        suffix=['<end_of_utterance>'],
        system_prefix=[['bos_token_id'], 'System: {{SYSTEM}}<end_of_utterance>\n'],
        template_cls=Idefics3Template,
    ))
