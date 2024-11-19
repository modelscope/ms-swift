# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch import nn

from swift.utils import get_env_args
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall, gather_list
from ..vision_utils import load_video_minicpmv_mplug_owl3, replace_video2image
from .llama import Llama3TemplateMeta
from .qwen import QwenTemplateMeta
from .utils import DEFAULT_SYSTEM


@dataclass
class MinicpmTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<s>{{SYSTEM}}'])
    prompt: Prompt = field(default_factory=lambda: ['<用户>{{QUERY}}<AI>'])
    chat_sep: Optional[Prompt] = field(default_factory=list)
    suffix: Prompt = field(default_factory=lambda: ['</s>'])


register_template(MinicpmTemplateMeta(LLMTemplateType.minicpm))


def _remove_idx(arr: List[int], idx_list: List[int]) -> List[int]:
    res = []
    idx_set = set(idx_list)
    for i, x in enumerate(arr):
        if i not in idx_set:
            res.append(x)
    return res


class MiniCPMVTemplate(Template):
    is_v2_5 = False

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            return ['(<image>./</image>)\n']
        else:
            return [[-100]]

    def _check_inputs(self, inputs):
        images = inputs.images or []
        if self.mode not in ('vllm', 'lmdeploy'):
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

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        inputs = super()._encode(inputs)
        if len(inputs) == 0:
            return inputs
        images = inputs.images
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = findall(input_ids, -100)
        idx = idx_list[0]
        config = model.config
        tgt_sizes = None
        slice_mode = getattr(config, 'slice_mode', False)
        if slice_mode:
            if self.is_v2_5:
                image_processor = self.processor.image_processor
                image_inputs = image_processor(images, return_tensors='pt').to(model.dtype)
                placeholder = image_processor.get_slice_image_placeholder(image_inputs.image_sizes[0][0])
                pixel_values = image_inputs['pixel_values']
                tgt_sizes = image_inputs['tgt_sizes']
            else:
                images, placeholder = model.get_slice_image_placeholder(images[0], self.processor)
                pixel_values = [[model.transform(img) for img in images]]
            placeholder += '\n'
            placeholder_id = self.processor.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            input_tensor_ids = torch.tensor(input_ids)
            image_start_idx = torch.where(input_tensor_ids == self.processor.im_start_id)[0]
            image_start_idx += 1
            image_end_idx = torch.where(input_tensor_ids == self.processor.im_end_id)[0]
            valid_image_nums = max(len(image_start_idx), len(image_end_idx))
            image_bound = [
                torch.hstack(
                    [image_start_idx[:valid_image_nums].unsqueeze(-1), image_end_idx[:valid_image_nums].unsqueeze(-1)])
            ]
        else:
            placeholder = '<image>' + '<unk>' * config.query_num + '</image>\n'
            placeholder_id = self.processor.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            image_bound = [torch.tensor([[idx, idx + config.query_num]])]
            pixel_values = [[model.transform(images[0])]]
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
        return inputs

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs_embeds, _ = model.get_vllm_embedding(inputs)
        return {'inputs_embeds': inputs_embeds[0]}


register_template(MinicpmTemplateMeta(MLLMTemplateType.minicpmv, template_cls=MiniCPMVTemplate))


class MiniCPMV2_5Template(MiniCPMVTemplate):
    is_v2_5 = True


register_template(Llama3TemplateMeta(MLLMTemplateType.minicpmv2_5, template_cls=MiniCPMV2_5Template))


class MiniCPMV2_6Template(MiniCPMVTemplate):

    def _check_inputs(self, inputs):
        pass

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type in {'image', 'video'}
        max_num_frames = get_env_args('max_num_frames', int, 64)
        load_video = partial(load_video_minicpmv_mplug_owl3, max_num_frames=max_num_frames)
        image_context = super().replace_tag('image', index, example)
        if media_type == 'image':
            return image_context
        elif media_type == 'video':
            return replace_video2image(load_video, example, lambda i: image_context)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        inputs, _ = Template._encode(self, inputs)
        if len(inputs) == 0:
            return inputs
        images = inputs.images
        use_video = bool(inputs.videos)
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

        image_processor = self.processor.image_processor
        image_inputs = image_processor([images], return_tensors='pt', max_slice_nums=max_slice_nums).to(model.dtype)

        res_input_ids = []
        res_labels = []
        for i in range(len(idx_list) - 1):
            placeholder = image_processor.get_slice_image_placeholder(
                image_inputs.image_sizes[0][i], image_idx=i, max_slice_nums=max_slice_nums, use_image_id=use_image_id)
            placeholder += '\n'
            placeholder_id = self.processor.encode(placeholder, add_special_tokens=False)
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
            unk_token = self.processor.encode('<unk>', add_special_tokens=False)[0]
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
        return inputs


register_template(QwenTemplateMeta(MLLMTemplateType.minicpmv2_6, template_cls=MiniCPMV2_6Template))