# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
from torch import nn

from swift.utils import get_env_args
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from ..vision_utils import load_video_minicpmv_mplug_owl3
from .llama import Llama3TemplateMeta
from .qwen import Qwen2_5TemplateMeta, QwenTemplateMeta


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
    use_model = True
    skip_prompt = False
    placeholder_tokens = ['<unk>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            return ['(<image>./</image>)\n']
        else:
            return [[-100]]

    async def prepare_lmdeploy_turbomind_inputs(self, inputs: Dict[str, Any]) -> None:
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
        await super().prepare_lmdeploy_turbomind_inputs(inputs)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, -100)
        idx = idx_list[0]
        tgt_sizes = None
        slice_mode = getattr(self.config, 'slice_mode', False)
        if slice_mode:
            if self.is_v2_5:
                image_processor = self.processor.image_processor
                image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
                placeholder = image_processor.get_slice_image_placeholder(image_inputs.image_sizes[0][0])
                pixel_values = image_inputs['pixel_values']
                tgt_sizes = image_inputs['tgt_sizes']
            else:
                images, placeholder = self.model.get_slice_image_placeholder(images[0], self.processor)
                pixel_values = [[self.model.transform(img) for img in images]]
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
            placeholder = '<image>' + '<unk>' * self.config.query_num + '</image>\n'
            placeholder_id = self.processor.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            image_bound = [torch.tensor([[idx, idx + self.config.query_num]])]
            pixel_values = [[self.model.transform(images[0])]]
        encoded = {
            'input_ids': input_ids,
            'labels': labels,
            'image_bound': image_bound,
            'pixel_values': pixel_values,
            'tgt_sizes': tgt_sizes
        }
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs_embeds, _ = model.get_vllm_embedding(inputs)
        return {'inputs_embeds': inputs_embeds}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = {}
        for k in ['pixel_values', 'image_bound', 'tgt_sizes']:
            res[k] = self.gather_list(batch, k)
        res.update(super()._data_collator(batch, padding_to=padding_to))
        return res


register_template(MinicpmTemplateMeta(MLLMTemplateType.minicpmv, template_cls=MiniCPMVTemplate))


class MiniCPMV2_5Template(MiniCPMVTemplate):
    is_v2_5 = True


register_template(Llama3TemplateMeta(
    MLLMTemplateType.minicpmv2_5,
    template_cls=MiniCPMV2_5Template,
))


class MiniCPMV2_6Template(MiniCPMVTemplate):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type in {'image', 'video'}
        max_num_frames = get_env_args('max_num_frames', int, 64)
        load_video = partial(load_video_minicpmv_mplug_owl3, max_num_frames=max_num_frames)
        image_context = super().replace_tag('image', index, inputs)
        if media_type == 'image':
            return image_context
        elif media_type == 'video':
            return self.replace_video2image(load_video, inputs, lambda i: image_context)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        images = inputs.images
        use_video = bool(inputs.videos)
        use_image_id = True
        max_slice_nums = get_env_args('max_slice_nums', int, None)
        video_max_slice_nums = get_env_args('video_max_slice_nums', int, 1)  # or 2
        if use_video:
            max_slice_nums = video_max_slice_nums
            use_image_id = False
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, -100)

        image_processor = self.processor.image_processor
        image_inputs = image_processor([images], return_tensors='pt',
                                       max_slice_nums=max_slice_nums).to(self.model_info.torch_dtype)

        def _get_new_tokens(i):
            placeholder = image_processor.get_slice_image_placeholder(
                image_inputs.image_sizes[0][i], image_idx=i, max_slice_nums=max_slice_nums, use_image_id=use_image_id)
            placeholder += '\n'
            return self.processor.encode(placeholder, add_special_tokens=False)

        input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)
        if inputs.images:
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

        encoded = {
            'input_ids': input_ids,
            'labels': labels,
            'image_bound': image_bound,
            'pixel_values': image_inputs['pixel_values'],
            'tgt_sizes': image_inputs['tgt_sizes']
        }
        return encoded


register_template(QwenTemplateMeta(
    MLLMTemplateType.minicpmv2_6,
    template_cls=MiniCPMV2_6Template,
))

register_template(Qwen2_5TemplateMeta(
    MLLMTemplateType.minicpmo2_6,
    template_cls=MiniCPMV2_6Template,
))
