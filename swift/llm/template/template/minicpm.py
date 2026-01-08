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
from .qwen import Qwen2_5TemplateMeta, Qwen3MixedTemplateMeta, QwenTemplateMeta
from .utils import ChatmlTemplateMeta


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

    support_padding_free = True

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """重写 _data_collator 以正确处理 packing 场景。
        
        MiniCPMV 的多模态数据格式特殊：
        - pixel_values: list[list[Tensor]] - 双层嵌套，外层对应图像数量，内层对应 slice
        - image_bound: list[Tensor[1, 2]] - 单层列表
        - tgt_sizes: list[Tensor[1, 2]] - 单层列表
        
        这些不能用 torch.concat，需要用 gather_list 保持列表格式。
        """
        if self.padding_free:
            # packing 场景：完全自己处理，避免调用父类的 _data_collator_mm_data
            self._update_dataset_progress(batch)
            
            # 1. 调用 packing_row 合并多个样本
            packed = self.packing_row(batch)
            batch[:] = [packed]
            
            # 2. 构建返回结果
            # 格式要求（与非 packing 一致，只是 batch_size=1）：
            # - input_ids: Tensor[1, seq_len] - 2D
            # - pixel_values: list[list[Tensor]] - 外层 len=1
            # - image_bound: list[Tensor] - 外层 len=1，Tensor 形状 [num_images, 2]
            # - tgt_sizes: list[Tensor] - 总 slice 数
            res = {}
            for k in ['input_ids', 'labels', 'position_ids', 'loss_scale']:
                v = packed.get(k)
                if v is not None:
                    # 转换成 2D Tensor [1, seq_len]
                    t = torch.tensor(v) if isinstance(v, list) else v
                    res[k] = t.unsqueeze(0)
            
            # channel 保持原样
            if packed.get('channel') is not None:
                res['channel'] = packed['channel']
            
            # 3. 多模态数据：包装成 batch_size=1 的格式
            # pixel_values: [内层slices列表] -> [[内层slices列表]]
            pv = packed.get('pixel_values')
            if pv is not None:
                res['pixel_values'] = [pv]  # 外层包一层，len=1
            
            # image_bound: [Tensor1, Tensor2, ...] -> [cat后的单个Tensor]
            ib = packed.get('image_bound')
            if ib is not None and len(ib) > 0:
                res['image_bound'] = [torch.cat(ib, dim=0)]  # 合并成 [N, 2]，外层包一层
            else:
                res['image_bound'] = [torch.empty(0, 2)]
            
            # tgt_sizes: 保持不变（总 slice 数）
            ts = packed.get('tgt_sizes')
            if ts is not None:
                res['tgt_sizes'] = ts
            
            return res
        else:
            # 非 packing 场景，使用原有逻辑
            return super()._data_collator(batch, padding_to=padding_to)

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理 packing 时的多模态数据合并。

        数据格式（单个样本）：
        - input_ids: list[int]
        - labels: list[int]  
        - image_bound: list[Tensor[1, 2]] len=1
        - pixel_values: list[list[Tensor[3, 14, X]]] len=1  # 双层嵌套！
        - tgt_sizes: list[Tensor[1, 2]] len=1
        - length: int
        
        Packing 后需要：
        - 拼接 input_ids, labels
        - 调整 image_bound 的偏移量
        - 合并 pixel_values, tgt_sizes 的外层列表
        """
        packed = {}
        length = []
        for r in row:
            length.append(r['length'])

        # 1. 处理文本相关字段
        for key in ['input_ids', 'labels', 'loss_scale']:
            values = [x.get(key) or [] for x in row]
            if any(values):
                packed[key] = sum(values, start=[])

        packed['length'] = sum(length)

        channels = [x.get('channel') for x in row]
        if any(c is not None for c in channels):
            packed['channel'] = channels

        sources = [x.get('_dataset_source') for x in row]
        if any(s is not None for s in sources):
            packed['_dataset_source'] = sources

        packed['position_ids'] = sum((list(range(x)) for x in length), start=[])

        # 2. 处理 image_bound - 需要根据累积的 token offset 调整索引
        # 输入格式：每个样本 image_bound = [Tensor[1, 2]]，长度为 1
        # 输出格式：合并后 image_bound = [Tensor[1, 2], Tensor[1, 2], ...]
        image_bounds = []
        offset = 0
        for r in row:
            bounds = r.get('image_bound')
            if bounds is not None:
                for bound in bounds:
                    if isinstance(bound, torch.Tensor) and bound.numel() > 0:
                        image_bounds.append(bound + offset)
                    elif isinstance(bound, (list, tuple)) and len(bound) > 0:
                        image_bounds.append(torch.tensor(bound) + offset)
            offset += r['length']
        packed['image_bound'] = image_bounds

        # 3. 拼接 pixel_values
        # 输入格式：每个样本 pixel_values = [[Tensor[3, 14, X]]]，双层嵌套，外层len=1
        # 输出格式：合并后 pixel_values = [Tensor1, Tensor2, ...]（扁平列表）
        # 然后在 _data_collator 中包装成 [[T1, T2, ...]]
        pixel_values = []
        for r in row:
            pv = r.get('pixel_values')
            if pv is not None:
                # pv 是 [[Tensor, ...]]，需要展开两层收集所有 Tensor
                for inner_list in pv:
                    pixel_values.extend(inner_list)
        packed['pixel_values'] = pixel_values

        # 4. 拼接 tgt_sizes  
        # 输入格式：每个样本 tgt_sizes = [Tensor[1, 2]]
        # 输出格式：合并后 tgt_sizes = [Tensor, Tensor, ...]
        tgt_sizes = []
        for r in row:
            ts = r.get('tgt_sizes')
            if ts is not None:
                tgt_sizes.extend(ts)
        packed['tgt_sizes'] = tgt_sizes

        return packed

    def init_env_args(self):
        super().init_env_args()
        self.max_num_frames = get_env_args('max_num_frames', int, 64)
        self.max_slice_nums = get_env_args('max_slice_nums', int, None)
        self.video_max_slice_nums = get_env_args('video_max_slice_nums', int, 1)  # or 2

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type in {'image', 'video'}
        load_video = partial(load_video_minicpmv_mplug_owl3, max_num_frames=self.max_num_frames)
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
        max_slice_nums = self.max_slice_nums
        if use_video:
            max_slice_nums = self.video_max_slice_nums
            use_image_id = False
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        idx_list = findall(input_ids, -100)

        image_processor = self.processor.image_processor
        image_inputs = image_processor([images], return_tensors='pt',
                                       max_slice_nums=max_slice_nums).to(self.model_info.torch_dtype)

        def _get_new_tokens(i):
            placeholder = image_processor.get_slice_image_placeholder(
                image_inputs.image_sizes[0][i], image_idx=i, max_slice_nums=max_slice_nums, use_image_id=use_image_id)
            placeholder += '\n'
            return self.processor.encode(placeholder, add_special_tokens=False)

        input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list, _get_new_tokens)

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
            'loss_scale': loss_scale,
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

register_template(ChatmlTemplateMeta(
    MLLMTemplateType.minicpmv4,
    template_cls=MiniCPMV2_6Template,
))


class MiniCPMV4_5Template(MiniCPMV2_6Template):

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理 packing 时的多模态数据合并。

        在 MiniCPMV2_6Template 基础上，额外处理 temporal_ids 字段（视频场景）。
        """
        packed = super().packing_row(row)

        # 处理 temporal_ids（视频场景）
        temporal_ids = []
        for r in row:
            tid = r.get('temporal_ids')
            if tid is not None:
                temporal_ids.extend(tid)
        packed['temporal_ids'] = temporal_ids

        return packed

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        images = inputs.images
        use_video = bool(inputs.videos)
        use_image_id = True
        max_slice_nums = self.max_slice_nums
        if use_video:
            max_slice_nums = self.video_max_slice_nums
            use_image_id = False
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        idx_list = findall(input_ids, -100)

        image_processor = self.processor.image_processor
        image_inputs = image_processor([images], return_tensors='pt',
                                       max_slice_nums=max_slice_nums).to(self.model_info.torch_dtype)

        def _get_new_tokens(i):
            placeholder = image_processor.get_slice_image_placeholder(
                image_inputs.image_sizes[0][i], image_idx=i, max_slice_nums=max_slice_nums, use_image_id=use_image_id)
            placeholder += '\n'
            return self.processor.encode(placeholder, add_special_tokens=False)

        input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list, _get_new_tokens)

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
            'loss_scale': loss_scale,
            'image_bound': image_bound,
            'pixel_values': image_inputs['pixel_values'],
            'tgt_sizes': image_inputs['tgt_sizes'],
            'temporal_ids': image_inputs['temporal_ids'],
        }
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = {}
        for k in ['pixel_values', 'image_bound', 'tgt_sizes', 'temporal_ids']:
            res[k] = self.gather_list(batch, k)
        res.update(Template._data_collator(self, batch, padding_to=padding_to))
        return res


register_template(
    Qwen3MixedTemplateMeta(
        MLLMTemplateType.minicpmv4_5,
        template_cls=MiniCPMV4_5Template,
        is_thinking=True,
        thinking_prefix='<think>\n',
    ))
