# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Literal, Optional, Tuple

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Context, findall, gather_list
from .utils import DEFAULT_SYSTEM

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
