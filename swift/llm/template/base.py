from ._base import Template as _Template

from .utils import Context, findall
from swift.llm import to_device
import inspect
from contextlib import contextmanager
from typing import List, Any, Dict
from functools import partial
import torch.nn as nn

class Template(_Template):

    def _pre_forward_hook(self, model, args, kwargs):
        if '_data' in kwargs:
            res_extra = []
            data = kwargs.pop('_data')
            for d in data:
                res_extra.append(self.post_encode(model, d))
            kwargs.update(to_device(self.data_collator(res_extra), model.device))
            if 'inputs_embeds' in kwargs:
                kwargs.pop('input_ids', None)

        parameters = inspect.signature(model.forward).parameters
        if 'position_ids' not in parameters:
            kwargs.pop('position_ids', None)
        return args, kwargs

    @contextmanager
    def training_context(models: List[nn.Module], templates: List['Template']):
        """This function is important for multi-modal training
            Some models need to convert or generate input_embeds before forward, and this part need gradients also.
            So this must happens after the template.encode and data_collator, and befores the forward operation.
        Args:
            models: List of Modules
        """
        handles = []
        for model, x in zip(models, templates):
            parameters = inspect.signature(model.register_forward_pre_hook).parameters
            if 'with_kwargs' in parameters:
                handle = model.register_forward_pre_hook(partial(_pre_forward_hook, template=template),
                                                         with_kwargs=True)
                handles.append(handle)

        _deepspeed_initialize = None
        if is_deepspeed_zero3_enabled():
            import deepspeed
            _deepspeed_initialize = deepspeed.initialize

            @wraps(_deepspeed_initialize)
            def _initialize(*args, **kwargs):
                res = _deepspeed_initialize(*args, **kwargs)
                for model, handle in zip(models, handles):
                    model._forward_pre_hooks.move_to_end(handle.id)
                return res

            deepspeed.initialize = _initialize
        yield
        for handle in handles:
            handle.remove()
        if _deepspeed_initialize:
            deepspeed.initialize = _deepspeed_initialize

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = Template._encode(self.template, example)
        if self.framework in ('vllm', 'lmdeploy'):
            inputs['images'] = example.get('images')
        return inputs, tokenizer_kwargs

    def check_example(self, example):
        if self.template.name in ('minicpm-v-v2_5', 'minicpm-v-v2_6', 'qwen-vl') and self.framework in ('vllm',
                                                                                                        'lmdeploy'):
            return
        return self.template.check_example(example)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        if media_type == 'image' and self.framework == 'lmdeploy':
            return [[-100]]
        if self.template.template_type == 'qwen-vl':
            if self.framework == 'lmdeploy':
                return [f'Picture {index + 1}: ', [-100], '\n']
            if self.framework == 'vllm':
                return [f'Picture {index + 1}: <img></img>\n']
        if 'internvl' in self.template.template_type:
            if self.framework == 'vllm':
                return ['<img><image></img>\n']
        if self.template.template_type == 'llava-yi':
            if self.framework == 'vllm':
                return [[64000], '\n']
        if self.template.template_type == 'paligemma':
            if self.framework == 'vllm':
                self.template.prompt = ['{{QUERY}}']
                return []
        if self.template.template_type == 'phi3-vl':
            if self.framework == 'vllm':
                return [f'<|image_{index + 1}|>\n']  # <|image_1|>\n
        if self.template.template_type in ('minicpm-v-v2_5', 'minicpm-v-v2_6'):
            if self.framework == 'vllm':
                return ['(<image>./</image>)\n']
        return self.template.replace_tag(media_type, index, example)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.template, name)

    async def _minicpm_v_prepare_lmdeploy_inputs(self, inputs: Dict[str, Any]) -> None:
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

    async def prepare_lmdeploy_inputs(self, inputs: Dict[str, Any]) -> None:
        if self.template.template_type == ('minicpm-v-v2_5', 'minicpm-v-v2_6'):
            await self._minicpm_v_prepare_lmdeploy_inputs(inputs)
        images = inputs.pop('images', None) or []
        if len(images) == 0:
            return
        from lmdeploy.vl.constants import IMAGE_DUMMY_TOKEN_INDEX
        input_ids = inputs['input_ids']
        idx_list = findall(input_ids, -100)
        assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
        idx_list.insert(0, -1)
        new_input_ids = []
        ranges = []
        for i in range(len(idx_list) - 1):
            _range = []
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            _range.append(len(new_input_ids))
            new_input_ids += [IMAGE_DUMMY_TOKEN_INDEX] * images[i].shape[0]
            _range.append(len(new_input_ids))
            ranges.append(_range)
        new_input_ids += input_ids[idx_list[-1] + 1:]
        inputs['input_embeddings'] = images
        inputs['input_embedding_ranges'] = ranges
        inputs['input_ids'] = new_input_ids
