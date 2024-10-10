import inspect
from contextlib import contextmanager
from functools import partial, wraps
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled

from swift.llm import to_device
from swift.utils import get_dist_setting, use_torchacc
from ._base import Template as _Template
from ._base import TemplateInputs
from .utils import Context, findall


class Template(_Template):
    """This class expands the capabilities of the template for train, vllm_infer, and lmdeploy_infer."""

    def _pre_forward_hook(self, args, kwargs, model):
        res_extra = []
        data = kwargs.pop('_data')
        for d in data:
            res_extra.append(self._post_encode(model, d))
        kwargs.update(to_device(self.data_collator(res_extra), model.device))
        if 'inputs_embeds' in kwargs:
            kwargs.pop('input_ids', None)

        if isinstance(model, PeftModel):
            parameters = inspect.signature(model.base_model.model.forward).parameters
        else:
            parameters = inspect.signature(model.forward).parameters

        if 'position_ids' not in parameters:
            kwargs.pop('position_ids', None)
        return args, kwargs

    @contextmanager
    def vllm_context(self):
        task = self.task
        self.task = 'infer_vllm'
        yield
        self.task = task

    @contextmanager
    def lmdeploy_context(self):
        task = self.task
        self.task = 'infer_lmdeploy'
        yield
        self.task = task

    @contextmanager
    def training_context(self, trainer, models: List[nn.Module]):
        """This function is important for multi-modal training, as it registers the post_encode method
            as a forward hook, converting input_ids into inputs_embeds.
        """
        # TODO:torch>=2.0
        task = self.task
        self.task = 'train'
        if not self.is_multimodal:
            yield
            self.task = task  # recover
            return

        trainer.data_collator = self._pre_data_collator
        handles = []
        for model in models:
            handle = model.register_forward_pre_hook(partial(self._pre_forward_hook, model=model), with_kwargs=True)
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
        trainer.data_collator = self.data_collator
        self.task = task
        for handle in handles:
            handle.remove()
        if _deepspeed_initialize:
            deepspeed.initialize = _deepspeed_initialize

    def _init_template(self,
                       tokenizer: PreTrainedTokenizerBase,
                       *,
                       default_system: Optional[str] = None,
                       sequence_parallel_size: int = 1) -> None:
        self.sequence_parallel_size = sequence_parallel_size
        return super()._init_template(tokenizer, default_system)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: TemplateInputs) -> List[Context]:
        if media_type == 'image' and self.task == 'infer_lmdeploy':
            return [[-100]]
        else:
            return super().replace_tag(media_type, index, inputs)

    def _pre_data_collator(self, batch: List[Dict[str, Any]], *args, **kwargs) -> Dict[str, Any]:
        """for multimodal LLM"""
        assert self.is_multimodal
        new_batch = [{'labels': b['labels']} for b in batch]
        res = self.data_collator(new_batch)  # only labels
        res['_data'] = batch
        return res

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        Args:
            batch(`List[Dict[str, Any]]`): The input data in batch
            padding_to(`int`, optional): Whether padding the batch to a fixed length, if none, the batch
                will be padded to the `longest`
        """
        tokenizer = self.tokenizer
        assert tokenizer.pad_token_id is not None
        padding_right = self.padding_side == 'right'
        res = {}

        if 'inputs_embeds' in batch[0]:
            inputs_embeds = [b['inputs_embeds'] for b in batch]
            res['inputs_embeds'] = inputs_embeds
            res['attention_mask'] = [
                torch.ones((inputs_embeds[i].shape[0]), dtype=torch.int64) for i in range(len(inputs_embeds))
            ]
        elif 'input_ids' in batch[0]:
            input_ids = [torch.tensor(b['input_ids']) for b in batch]
            res['input_ids'] = input_ids
            res['attention_mask'] = [torch.ones(len(input_ids[i]), dtype=torch.int64) for i in range(len(input_ids))]

        for key in ['labels', 'loss_scale', 'position_ids']:
            if key in batch[0]:
                res[key] = [torch.tensor(b[key]) for b in batch]

        if padding_to is not None:
            assert 'input_ids' in res
            padding_len = padding_to - res['input_ids'][0].shape[-1]
            if padding_len > 0:
                for key, value in zip(['input_ids', 'attention_mask', 'labels', 'loss_scale', 'position_ids'],
                                      [tokenizer.pad_token_id, 0, -100, 0., -1]):
                    if key in res:
                        res[key][0] = F.pad(res[key][0], (0, padding_len) if padding_right else (padding_len, 0),
                                            'constant', value)
        for key, value in zip(['input_ids', 'inputs_embeds', 'attention_mask', 'labels', 'loss_scale', 'position_ids'],
                              [tokenizer.pad_token_id, 0., 0, -100, 0., -1]):
            if key in res:
                res[key] = self._pad_sequence(res[key], value, self.padding_side)

        # multimodal
        pixel_values = [b['pixel_values'] for b in batch if b.get('pixel_values') is not None]
        if len(pixel_values) > 0:
            res['pixel_values'] = torch.concat(pixel_values)

            image_sizes = [b['image_sizes'] for b in batch if b.get('image_sizes') is not None]
            if len(image_sizes) > 0:
                res['image_sizes'] = torch.concat(image_sizes)

        pixel_values_videos = [b['pixel_values_videos'] for b in batch if b.get('pixel_values_videos') is not None]
        if len(pixel_values_videos) > 0:
            res['pixel_values_videos'] = torch.concat(pixel_values_videos)

        # torchacc & xtuner
        input_ids = res.get('input_ids')
        attention_mask = res.get('attention_mask')
        labels = res.get('labels')
        loss_scale = res.get('loss_scale')
        if use_torchacc():
            from swift.utils.torchacc_utils import pad_and_split_batch
            rank, _, world_size, _ = get_dist_setting()
            input_ids, attention_mask, labels, loss_scale = pad_and_split_batch(
                padding_to,
                input_ids,
                attention_mask,
                labels,
                loss_scale,
                self.max_length,
                self.tokenizer,
                rank,
                world_size,
                padding_right=padding_right)
        if self.sequence_parallel_size > 1 and input_ids is not None:
            bs, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)
            assert padding_right or bs == 1, 'Sequence parallel only support padding_side=right'
            from swift.trainers.xtuner import get_xtuner_sequence_parallel_world_size
            if get_xtuner_sequence_parallel_world_size() > 1:
                from swift.trainers.xtuner import pad_and_split_for_sequence_parallel
                input_ids, labels, position_ids, attention_mask, loss_scale = \
                    pad_and_split_for_sequence_parallel(
                        tokenizer, input_ids, labels, position_ids, attention_mask, loss_scale)
            res['position_ids'] = position_ids
        _local_var = locals()
        for key in ['input_ids', 'attention_mask', 'labels', 'loss_scale']:
            value = _local_var[key]
            if value is not None:
                res[key] = value
        return res

    async def prepare_lmdeploy_inputs(self, inputs: Dict[str, Any]) -> None:
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

    @staticmethod
    def _pad_sequence(sequences: List[torch.Tensor],
                      padding_value: float = 0.,
                      padding_side: Literal['right', 'left'] = 'right') -> torch.Tensor:
        """Pad sequence by some side

        Args:
            sequences: The input sequences in tensor.
            padding_value: The padding value
            padding_side: The padding side

        Returns:
            A tensor after padding
        """
        padding_right = padding_side == 'right'
        if padding_right:
            return pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        max_len = max([s.size(0) for s in sequences])

        padded_sequences = []
        for seq in sequences:
            pad_length = max_len - seq.size(0)
            pad_tuple = [0] * ((seq.dim() - 1) * 2) + [pad_length, 0]
            padded_seq = F.pad(seq, tuple(pad_tuple), 'constant', padding_value)
            padded_sequences.append(padded_seq)

        return torch.stack(padded_sequences)
