import inspect
from contextlib import contextmanager
from functools import partial, wraps
from typing import Any, Dict, List, Literal, Optional, Union

import accelerate
import torch
import transformers
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedTokenizerBase, trainer
from transformers.integrations import is_deepspeed_zero3_enabled

from swift import get_logger
from swift.llm import Template
from swift.llm.utils import to_device
from swift.utils import get_dist_setting, is_ddp_plus_mp, use_torchacc
from swift.utils.torch_utils import _get_max_memory, _sync_max_memory

logger = get_logger()


def patch_ddp_mp():
    """Patch ddp with device_map.
    After patching, the ddp can run with the device_map.
    This should be called before any training starts.
    """
    if is_ddp_plus_mp():
        from accelerate.utils.modeling import get_balanced_memory, infer_auto_device_map

        @wraps(infer_auto_device_map)
        def _infer_auto_device_map_patch(model: Module,
                                         max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
                                         **kwargs) -> Dict[str, Union[int, str, torch.device]]:
            """The auxiliary function for supports DDP+MP. Monkey Patching.
            add feat in accelerate to support DDP + MP"""
            verbose = kwargs.pop('verbose', False)
            n_gpu = torch.cuda.device_count()
            _, local_rank, _, local_world_size = get_dist_setting()
            device_ids = list(range(local_rank, n_gpu, local_world_size))
            max_memory = _get_max_memory(device_ids)
            max_memory = _sync_max_memory(max_memory)
            max_memory = get_balanced_memory(model, max_memory, low_zero=False, **kwargs)
            max_memory = {k: v for k, v in max_memory.items() if v > 0}
            return infer_auto_device_map(model, max_memory, verbose=verbose, **kwargs)

        _old_ddp_init = DDP.__init__
        accelerate.accelerator.torch.nn.parallel.DistributedDataParallel.__init__ = (
            lambda self, model, device_ids, output_device, *args, **kwargs: _old_ddp_init(self, model, *args, **kwargs))
        transformers.modeling_utils.get_balanced_memory = lambda *args, **kwargs: None
        transformers.modeling_utils.infer_auto_device_map = _infer_auto_device_map_patch

    if is_ddp_plus_mp() or use_torchacc():
        _old_accelerator_init = trainer.Accelerator.__init__
        trainer.Accelerator.__init__ = (lambda self, device_placement=False, *args, **kwargs: _old_accelerator_init(
            self, device_placement=device_placement, *args, **kwargs))
        trainer.Accelerator.verify_device_map = lambda *args, **kwargs: False


class TrainTemplate:

    def __init__(self, template: Template, **kwargs):
        self.template = template
        self.sequence_parallel_size = 1

    def init_template(self,
                      tokenizer: PreTrainedTokenizerBase,
                      default_system: Optional[str] = None,
                      max_length: Optional[int] = None,
                      truncation_strategy: Literal['delete', 'truncation_left'] = 'delete',
                      loss_scale: str = 'default',
                      rescale_image: int = -1,
                      **kwargs) -> None:
        self.sequence_parallel_size = kwargs.pop('sequence_parallel_size', 1)
        return self.template.init_template(tokenizer, default_system, max_length, truncation_strategy, loss_scale,
                                           rescale_image, **kwargs)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.template, name)

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        padding_right = self.template.padding_side == 'right'
        res = self.template.data_collator(batch, padding_to)
        input_ids = res.get('input_ids')
        attention_mask = res.get('attention_mask')
        labels = res.get('labels')
        loss_scale = res.get('loss_scale')
        if use_torchacc():
            rank, _, world_size, _ = get_dist_setting()
            from swift.torchacc_utils import pad_and_split_batch
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
                        self.template.tokenizer, input_ids, labels, position_ids, attention_mask, loss_scale)
            res['position_ids'] = position_ids
        _local_var = locals()
        for key in ['input_ids', 'attention_mask', 'labels', 'loss_scale']:
            value = _local_var[key]
            if value is not None:
                res[key] = value
        return res
