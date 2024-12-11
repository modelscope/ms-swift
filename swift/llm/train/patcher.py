# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import wraps
from typing import Dict, Optional, Union

import accelerate
import torch
import transformers
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import trainer

from swift.utils import get_dist_setting, get_logger, is_mp_ddp, use_torchacc
from swift.utils.torch_utils import _get_max_memory, _sync_max_memory

logger = get_logger()


def patch_ddp_mp():
    """Patch ddp with device_map.
    After patching, the ddp can run with the device_map.
    This should be called before any training starts.
    """
    if is_mp_ddp():
        from accelerate.utils.modeling import get_balanced_memory, infer_auto_device_map

        @wraps(infer_auto_device_map)
        def _infer_auto_device_map_patch(model: Module,
                                         max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
                                         **kwargs) -> Dict[str, Union[int, str, torch.device]]:
            """The auxiliary function for supports MP + DDP. Monkey Patching.
            add feat in accelerate to support MP + DDP"""
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

    if is_mp_ddp() or use_torchacc():
        _old_accelerator_init = trainer.Accelerator.__init__
        trainer.Accelerator.__init__ = (lambda self, device_placement=False, *args, **kwargs: _old_accelerator_init(
            self, device_placement=device_placement, *args, **kwargs))
        trainer.Accelerator.verify_device_map = lambda *args, **kwargs: False


patch_ddp_mp()
