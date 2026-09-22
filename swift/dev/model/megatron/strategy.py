from __future__ import annotations

import logging
from twinkle.model.megatron.strategy import MegatronStrategy
from typing import Any, Dict, List

from .bridge import BridgeBackend, MCoreBridgeBackend

logger = logging.getLogger(__name__)


class DevMegatronStrategy(MegatronStrategy):
    """MegatronStrategy variant whose model construction is backend-pluggable."""

    def __init__(self,
                 *args,
                 backend: BridgeBackend = None,
                 attn_impl: str = None,
                 align_grad_reduce: bool = True,
                 nccl_comm_warmup: bool = False,
                 **kwargs):
        # Default to mcore-bridge so behavior matches a plain MegatronStrategy when no
        # backend is passed. Set BEFORE super().__init__ because the parent __init__ calls
        # get_model_config, which needs self._backend already present.
        self._backend = backend or MCoreBridgeBackend()
        self._align_grad_reduce = align_grad_reduce
        self._nccl_comm_warmup = nccl_comm_warmup
        # FlashAttention version pin (flash_3 / flash_attention_3 / ...). Applied HERE rather than in
        # build_model because it works by flipping transformer_engine module globals, which only
        # affect the current process -- and in Ray mode build_model runs on the driver while this
        # runs on the worker that actually builds and runs the model. attn_impl is popped from the
        # kwargs either way: it is dev's own field name and would otherwise reach TransformerConfig,
        # which has no such attribute (the kernel choice arrives separately as attention_backend).
        from swift.dev.naming import apply_flash_version_pin
        pinned = apply_flash_version_pin(attn_impl)
        if pinned is not None:
            logger.info(f'Forcing Flash Attention v{pinned} as the attention backend.')
        super().__init__(*args, **kwargs)
        if self._nccl_comm_warmup:
            self._warmup_communicators()

    @staticmethod
    def _warmup_communicators() -> None:
        import torch
        from megatron.core import mpu
        from twinkle import torch_util

        dummy = torch.zeros(1, device=torch_util.get_current_device())
        warmed = 0
        for getter, kwargs in (
            (mpu.get_data_parallel_group, {
                'with_context_parallel': True
            }),
            (mpu.get_data_parallel_group, {}),
            (mpu.get_context_parallel_group, {}),
            (mpu.get_tensor_model_parallel_group, {}),
            (mpu.get_pipeline_model_parallel_group, {}),
            (mpu.get_model_parallel_group, {}),
            (mpu.get_embedding_group, {}),
            (mpu.get_position_embedding_group, {}),
        ):
            try:
                groups = getter(**kwargs)
            except (AssertionError, ValueError, TypeError):
                continue
            for group in groups if isinstance(groups, list) else [groups]:
                if group is not None:
                    torch.distributed.all_reduce(dummy, group=group)
                    warmed += 1
        torch_util.synchronize()
        logger.info(f'NCCL communicator warm-up done ({warmed} groups).')

    def finish_param_config(self, model: List[Any], optimizer: Any):
        super().finish_param_config(model, optimizer)
        if not self._align_grad_reduce and self.ddp_config.get('overlap_grad_reduce'):
            self.config.grad_sync_func = None

    @property
    def backend(self) -> BridgeBackend:
        return self._backend

    def get_model_config(self, hf_config: Any, parallel_kwargs: Dict[str, Any], **kwargs):
        return self._backend.build_model_config(hf_config, parallel_kwargs, self, **kwargs)

    def create_megatron_model(self, load_weights: bool = True) -> List[Any]:
        return self._backend.create_model(
            self.config, self.model_dir, load_weights=load_weights, move_to_gpu=self._move_model_to_gpu)
