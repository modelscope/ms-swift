from __future__ import annotations

import logging
from twinkle.model.megatron.strategy import MegatronStrategy
from typing import Any, Dict, List

from .bridge import BridgeBackend, MCoreBridgeBackend

logger = logging.getLogger(__name__)


class DevMegatronStrategy(MegatronStrategy):
    """MegatronStrategy variant whose model construction is backend-pluggable."""

    def __init__(self, *args, backend: BridgeBackend = None, attn_impl: str = None, **kwargs):
        # Default to mcore-bridge so behavior matches a plain MegatronStrategy when no
        # backend is passed. Set BEFORE super().__init__ because the parent __init__ calls
        # get_model_config, which needs self._backend already present.
        self._backend = backend or MCoreBridgeBackend()
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

    @property
    def backend(self) -> BridgeBackend:
        return self._backend

    def get_model_config(self, hf_config: Any, parallel_kwargs: Dict[str, Any], **kwargs):
        return self._backend.build_model_config(hf_config, parallel_kwargs, self, **kwargs)

    def create_megatron_model(self, load_weights: bool = True) -> List[Any]:
        return self._backend.create_model(
            self.config, self.model_dir, load_weights=load_weights, move_to_gpu=self._move_model_to_gpu)
