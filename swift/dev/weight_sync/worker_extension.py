# Copyright (c) ModelScope Contributors. All rights reserved.
"""twinkle's vLLM worker extension, extended for quantized weight refits.

dev delivers every weight sync through twinkle's ``TwinkleWorkerExtension``
(ZMQ + CUDA IPC into the vLLM worker subprocess). Its base path calls
``model.load_weights`` directly, which cannot land a BF16 stream on an engine
that serves a quantized checkpoint: the kernel post-processing rewrote the
parameters into an inference layout the checkpoint tensors no longer fit, and
the scales must be re-derived on the fly.

This subclass keeps the receive protocol byte-for-byte and only swaps the load
itself, bracketing one transfer with the scheme's stage/process pair:

* ``__new__`` installs the vLLM post-load patches before the model is built, so
  each layer's checkpoint layout is recorded during the engine's own initial load.
* ``update_weights_from_ipc`` / ``load_synced_weights`` stage the modules once
  per base sync, and replay the kernel layout once the last bucket has landed.
* ``_load_weights`` quantizes each bucket and loads it into the staged buffers.

LoRA syncs are untouched: a merged adapter rides the same base path (and is
therefore quantized), while the incremental adapter channel never touches the
quantized base.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension

from swift.dev.utils.logger import get_logger
from swift.dev.weight_sync.manager import (
    apply_vllm_quant_patches,
    is_quantized_sync_model,
    load_quanted_weights,
    prepare_quanted_weights_for_loading,
    process_quanted_weights_after_loading,
)

logger = get_logger()

#: Attributes twinkle's ``_load_weights`` drops before loading a base sync: the
#: adapter channel owns them, the base model has no slot for them.
_LORA_NAME_MARKERS = ('lora_A', 'lora_B', 'lora_embedding')


def _base_sync(peft_config: Optional[Dict], base_sync_done: bool) -> bool:
    """Whether this call transfers base weights rather than an adapter.

    Same predicate twinkle uses for its LoRA branch: ``peft_config`` plus
    ``base_sync_done`` means the base is already in place and only the adapter
    moved, so nothing quantized is involved.
    """
    return not (peft_config and base_sync_done)


def _strip_adapter_weights(weights: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    """Drop adapter tensors, mirroring twinkle's base-model filter."""
    return [(name, tensor) for name, tensor in weights if not any(marker in name for marker in _LORA_NAME_MARKERS)]


class QuantSyncWorkerExtension(TwinkleWorkerExtension):
    """Weight-sync worker extension that can refit a quantized vLLM engine.

    Selected via the ``worker_extension_cls`` engine arg (see
    :data:`swift.dev.weight_sync.QUANT_SYNC_WORKER_EXTENSION`); a no-op wrapper
    when the engine is not quantized, so it is installed for every RL sampler.
    """

    #: Opaque per-refit state returned by ``prepare_quanted_weights_for_loading``;
    #: None means no quantized refit is open (non-quantized engine, LoRA sync).
    _quant_reload_state: Optional[Dict[str, Any]] = None

    def __new__(cls, *args, **kwargs):
        # The patched hooks run during the engine's initial weight load, which
        # happens after this worker is constructed and before any sync. This is
        # the earliest in-process point that reaches every worker subprocess.
        apply_vllm_quant_patches()
        # Forward the construction kwargs: the base __new__ reads ``vllm_config`` off them to
        # register the delta transfer engines -- swallowing them here disabled that registration.
        return super().__new__(cls, *args, **kwargs)

    # -----------------------------------------------------------------
    # Quantized-refit plumbing
    # -----------------------------------------------------------------

    def _quantized_base_sync(self, peft_config: Optional[Dict], base_sync_done: bool) -> bool:
        """Whether this transfer has to be quantized on the way in."""
        if not _base_sync(peft_config, base_sync_done):
            return False
        cached = getattr(self, '_is_quantized_sync', None)
        if cached is None:
            vllm_config = getattr(self.model_runner, 'vllm_config', None)
            cached = is_quantized_sync_model(vllm_config)
            self._is_quantized_sync = cached
            if cached:
                logger.info('Detected a quantized rollout engine; base weight sync will quantize on the fly.')
        return cached

    def _open_quant_reload(self, peft_config: Optional[Dict], base_sync_done: bool) -> bool:
        """Stage checkpoint-layout buffers for one base sync; returns whether it opened."""
        if not self._quantized_base_sync(peft_config, base_sync_done):
            return False
        if self._quant_reload_state is None:
            self._quant_reload_state = prepare_quanted_weights_for_loading(self.model_runner.model)
        return True

    def _close_quant_reload(self) -> None:
        """Replay the kernel layout into the live storage and drop the staging state."""
        state = self._quant_reload_state
        if state is None:
            return
        try:
            process_quanted_weights_after_loading(self.model_runner.model, state)
        finally:
            self._quant_reload_state = None

    # -----------------------------------------------------------------
    # Public API overrides
    # -----------------------------------------------------------------

    def update_weights_from_ipc(self,
                                peft_config: Optional[Dict] = None,
                                base_sync_done: bool = False,
                                use_shm: bool = False,
                                zmq_handle: Optional[str] = None) -> None:
        """Load one bucketed transfer, quantizing a base sync before it lands."""
        opened = self._open_quant_reload(peft_config, base_sync_done)
        try:
            return super().update_weights_from_ipc(
                peft_config=peft_config, base_sync_done=base_sync_done, use_shm=use_shm, zmq_handle=zmq_handle)
        finally:
            if opened:
                self._close_quant_reload()

    def load_synced_weights(self,
                            weights: Dict[str, Any],
                            peft_config: Optional[Dict] = None,
                            base_sync_done: bool = False) -> None:
        """Dict-shaped sync path (older callers); same quantized bracket."""
        opened = self._open_quant_reload(peft_config, base_sync_done)
        try:
            return super().load_synced_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done)
        finally:
            if opened:
                self._close_quant_reload()

    def _load_weights(self, weights: List[Tuple[str, Any]], peft_config: Optional[Dict],
                      base_sync_done: bool) -> None:
        """Load one bucket; quantize it when a quantized base sync is open."""
        base = _strip_adapter_weights(weights)
        if not base:
            return
        if self._quant_reload_state is None:
            # Non-quantized engine (or an adapter sync): twinkle's own loader handles
            # the fused-parameter merging and prefix mapping.
            return super()._load_weights(weights, peft_config=peft_config, base_sync_done=base_sync_done)

        vllm_config = self.model_runner.vllm_config
        load_quanted_weights(
            base,
            self.model_runner.model,
            vllm_config.quant_config,
            dtype=vllm_config.model_config.dtype,
            model_dir=vllm_config.model_config.model,
            reload_state=self._quant_reload_state)


__all__ = ['QuantSyncWorkerExtension']
