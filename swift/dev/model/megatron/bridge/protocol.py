"""BridgeBackend: backend-agnostic protocol over mcore-bridge / megatron-bridge.

The two bridge libraries construct a Megatron model very differently:

- mcore-bridge: ``hf_to_mcore_config(hf_config) -> ModelConfig(TransformerConfig)`` then
  ``get_mcore_model(config)`` + ``config.bridge.load_weights(models, model_dir)``. The
  ``bridge`` is auto-attached to the config and drives runtime load/save/export.
- megatron-bridge: ``AutoBridge.from_hf_pretrained(path) ->
  to_megatron_provider(load_weights=False) -> apply_overrides_and_finalize(...) ->
  provide_distributed_model(...)`` + ``bridge.load_hf_weights(models, path)``. Runtime
  save/export go through ``bridge.save_hf_weights`` / ``bridge.export_hf_weights``.

A backend therefore owns TWO construction steps, mirroring twinkle's
``MegatronStrategy.get_model_config`` / ``create_megatron_model`` seam:

1. ``build_model_config(hf_config, parallel_kwargs, **kwargs)`` -> an opaque config object.
   For mcore this is the mcore ``ModelConfig`` (which twinkle's strategy stores as
   ``self.config`` and reads ``self.config.bridge`` from). Callers must NOT assume any
   specific config type -- they only pass it back into ``create_model``.
2. ``create_model(config, model_dir, load_weights, move_to_gpu)`` -> ``list[nn.Module]``.

Runtime weight lifecycle (load/save/export) is intentionally NOT in this protocol: twinkle's
Model reads them via ``strategy.bridge`` (== ``config.bridge``), so a backend only has to make
``build_model_config`` return a config carrying a ``bridge`` that exposes the mcore-compatible
``load_weights``/``save_weights``/``export_weights`` surface. mcore attaches this natively;
megatron-bridge (which has no ``config.bridge`` and different method names/signatures) attaches
a small ``_MCoreCompatBridgeShim`` to the provider's ``.bridge`` instead. Both keep the
inherited Model load/save/export working unchanged -- confirming these two construction methods
are enough and the protocol does NOT need to grow explicit weight-lifecycle methods.
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class BridgeBackend(Protocol):
    backend_name: str

    def build_model_config(self, hf_config: Any, parallel_kwargs: Dict[str, Any], strategy: Any, **kwargs) -> Any:
        """HF config -> backend-specific model config (opaque to the caller).

        ``strategy`` gives access to strategy-owned settings the config depends on
        (params_dtype, sequence_parallel, variable_seq_lengths, ...) without the backend
        re-deriving them. The returned object is stored as ``strategy.config`` and passed
        back into ``create_model``.
        """
        ...

    def create_model(self, config: Any, model_dir: str, *, load_weights: bool, move_to_gpu) -> List[Any]:
        """Build the Megatron model(s) from ``config`` and optionally load HF weights.

        ``move_to_gpu`` is the strategy's per-model placement callable (kept on the
        strategy so device policy stays in one place). Returns the list of model chunks.
        """
        ...

    @property
    def is_multimodal(self) -> bool:
        """Whether the constructed model is multimodal (drives VL module discovery)."""
        ...
