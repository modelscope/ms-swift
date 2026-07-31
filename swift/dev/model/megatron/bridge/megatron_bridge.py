"""MegatronBridgeBackend: build a Megatron model through NVIDIA megatron-bridge (AutoBridge).

Supports full-parameter and LoRA text causal LM. LoRA goes through twinkle's HF ``get_peft_model``
(same as the mcore backend); multimodal is not supported yet.

``AutoBridge`` has no ``config.bridge`` and different method names/signatures than what twinkle's
Model expects (``load_hf_weights`` / ``save_hf_weights`` / ``export_hf_weights``, no
``peft_format`` / ``adapter_name`` / ``target_device`` / ``only_master_rank`` kwargs). To keep
twinkle's Model load/save/export working unchanged, ``build_model_config`` attaches a
``_MCoreCompatBridgeShim`` as ``provider.bridge``; the shim exposes the mcore-compatible surface
and forwards to ``AutoBridge``.

``AutoBridge.to_megatron_provider(load_weights=False)`` returns a ``GPTModelProvider`` (subclass of
megatron-core ``TransformerConfig``) that already carries the fields twinkle's strategy reads off
``self.config``, so the provider itself is used as the config-like object.

AutoBridge import path: ``megatron.bridge.models.conversion.auto_bridge.AutoBridge`` (the top-level
``megatron.bridge`` does not re-export it).
"""
from __future__ import annotations

from typing import Any, Dict, List


class _MCoreCompatBridgeShim:
    """mcore-compatible facade over a megatron-bridge ``AutoBridge``.

    Exposes ``load_weights`` / ``save_weights`` / ``export_weights`` with the signatures twinkle's
    Model calls, forwarding to ``AutoBridge.load_hf_weights`` / ``save_hf_weights`` /
    ``export_hf_weights``. For a non-default ``adapter_name`` (``peft_format=True``), save/export go
    through peft's ``get_peft_model_state_dict(adapter_name=...)`` to write only that adapter's LoRA
    delta (AutoBridge has no per-adapter export). A ``converter`` hook and ``only_master_rank=True``
    are not supported and fail-fast.
    """

    def __init__(self, auto_bridge: Any, hf_config: Any):
        self._bridge = auto_bridge
        self._hf_config = hf_config

    @staticmethod
    def _peft_state_dict(mg_models, adapter_name: str):
        """Return exactly ``adapter_name``'s LoRA weights from the PeftModel, in HF PEFT layout.

        ``get_peft_model_state_dict`` returns only that adapter's ``lora_*`` params with the adapter
        name stripped from the keys, which is the ``adapter_model.safetensors`` content
        ``peft.PeftModel.from_pretrained`` expects.
        """
        from peft import PeftModel
        from peft.utils.save_and_load import get_peft_model_state_dict

        peft_model = next((m for m in mg_models if isinstance(m, PeftModel)), None)
        if peft_model is None:
            raise RuntimeError('peft_format=True save/export expects a PeftModel; got '
                               f'{[type(m).__name__ for m in mg_models]}.')
        name = adapter_name or 'default'
        if name not in peft_model.peft_config:
            raise KeyError(f'adapter {name!r} not on the model; available: {list(peft_model.peft_config)}')
        return get_peft_model_state_dict(peft_model, adapter_name=name)

    @staticmethod
    def _reject_sharded_peft() -> None:
        """Fail-fast on per-adapter save/export under TP or PP > 1.

        ``get_peft_model_state_dict`` returns each rank's local LoRA params. Under DP the LoRA
        weights are replicated, so rank 0's copy is complete; under TP/PP they are sharded and this
        shim does not all-gather them, so rank 0 would write an incomplete adapter. Only DP is
        supported here.
        """
        try:
            from megatron.core import parallel_state as mpu
        except Exception:
            return
        if not mpu.model_parallel_is_initialized():
            return
        tp = mpu.get_tensor_model_parallel_world_size()
        pp = mpu.get_pipeline_model_parallel_world_size()
        if tp > 1 or pp > 1:
            raise NotImplementedError(f'per-adapter save/export is only implemented for DP; got tensor_parallel={tp}, '
                                      f'pipeline_parallel={pp}. LoRA weights are sharded under TP/PP and are not '
                                      'all-gathered here, so the written adapter would be incomplete.')

    def load_weights(self,
                     mg_models,
                     hf_model_dir: str,
                     peft_format: bool = False,
                     adapter_name: str = 'default',
                     converter=None):
        """Forwards to ``AutoBridge.load_hf_weights``. Adapter reload is handled by twinkle's
        PeftModel, so this base-weight path never runs with ``peft_format=True``."""
        if converter is not None:
            raise NotImplementedError('load_weights does not support a converter hook.')
        if peft_format:
            raise NotImplementedError('load_weights(peft_format=True) is not used on this backend.')
        self._bridge.load_hf_weights(mg_models, hf_model_dir)

    def save_weights(self,
                     mg_models,
                     output_dir: str,
                     peft_format: bool = False,
                     adapter_name: str = 'default',
                     converter=None,
                     max_shard_size: str = '5GB'):
        """peft_format=False: forward to ``AutoBridge.save_hf_weights`` (full HF weights).
        peft_format=True: write only ``adapter_name``'s LoRA delta as ``adapter_model.safetensors``
        (twinkle writes the matching ``adapter_config.json`` / hf_config / tokenizer separately)."""
        if converter is not None:
            raise NotImplementedError('save_weights does not support a converter hook.')
        if peft_format:
            import torch.distributed as dist
            self._reject_sharded_peft()
            state_dict = self._peft_state_dict(mg_models, adapter_name)
            is_global_zero = (not dist.is_initialized()) or dist.get_rank() == 0
            if is_global_zero:
                import os
                from safetensors.torch import save_file
                os.makedirs(output_dir, exist_ok=True)
                cpu_sd = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
                save_file(cpu_sd, os.path.join(output_dir, 'adapter_model.safetensors'))
            if dist.is_initialized():
                dist.barrier()
            return
        # save_hf_weights decides sharding internally; it takes no max_shard_size.
        self._bridge.save_hf_weights(mg_models, output_dir)

    def export_weights(self,
                       mg_models,
                       target_device=None,
                       only_master_rank: bool = False,
                       peft_format: bool = False,
                       adapter_name: str = 'default',
                       converter=None,
                       tqdm_desc: str = 'Exporting: ',
                       disable_tqdm: bool = True,
                       _is_saving: bool = False):
        """Yields ``(hf_name, tensor)``. peft_format=False: full HF weights via
        ``AutoBridge.export_hf_weights``. peft_format=True: only ``adapter_name``'s LoRA delta.
        ``only_master_rank=True`` is unsupported (AutoBridge gathers full tensors to all ranks)."""
        if converter is not None:
            raise NotImplementedError('export_weights does not support a converter hook.')
        if peft_format:
            self._reject_sharded_peft()
            state_dict = self._peft_state_dict(mg_models, adapter_name)
            to_cpu = (target_device == 'cpu')
            for name, tensor in state_dict.items():
                yield name, (tensor.detach().cpu() if to_cpu else tensor.detach())
            return
        if only_master_rank:
            raise NotImplementedError('export_weights only supports only_master_rank=False.')
        cpu = (target_device == 'cpu')
        for name, tensor in self._bridge.export_hf_weights(mg_models, cpu=cpu, show_progress=not disable_tqdm):
            yield name, tensor


class MegatronBridgeBackend:
    """Wraps NVIDIA megatron-bridge (AutoBridge) construction behind the BridgeBackend protocol."""

    backend_name = 'megatron-bridge'

    @property
    def is_multimodal(self) -> bool:
        # TODO: MLLM
        return False

    def build_model_config(self, hf_config: Any, parallel_kwargs: Dict[str, Any], strategy: Any, **kwargs) -> Any:
        """HF config -> a finalized GPTModelProvider with a shim attached as ``.bridge``.

        Override values (parallel sizes, dtype, sequence_parallel, variable_seq_lengths, and the
        backward/MoE defaults calculate_per_token_loss / moe_token_dispatcher_type) are taken from
        the same strategy the mcore backend reads, so both backends build the same-shaped model.
        """
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge

        # hf_config carries the local snapshot path (twinkle resolves it before calling).
        model_path = getattr(hf_config, 'name_or_path', None) or getattr(hf_config, '_name_or_path', None)
        if not model_path:
            raise ValueError('MegatronBridgeBackend needs hf_config.name_or_path (a local model dir).')
        trust_remote_code = bool(kwargs.pop('trust_remote_code', True))
        bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=trust_remote_code)
        provider = bridge.to_megatron_provider(load_weights=False)

        overrides: Dict[str, Any] = dict(parallel_kwargs)
        overrides['sequence_parallel'] = strategy.sequence_parallel
        overrides['variable_seq_lengths'] = strategy.variable_seq_lengths
        # attention_backend is NOT set here. It used to be pinned to AttnBackend.flash on this path
        # only, while the mcore-bridge path left it at mcore's AttnBackend.auto -- so the same dev
        # Config ran a different attention kernel depending on bridge_backend. It is now resolved once
        # in builders/model.py (from ModelConfig.attn_impl, defaulting to flash like legacy) and
        # arrives through kwargs below, which both backends share.
        # Fold through explicit config kwargs the provider declares; apply_overrides_and_finalize
        # raises on unknown attrs, so drop keys the provider does not model.
        for k, v in kwargs.items():
            if hasattr(provider, k):
                overrides[k] = v

        # Match the mcore backend's backward/MoE defaults. calculate_per_token_loss defaults to
        # False in core (mcore forces True for per-token-mean grad normalization);
        # moe_token_dispatcher_type follows the same variable_seq_lengths gate (MoE only).
        overrides.setdefault('calculate_per_token_loss', True)
        overrides.setdefault('moe_token_dispatcher_type', 'alltoall' if strategy.variable_seq_lengths else 'allgather')
        # Same fusion-flag alignment as the mcore backend (see mcore.py for the rationale): legacy
        # Megatron-SWIFT defaults these True, mcore's TransformerConfig defaults them False, and
        # bias_activation_fusion in particular changes SwiGLU arithmetic in low precision. Only fold in
        # flags the provider actually declares; gradient_accumulation_fusion is left out for the
        # APEX-availability reason documented in mcore.py.
        for _fusion_flag in ('bias_activation_fusion', 'masked_softmax_fusion', 'bias_dropout_fusion'):
            if hasattr(provider, _fusion_flag):
                overrides.setdefault(_fusion_flag, True)

        provider.apply_overrides_and_finalize(dtype=strategy.params_type, overrides=overrides)
        provider.bridge = _MCoreCompatBridgeShim(bridge, hf_config)
        return provider

    def create_model(self, config: Any, model_dir: str, *, load_weights: bool, move_to_gpu) -> List[Any]:
        """provide_distributed_model(wrap_with_ddp=False) -> bare models, move_to_gpu, then
        (if load_weights) AutoBridge.load_hf_weights. twinkle's strategy.wrap_model does the DDP
        wrap later. MPU is already initialized by twinkle's strategy, so it is reused here."""
        import torch.distributed as dist

        provider = config
        mg_models = provider.provide_distributed_model(wrap_with_ddp=False, fp16=provider.fp16, bf16=provider.bf16)
        if not isinstance(mg_models, list):
            mg_models = [mg_models]
        if dist.is_initialized():
            dist.barrier()

        models = [move_to_gpu(m) for m in mg_models]

        if load_weights:
            # config.bridge is the shim; use the underlying AutoBridge for the raw load.
            config.bridge._bridge.load_hf_weights(mg_models, model_dir)
        return models
