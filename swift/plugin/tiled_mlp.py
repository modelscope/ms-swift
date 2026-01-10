# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Tiled MLP implementation for memory-efficient training.

This module provides a tiled MLP implementation that is compatible with FSDP2.
- FSDP2: Uses custom TiledMLP implementation (this file)
- DeepSpeed/Single GPU: Uses liger_kernel's LigerTiledSwiGLUMLP
- FSDP1: Raises error (not compatible)
"""
import os
import threading
from typing import List, Optional

import torch
import torch.nn as nn

from swift.utils import get_logger

logger = get_logger()

# ============================================================================
# FSDP2 Compatible TiledMLP Implementation
# ============================================================================


class GradientAccumulator:
    """Gradient accumulator for TiledMLP (FSDP2 compatible)"""

    def __init__(self, params: List[torch.nn.Parameter], total_shards: int, dtype: torch.dtype = None):
        self.params = params
        self.total_shards = total_shards
        self.grad_accumulation_dtype = dtype or torch.float32
        self.accumulated_grads = {}
        self.hooks = []
        self.lock = threading.Lock()

        for param in self.params:
            if param.grad is not None:
                self.accumulated_grads[param] = param.grad.to(self.grad_accumulation_dtype)
                param.grad = None
            else:
                self.accumulated_grads[param] = torch.zeros_like(param, dtype=self.grad_accumulation_dtype)

    def install_hooks(self, is_last_shard: bool):
        self._remove_hooks()

        def create_hook(param):

            def hook(grad):
                with self.lock:
                    grad_to_accum_dtype = grad.to(self.grad_accumulation_dtype)
                    self.accumulated_grads[param] += grad_to_accum_dtype

                    if is_last_shard:
                        param.grad = None  # Critical: prevent double accumulation
                        final_grad = self.accumulated_grads[param].to(param.dtype)
                        return final_grad
                    return None

            return hook

        for param in self.params:
            if param.requires_grad:
                hook = param.register_hook(create_hook(param))
                self.hooks.append(hook)

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def cleanup(self):
        self._remove_hooks()


class TiledMLPFunction(torch.autograd.Function):
    """TiledMLP autograd function for FSDP2 compatibility"""

    @staticmethod
    def forward(ctx, fn, self, x, shards, compute_params):
        ctx.fn = fn
        ctx.self = self
        ctx.shards = shards
        ctx.compute_params = [p for p in compute_params if p.requires_grad]
        ctx.save_for_backward(x)

        # Split on dim=-2 (seqlen dimension)
        x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
        with torch.no_grad():
            output_shards = [fn(self, x_shard) for x_shard in x_shards]
        output_unsharded = torch.cat(output_shards, dim=-2)
        return output_unsharded

    @staticmethod
    def backward(ctx, *grads):
        fn = ctx.fn
        (x, ) = ctx.saved_tensors
        self = ctx.self
        shards = ctx.shards
        compute_params = ctx.compute_params

        x_requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad_(x_requires_grad)

        # Flatten to [bs*seqlen, hidden_size]
        hidden_size = x.shape[-1]
        x_shape_orig = x.shape
        x = x.view(-1, hidden_size)
        incoming_grad = grads[0].view(-1, hidden_size)

        # Pre-allocate input gradient
        x_grad = torch.zeros_like(x)

        # Split on dim=0
        x_shards = list(torch.chunk(x, chunks=shards, dim=0))

        grad_accumulator = GradientAccumulator(compute_params, shards, dtype=x.dtype)

        for i, x_shard in enumerate(x_shards):
            x_shard.requires_grad_(x_requires_grad)

            shard_step = x_shards[i].shape[0]
            shard_offset = i * x_shards[0].shape[0]

            # narrow(0, ...) creates a view that can correctly receive gradients
            x_shard.grad = x_grad.narrow(0, shard_offset, shard_step)
            incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step)

            is_last_shard = i + 1 == shards
            grad_accumulator.install_hooks(is_last_shard)

            with torch.enable_grad():
                output = fn(self, x_shard)
            torch.autograd.backward(output, incoming_grad_shard)

        grad_accumulator.cleanup()
        del grad_accumulator

        # Restore original shape
        x_grad = x_grad.view(x_shape_orig) if x_requires_grad else None
        return (None, None, x_grad, None, None)


class TiledSwiGLUMLP(nn.Module):
    """
    Memory-efficient SwiGLU MLP using tiled computation for FSDP2.

    This module combines SwiGLU activation with tiled processing to handle
    very long sequences efficiently. The forward pass is recomputed during
    backward to save memory.

    Args:
        config: Model configuration with hidden_size and intermediate_size attributes
        num_shards: Number of shards to split the sequence. If None, automatically
                   calculated as ceil(seqlen / hidden_size)
    """

    def __init__(self, config, num_shards: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_shards = num_shards or 4  # Default to 4 shards

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act = nn.SiLU()

    def _mlp_forward(self, module, x):
        """Internal MLP forward function for tiled computation."""
        gate = module.gate_proj(x)
        up = module.up_proj(x)
        return module.down_proj(module.act(gate) * up)

    def forward(self, x):
        """
        Forward pass with tiled computation.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
               or [seq_len, hidden_size]
        Returns:
            Output tensor of the same shape as input
        """
        compute_params = [
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
        ]
        return TiledMLPFunction.apply(
            self._mlp_forward,
            self,
            x,
            self.num_shards,
            compute_params,
        )


# ============================================================================
# Environment Detection Functions
# ============================================================================


def is_fsdp2_enabled() -> bool:
    """Check if FSDP2 is enabled via accelerate config."""
    # Check environment variable set by accelerate
    if os.environ.get('ACCELERATE_USE_FSDP', 'false').lower() == 'true':
        # Check fsdp_version from accelerate config
        # FSDP_VERSION is set by accelerate when fsdp_version is specified in config
        fsdp_version = os.environ.get('FSDP_VERSION', '1')
        if fsdp_version == '2':
            return True
        # Also check accelerate state if available
        try:
            from accelerate import PartialState
            state = PartialState()
            if hasattr(state, 'fsdp_plugin') and state.fsdp_plugin is not None:
                # Check if fsdp_version is 2 in the plugin
                if hasattr(state.fsdp_plugin, 'fsdp_version'):
                    return state.fsdp_plugin.fsdp_version == 2
        except Exception:
            pass
    return False


def is_fsdp1_enabled() -> bool:
    """Check if FSDP1 is enabled via accelerate config."""
    if os.environ.get('ACCELERATE_USE_FSDP', 'false').lower() == 'true':
        fsdp_version = os.environ.get('FSDP_VERSION', '1')
        if fsdp_version == '2':
            return False
        # Also check accelerate state if available
        try:
            from accelerate import PartialState
            state = PartialState()
            if hasattr(state, 'fsdp_plugin') and state.fsdp_plugin is not None:
                if hasattr(state.fsdp_plugin, 'fsdp_version'):
                    return state.fsdp_plugin.fsdp_version != 2
        except Exception:
            pass
        return True
    return False


def is_deepspeed_enabled() -> bool:
    """Check if DeepSpeed is enabled."""
    from swift.utils import is_deepspeed_enabled as _is_deepspeed_enabled
    return _is_deepspeed_enabled()


def get_tiled_mlp_mode() -> str:
    """
    Determine which tiled MLP implementation to use.

    Returns:
        'fsdp2': Use custom TiledSwiGLUMLP implementation
        'liger': Use liger_kernel's LigerTiledSwiGLUMLP
        'error': FSDP1 detected, should raise error
    """
    if is_fsdp2_enabled():
        return 'fsdp2'
    elif is_fsdp1_enabled():
        return 'error'
    else:
        # DeepSpeed, Single GPU, or DDP - use liger kernel
        return 'liger'


# ============================================================================
# MLP Replacement Functions
# ============================================================================

# Supported model types for tiled MLP
SUPPORTED_MODEL_TYPES = {
    'qwen2',
    'qwen2_5',
    'qwen3',
    'qwen3_vl',
}


def _get_mlp_class_for_model(model_type: str) -> str:
    """Get the MLP class name for different model architectures."""
    # Map model types to their MLP class names
    mlp_class_mapping = {
        'qwen2': 'Qwen2MLP',
        'qwen2_5': 'Qwen2MLP',
        'qwen3': 'Qwen3MLP',
        'qwen3_vl': 'Qwen3VLTextMLP',
    }

    if model_type in mlp_class_mapping:
        return mlp_class_mapping[model_type]

    # Fallback: capitalize model_type and append 'MLP'
    # e.g., 'mistral' -> 'MistralMLP'
    return model_type.capitalize() + 'MLP'


def apply_tiled_mlp(model_type: str, num_shards: Optional[int] = None):
    """
    Apply tiled MLP replacement before model instantiation.

    This function should be called BEFORE loading the model to replace
    the MLP class in the transformers module.

    Args:
        model_type: The model type (e.g., 'llama', 'qwen2')
        num_shards: Number of shards for tiled computation

    Raises:
        ValueError: If FSDP1 is detected (not compatible)
    """
    mode = get_tiled_mlp_mode()

    if mode == 'error':
        raise ValueError('Tiled MLP is not compatible with FSDP1. '
                         'Please use FSDP2 (set fsdp_version: 2 in accelerate config) or DeepSpeed.')

    if mode == 'fsdp2':
        _apply_custom_tiled_mlp(model_type, num_shards)
    elif mode == 'liger':
        _apply_liger_tiled_mlp(model_type, num_shards)


def _apply_custom_tiled_mlp(model_type: str, num_shards: Optional[int] = None):
    """Apply custom FSDP2-compatible tiled MLP."""
    num_shards = num_shards or 4
    mlp_class_name = _get_mlp_class_for_model(model_type)

    # Get the transformers module for this model
    model_module = _get_transformers_module(model_type)
    if model_module is None:
        raise ValueError(f'Tiled MLP: Could not find transformers module for model_type={model_type}. '
                         f'Supported model types: {SUPPORTED_MODEL_TYPES}')

    # Check if MLP class exists in the module
    original_mlp_class = getattr(model_module, mlp_class_name, None)
    if original_mlp_class is None:
        raise ValueError(f'Tiled MLP: Could not find {mlp_class_name} in {model_module.__name__}. '
                         f'model_type={model_type} may not be supported.')

    # Create a wrapper class that uses TiledSwiGLUMLP
    class TiledMLPWrapper(TiledSwiGLUMLP):

        def __init__(self, config, **kwargs):
            super().__init__(config, num_shards=num_shards)

    # Replace the MLP class
    setattr(model_module, mlp_class_name, TiledMLPWrapper)
    logger.info(f'Tiled MLP: Replaced {mlp_class_name} with TiledSwiGLUMLP (FSDP2 mode, num_shards={num_shards})')


def _apply_liger_tiled_mlp(model_type: str, num_shards: Optional[int] = None):
    """Apply liger_kernel's tiled MLP implementation."""
    try:
        from liger_kernel.transformers.tiled_mlp import LigerTiledSwiGLUMLP
    except ImportError:
        raise ImportError('Tiled MLP: liger_kernel not installed or LigerTiledSwiGLUMLP not available. '
                          'Please install liger-kernel: pip install liger-kernel')

    num_shards = num_shards or 4
    mlp_class_name = _get_mlp_class_for_model(model_type)

    model_module = _get_transformers_module(model_type)
    if model_module is None:
        raise ValueError(f'Tiled MLP: Could not find transformers module for model_type={model_type}. '
                         f'Supported model types: {SUPPORTED_MODEL_TYPES}')

    # Check if MLP class exists in the module
    original_mlp_class = getattr(model_module, mlp_class_name, None)
    if original_mlp_class is None:
        raise ValueError(f'Tiled MLP: Could not find {mlp_class_name} in {model_module.__name__}. '
                         f'model_type={model_type} may not be supported.')

    # Create a wrapper class
    class LigerTiledMLPWrapper(LigerTiledSwiGLUMLP):

        def __init__(self, config, **kwargs):
            super().__init__(config, num_shards=num_shards)

    setattr(model_module, mlp_class_name, LigerTiledMLPWrapper)
    logger.info(f'Tiled MLP: Replaced {mlp_class_name} with LigerTiledSwiGLUMLP (liger mode, num_shards={num_shards})')


def _get_transformers_module(model_type: str):
    """Get the transformers modeling module for a given model type."""
    import importlib

    module_mapping = {
        'qwen2': 'transformers.models.qwen2.modeling_qwen2',
        'qwen2_5': 'transformers.models.qwen2.modeling_qwen2',
        'qwen3': 'transformers.models.qwen3.modeling_qwen3',
        'qwen3_vl': 'transformers.models.qwen3_vl.modeling_qwen3_vl',
    }

    module_name = module_mapping.get(model_type)

    # Fallback: try to construct module name from model_type
    if module_name is None:
        base_type = model_type
        module_name = f'transformers.models.{base_type}.modeling_{base_type}'

    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None
