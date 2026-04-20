# Copyright (c) ModelScope Contributors. All rights reserved.
"""
NPU Flash Attention Registration Module for ms-swift

This module registers NPU Flash Attention to transformers' ALL_ATTENTION_FUNCTIONS
when NPU is available, enabling users to use attn_implementation='npu_flash_attention'.

Usage:
    from swift.model.npu_flash_attention import register_npu_flash_attention
    register_npu_flash_attention()  # Auto-detects NPU and registers if available

    # Then in model loading:
    model = AutoModelForCausalLM.from_pretrained(
        ...,
        attn_implementation='npu_flash_attention'
    )
"""

import torch
from typing import Optional, Tuple
from swift.utils import get_logger

logger = get_logger()

# Global flag to track if NPU FA has been registered
_NPU_FA_REGISTERED = False


def is_torch_npu_available(check_device: bool = True) -> bool:
    """Check if Ascend NPU is available for PyTorch operations."""
    try:
        if not hasattr(torch, "npu"):
            return False
        if check_device:
            return torch.npu.is_available()
        return True
    except ImportError:
        return False


def npu_flash_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs
) -> Tuple[torch.Tensor, None]:
    """
    NPU Flash Attention forward function compatible with transformers interface.
    
    Args:
        module: The attention module (passed by transformers)
        query: Query tensor of shape (batch, seq_len, num_heads, head_dim) - bshd format
        key: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        value: Value tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        attention_mask: Optional attention mask
        scaling: Optional scaling factor (default: 1/sqrt(head_dim))
        dropout: Dropout probability
        **kwargs: Additional arguments (position_ids, etc.)
    
    Returns:
        Tuple of (attn_output, None) - attn_output has shape (batch, seq_len, num_heads, head_dim)
    """
    # Import NPU flash attention function
    from transformers.integrations.npu_flash_attention import npu_flash_attn_func
    
    batch_size, seq_len, num_heads, head_dim = query.shape
    _, _, num_kv_heads, _ = key.shape
    
    # Handle GQA (Grouped Query Attention) - expand KV heads to match Q heads
    if num_heads != num_kv_heads:
        n_rep = num_heads // num_kv_heads
        key = key.unsqueeze(3).expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)
        key = key.reshape(batch_size, seq_len, num_heads, head_dim)
        value = value.unsqueeze(3).expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)
        value = value.reshape(batch_size, seq_len, num_heads, head_dim)
    
    # Convert from bshd to bsnd format: (b,s,h,d) -> (b,h,s,d)
    query_bsnd = query.permute(0, 2, 1, 3).contiguous()
    key_bsnd = key.permute(0, 2, 1, 3).contiguous()
    value_bsnd = value.permute(0, 2, 1, 3).contiguous()
    
    # Determine if causal (auto-detect from module config if available)
    is_causal = True
    if hasattr(module, 'config') and hasattr(module.config, 'is_decoder'):
        is_causal = module.config.is_decoder
    
    # Call native NPU Flash Attention
    attn_output = npu_flash_attn_func(
        query_bsnd,
        key_bsnd,
        value_bsnd,
        dropout_p=dropout,
        softmax_scale=scaling if scaling is not None else (head_dim ** -0.5),
        causal=is_causal,
    )
    
    # Convert back to bshd format: (b,h,s,d) -> (b,s,h,d)
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
    
    return attn_output, None


def register_npu_flash_attention(force: bool = False) -> bool:
    """
    Register NPU Flash Attention to transformers' ALL_ATTENTION_FUNCTIONS.
    
    This function registers 'npu_flash_attention' as an available attention
    implementation when NPU is detected. After registration, users can load
    models with attn_implementation='npu_flash_attention'.
    
    Args:
        force: If True, re-register even if already registered
    
    Returns:
        bool: True if registration successful, False otherwise
    """
    global _NPU_FA_REGISTERED
    
    # Check if already registered and not forcing re-registration
    if _NPU_FA_REGISTERED and not force:
        return True
    
    # Check NPU availability
    if not is_torch_npu_available():
        logger.info("NPU Flash Attention: NPU not available, skipping registration")
        return False
    
    try:
        # Import transformers utilities
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.integrations.npu_flash_attention import is_torch_npu_available as _is_npu_available
        
        # Verify NPU is truly available from transformers' perspective
        if not _is_npu_available():
            logger.warning("NPU Flash Attention: transformers reports NPU unavailable")
            return False
        
        # Register the NPU Flash Attention function
        ALL_ATTENTION_FUNCTIONS["npu_flash_attention"] = npu_flash_attention_forward
        _NPU_FA_REGISTERED = True
        
        logger.info("✅ NPU Flash Attention registered successfully! "
                   "Use attn_implementation='npu_flash_attention' to enable it.")
        return True
        
    except ImportError as e:
        logger.warning(f"NPU Flash Attention: Failed to import required modules: {e}")
        return False
    except Exception as e:
        logger.warning(f"NPU Flash Attention: Registration failed: {e}")
        return False


def auto_register_npu_flash_attention() -> bool:
    """
    Automatically register NPU Flash Attention on module import if NPU is available.
    
    This is the recommended entry point for automatic NPU FA enablement.
    """
    # Only auto-register if NPU environment variable is not explicitly disabled
    import os
    if os.environ.get('SWIFT_DISABLE_NPU_FA', '0').lower() in ('1', 'true', 'yes'):
        logger.info("NPU Flash Attention: Auto-registration disabled by environment variable")
        return False
    
    return register_npu_flash_attention()


# Auto-register on import (can be disabled via SWIFT_DISABLE_NPU_FA=1)
_AUTO_REGISTERED = auto_register_npu_flash_attention()
