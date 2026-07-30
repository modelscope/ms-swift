"""Bridge backends: the abstraction over mcore-bridge / megatron-bridge."""
from __future__ import annotations

from .mcore import MCoreBridgeBackend
from .megatron_bridge import MegatronBridgeBackend
from .protocol import BridgeBackend

__all__ = ['BridgeBackend', 'MCoreBridgeBackend', 'MegatronBridgeBackend']
