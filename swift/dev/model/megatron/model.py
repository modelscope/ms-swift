from __future__ import annotations

from functools import partial
from twinkle.model.megatron import MegatronModel as TwinkleMegatronModel
from unittest.mock import patch

from .bridge import BridgeBackend
from .strategy import DevMegatronStrategy

# Module that reads the ``MegatronStrategy`` name inside ``MegatronModel.__init__``.
_TWINKLE_MEGATRON_MODULE = 'twinkle.model.megatron.megatron'


class MegatronModel(TwinkleMegatronModel):
    """MegatronModel whose Megatron construction is routed through a BridgeBackend."""

    def __init__(self, *args, backend: BridgeBackend = None, **kwargs):
        strategy_cls = partial(DevMegatronStrategy, backend=backend)
        with patch(f'{_TWINKLE_MEGATRON_MODULE}.MegatronStrategy', strategy_cls):
            super().__init__(*args, **kwargs)
