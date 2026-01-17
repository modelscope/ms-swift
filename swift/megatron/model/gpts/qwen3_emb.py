# Copyright (c) ModelScope Contributors. All rights reserved.

import megatron.core
from packaging import version

from swift.model import ModelType
from ..constant import MegatronModelType
from ..gpt_bridge import GPTBridge
from ..register import MegatronModelMeta, register_megatron_model


class Qwen3EmbBridge(GPTBridge):

    def _convert_hf_state_dict(self, hf_state_dict, to_mcore):
        res = super()._convert_hf_state_dict(hf_state_dict, to_mcore)
        if to_mcore:
            res = self._add_prefix(res, 'model.')
        elif not to_mcore:
            res = self._remove_prefix(res, 'model.')
        return res


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.qwen3_emb,
        [
            ModelType.qwen3_emb,
        ],
        bridge_cls=Qwen3EmbBridge,
    ))
