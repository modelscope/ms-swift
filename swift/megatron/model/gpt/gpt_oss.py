from swift.llm import ModelType
from ..constant import MegatronModelType
from ..gpt_bridge import GPTBridge
from ..register import MegatronModelMeta, register_megatron_model


class GptOssBridge(GPTBridge):
    pass

register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.gpt_oss,
        [
            ModelType.gpt_oss,
        ],
        bridge_cls=GptOssBridge,
    ))
