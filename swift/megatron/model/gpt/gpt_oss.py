from swift.llm import ModelType
from ..constant import MegatronModelType
from ..register import MegatronModelMeta, register_megatron_model

register_megatron_model(MegatronModelMeta(
    MegatronModelType.gpt_oss,
    [
        ModelType.gpt_oss,
    ],
))
