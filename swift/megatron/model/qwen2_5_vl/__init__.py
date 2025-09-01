from swift.llm import ModelType
from ..constant import MegatronModelType
from ..gpt import GPTMegatronModelMeta
from ..register import register_megatron_model
from .convert import convert_hf2mcore_qwen2_5_vl, convert_mcore2hf_qwen2_5_vl
from .vit import Qwen2_5VL_Vit

register_megatron_model(
    GPTMegatronModelMeta(
        MegatronModelType.qwen2_5_vl, [
            ModelType.qwen2_5_vl,
        ],
        convert_hf2mcore=convert_hf2mcore_qwen2_5_vl,
        convert_mcore2hf=convert_mcore2hf_qwen2_5_vl,
        visual=Qwen2_5VL_Vit))
