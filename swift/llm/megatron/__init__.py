from .argument import MegatronArguments
from .convert import convert_hf_to_megatron, convert_megatron_to_hf, model_provider
from .utils import get_model_seires, init_megatron_env, patch_megatron

init_megatron_env()
