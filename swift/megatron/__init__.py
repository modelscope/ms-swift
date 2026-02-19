# Copyright (c) ModelScope Contributors. All rights reserved.

try:
    from transformers.utils import is_torch_npu_available

    if is_torch_npu_available():
        # Enable Megatron on Ascend NPU
        import mindspeed.megatron_adaptor  # F401
    from .init import init_megatron_env
    init_megatron_env()
except Exception:
    # allows lint pass.
    raise

from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .pipelines import megatron_export_main, megatron_sft_main, megatron_pretrain_main, megatron_rlhf_main
    from .convert import convert_hf2mcore, convert_mcore2hf
    from .utils import prepare_mcore_model, initialize_megatron
    from .arguments import (MegatronSftArguments, MegatronPretrainArguments, MegatronRLHFArguments,
                            MegatronExportArguments, MegatronArguments)
    from .model import (MegatronModelType, MegatronModelMeta, get_megatron_model_meta, register_megatron_model,
                        get_mcore_model_config, convert_hf_config, get_mcore_model)
    from .trainers import MegatronTrainer, MegatronDPOTrainer
    from .tuners import LoraParallelLinear
else:
    _import_structure = {
        'pipelines': ['megatron_sft_main', 'megatron_pretrain_main', 'megatron_rlhf_main', 'megatron_export_main'],
        'convert': ['convert_hf2mcore', 'convert_mcore2hf'],
        'utils': ['prepare_mcore_model', 'initialize_megatron'],
        'arguments': [
            'MegatronSftArguments', 'MegatronPretrainArguments', 'MegatronRLHFArguments', 'MegatronExportArguments',
            'MegatronArguments'
        ],
        'model': [
            'MegatronModelType', 'MegatronModelMeta', 'get_megatron_model_meta', 'register_megatron_model',
            'get_mcore_model_config', 'convert_hf_config', 'get_mcore_model'
        ],
        'trainers': ['MegatronTrainer', 'MegatronDPOTrainer'],
        'tuners': ['LoraParallelLinear'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
