# Copyright (c) Alibaba, Inc. and its affiliates.

try:
    from .init import init_megatron_env
    init_megatron_env()
except Exception:
    # allows lint pass.
    raise

from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .train import megatron_sft_main, megatron_pt_main, megatron_rlhf_main
    from .export import megatron_export_main
    from .convert import convert_hf2mcore, convert_mcore2hf
    from .utils import prepare_mcore_model, adapter_state_dict_context, convert_hf_config
    from .argument import MegatronTrainArguments, MegatronRLHFArguments, MegatronExportArguments, MegatronArguments
    from .model import MegatronModelType, MegatronModelMeta, get_megatron_model_meta, register_megatron_model
    from .trainers import MegatronTrainer, MegatronDPOTrainer
    from .tuners import LoraParallelLinear
else:
    _import_structure = {
        'train': ['megatron_sft_main', 'megatron_pt_main', 'megatron_rlhf_main'],
        'export': ['megatron_export_main'],
        'convert': ['convert_hf2mcore', 'convert_mcore2hf'],
        'utils': ['prepare_mcore_model', 'adapter_state_dict_context', 'convert_hf_config'],
        'argument': ['MegatronTrainArguments', 'MegatronRLHFArguments', 'MegatronExportArguments', 'MegatronArguments'],
        'model': ['MegatronModelType', 'MegatronModelMeta', 'get_megatron_model_meta', 'register_megatron_model'],
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
