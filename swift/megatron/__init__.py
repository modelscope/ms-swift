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
    from .train import megatron_sft_main, megatron_pt_main
    from .utils import convert_hf2mcore, convert_mcore2hf
    from .argument import MegatronTrainArguments
    from .model import MegatronModelType, MegatronModelMeta, get_megatron_model_meta, register_megatron_model
else:
    _import_structure = {
        'train': ['megatron_sft_main', 'megatron_pt_main'],
        'utils': ['convert_hf2mcore', 'convert_mcore2hf'],
        'argument': ['MegatronTrainArguments'],
        'model': ['MegatronModelType', 'MegatronModelMeta', 'get_megatron_model_meta', 'register_megatron_model']
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
