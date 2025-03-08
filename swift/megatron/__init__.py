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
    from .train import megatron_sft_main
    from .convert import convert_hf2megatron, convert_megatron2hf
    from .argument import MegatronTrainArguments
else:
    _import_structure = {
        'train': ['megatron_sft_main'],
        'convert': ['convert_hf2megatron', 'convert_megatron2hf'],
        'argument': ['MegatronTrainArguments'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
