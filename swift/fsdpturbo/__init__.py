# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .arguments import FSDPTurboArguments, FSDPTurboSftArguments
    from .model_specs import FSDPTurboModelSpec, get_model_spec
    from .pipeline import FSDPTurboSft, fsdpturbo_sft_main
    from .trainer import FSDPTurboTrainer
else:
    _import_structure = {
        'arguments': ['FSDPTurboArguments', 'FSDPTurboSftArguments'],
        'model_specs': ['FSDPTurboModelSpec', 'get_model_spec'],
        'pipeline': ['FSDPTurboSft', 'fsdpturbo_sft_main'],
        'trainer': ['FSDPTurboTrainer'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
