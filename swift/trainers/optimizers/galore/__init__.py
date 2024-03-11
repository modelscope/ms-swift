# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .utils import create_optimizer_and_scheduler, GaLoreConfig
    from .adafactor import GaLoreAdafactor
    from .adamw8bit import GaLoreAdamW8bit
    from .adamw import GaLoreAdamW
else:
    _import_structure = {
        'utils': ['GaLoreConfig', 'create_optimizer_and_scheduler'],
        'adafactor': ['GaLoreAdafactor'],
        'adamw8bit': ['GaLoreAdamW8bit'],
        'adamw': ['GaLoreAdamW'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
