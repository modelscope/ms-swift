# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .galore import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit
else:
    _import_structure = {
        'galore': ['GaLoreAdafactor', 'GaLoreAdamW', 'GaLoreAdamW8bit'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )