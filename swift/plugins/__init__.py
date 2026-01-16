# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .callback import extra_callbacks, EarlyStopCallback
    from .tuner import Tuner, extra_tuners, PeftTuner

else:
    _import_structure = {
        'callback': ['extra_callbacks', 'EarlyStopCallback'],
        'tuner': ['Tuner', 'extra_tuners', 'PeftTuner'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
