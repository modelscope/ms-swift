# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .prm import prms, PRM
    from .orm import orms, ORM, AsyncORM, MathAccuracy
    from .rm_plugin import rm_plugins, DefaultRMPlugin, GenRMPlugin

else:
    _import_structure = {
        'prm': ['prms', 'PRM'],
        'orm': ['orms', 'ORM', 'AsyncORM', 'MathAccuracy'],
        'rm_plugin': ['rm_plugins', 'DefaultRMPlugin', 'GenRMPlugin'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
