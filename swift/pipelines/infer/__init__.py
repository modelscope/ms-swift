# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .infer import infer_main, SwiftInfer
    from .rollout import rollout_main
    from .deploy import deploy_main, SwiftDeploy, run_deploy
else:
    _import_structure = {
        'rollout': ['rollout_main'],
        'infer': ['infer_main', 'SwiftInfer'],
        'deploy': ['deploy_main', 'SwiftDeploy', 'run_deploy'],
        'protocol': ['RequestConfig', 'Function'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
