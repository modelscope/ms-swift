# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .infer import infer_main, SwiftInfer
    from .deploy import deploy_main, SwiftDeploy, run_deploy
    from .protocol import RequestConfig
    from .utils import prepare_pt_engine_template
    from .infer_engine import (InferEngine, VllmEngine, LmdeployEngine, PtEngine, LoRARequest, InferClient,
                               prepare_generation_config)
else:
    _extra_objects = {k: v for k, v in globals().items() if not k.startswith('_')}
    _import_structure = {
        'infer': ['infer_main', 'SwiftInfer'],
        'deploy': ['deploy_main', 'SwiftDeploy', 'run_deploy'],
        'protocol': ['RequestConfig'],
        'utils': ['prepare_pt_engine_template'],
        'infer_engine': [
            'InferEngine', 'VllmEngine', 'LmdeployEngine', 'PtEngine', 'LoRARequest', 'InferClient',
            'prepare_generation_config'
        ],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extra_objects,
    )
