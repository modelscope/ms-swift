# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .infer import infer_main, SwiftInfer
    from .rollout import rollout_main
    from .deploy import deploy_main, SwiftDeploy, run_deploy
    from .protocol import RequestConfig, Function
    from .utils import prepare_model_template
    from .infer_engine import (InferEngine, VllmEngine, LmdeployEngine, SglangEngine, PtEngine, InferClient,
                               prepare_generation_config, AdapterRequest, BaseInferEngine)
else:
    _import_structure = {
        'rollout': ['rollout_main'],
        'infer': ['infer_main', 'SwiftInfer'],
        'deploy': ['deploy_main', 'SwiftDeploy', 'run_deploy'],
        'protocol': ['RequestConfig', 'Function'],
        'utils': ['prepare_model_template'],
        'infer_engine': [
            'InferEngine', 'VllmEngine', 'LmdeployEngine', 'SglangEngine', 'PtEngine', 'InferClient',
            'prepare_generation_config', 'AdapterRequest', 'BaseInferEngine'
        ],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
