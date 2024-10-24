# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .infer import InferPipeline
    from .deploy import DeployApp
    from .protocol import InferRequest, RequestConfig
    from .infer_engine import (InferEngine, VllmEngine, LmdeployEngine, PtEngine, InferStats)
else:
    _extra_objects = {k: v for k, v in globals().items() if not k.startswith('_')}
    _import_structure = {
        'deploy': ['DeployApp'],
        'infer': ['InferPipeline'],
        'protocol': ['InferRequest', 'RequestConfig'],
        'infer_engine': ['InferEngine', 'VllmEngine', 'LmdeployEngine', 'PtEngine', 'InferStats'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extra_objects,
    )
