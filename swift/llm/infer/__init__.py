# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .deploy import deploy_main
    from .infer import infer_main, merge_lora_main
    from .protocol import InferRequest, RequestConfig
    from .infer_engine import (InferEngine, VllmEngine, LmdeployEngine, PtEngine, InferStats)
else:
    _extra_objects = {k: v for k, v in globals().items() if not k.startswith('_')}
    _import_structure = {
        'deploy': ['deploy_main'],
        'infer': ['infer_main', 'merge_lora_main', 'merge_lora'],
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
