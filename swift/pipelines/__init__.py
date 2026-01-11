# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    # Recommend using `xxx_main`
    from .infer import (infer_main, deploy_main, run_deploy, rollout_main)
    from .export import (export_main, merge_lora, quantize_model, export_to_ollama)
    from .eval import eval_main
    from .app import app_main
    from .train import sft_main, pretrain_main, rlhf_main, SwiftSft
    from .sampling import sampling_main
    from .base import SwiftPipeline
    from .utils import prepare_model_template
else:
    _import_structure = {
        'infer': [
            'deploy_main',
            'infer_main',
            'run_deploy',
            'rollout_main',
        ],
        'export': ['export_main', 'merge_lora', 'quantize_model', 'export_to_ollama'],
        'app': ['app_main'],
        'eval': ['eval_main'],
        'train': ['sft_main', 'pretrain_main', 'rlhf_main', 'SwiftSft'],
        'sampling': ['sampling_main'],
        'base': ['SwiftPipeline'],
        'utils': ['prepare_model_template'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
