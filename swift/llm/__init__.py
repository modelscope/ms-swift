# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule
from .utils import *

if TYPE_CHECKING:
    # Recommend using `xxx_main`
    from .app_ui import gradio_chat_demo, gradio_generation_demo, app_ui_main
    from .infer.deploy import deploy_main
    from .export.export import export_main
    from swift.llm.train.rlhf import rlhf_main
else:
    _extra_objects = {k: v for k, v in globals().items() if not k.startswith('_')}
    _import_structure = {
        'app_ui': ['gradio_chat_demo', 'gradio_generation_demo', 'app_ui_main'],
        'deploy': ['deploy_main'],
        'rlhf': ['rlhf_main'],
        'infer': ['merge_lora', 'prepare_model_template', 'infer_main', 'merge_lora_main'],
        'sft': ['sft_main', 'pt_main'],
        'export': ['export_main'],
        'eval': ['eval_main'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extra_objects,
    )
