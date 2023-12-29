# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule
from .utils import *

if TYPE_CHECKING:
    from .app_ui import gradio_chat_demo, gradio_generation_demo, llm_app_ui
    from .infer import llm_infer, merge_lora, prepare_model_template
    from .rome import rome_infer
    # Recommend using `xxx_main`
    from .run import (app_ui_main, dpo_main, infer_main, merge_lora_main,
                      rome_main, sft_main)
    from .sft import llm_sft
else:
    _import_structure = {
        'app_ui': ['gradio_chat_demo', 'gradio_generation_demo', 'llm_app_ui'],
        'infer': ['llm_infer', 'merge_lora', 'prepare_model_template'],
        'rome': ['rome_infer'],
        'run': [
            'app_ui_main', 'dpo_main', 'infer_main', 'merge_lora_main',
            'rome_main', 'sft_main'
        ],
        'sft': ['llm_sft'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
