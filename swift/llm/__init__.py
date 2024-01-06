# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule
from .utils import *

if TYPE_CHECKING:
    # Recommend using `xxx_main`
    from .app_ui import gradio_chat_demo, gradio_generation_demo, llm_app_ui, app_ui_main
    from .deploy import llm_deploy, deploy_main
    from .dpo import dpo_main, llm_dpo
    from .infer import llm_infer, merge_lora, prepare_model_template, infer_main, merge_lora_main
    from .rome import rome_infer, rome_main
    from .sft import llm_sft, sft_main
else:
    _extra_objects = {
        k: v
        for k, v in globals().items() if not k.startswith('_')
    }
    _import_structure = {
        'app_ui': [
            'gradio_chat_demo', 'gradio_generation_demo', 'llm_app_ui',
            'app_ui_main'
        ],
        'deploy': ['llm_deploy', 'deploy_main'],
        'dpo': ['dpo_main', 'llm_dpo'],
        'infer': [
            'llm_infer', 'merge_lora', 'prepare_model_template', 'infer_main',
            'merge_lora_main'
        ],
        'rome': ['rome_infer', 'rome_main'],
        'sft': ['llm_sft', 'sft_main'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extra_objects,
    )
