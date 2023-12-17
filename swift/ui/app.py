import os
from functools import partial

import gradio as gr

from swift.ui.base import BaseUI, all_langs
from swift.ui.llm_infer.llm_infer import LLMInfer
from swift.ui.llm_train.llm_train import LLMTrain


lang = os.environ.get('SWIFT_UI_LANG', all_langs[0])


def reload_locale(base_tab: BaseUI):
    """Reload labels"""
    updates = []
    for elem_id, element in base_tab.elements().items():
        locale = base_tab.locale(elem_id, lang)
        updates.append(gr.update(**locale))
    base_tab.set_lang(lang)
    return updates


def run_ui():
    with gr.Blocks() as app:
        with gr.Tabs():
            LLMTrain.build_ui(LLMTrain)
            LLMInfer.build_ui(LLMInfer)

        app.load(partial(reload_locale, base_tab=LLMTrain), outputs=list(LLMTrain.elements().values()))
        app.load(partial(reload_locale, base_tab=LLMInfer), outputs=list(LLMInfer.elements().values()))

    app.queue().launch(height=800, share=False)
