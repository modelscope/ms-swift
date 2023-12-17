from functools import partial

import gradio as gr

from swift.ui.base import BaseUI
from swift.ui.llm_infer.llm_infer import LLMInfer
from swift.ui.llm_train.llm_train import LLMTrain

all_langs = ['中文', 'en']

locale_dict = {
    'locale': {
        'label': {
            'zh': '选择语言',
            'en': 'Select Language',
        }
    }
}


def reload_locale(lang, base_tab: BaseUI):
    """Reload labels"""
    updates = []
    for element in base_tab.elements():
        elem_id = element.elem_id
        locale = base_tab.locale(elem_id, lang)
        updates.append(gr.update(**locale))
    LLMTrain.set_lang(lang)
    LLMInfer.set_lang(lang)
    return updates


def run_ui():
    with gr.Blocks() as app:
        locale = gr.Dropdown(elem_id='locale', choices=all_langs)
        with gr.Tabs():
            LLMTrain.build_ui(LLMTrain)
            LLMInfer.build_ui(LLMInfer)

        locale.change(partial(reload_locale, base_tab=LLMTrain), [locale], list(LLMTrain.elements().values()))
        locale.change(partial(reload_locale, base_tab=LLMInfer), [locale], list(LLMInfer.elements().values()))

    app.queue().launch(height=800, share=False)
