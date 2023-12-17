import os
from functools import partial

import gradio as gr

from swift.ui.base import BaseUI, all_langs
from swift.ui.llm_infer.llm_infer import LLMInfer
from swift.ui.llm_train.llm_train import LLMTrain


def run_ui():
    with gr.Blocks() as app:
        with gr.Tabs():
            LLMTrain.build_ui(LLMTrain)
            LLMInfer.build_ui(LLMInfer)

    app.queue().launch(height=800, share=False)
