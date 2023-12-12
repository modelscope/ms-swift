import gradio as gr

from swift.llm import MODEL_MAPPING
from swift.ui.element import elements, extras


def advanced():
    with gr.Accordion(elem_id="advanced_param", label=extras['advanced_param'], open=False):
        with gr.Blocks() as block:
            with gr.Row():
                optim = gr.Textbox(elem_id='optim', lines=1, scale=20)
                weight_decay = gr.Textbox(elem_id='weight_decay', lines=1, scale=20)
                logging_steps = gr.Textbox(elem_id='logging_steps', lines=1, scale=20)
                use_flash_attn = gr.Textbox(elem_id='use_flash_attn', lines=1, scale=20)
            with gr.Row():
                more_params = gr.Textbox(elem_id='more_params', lines=1, scale=20)
