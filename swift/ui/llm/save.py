import gradio as gr

from swift.llm import MODEL_MAPPING
from swift.ui.element import elements


def save():
    with gr.Accordion(elem_id="save_param", label='save_param', open=False):
        with gr.Blocks() as block:
            with gr.Row():
                save_steps = gr.Textbox(elem_id='save_steps', lines=1, scale=20)
                output_dir = gr.Textbox(elem_id='output_dir', lines=1, scale=20)
