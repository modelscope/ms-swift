import gradio as gr

from swift.llm import MODEL_MAPPING
from swift.ui.element import elements, extras


def hyper():
    with gr.Accordion(elem_id="hyper_param", label=extras['hyper_param'], open=False):
        with gr.Blocks() as block:
            with gr.Row():
                batch_size = gr.Textbox(elem_id='batch_size', lines=1, scale=20)
                learning_rate = gr.Textbox(elem_id='learning_rate', lines=1, scale=20)
                max_tokens = gr.Textbox(elem_id='max_tokens', lines=1, scale=20)
                eval_steps = gr.Textbox(elem_id='eval_steps', lines=1, scale=20)
                num_train_epochs = gr.Textbox(elem_id='num_train_epochs', lines=1, scale=20)
                gradient_accumulation_steps = gr.Textbox(elem_id='gradient_accumulation_steps', lines=1, scale=20)
