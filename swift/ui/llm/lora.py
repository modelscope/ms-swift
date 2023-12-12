import gradio as gr

from swift.llm import MODEL_MAPPING
from swift.ui.element import elements,extras


def lora():
    with gr.Accordion(elem_id='lora_tab', label=extras['lora_tab'], open=False):
        with gr.Blocks() as block:
            with gr.Row():
                lora_module = gr.Textbox(
                    elem_id='lora_module', lines=1, scale=20)
            with gr.Row():
                lora_rank = gr.Slider(elem_id='lora_rank', value=32, minimum=1, maximum=512, step=8)
                lora_alpha = gr.Slider(
                    elem_id='lora_alpha', value=8, minimum=1, maximum=512, step=8)
                lora_dropout_p = gr.Textbox(
                    elem_id='lora_dropout_p', value=1.0, minimum=0.0, maximum=1.0, step=0.05)

        def update_lora(choice):
            return MODEL_MAPPING[choice]['lora_target_modules']

        elements['model_code'].change(
            update_lora,
            inputs=[elements['model_code']],
            outputs=[lora_module])
