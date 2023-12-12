import gradio as gr


def lora_train():
    with gr.Accordion(elem_id="lora_tab", label='asdf', open=False):
        with gr.Blocks() as block:
            with gr.Row():
                lora_rank = gr.Textbox(elem_id='lora_rank', lines=1, scale=20)
                lora_alpha = gr.Textbox(elem_id='lora_alpha', lines=1, scale=20)
                lora_dropout_p = gr.Textbox(elem_id='lora_dropout_p', lines=1, scale=20)