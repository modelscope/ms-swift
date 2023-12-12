import gradio as gr

from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING


def model():
    with gr.Row():
        model_code = gr.Dropdown(elem_id='model_code', choices=list(MODEL_MAPPING.keys()), scale=20)
        model_id = gr.Textbox(elem_id='model_id', lines=1, scale=20)
    with gr.Row():
        model_system = gr.Textbox(elem_id='model_system', lines=1, scale=20)

    def update_input_model(choice):
        return MODEL_MAPPING[choice]['model_id_or_path'], TEMPLATE_MAPPING[MODEL_MAPPING[choice]['template']].default_system

    model_code.change(update_input_model, inputs=[model_code], outputs=[model_id, model_system])
