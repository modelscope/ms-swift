import gradio as gr

from swift.llm import DATASET_MAPPING


def dataset():
    with gr.Row():
        dataset_codes = gr.Dropdown(elem_id='dataset_code',
                                    multiselect=True,
                                    choices=list(DATASET_MAPPING.keys()), scale=20)
        custom_dataset_codes = gr.Textbox(elem_id='custom_dataset', scale=20)
