import os

import gradio as gr
import json
from .i18n import extras
from swift.llm.utils.model import MODEL_MAPPING
from swift.llm.utils.dataset import DATASET_MAPPING
from .lora_hparams import lora_train


def llm_train():
    with gr.Blocks() as block:
        with gr.Row():
            model_code = gr.Dropdown(elem_id='model_code', choices=list(MODEL_MAPPING.keys()), scale=20)
            model_id = gr.Textbox(elem_id='model_id', lines=1, scale=20)

        def update_input_model(choice):
            return MODEL_MAPPING[choice]['model_id_or_path']

        model_code.change(update_input_model, inputs=[model_code], outputs=[model_id])

        with gr.Row():
            dataset_codes = gr.Dropdown(elem_id='dataset_code',
                                        multiselect=True,
                                        choices=list(DATASET_MAPPING.keys()), scale=20)
            custom_dataset_codes = gr.Textbox(elem_id='custom_dataset', scale=20)
        with gr.Row():
            model_code = gr.Dropdown(elem_id='sft_type',
                                     choices=[extras['full_param'], extras['lora']], scale=20)

        lora_train()
