import gradio as gr

from swift.ui.element import extras
from swift.ui.llm.advanced import advanced
from swift.ui.llm.dataset import dataset
from swift.ui.llm.hyper import hyper
from swift.ui.llm.lora import lora
from swift.ui.llm.model import model
from swift.ui.llm.save import save


def llm_train():
    with gr.Blocks():
        model()
        dataset()
        with gr.Row():
            gr.Dropdown(
                elem_id='sft_type',
                choices=[extras['full_param'], extras['lora']],
                scale=4)
            gr.Slider(elem_id='neftune_alpha', minimum=0.0, maximum=1.0, step=0.05, scale=4)
            gr.Textbox(elem_id='output_dir', scale=20)
        lora()
        hyper()
        advanced()
        save()
