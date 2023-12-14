import gradio as gr

from swift.ui.i18n import get_i18n_labels
from swift.ui.llm.utils import get_choices


def dataset():
    get_i18n_labels(i18n)
    with gr.Accordion(elem_id='quantization_tab', open=False):
        with gr.Row():
            gr.Slider(elem_id='quantization_bit', minimum=0, maximum=8, step=4)
            gr.Dropdown(
                elem_id='bnb_4bit_comp_dtype',
                choices=get_choices('bnb_4bit_comp_dtype'))
            gr.Dropdown(
                elem_id='bnb_4bit_quant_type',
                choices=get_choices('bnb_4bit_quant_type'))
            gr.Checkbox(elem_id='bnb_4bit_use_double_quant')


i18n = {
    'quantization_tab': {
        'label': {
            'zh': '量化参数',
            'en': 'Quantization'
        },
    },
    'quantization_bit': {
        'label': {
            'zh': '量化bit数',
            'en': 'Quantization bit'
        },
        'info': {
            'zh': '设置量化bit数, 0代表不进行量化',
            'en': 'Set the quantization bit, 0 for no quantization'
        }
    },
    'bnb_4bit_comp_dtype': {
        'label': {
            'zh': 'bnb_4bit_comp_dtype',
            'en': 'bnb_4bit_comp_dtype'
        },
    },
    'bnb_4bit_quant_type': {
        'label': {
            'zh': 'bnb_4bit_quant_type',
            'en': 'bnb_4bit_quant_type'
        },
    },
    'bnb_4bit_use_double_quant': {
        'label': {
            'zh': 'bnb_4bit_use_double_quant',
            'en': 'bnb_4bit_use_double_quant'
        },
    },
}
