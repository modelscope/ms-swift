# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Quantization(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'quantization_tab': {
            'label': {
                'zh': '量化参数设置',
                'en': 'Quantization settings'
            },
        },
        'quant_method': {
            'label': {
                'zh': '量化方式',
                'en': 'Quantization method'
            },
            'info': {
                'zh': '如果制定了量化位数，本参数默认为bnb',
                'en': 'Default is bnb if quantization_bit is specified'
            }
        },
        'quant_bits': {
            'label': {
                'zh': '量化bit数',
                'en': 'Quantization bit'
            },
            'info': {
                'zh': '设置量化bit数, 0代表不进行量化',
                'en': 'Set the quantization bit, 0 for no quantization'
            }
        },
        'bnb_4bit_compute_dtype': {
            'label': {
                'zh': '计算数据类型',
                'en': 'Computational data type'
            },
        },
        'bnb_4bit_quant_type': {
            'label': {
                'zh': '量化数据类型',
                'en': 'Quantization data type'
            },
        },
        'bnb_4bit_use_double_quant': {
            'label': {
                'zh': '使用嵌套量化',
                'en': 'Use double quantization'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='quantization_tab'):
            with gr.Row():
                gr.Dropdown(elem_id='quant_bits', value=None)
                gr.Dropdown(elem_id='quant_method', value=None)
                gr.Dropdown(elem_id='bnb_4bit_compute_dtype', value=None)
                gr.Dropdown(elem_id='bnb_4bit_quant_type', value=None)
                gr.Checkbox(elem_id='bnb_4bit_use_double_quant', value=None)
