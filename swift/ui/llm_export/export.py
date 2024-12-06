# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.llm.dataset.register import get_dataset_list
from swift.ui.base import BaseUI


class Export(BaseUI):

    group = 'llm_export'

    locale_dict = {
        'merge_lora': {
            'label': {
                'zh': '合并lora',
                'en': 'Merge lora'
            },
            'info': {
                'zh':
                'lora合并的路径在填入的checkpoint同级目录，请查看运行时log获取更具体的信息',
                'en':
                'The output path is in the sibling directory as the input checkpoint. '
                'Please refer to the runtime log for more specific information.'
            },
        },
        'device_map': {
            'label': {
                'zh': '合并lora使用的device_map',
                'en': 'The device_map when merge-lora'
            },
            'info': {
                'zh': '如果显存不够请填入cpu',
                'en': 'If GPU memory is not enough, fill in cpu'
            },
        },
        'quant_bits': {
            'label': {
                'zh': '量化比特数',
                'en': 'Quantize bits'
            },
        },
        'quant_method': {
            'label': {
                'zh': '量化方法',
                'en': 'Quantize method'
            },
        },
        'quant_n_samples': {
            'label': {
                'zh': '量化集采样数',
                'en': 'Sampled rows from calibration dataset'
            },
        },
        'max_length': {
            'label': {
                'zh': '量化集的max-length',
                'en': 'The quantize sequence length'
            },
        },
        'output_dir': {
            'label': {
                'zh': '输出路径',
                'en': 'Output dir'
            },
        },
        'dataset': {
            'label': {
                'zh': '校准数据集',
                'en': 'Calibration datasets'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row():
            gr.Checkbox(elem_id='merge_lora', scale=10)
            gr.Textbox(elem_id='device_map', scale=20)
        with gr.Row():
            gr.Dropdown(elem_id='quant_bits', scale=20)
            gr.Dropdown(elem_id='quant_method', scale=20)
            gr.Textbox(elem_id='quant_n_samples', scale=20)
            gr.Textbox(elem_id='max_length', scale=20)
        with gr.Row():
            gr.Textbox(elem_id='output_dir', scale=20)
            gr.Dropdown(
                elem_id='dataset', multiselect=True, allow_custom_value=True, choices=get_dataset_list(), scale=20)
