from typing import Type

import gradio as gr

from swift.llm import DATASET_MAPPING
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
        'merge_device_map': {
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
        'quant_seqlen': {
            'label': {
                'zh': '量化集的max-length',
                'en': 'The quantize sequence length'
            },
        },
        'quant_output_dir': {
            'label': {
                'zh': '量化输出路径',
                'en': 'Output dir for quantization'
            },
            'info': {
                'zh':
                '如果仅merge-lora不需要修改这里, 留空时量化输出在当前目录的<model-type>-<quant_method>-<quant_bits>下',
                'en':
                'If only merging LoRA, no need to modify this. When left blank, '
                'the output will be in the current directory under <model-type>-<quant_method>-<quant_bits>'
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
            gr.Textbox(elem_id='merge_device_map', scale=20)
        with gr.Row():
            gr.Textbox(elem_id='quant_bits', scale=20)
            gr.Dropdown(elem_id='quant_method', scale=20)
            gr.Textbox(elem_id='quant_n_samples', scale=20)
            gr.Textbox(elem_id='quant_seqlen', scale=20)
        with gr.Row():
            gr.Textbox(elem_id='quant_output_dir', scale=20)
            gr.Dropdown(
                elem_id='dataset',
                multiselect=True,
                allow_custom_value=True,
                choices=list(DATASET_MAPPING.keys()),
                scale=20)
