# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Optimizer(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'galore_tab': {
            'label': {
                'zh': 'Galore参数设置',
                'en': 'Galore Settings'
            },
        },
        'use_galore': {
            'label': {
                'zh': '使用GaLore',
                'en': 'Use GaLore'
            },
            'info': {
                'zh': '使用Galore来减少全参数训练的显存消耗',
                'en': 'Use Galore to reduce GPU memory usage in full parameter training'
            }
        },
        'galore_rank': {
            'label': {
                'zh': 'Galore的秩',
                'en': 'The rank of Galore'
            },
        },
        'galore_update_proj_gap': {
            'label': {
                'zh': 'galore_update_proj_gap',
                'en': 'galore_update_proj_gap'
            },
            'info': {
                'zh': 'Galore project matrix更新频率',
                'en': 'The updating gap of the project matrix'
            },
        },
        'galore_optim_per_parameter': {
            'label': {
                'zh': 'galore_optim_per_parameter',
                'en': 'galore_optim_per_parameter'
            },
            'info': {
                'zh': '为每个Galore Parameter创建单独的optimizer',
                'en': 'Create unique optimizer for per Galore parameter'
            },
        },
        'lorap_tab': {
            'label': {
                'zh': 'lora+参数设置',
                'en': 'lora+ parameters settings'
            },
        },
        'lorap_lr_ratio': {
            'label': {
                'zh': 'LoRA+参数',
                'en': 'LoRA+ Parameters'
            },
            'info': {
                'zh': '使用lora时指定该参数可使用lora+，建议值10～16',
                'en': 'When using lora, specify this parameter to use lora+, and the recommended value is 10 to 16'
            }
        },
        'muon_tab': {
            'label': {
                'zh': 'Muon参数设置',
                'en': 'Muon Settings'
            },
        },
        'multimodal_tab': {
            'label': {
                'zh': 'Multimodal参数设置',
                'en': 'Multimodal Settings'
            },
        },
        'vit_lr': {
            'label': {
                'zh': 'vit的学习率',
                'en': 'Learning rate of vit'
            },
        },
        'aligner_lr': {
            'label': {
                'zh': 'aligner的学习率',
                'en': 'Learning rate of aligner'
            },
        },
        'optimizer': {
            'label': {
                'zh': '优化器',
                'en': 'optimizer'
            },
            'info': {
                'zh': 'plugin的自定义optimizer名称',
                'en': 'Custom optimizer name in plugin'
            }
        },
        'optimizer_params': {
            'label': {
                'zh': '优化器参数',
                'en': 'Optimizer params'
            },
        },
    }

    tabs_to_filter = {
        'galore': ['use_galore', 'galore_optim_per_parameter', 'galore_rank', 'galore_update_proj_gap'],
        'lorap': ['lorap_lr_ratio'],
        'multimodal': ['vit_lr', 'aligner_lr']
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='optimizer_params', open=False):
            gr.Dropdown(
                elem_id='optimizer',
                choices=['galore', 'lorap', 'muon', 'multimodal'],
                value='',
                allow_custom_value=True)
            with gr.Tabs():
                with gr.TabItem(elem_id='galore_tab'):
                    with gr.Row():
                        gr.Checkbox(elem_id='use_galore', scale=4)
                        gr.Checkbox(elem_id='galore_optim_per_parameter', scale=4)
                        gr.Slider(elem_id='galore_rank', minimum=8, maximum=256, step=8, scale=4)
                        gr.Slider(elem_id='galore_update_proj_gap', minimum=10, maximum=1000, step=50, scale=4)
                with gr.TabItem(elem_id='lorap_tab'):
                    with gr.Row():
                        gr.Textbox(elem_id='lorap_lr_ratio', scale=4)
                with gr.TabItem(elem_id='multimodal_tab'):
                    with gr.Row():
                        gr.Textbox(elem_id='vit_lr', lines=1, scale=20)
                        gr.Textbox(elem_id='aligner_lr', lines=1, scale=20)
