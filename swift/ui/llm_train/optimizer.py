# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Optimizer(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'galore_tab': {
            'label': {
                'zh': 'GaLore参数设置',
                'en': 'GaLore Settings'
            },
        },
        'use_galore': {
            'label': {
                'zh': '使用GaLore',
                'en': 'Use GaLore'
            },
            'info': {
                'zh': '使用GaLore来减少全参数训练的显存消耗',
                'en': 'Use GaLore to reduce GPU memory usage in full parameter training'
            }
        },
        'galore_rank': {
            'label': {
                'zh': 'GaLore的秩',
                'en': 'The rank of GaLore'
            },
        },
        'galore_update_proj_gap': {
            'label': {
                'zh': '投影矩阵更新间隔',
                'en': 'Projection matrix update interval'
            },
            'info': {
                'zh': 'GaLore分解矩阵的更新间隔',
                'en': 'Update interval of GaLore decomposition matrix'
            },
        },
        'galore_with_embedding': {
            'label': {
                'zh': '对嵌入层应用GaLore',
                'en': 'Use GaLore with embedding'
            },
            'info': {
                'zh': '是否对嵌入层应用GaLore',
                'en': 'Whether to apply GaLore to embedding'
            },
        },
        'lorap_tab': {
            'label': {
                'zh': 'LoRA+参数设置',
                'en': 'LoRA+ settings'
            },
        },
        'lorap_lr_ratio': {
            'label': {
                'zh': 'LoRA+学习率比率',
                'en': 'LoRA+ lr ratio'
            },
            'info': {
                'zh': '使用LoRA时指定该参数可使用LoRA+，建议值10～16',
                'en': 'When using LoRA, specify this parameter to use LoRA+, and the recommended value is 10 to 16'
            }
        },
        'muon_tab': {
            'label': {
                'zh': 'Muon参数设置',
                'en': 'Muon Settings'
            },
        },
        'use_muon': {
            'label': {
                'zh': '使用Muon',
                'en': 'Use Muon'
            },
            'info': {
                'zh': '使用Muon优化器，将在命令行参数中设置`--optimizer muon`',
                'en': 'Using the Muon optimizer, set `--optimizer muon` in the command line'
            }
        },
        'multimodal_tab': {
            'label': {
                'zh': '多模态参数设置',
                'en': 'Multimodal Settings'
            },
        },
        'vit_lr': {
            'label': {
                'zh': 'ViT的学习率',
                'en': 'Learning rate of ViT'
            },
        },
        'aligner_lr': {
            'label': {
                'zh': 'Aligner的学习率',
                'en': 'Learning rate of aligner'
            },
        },
        'optimizer_params': {
            'label': {
                'zh': '优化器参数',
                'en': 'Optimizer params'
            },
        },
    }

    tabs_to_filter = {
        'galore': ['use_galore', 'galore_with_embedding', 'galore_rank', 'galore_update_proj_gap'],
        'lorap': ['lorap_lr_ratio'],
        'multimodal': ['vit_lr', 'aligner_lr'],
        'muon': ['use_muon']
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='optimizer_params', open=False):
            with gr.Tabs():
                with gr.TabItem(elem_id='galore_tab'):
                    with gr.Row():
                        gr.Checkbox(elem_id='use_galore', scale=4)
                        gr.Checkbox(elem_id='galore_with_embedding', scale=4)
                        gr.Slider(elem_id='galore_rank', minimum=8, maximum=256, step=8, scale=4)
                        gr.Slider(elem_id='galore_update_proj_gap', minimum=10, maximum=1000, step=50, scale=4)
                with gr.TabItem(elem_id='lorap_tab'):
                    with gr.Row():
                        gr.Textbox(elem_id='lorap_lr_ratio', scale=4)
                with gr.TabItem(elem_id='multimodal_tab'):
                    with gr.Row():
                        gr.Textbox(elem_id='vit_lr', lines=1, scale=20)
                        gr.Textbox(elem_id='aligner_lr', lines=1, scale=20)
                with gr.TabItem(elem_id='muon_tab'):
                    with gr.Row():
                        gr.Checkbox(elem_id='use_muon', scale=4)
