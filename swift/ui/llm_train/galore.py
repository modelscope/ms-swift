from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Galore(BaseUI):

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
                'zh': 'Galore project matrix更新频率',
                'en': 'The updating gap of the project matrix'
            },
        },
        'galore_optim_per_parameter': {
            'label': {
                'zh': '为每个Galore Parameter创建单独的optimizer',
                'en': 'Create unique optimizer for per Galore parameter'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='galore_tab', open=False):
            with gr.Blocks():
                with gr.Row():
                    gr.Checkbox(elem_id='use_galore', scale=4)
                    gr.Slider(elem_id='galore_rank', minimum=8, maximum=256, step=8, scale=4)
                    gr.Slider(elem_id='galore_update_proj_gap', minimum=10, maximum=1000, step=50, scale=4)
                    gr.Checkbox(elem_id='galore_optim_per_parameter', scale=4)
