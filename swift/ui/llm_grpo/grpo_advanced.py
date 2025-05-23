# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class GRPOAdvanced(BaseUI):
    group = 'llm_grpo'

    locale_dict = {
        'grpo_advanced_param': {
            'label': {
                'zh': 'GRPO高级参数设置',
                'en': 'GRPO Advanced settings'
            },
        },
        'loss_type': {
            'label': {
                'zh': 'loss归一化类型',
                'en': 'Loss normalization type'
            }
        },
        'epsilon': {
            'label': {
                'zh': 'clip系数',
                'en': 'clip coefficient'
            }
        },
        'epsilon_high': {
            'label': {
                'zh': 'upper clip系数',
                'en': 'upper clip coefficient'
            }
        },
        'beta': {
            'label': {
                'zh': 'KL正则项系数',
                'en': 'KL regularization coefficient'
            }
        },
        'num_iterations': {
            'label': {
                'zh': '每个批次代更新次数',
                'en': 'Number of updates per batch'
            }
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='grpo_advanced_param', open=False):
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(elem_id='loss_type', choices=['grpo', 'bnpo', 'dr_grpo'], value='grpo', scale=20)
                    gr.Textbox(elem_id='epsilon', value=0.2, lines=1, scale=20)
                    gr.Textbox(elem_id='epsilon_high', value=None, lines=1, scale=20)
                    gr.Textbox(elem_id='beta', value=0.04, lines=1, scale=20)
                    gr.Textbox(elem_id='num_iterations', value=1, lines=1, scale=20)
