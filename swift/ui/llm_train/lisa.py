# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Lisa(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'lisa_tab': {
            'label': {
                'zh': 'LISA参数设置',
                'en': 'LISA settings'
            },
        },
        'lisa_activated_layers': {
            'label': {
                'zh': 'LISA激活层数',
                'en': 'LoRA activated layers'
            },
            'info': {
                'zh': 'LISA每次训练的模型层数，调整为正整数代表使用LISA',
                'en': 'Num of layers activated each time, a positive value means using lisa'
            }
        },
        'lisa_step_interval': {
            'label': {
                'zh': 'LISA切换layers间隔',
                'en': 'The interval of lisa layers switching'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='lisa_tab', open=False):
            with gr.Blocks():
                with gr.Row():
                    gr.Textbox(elem_id='lisa_activated_layers')
                    gr.Textbox(elem_id='lisa_step_interval')
