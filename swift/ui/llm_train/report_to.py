# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class ReportTo(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'reporter': {
            'label': {
                'zh': '训练记录',
                'en': 'Training report'
            },
        },
        'report_to': {
            'label': {
                'zh': '训练记录方式',
                'en': 'Report to'
            },
        },
        'swanlab_token': {
            'label': {
                'zh': 'swanlab登录token',
                'en': 'The login token of swanlab'
            },
        },
        'swanlab_project': {
            'label': {
                'zh': 'swanlab项目名称',
                'en': 'Project of swanlab'
            },
        },
        'swanlab_workspace': {
            'label': {
                'zh': 'swanlab工作空间',
                'en': 'Workspace of swanlab'
            },
        },
        'swanlab_exp_name': {
            'label': {
                'zh': 'swanlab实验名称',
                'en': 'Experiment of swanlab'
            },
        },
        'swanlab_mode': {
            'label': {
                'zh': 'swanlab工作模式',
                'en': 'Work mode of swanlab'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='reporter', open=False):
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(
                        elem_id='report_to',
                        multiselect=True,
                        is_list=True,
                        choices=['tensorboard', 'wandb', 'swanlab'],
                        allow_custom_value=True,
                        scale=20)
                    gr.Textbox(elem_id='swanlab_token', lines=1, scale=20)
                    gr.Textbox(elem_id='swanlab_project', lines=1, scale=20)
                with gr.Row():
                    gr.Textbox(elem_id='swanlab_workspace', lines=1, scale=20)
                    gr.Textbox(elem_id='swanlab_exp_name', lines=1, scale=20)
                    gr.Dropdown(elem_id='swanlab_mode', scale=20)
