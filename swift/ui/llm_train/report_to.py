# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class ReportTo(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'reporter_tab': {
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
                'zh': 'SwanLab登录token',
                'en': 'The login token of SwanLab'
            },
        },
        'swanlab_project': {
            'label': {
                'zh': 'SwanLab项目名称',
                'en': 'Project of SwanLab'
            },
        },
        'swanlab_workspace': {
            'label': {
                'zh': 'SwanLab工作空间',
                'en': 'Workspace of SwanLab'
            },
        },
        'swanlab_exp_name': {
            'label': {
                'zh': 'SwanLab实验名称',
                'en': 'Experiment of SwanLab'
            },
        },
        'swanlab_lark_webhook_url': {
            'label': {
                'zh': 'SwanLab飞书Webhook地址',
                'en': 'Webhook URL of SwanLab Lark Callback'
            },
        },
        'swanlab_lark_secret': {
            'label': {
                'zh': 'SwanLab飞书Secret',
                'en': 'Secret of SwanLab Lark Callback'
            },
        },
        'swanlab_mode': {
            'label': {
                'zh': 'SwanLab工作模式',
                'en': 'Work mode of SwanLab'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='reporter_tab'):
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
                    gr.Textbox(elem_id='swanlab_lark_webhook_url', lines=1, scale=20)
                    gr.Textbox(elem_id='swanlab_lark_secret', lines=1, scale=20)
                with gr.Row():
                    gr.Textbox(elem_id='swanlab_workspace', lines=1, scale=20)
                    gr.Textbox(elem_id='swanlab_exp_name', lines=1, scale=20)
                    gr.Dropdown(elem_id='swanlab_mode', scale=20)
