# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class SelfCog(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'self_cognition': {
            'label': {
                'zh': '自我认知任务参数设置',
                'en': 'Self cognition settings'
            },
        },
        'self_cognition_sample': {
            'label': {
                'zh': '数据及采样条数',
                'en': 'Dataset sample size'
            },
            'info': {
                'zh': '设置数据集采样的条数',
                'en': 'Set the dataset sample size'
            }
        },
        'model_name': {
            'label': {
                'zh': '模型认知名称',
                'en': 'Model name'
            },
            'info': {
                'zh': '设置模型应当认知自己的名字, 格式为:中文名字 英文名字,中间以空格分隔',
                'en': 'Set the name of the model think itself of, the format is Chinesename Englishname, split by space'
            }
        },
        'model_author': {
            'label': {
                'zh': '模型作者',
                'en': 'Model author'
            },
            'info': {
                'zh': '设置模型认知的自己的作者, 格式为:中文作者 英文作者,中间以空格分隔',
                'en': 'Set the author of the model, the format is Chineseauthor Englishauthor, split by space'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='self_cognition', open=False):
            with gr.Row():
                gr.Textbox(elem_id='model_name', scale=20, is_list=True)
                gr.Textbox(elem_id='model_author', scale=20, is_list=True)
