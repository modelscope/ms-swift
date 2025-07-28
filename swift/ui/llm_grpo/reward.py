# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Type

import gradio as gr

from swift.llm import BaseArguments, ModelType
from swift.llm.model.register import get_all_models
from swift.ui.base import BaseUI


class Reward(BaseUI):
    group = 'llm_grpo'

    locale_dict = {
        'reward_funcs': {
            'label': {
                'zh': '奖励函数',
                'en': 'Reward functions'
            },
            'info': {
                'zh': 'GRPO算法奖励函数',
                'en': 'GRPO algorithm reward function'
            }
        },
        'reward_weights': {
            'label': {
                'zh': '奖励函数权重',
                'en': 'The weight of each reward function'
            },
            'info': {
                'zh': '各奖励函数的权重之间用空格隔开',
                'en': 'The weights of each reward function are separated by spaces'
            }
        },
        'reward_param': {
            'label': {
                'zh': '奖励模型设置(更多参数->GRPO高级参数设置)',
                'en': 'Reward settings(more params->GRPO advanced settings)'
            },
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='reward_param', open=True):
            with gr.Row():
                gr.Dropdown(
                    elem_id='reward_funcs',
                    multiselect=True,
                    choices=['accuracy', 'format', 'cosine', 'repetition', 'soft_overlong'],
                    scale=2,
                    allow_custom_value=True)
                gr.Textbox(elem_id='reward_weights', lines=1, scale=2)
