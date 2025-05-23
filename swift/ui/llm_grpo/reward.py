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
        'reward_model_plugin': {
            'label': {
                'zh': '奖励模型逻辑',
                'en': 'Reward model logic'
            },
            'info': {
                'zh': '利用reward_model_plugin自定义奖励模型的处理逻辑',
                'en': 'Use reward_model_plugin to customize the processing logic of the reward model'
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
        'reward_model': {
            'label': {
                'zh': '奖励模型id或路径',
                'en': 'Reward Model id or path'
            },
            'info': {
                'zh': '实际的模型id',
                'en': 'The actual model id or model path'
            }
        },
        'reward_model_type': {
            'label': {
                'zh': 'reward模型类型',
                'en': 'Select Reward Model Type'
            },
            'info': {
                'zh': 'SWIFT已支持的模型类型',
                'en': 'Base model type supported by SWIFT'
            }
        },
        'reward_param': {
            'label': {
                'zh': 'reward设置',
                'en': 'Reward settings'
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
                gr.Textbox(elem_id='reward_model_plugin', lines=1, scale=3)
            with gr.Row():
                gr.Dropdown(elem_id='reward_model', multiselect=True, choices=get_all_models(), scale=20)
                gr.Dropdown(
                    elem_id='reward_model_type',
                    multiselect=True,
                    choices=ModelType.get_model_name_list(),
                    allow_custom_value=True,
                    scale=20)

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('reward_model').change(
            partial(cls.update_input_models, allow_keys=['reward_model_type'], is_reward_model=True, has_record=False),
            inputs=[cls.element('reward_model')],
            outputs=[cls.element('reward_model_type')])

    @classmethod
    def update_input_models(cls,
                            models,
                            allow_keys=None,
                            has_record=False,
                            arg_cls=BaseArguments,
                            is_reward_model=False):
        if models is None:
            return gr.update()
        rm_type_str = ''
        for model in models:
            rm_type_str = ' '.join([
                rm_type_str,
                cls.update_input_model(
                    model,
                    allow_keys=allow_keys,
                    has_record=has_record,
                    arg_cls=arg_cls,
                    is_reward_model=is_reward_model)['value']
            ])

        return gr.update(value=rm_type_str.strip())
