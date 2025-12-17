# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Type

import gradio as gr

from swift.llm import ModelType
from swift.llm.model.register import get_all_models
from swift.ui.base import BaseUI


class RLHF(BaseUI):

    group = 'llm_rlhf'

    locale_dict = {
        'rlhf_tab': {
            'label': {
                'zh': '对齐参数设置',
                'en': 'Alignment params settings'
            },
        },
        'ref_model': {
            'label': {
                'zh': 'Ref模型id或路径',
                'en': 'Ref model id or path'
            },
            'info': {
                'zh': '实际的模型id或路径',
                'en': 'The actual model id or path'
            }
        },
        'ref_model_type': {
            'label': {
                'zh': 'Ref模型类型',
                'en': 'Ref model type'
            },
            'info': {
                'zh': 'SWIFT已支持的模型类型',
                'en': 'Model type supported by SWIFT'
            }
        },
        'reward_model': {
            'label': {
                'zh': '奖励模型id或路径',
                'en': 'Reward model id or path'
            },
            'info': {
                'zh': '实际的模型id或路径',
                'en': 'The actual model id or path'
            }
        },
        'reward_model_type': {
            'label': {
                'zh': '奖励模型类型',
                'en': 'Reward model type'
            },
            'info': {
                'zh': 'SWIFT已支持的模型类型',
                'en': 'Model type supported by SWIFT'
            }
        },
        'teacher_model': {
            'label': {
                'zh': '教师模型id或路径',
                'en': 'Teacher model id or path'
            },
            'info': {
                'zh': '实际的模型id或路径',
                'en': 'The actual model id or path'
            }
        },
        'teacher_model_type': {
            'label': {
                'zh': '教师模型类型',
                'en': 'Teacher model type'
            },
            'info': {
                'zh': 'SWIFT已支持的模型类型',
                'en': 'Model type supported by SWIFT'
            }
        },
        'beta': {
            'label': {
                'zh': 'KL正则项系数',
                'en': 'KL regression ratio'
            },
        },
        'max_completion_length': {
            'label': {
                'zh': '最大生成长度',
                'en': 'Max completion length'
            },
        },
        'loss_scale': {
            'label': {
                'zh': '损失权重设置',
                'en': 'Loss weights setting'
            },
        },
        'lmbda': {
            'label': {
                'zh': 'GKD学生数据比例',
                'en': 'GKD student data ratio'
            },
        },
        'cpo_alpha': {
            'label': {
                'zh': 'CPO/SimPO中NLL损失系数',
                'en': 'CPO/SimPO NLL loss coefficient'
            },
        },
        'rpo_alpha': {
            'label': {
                'zh': 'DPO中混合sft交叉熵的系数',
                'en': 'DPO Cross Entropy ratio'
            },
        },
        'simpo_gamma': {
            'label': {
                'zh': 'SimPO reward margin',
                'en': 'SimPO reward margin'
            },
        },
        'desirable_weight': {
            'label': {
                'zh': 'KTO符合项系数',
                'en': 'KTO desirable ratio'
            },
        },
        'undesirable_weight': {
            'label': {
                'zh': 'KTO不符合项系数',
                'en': 'KTO undesirable ratio'
            },
        }
    }

    rlhf_args_dict = {
        'dpo': ['rpo_alpha', 'ref_model', 'ref_model_type'],
        'cpo': ['cpo_alpha'],
        'kto': ['desirable_weight', 'undesirable_weight', 'ref_model', 'ref_model_type'],
        'simpo': ['simpo_gamma', 'cpo_alpha'],
        'gkd': ['teacher_model', 'teacher_model_type', 'max_completion_length', 'lmbda'],
        'ppo': ['reward_model', 'reward_model_type', 'max_completion_length', 'ref_model', 'ref_model_type']
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='rlhf_tab', open=False):
            with gr.Blocks():
                with gr.Row():
                    gr.Slider(elem_id='beta', minimum=0., maximum=5.0, step=0.1, value=0.1, scale=10)
                    gr.Slider(elem_id='rpo_alpha', minimum=0., maximum=2, step=0.1, scale=10)
                    gr.Slider(elem_id='lmbda', minimum=0., maximum=1.0, step=0.1, scale=10)
                    gr.Slider(elem_id='simpo_gamma', minimum=0., maximum=2.0, step=0.1, scale=10)
                    gr.Slider(elem_id='desirable_weight', minimum=0., maximum=2.0, step=0.1, scale=10)
                    gr.Slider(elem_id='undesirable_weight', minimum=0., maximum=2.0, step=0.1, scale=10)
                with gr.Row():
                    gr.Textbox(elem_id='max_completion_length', scale=10)
                    gr.Textbox(elem_id='loss_scale', scale=10)
                    gr.Slider(elem_id='cpo_alpha', minimum=0., maximum=1, step=0.1, scale=10)
                    gr.Dropdown(
                        elem_id='teacher_model',
                        scale=20,
                        value=None,
                        choices=get_all_models(),
                        allow_custom_value=True)
                    gr.Dropdown(
                        elem_id='teacher_model_type',
                        choices=ModelType.get_model_name_list(),
                        value=None,
                        scale=10,
                        allow_custom_value=True)
                with gr.Row():
                    gr.Dropdown(
                        elem_id='ref_model', scale=20, value=None, choices=get_all_models(), allow_custom_value=True)
                    gr.Dropdown(
                        elem_id='ref_model_type',
                        choices=ModelType.get_model_name_list(),
                        value=None,
                        scale=10,
                        allow_custom_value=True)
                    gr.Dropdown(
                        elem_id='reward_model', scale=20, value=None, choices=get_all_models(), allow_custom_value=True)
                    gr.Dropdown(
                        elem_id='reward_model_type',
                        choices=ModelType.get_model_name_list(),
                        value=None,
                        scale=10,
                        allow_custom_value=True)

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('ref_model').change(
            partial(cls.update_input_model, allow_keys=['ref_model_type'], has_record=False, is_ref_model=True),
            inputs=[cls.element('ref_model')],
            outputs=[cls.element('ref_model_type')])
        cls.element('reward_model').change(
            partial(cls.update_input_model, allow_keys=['reward_model_type'], has_record=False, is_ref_model=True),
            inputs=[cls.element('reward_model')],
            outputs=[cls.element('reward_model_type')])
        cls.element('teacher_model').change(
            partial(cls.update_input_model, allow_keys=['teacher_model_type'], has_record=False, is_ref_model=True),
            inputs=[cls.element('teacher_model')],
            outputs=[cls.element('teacher_model_type')])

    @staticmethod
    def update_beta(rlhf_type):
        beta_value_dict = {'simpo': 2., 'gkd': 0.5, 'grpo': 0.04}
        return beta_value_dict.get(rlhf_type, 0.1) if rlhf_type else 0.1
