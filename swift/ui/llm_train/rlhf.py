# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Type

import gradio as gr

from swift.llm import ModelType
from swift.llm.model.register import get_all_models
from swift.ui.base import BaseUI


class RLHF(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'rlhf_tab': {
            'label': {
                'zh': '人类对齐参数设置',
                'en': 'RLHF settings'
            },
        },
        'rlhf_type': {
            'label': {
                'zh': '人类对齐算法类型',
                'en': 'RLHF type'
            },
        },
        'ref_model_type': {
            'label': {
                'zh': '选择ref模型',
                'en': 'Select ref model'
            },
            'info': {
                'zh': 'SWIFT已支持的模型名称',
                'en': 'Base model supported by SWIFT'
            }
        },
        'ref_model': {
            'label': {
                'zh': 'ref模型id或路径',
                'en': 'Ref model id or path'
            },
            'info': {
                'zh': '实际的模型id或路径',
                'en': 'The actual model id or path'
            }
        },
        'beta': {
            'label': {
                'zh': 'KL正则项系数',
                'en': 'KL regression ratio'
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

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='rlhf_tab', open=False):
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(elem_id='rlhf_type', value=None)
                    gr.Dropdown(
                        elem_id='ref_model', scale=20, value=None, choices=get_all_models(), allow_custom_value=True)
                    gr.Dropdown(elem_id='ref_model_type', choices=ModelType.get_model_name_list(), value=None, scale=20)
                with gr.Row():
                    gr.Slider(elem_id='beta', minimum=0., maximum=5.0, step=0.1, scale=20)
                    gr.Slider(elem_id='rpo_alpha', minimum=0., maximum=2, step=0.1, scale=20)
                    gr.Slider(elem_id='simpo_gamma', minimum=0., maximum=2.0, step=0.1, scale=20)
                    gr.Slider(elem_id='desirable_weight', minimum=0., maximum=2.0, step=0.1, scale=20)
                    gr.Slider(elem_id='undesirable_weight', minimum=0., maximum=2.0, step=0.1, scale=20)

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('ref_model').change(
            partial(cls.update_input_model, allow_keys=['ref_model_type'], has_record=False, is_ref_model=True),
            inputs=[cls.element('ref_model')],
            outputs=[cls.element('ref_model_type')])
