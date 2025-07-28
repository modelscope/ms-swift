# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Advanced(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'advanced_tab': {
            'label': {
                'zh': '高级参数设置',
                'en': 'Advanced settings'
            },
        },
        'tuner_backend': {
            'label': {
                'zh': 'Tuner backend',
                'en': 'Tuner backend'
            },
            'info': {
                'zh': 'Tuner实现框架',
                'en': 'The tuner backend'
            }
        },
        'weight_decay': {
            'label': {
                'zh': '权重衰减',
                'en': 'Weight decay'
            },
            'info': {
                'zh': '设置weight decay',
                'en': 'Set the weight decay'
            }
        },
        'logging_steps': {
            'label': {
                'zh': '日志打印步数',
                'en': 'Logging steps'
            },
            'info': {
                'zh': '设置日志打印的步数间隔',
                'en': 'Set the logging interval'
            }
        },
        'lr_scheduler_type': {
            'label': {
                'zh': 'LrScheduler类型',
                'en': 'The LrScheduler type'
            },
            'info': {
                'zh': '设置LrScheduler类型',
                'en': 'Set the LrScheduler type'
            }
        },
        'warmup_ratio': {
            'label': {
                'zh': '学习率warmup比例',
                'en': 'Lr warmup ratio'
            },
            'info': {
                'zh': '设置学习率warmup比例',
                'en': 'Set the warmup ratio in total steps'
            }
        },
        'truncation_strategy': {
            'label': {
                'zh': '数据集超长策略',
                'en': 'Dataset truncation strategy'
            },
            'info': {
                'zh': '如果token超长该如何处理',
                'en': 'How to deal with the rows exceed the max length'
            }
        },
        'max_steps': {
            'label': {
                'zh': '最大迭代步数',
                'en': 'Max steps',
            },
            'info': {
                'zh': '设置最大迭代步数，该值如果大于零则数据集迭代次数不生效',
                'en': 'Set the max steps, if the value > 0 then num_train_epochs has no effects',
            }
        },
        'max_grad_norm': {
            'label': {
                'zh': '梯度裁剪',
                'en': 'Max grad norm',
            },
            'info': {
                'zh': '设置梯度裁剪',
                'en': 'Set the max grad norm',
            }
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='advanced_tab'):
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(elem_id='tuner_backend', scale=20)
                    gr.Textbox(elem_id='weight_decay', lines=1, scale=20)
                    gr.Textbox(elem_id='logging_steps', lines=1, scale=20)
                    gr.Textbox(elem_id='lr_scheduler_type', lines=1, scale=20)
                with gr.Row():
                    gr.Dropdown(elem_id='truncation_strategy', value=None, scale=20)
                    gr.Textbox(elem_id='max_steps', lines=1, scale=20)
                    gr.Textbox(elem_id='max_grad_norm', lines=1, scale=20)
                    gr.Slider(elem_id='warmup_ratio', minimum=0.0, maximum=1.0, step=0.05, scale=20)
