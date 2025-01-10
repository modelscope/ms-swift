# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Advanced(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'advanced_param': {
            'label': {
                'zh': '高级参数设置',
                'en': 'Advanced settings'
            },
        },
        'optim': {
            'label': {
                'zh': 'Optimizer类型',
                'en': 'The Optimizer type'
            },
            'info': {
                'zh': '设置Optimizer类型',
                'en': 'Set the Optimizer type'
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
        'more_params': {
            'label': {
                'zh': '其他高级参数',
                'en': 'Other params'
            },
            'info': {
                'zh': '以json格式或--xxx xxx命令行格式填入',
                'en': 'Fill in with json format or --xxx xxx cmd format'
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
        'per_device_eval_batch_size': {
            'label': {
                'zh': '验证batch size',
                'en': 'Val batch size',
            },
            'info': {
                'zh': '验证的batch size',
                'en': 'Set the val batch size',
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
        },
        'predict_with_generate': {
            'label': {
                'zh': '使用生成指标代替loss',
                'en': 'Use generate metric instead of loss',
            },
            'info': {
                'zh': '验证时使用generate/Rouge代替loss',
                'en': 'Use model.generate/Rouge instead of loss',
            }
        },
        'deepspeed': {
            'label': {
                'zh': 'deepspeed',
                'en': 'deepspeed',
            },
            'info': {
                'zh': '可以选择下拉列表，也支持传入路径',
                'en': 'Choose from the dropbox or fill in a valid path',
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='advanced_param', open=False):
            with gr.Blocks():
                with gr.Row():
                    gr.Textbox(elem_id='optim', lines=1, scale=20)
                    gr.Textbox(elem_id='weight_decay', lines=1, scale=20)
                    gr.Textbox(elem_id='logging_steps', lines=1, scale=20)
                    gr.Textbox(elem_id='lr_scheduler_type', lines=1, scale=20)
                    gr.Textbox(elem_id='max_steps', lines=1, scale=20)
                    gr.Slider(elem_id='warmup_ratio', minimum=0.0, maximum=1.0, step=0.05, scale=20)
                with gr.Row():
                    gr.Dropdown(elem_id='truncation_strategy', scale=20)
                    gr.Slider(elem_id='per_device_eval_batch_size', minimum=1, maximum=256, step=2, scale=20)
                    gr.Textbox(elem_id='max_grad_norm', lines=1, scale=20)
                    gr.Dropdown(
                        elem_id='deepspeed',
                        scale=20,
                        allow_custom_value=True,
                        value=None,
                        choices=['zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'])
                with gr.Row():
                    gr.Textbox(elem_id='more_params', lines=4, scale=20)
