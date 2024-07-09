from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Hyper(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'hyper_param': {
            'label': {
                'zh': '超参数设置(更多参数->高级参数设置)',
                'en': 'Hyper settings(more params->Advanced settings)',
            },
        },
        'batch_size': {
            'label': {
                'zh': '训练batch size',
                'en': 'Train batch size',
            },
            'info': {
                'zh': '训练的batch size',
                'en': 'Set the train batch size',
            }
        },
        'learning_rate': {
            'label': {
                'zh': '学习率',
                'en': 'Learning rate',
            },
            'info': {
                'zh': '设置学习率',
                'en': 'Set the learning rate',
            }
        },
        'eval_steps': {
            'label': {
                'zh': '交叉验证步数',
                'en': 'Eval steps',
            },
            'info': {
                'zh': '设置每隔多少步数进行一次验证',
                'en': 'Set the step interval to validate',
            }
        },
        'num_train_epochs': {
            'label': {
                'zh': '数据集迭代轮次',
                'en': 'Train epoch',
            },
            'info': {
                'zh': '设置对数据集训练多少轮次',
                'en': 'Set the max train epoch',
            }
        },
        'gradient_accumulation_steps': {
            'label': {
                'zh': '梯度累计步数',
                'en': 'Gradient accumulation steps',
            },
            'info': {
                'zh': '设置梯度累计步数以减小显存占用',
                'en': 'Set the gradient accumulation steps',
            }
        },
        'use_flash_attn': {
            'label': {
                'zh': '使用Flash Attention',
                'en': 'Use Flash Attention',
            },
            'info': {
                'zh': '使用Flash Attention减小显存占用',
                'en': 'Use Flash Attention to reduce memory',
            }
        },
        'neftune_noise_alpha': {
            'label': {
                'zh': 'neftune_noise_alpha',
                'en': 'neftune_noise_alpha'
            },
            'info': {
                'zh': '使用neftune提升训练效果, 一般设置为5或者10',
                'en': 'Use neftune to improve performance, normally the value should be 5 or 10'
            }
        },
        'save_steps': {
            'label': {
                'zh': '存储步数',
                'en': 'save steps',
            },
            'info': {
                'zh': '设置每个多少步数进行存储',
                'en': 'Set the save steps',
            }
        },
        'output_dir': {
            'label': {
                'zh': '存储目录',
                'en': 'The output dir',
            },
            'info': {
                'zh': '设置输出模型存储在哪个文件夹下',
                'en': 'Set the output folder',
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='hyper_param', open=False):
            with gr.Blocks():
                with gr.Row():
                    gr.Slider(elem_id='batch_size', minimum=1, maximum=256, step=2, scale=20)
                    gr.Textbox(elem_id='learning_rate', value='1e-4', lines=1, scale=20)
                    gr.Textbox(elem_id='num_train_epochs', lines=1, scale=20)
                    gr.Checkbox(elem_id='use_flash_attn', scale=20)
                    gr.Slider(elem_id='gradient_accumulation_steps', minimum=1, maximum=256, step=2, value=16, scale=20)
                with gr.Row():
                    gr.Textbox(elem_id='eval_steps', lines=1, value='500', scale=20)
                    gr.Textbox(elem_id='save_steps', value='500', lines=1, scale=20)
                    gr.Textbox(elem_id='output_dir', scale=20)
                    gr.Slider(elem_id='neftune_noise_alpha', minimum=0.0, maximum=20.0, step=0.5, scale=20)

    @staticmethod
    def update_lr(sft_type):
        if sft_type == 'full':
            return 1e-5
        else:
            return 1e-4
