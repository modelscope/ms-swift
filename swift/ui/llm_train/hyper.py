from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Hyper(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'hyper_param': {
            'label': {
                'zh': '超参数',
                'en': 'Hyper settings',
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
        'eval_batch_size': {
            'label': {
                'zh': '验证batch size',
                'en': 'Val batch size',
            },
            'info': {
                'zh': '验证的batch size',
                'en': 'Set the val batch size',
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
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='hyper_param', open=True):
            with gr.Blocks():
                with gr.Row():
                    gr.Slider(elem_id='batch_size', minimum=1, maximum=256, step=2, scale=20)
                    learning_rate = gr.Textbox(elem_id='learning_rate', value='1e-4', lines=1, scale=20)
                    gr.Textbox(elem_id='num_train_epochs', lines=1, scale=20)
                    gr.Textbox(elem_id='max_steps', lines=1, scale=20)
                    gr.Slider(elem_id='gradient_accumulation_steps', minimum=1, maximum=256, step=2, value=16, scale=20)
                with gr.Row():
                    gr.Slider(elem_id='eval_batch_size', minimum=1, maximum=256, step=2, scale=20)
                    gr.Textbox(elem_id='eval_steps', lines=1, value='500', scale=20)
                    gr.Textbox(elem_id='max_grad_norm', lines=1, scale=20)
                    gr.Checkbox(elem_id='predict_with_generate', scale=20)
                    gr.Checkbox(elem_id='use_flash_attn', scale=20)

            def update_lr(sft_type):
                if sft_type == 'full':
                    return 1e-5
                else:
                    return 1e-4

            base_tab.element('sft_type').change(
                update_lr, inputs=[base_tab.element('sft_type')], outputs=[learning_rate])
