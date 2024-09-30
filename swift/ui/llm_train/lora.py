from typing import Type

import gradio as gr

from swift.llm import MODEL_MAPPING, get_default_lora_target_modules
from swift.ui.base import BaseUI


class LoRA(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'lora_tab': {
            'label': {
                'zh': 'LoRA参数设置',
                'en': 'LoRA settings'
            },
        },
        'lora_target_modules': {
            'label': {
                'zh': 'LoRA目标模块',
                'en': 'LoRA target modules'
            },
            'info': {
                'zh': '设置LoRA目标模块，如训练所有Linear请改为ALL',
                'en': 'Set the LoRA target modules, fill in ALL if train all Linears'
            }
        },
        'lora_rank': {
            'label': {
                'zh': 'LoRA的秩',
                'en': 'The LoRA rank'
            }
        },
        'lora_alpha': {
            'label': {
                'zh': 'LoRA的alpha',
                'en': 'The LoRA alpha'
            }
        },
        'lora_dropout': {
            'label': {
                'zh': 'LoRA的dropout',
                'en': 'The LoRA dropout'
            }
        },
        'use_rslora': {
            'label': {
                'zh': '使用rslora',
                'en': 'Use rslora'
            }
        },
        'use_dora': {
            'label': {
                'zh': '使用dora',
                'en': 'Use dora'
            }
        },
        'lora_dtype': {
            'label': {
                'zh': 'lora部分的参数类型',
                'en': 'The dtype of lora parameters'
            }
        },
        'init_lora_weights': {
            'label': {
                'zh': 'lora初始化方法',
                'en': 'init lora weights'
            },
            'info': {
                'zh': 'gaussian/pissa/pissa_niter_[n]/olora/loftq/true/false',
                'en': 'gaussian/pissa/pissa_niter_[n]/olora/loftq/true/false',
            }
        },
        'lora_lr_ratio': {
            'label': {
                'zh': 'Lora+学习率倍率',
                'en': 'The lr ratio of Lora+'
            },
            'info': {
                'zh': '建议值16.0',
                'en': 'Suggested value: 16.0'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='lora_tab', open=True):
            with gr.Blocks():
                with gr.Row():
                    lora_target_modules = gr.Textbox(
                        elem_id='lora_target_modules', lines=1, scale=5, value='ALL', is_list=True)
                    gr.Slider(elem_id='lora_rank', value=32, minimum=1, maximum=512, step=8, scale=2)
                    gr.Slider(elem_id='lora_alpha', value=8, minimum=1, maximum=512, step=8, scale=2)
                    gr.Textbox(elem_id='lora_dropout', scale=2)
                with gr.Row():
                    gr.Dropdown(elem_id='lora_dtype', scale=2)
                    gr.Textbox(elem_id='lora_lr_ratio', scale=2)
                    gr.Checkbox(elem_id='use_rslora', scale=2)
                    gr.Checkbox(elem_id='use_dora', scale=2)
                    gr.Textbox(elem_id='init_lora_weights', scale=4)

            def update_lora(choice):
                if choice is not None:
                    target_modules = get_default_lora_target_modules(choice)
                    if isinstance(target_modules, list):
                        target_modules = 'ALL'  # llm
                    return target_modules
                return None

            base_tab.element('model_type').change(
                update_lora, inputs=[base_tab.element('model_type')], outputs=[lora_target_modules])
