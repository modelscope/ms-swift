# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Target(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'target_params': {
            'label': {
                'zh': 'Target模块参数',
                'en': 'Tuner modules params'
            }
        },
        'freeze_llm': {
            'label': {
                'zh': '冻结LLM',
                'en': 'Freeze LLM'
            },
        },
        'freeze_aligner': {
            'label': {
                'zh': '冻结aligner',
                'en': 'Freeze aligner'
            },
        },
        'freeze_vit': {
            'label': {
                'zh': '冻结ViT',
                'en': 'Freeze ViT'
            },
        },
        'target_modules': {
            'label': {
                'zh': '指定tuner模块',
                'en': 'Specify the tuner module'
            }
        },
        'target_regex': {
            'label': {
                'zh': 'Tuner模块regex表达式',
                'en': 'Tuner module regex expression'
            }
        },
        'modules_to_save': {
            'label': {
                'zh': '额外训练和存储的原模型模块',
                'en': 'Original model modules to train and save'
            }
        },
        'init_weights': {
            'label': {
                'zh': 'Tuner初始化方法',
                'en': 'Init tuner weights'
            },
            'info': {
                'zh': ('LoRA: gaussian/pissa/pissa_niter_[n]/olora/loftq/lora-ga/true/false,'
                       'Bone: bat/true/false'),
                'en': ('LoRA: gaussian/pissa/pissa_niter_[n]/olora/loftq/lora-ga/true/false,'
                       'Bone: bat/true/false'),
            }
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Blocks():
            with gr.Row():
                gr.Textbox(elem_id='target_modules', lines=1, value='all-linear', is_list=True, scale=5)
                gr.Checkbox(elem_id='freeze_llm', scale=5)
                gr.Checkbox(elem_id='freeze_aligner', scale=5)
                gr.Checkbox(elem_id='freeze_vit', scale=5)
            with gr.Row():
                gr.Textbox(elem_id='target_regex', scale=5)
                gr.Textbox(elem_id='modules_to_save', scale=5)
                gr.Textbox(elem_id='init_weights', scale=5)
