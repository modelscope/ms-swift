# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Type

import gradio as gr

from swift.llm import TEMPLATE_MAPPING, ModelType, RLHFArguments
from swift.llm.model.register import get_all_models
from swift.ui.base import BaseUI


class Model(BaseUI):
    group = 'llm_train'

    locale_dict = {
        'model_type': {
            'label': {
                'zh': '模型类型',
                'en': 'Select Model Type'
            },
            'info': {
                'zh': 'SWIFT已支持的模型类型',
                'en': 'Base model type supported by SWIFT'
            }
        },
        'model': {
            'label': {
                'zh': '模型id或路径',
                'en': 'Model id or path'
            },
            'info': {
                'zh': '实际的模型id',
                'en': 'The actual model id or model path'
            }
        },
        'template': {
            'label': {
                'zh': '模型Prompt模板类型',
                'en': 'Prompt template type'
            },
            'info': {
                'zh': '选择匹配模型的Prompt模板',
                'en': 'Choose the template type of the model'
            }
        },
        'system': {
            'label': {
                'zh': 'system字段',
                'en': 'system'
            },
            'info': {
                'zh': '选择system字段的内容',
                'en': 'Choose the content of the system field'
            }
        },
        'reset': {
            'value': {
                'zh': '恢复模型初始值',
                'en': 'Reset model default'
            },
        },
        'train_record': {
            'label': {
                'zh': '训练记录',
                'en': 'Train record'
            },
            'info': {
                'zh': '展示使用web-ui的历史训练及参数',
                'en': 'Show the training history and parameters'
            }
        },
        'clear_cache': {
            'value': {
                'zh': '删除训练记录',
                'en': 'Delete train records'
            },
        },
        'model_param': {
            'label': {
                'zh': '模型设置',
                'en': 'Model settings'
            },
        },
        'checkpoint': {
            'value': {
                'zh': '训练后的模型',
                'en': 'Trained model'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='model_param', open=True):
            with gr.Row():
                model = gr.Dropdown(
                    elem_id='model',
                    scale=20,
                    choices=get_all_models(),
                    value='Qwen/Qwen2.5-7B-Instruct',
                    allow_custom_value=True)
                gr.Dropdown(elem_id='model_type', choices=ModelType.get_model_name_list(), scale=20)
                gr.Dropdown(elem_id='template', choices=list(TEMPLATE_MAPPING.keys()), scale=20)
                train_record = gr.Dropdown(elem_id='train_record', choices=[], scale=20)
                clear_cache = gr.Button(elem_id='clear_cache', scale=2)
            with gr.Row():
                gr.Textbox(elem_id='system', lines=1, scale=20)

        def clear_record(model):
            if model:
                cls.clear_cache(model)
                return gr.update(choices=[])
            return gr.update()

        clear_cache.click(clear_record, inputs=[model], outputs=[train_record])

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('model').change(
            partial(base_tab.update_input_model, arg_cls=RLHFArguments),
            inputs=[cls.element('model')],
            outputs=[cls.element('train_record')] + list(base_tab.valid_elements().values()))

        cls.element('train_record').change(
            partial(base_tab.update_all_settings, base_tab=base_tab),
            inputs=[cls.element('model'), cls.element('train_record')],
            outputs=list(base_tab.valid_elements().values()))
