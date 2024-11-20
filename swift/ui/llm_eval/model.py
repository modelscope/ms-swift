import os.path
from typing import Type

import gradio as gr

from swift.llm import MODEL_MAPPING, ModelType
from swift.ui.base import BaseUI


class Model(BaseUI):

    group = 'llm_eval'

    locale_dict = {
        'checkpoint': {
            'value': {
                'zh': '训练后的模型',
                'en': 'Trained model'
            }
        },
        'model_type': {
            'label': {
                'zh': '选择模型',
                'en': 'Select Model'
            },
            'info': {
                'zh': 'SWIFT已支持的模型名称',
                'en': 'Base model supported by SWIFT'
            }
        },
        'model_id_or_path': {
            'label': {
                'zh': '模型id或路径',
                'en': 'Model id or path'
            },
            'info': {
                'zh': '实际的模型id，如果是训练后的模型请填入checkpoint-xxx的目录',
                'en': 'The actual model id or path, if is a trained model, please fill in the checkpoint-xxx dir'
            }
        },
        'reset': {
            'value': {
                'zh': '恢复初始值',
                'en': 'Reset to default'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row():
            model_type = gr.Dropdown(
                elem_id='model_type',
                choices=[base_tab.locale('checkpoint', cls.lang)['value']] + ModelType.get_model_name_list()
                + cls.get_custom_name_list(),
                value=base_tab.locale('checkpoint', cls.lang)['value'],
                allow_custom_value=True,
                scale=20)
            model_id_or_path = gr.Textbox(elem_id='model_id_or_path', lines=1, scale=20, interactive=True)
            reset_btn = gr.Button(elem_id='reset', scale=2)
            model_state = gr.State({})

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('model').change(
            partial(cls.update_input_model, has_record=False),
            inputs=[cls.element('model')],
            outputs=list(cls.valid_elements().values()))
