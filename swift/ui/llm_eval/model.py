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
                'zh': '实际的模型id',
                'en': 'The actual model id or model path'
            }
        },
        'more_params': {
            'label': {
                'zh': '更多参数',
                'en': 'More params'
            },
            'info': {
                'zh': '以json格式填入',
                'en': 'Fill in with json format'
            }
        },
        'reset': {
            'value': {
                'zh': '恢复初始值',
                'en': 'Reset to default'
            },
        },
        'evaluate': {
            'value': {
                'zh': '开始评估',
                'en': 'Begin Evaluation'
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
                scale=20)
            model_id_or_path = gr.Textbox(elem_id='model_id_or_path', lines=1, scale=20, interactive=True)
            reset_btn = gr.Button(elem_id='reset', scale=2)
            model_state = gr.State({})
        with gr.Row():
            gr.Textbox(elem_id='more_params', lines=4, scale=20)
            gr.Button(elem_id='evaluate', scale=2, variant='primary')

        def update_input_model(choice, model_state=None):
            if choice == base_tab.locale('checkpoint', cls.lang)['value']:
                if model_state and choice in model_state:
                    model_id_or_path = model_state[choice]
                else:
                    model_id_or_path = None
            else:
                if model_state and choice in model_state:
                    model_id_or_path = model_state[choice]
                else:
                    model_id_or_path = MODEL_MAPPING[choice]['model_id_or_path']
            return model_id_or_path

        def update_model_id_or_path(model_type, path, model_state):
            if not path or not os.path.exists(path):
                return gr.update()
            model_state[model_type] = path
            return model_state

        model_type.change(
            update_input_model, inputs=[model_type, model_state], outputs=[model_id_or_path])

        model_id_or_path.change(
            update_model_id_or_path,
            inputs=[model_type, model_id_or_path, model_state],
            outputs=[model_state])

        def reset(model_type):
            model_id_or_path = update_input_model(model_type)
            return model_id_or_path, {}

        reset_btn.click(reset, inputs=[model_type], outputs=[model_id_or_path, model_state])
