from typing import Type

import gradio as gr

from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING, ModelType
from swift.ui.base import BaseUI


class Model(BaseUI):
    group = 'llm_train'

    locale_dict = {
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
        'template_type': {
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
                'zh': '恢复初始值',
                'en': 'Reset to default'
            },
        },
        'model_param': {
            'label': {
                'zh': '模型设置',
                'en': 'Model settings'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='model_param', open=True):
            with gr.Row():
                model_type = gr.Dropdown(
                    elem_id='model_type',
                    choices=ModelType.get_model_name_list() + cls.get_custom_name_list(),
                    scale=20)
                model_id_or_path = gr.Textbox(elem_id='model_id_or_path', lines=1, scale=20, interactive=True)
                template_type = gr.Dropdown(
                    elem_id='template_type', choices=list(TEMPLATE_MAPPING.keys()) + ['AUTO'], scale=20)
                reset_btn = gr.Button(elem_id='reset', scale=2)
                model_state = gr.State({})
            with gr.Row():
                system = gr.Textbox(elem_id='system', lines=1, scale=20)

        def update_input_model(choice, model_state=None):
            if choice is None:
                return None, None, None
            if model_state and choice in model_state:
                model_id_or_path = model_state[choice]
            else:
                model_id_or_path = MODEL_MAPPING[choice]['model_id_or_path']
            default_system = getattr(TEMPLATE_MAPPING[MODEL_MAPPING[choice]['template']]['template'], 'default_system',
                                     None)
            template = MODEL_MAPPING[choice]['template']
            return model_id_or_path, default_system, template

        def update_model_id_or_path(model_type, model_id_or_path, model_state):
            if model_type is None or isinstance(model_type, list):
                return model_state
            model_state[model_type] = model_id_or_path
            return model_state

        def reset(model_type):
            model_id_or_path, default_system, template = update_input_model(model_type)
            return model_id_or_path, default_system, template, {}

        model_type.change(
            update_input_model, inputs=[model_type, model_state], outputs=[model_id_or_path, system, template_type])

        model_id_or_path.change(
            update_model_id_or_path, inputs=[model_type, model_id_or_path, model_state], outputs=[model_state])

        reset_btn.click(reset, inputs=[model_type], outputs=[model_id_or_path, system, template_type, model_state])
