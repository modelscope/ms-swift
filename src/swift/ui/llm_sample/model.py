# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Type

import gradio as gr

from swift.llm import TEMPLATE_MAPPING, ModelType, SamplingArguments
from swift.llm.model.register import get_all_models
from swift.ui.base import BaseUI


class Model(BaseUI):

    group = 'llm_sample'

    locale_dict = {
        'model_type': {
            'label': {
                'zh': '选择模型类型',
                'en': 'Select Model Type'
            },
            'info': {
                'zh': 'SWIFT已支持的模型类型，model是服务名称时请置空',
                'en': 'Base model type supported by SWIFT, Please leave it blank if model is the service name'
            }
        },
        'model': {
            'label': {
                'zh': '模型id、路径或模型服务名称',
                'en': 'Model id, path or server name'
            },
            'info': {
                'zh':
                '实际的模型id，如果是训练后的模型请填入checkpoint-xxx的目录，如果是模型服务请填入模型服务名称',
                'en': ('The actual model id or path, if is a trained model, please fill in the checkpoint-xxx dir'
                       'if is a model service, please fill in the server name')
            }
        },
        'template': {
            'label': {
                'zh': '模型Prompt模板类型',
                'en': 'Prompt template type'
            },
            'info': {
                'zh': '选择匹配模型的Prompt模板，model是服务名称时请置空',
                'en': 'Choose the template type of the model, Please leave it blank if model is the service name'
            }
        },
        'system': {
            'label': {
                'zh': 'System字段',
                'en': 'System'
            },
            'info': {
                'zh': 'System字段支持在加载模型后修改',
                'en': 'System can be modified after the model weights loaded'
            }
        },
        'prm_model': {
            'label': {
                'zh': '过程奖励模型',
                'en': 'Process Reward Model'
            },
            'info': {
                'zh': '可以是模型id，或者plugin中定义的prm key',
                'en': 'It can be a model id, or a prm key defined in the plugin'
            }
        },
        'orm_model': {
            'label': {
                'zh': '结果奖励模型',
                'en': 'Outcome Reward Model'
            },
            'info': {
                'zh': '通常是通配符或测试用例等，定义在plugin中',
                'en': 'Usually a wildcard or test case, etc., defined in the plugin'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row(equal_height=True):
            gr.Dropdown(
                elem_id='model',
                scale=20,
                choices=get_all_models(),
                value='Qwen/Qwen2.5-7B-Instruct',
                allow_custom_value=True)
            gr.Dropdown(elem_id='model_type', choices=ModelType.get_model_name_list(), scale=20)
            gr.Dropdown(elem_id='template', choices=list(TEMPLATE_MAPPING.keys()), scale=20)
        with gr.Row():
            gr.Textbox(elem_id='system', lines=1)
        with gr.Row():
            gr.Textbox(elem_id='prm_model', scale=20)
            gr.Textbox(elem_id='orm_model', scale=20)

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('model').change(
            partial(cls.update_input_model, arg_cls=SamplingArguments, has_record=False),
            inputs=[cls.element('model')],
            outputs=list(cls.valid_elements().values()))
