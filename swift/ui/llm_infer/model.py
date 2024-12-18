# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Type

import gradio as gr

from swift.llm import TEMPLATE_MAPPING, DeployArguments, ModelType
from swift.llm.model.register import get_all_models
from swift.ui.base import BaseUI
from swift.ui.llm_infer.generate import Generate


class Model(BaseUI):

    llm_train = 'llm_infer'

    sub_ui = [Generate]

    locale_dict = {
        'model_type': {
            'label': {
                'zh': '选择模型类型',
                'en': 'Select Model Type'
            },
            'info': {
                'zh': 'SWIFT已支持的模型类型',
                'en': 'Base model type supported by SWIFT'
            }
        },
        'load_checkpoint': {
            'value': {
                'zh': '部署模型',
                'en': 'Deploy model',
            }
        },
        'model': {
            'label': {
                'zh': '模型id或路径',
                'en': 'Model id or path'
            },
            'info': {
                'zh': '实际的模型id，如果是训练后的模型请填入checkpoint-xxx的目录',
                'en': 'The actual model id or path, if is a trained model, please fill in the checkpoint-xxx dir'
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
        'merge_lora': {
            'label': {
                'zh': '合并lora',
                'en': 'merge lora'
            },
            'info': {
                'zh': '仅在sft_type=lora时可用',
                'en': 'Only available when sft_type=lora'
            }
        },
        'lora_modules': {
            'label': {
                'zh': '外部lora模块',
                'en': 'More lora modules'
            },
            'info': {
                'zh': '空格分割的name=/path1/path2键值对',
                'en': 'name=/path1/path2 split by blanks'
            }
        },
        'more_params': {
            'label': {
                'zh': '更多参数',
                'en': 'More params'
            },
            'info': {
                'zh': '以json格式或--xxx xxx命令行格式填入',
                'en': 'Fill in with json format or --xxx xxx cmd format'
            }
        },
        'reset': {
            'value': {
                'zh': '恢复初始值',
                'en': 'Reset to default'
            },
        },
        'infer_backend': {
            'label': {
                'zh': '推理框架',
                'en': 'Infer backend'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row():
            gr.Dropdown(
                elem_id='model',
                scale=20,
                choices=get_all_models(),
                value='Qwen/Qwen2.5-7B-Instruct',
                allow_custom_value=True)
            gr.Dropdown(elem_id='model_type', choices=ModelType.get_model_name_list(), scale=20)
            gr.Dropdown(elem_id='template', choices=list(TEMPLATE_MAPPING.keys()), scale=20)
            gr.Checkbox(elem_id='merge_lora', scale=4)
            gr.Button(elem_id='reset', scale=2)
        with gr.Row():
            gr.Dropdown(elem_id='infer_backend', value='pt', scale=5)
        Generate.build_ui(base_tab)
        with gr.Row():
            gr.Textbox(elem_id='lora_modules', lines=1, is_list=True, scale=40)
            gr.Textbox(elem_id='more_params', lines=1, scale=20)
            gr.Button(elem_id='load_checkpoint', scale=2, variant='primary')

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('model').change(
            partial(cls.update_input_model, arg_cls=DeployArguments, has_record=False),
            inputs=[cls.element('model')],
            outputs=list(cls.valid_elements().values()))
