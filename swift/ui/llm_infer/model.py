import os.path
from functools import partial
from typing import Type

import gradio as gr
import json

from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING, ModelType
from swift.ui.base import BaseUI
from swift.ui.llm_infer.generate import Generate


class Model(BaseUI):

    llm_train = 'llm_infer'

    sub_ui = [Generate]

    is_inference = os.environ.get('USE_INFERENCE') == '1' or os.environ.get('MODELSCOPE_ENVIRONMENT') == 'studio'

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
        'load_checkpoint': {
            'value': {
                'zh': '加载模型' if is_inference else '部署模型',
                'en': 'Load model' if is_inference else 'Deploy model',
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
                'zh': 'system字段支持在加载模型后修改',
                'en': 'system can be modified after the model weights loaded'
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
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row():
            model = gr.Dropdown(elem_id='model', lines=1, scale=20,
                                choices=[base_tab.locale('checkpoint', cls.lang)['value']] + )
            model_type = gr.Dropdown(
                elem_id='model_type',
                choices=ModelType.get_model_name_list() + cls.get_custom_name_list(),
                value=base_tab.locale('checkpoint', cls.lang)['value'],
                scale=20)
            model_type = gr.Dropdown(
                elem_id='model_type',
                choices=ModelType.get_model_name_list() + cls.get_custom_name_list(),
                scale=20)
            template = gr.Dropdown(
                elem_id='template', choices=list(TEMPLATE_MAPPING.keys()) + ['AUTO'], scale=20)
            gr.Checkbox(elem_id='merge_lora', scale=4)
            reset_btn = gr.Button(elem_id='reset', scale=2)
            model_state = gr.State({})
        with gr.Row():
            system = gr.Textbox(elem_id='system', lines=4, scale=20)
        Generate.build_ui(base_tab)
        with gr.Row():
            gr.Textbox(elem_id='lora_modules', lines=1, is_list=True, scale=40)
            gr.Textbox(elem_id='more_params', lines=1, scale=20)
            gr.Button(elem_id='load_checkpoint', scale=2, variant='primary')

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('model').change(
            partial(cls.update_input_model, has_record=False),
            inputs=[cls.element('model')],
            outputs=list(cls.valid_elements().values()))
