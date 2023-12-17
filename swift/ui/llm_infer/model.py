import os.path

import gradio as gr
import json

from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING
from swift.ui.base import BaseUI


class Model(BaseUI):

    llm_train = 'llm_infer'

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
        'more_params': {
            'label': {
                'zh': '更多参数',
                'en': 'More params'
            },
            'info': {
                'zh': '以json格式填入',
                'en': 'Fill in with json format'
            }
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: 'BaseUI'):
        with gr.Row():
            model_type = gr.Dropdown(
                elem_id='model_type',
                choices=list(MODEL_MAPPING.keys())
                + [base_tab.locale('checkpoint', cls.lang)],
                value=base_tab.locale('checkpoint', cls.lang),
                scale=20)
            model_id_or_path = gr.Textbox(
                elem_id='model_id_or_path',
                lines=1,
                scale=20,
                interactive=True)
            template_type = gr.Dropdown(
                elem_id='template_type',
                choices=list(TEMPLATE_MAPPING.keys()) + ['AUTO'],
                scale=20)
        with gr.Row():
            system = gr.Textbox(elem_id='system', lines=1, scale=20)
        with gr.Row():
            gr.Textbox(elem_id='more_params', lines=4, scale=20)
            gr.Button(elem_id='load_checkpoint', scale=2, variant='primary')

        def update_input_model(choice):
            if choice == base_tab.locale('checkpoint', cls.lang):
                model_id_or_path = None
                default_system = None
                template = None
            else:
                model_id_or_path = MODEL_MAPPING[choice]['model_id_or_path']
                default_system = getattr(
                    TEMPLATE_MAPPING[MODEL_MAPPING[choice]['template']],
                    'default_system', None)
                template = MODEL_MAPPING[choice]['template']
            return model_id_or_path, default_system, template, \
                gr.update(interactive=choice == base_tab.locale('checkpoint', cls.lang))

        def update_interactive(choice):
            return gr.update(
                interactive=choice == base_tab.locale('checkpoint', cls.lang))

        def update_model_id_or_path(path):
            with open(os.path.join(path, 'sft_args.json'), 'r') as f:
                sft_args = json.load(f)
            base_model_type = sft_args['model_type']
            system = getattr(
                TEMPLATE_MAPPING[MODEL_MAPPING[base_model_type]['template']],
                'default_system', None)
            return sft_args['system'] or system, sft_args['template_type']

        model_type.change(
            update_input_model,
            inputs=[model_type],
            outputs=[model_id_or_path, system, template_type])

        model_type.change(
            update_interactive,
            inputs=[model_type],
            outputs=[model_id_or_path])

        model_id_or_path.change(
            update_model_id_or_path,
            inputs=[model_id_or_path],
            outputs=[system, template_type])
