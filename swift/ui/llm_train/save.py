from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Save(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'save_param': {
            'label': {
                'zh': '存储参数',
                'en': 'Saving settings'
            },
        },
        'save_steps': {
            'label': {
                'zh': '存储步数',
                'en': 'save steps',
            },
            'info': {
                'zh': '设置每个多少步数进行存储',
                'en': 'Set the save steps',
            }
        },
        'output_dir': {
            'label': {
                'zh': '存储目录',
                'en': 'The output dir',
            },
            'info': {
                'zh': '设置输出模型存储在哪个文件夹下',
                'en': 'Set the output folder',
            }
        },
        'push_to_hub': {
            'label': {
                'zh': '推送魔搭Hub',
                'en': 'Push to modelscope hub',
            },
            'info': {
                'zh': '是否推送魔搭的模型库',
                'en': 'Whether push the output model to modelscope hub',
            }
        },
        'hub_model_id': {
            'label': {
                'zh': '魔搭模型id',
                'en': 'The model-id in modelscope',
            },
            'info': {
                'zh': '设置魔搭的模型id',
                'en': 'Set the model-id of modelscope',
            }
        },
        'hub_private_repo': {
            'label': {
                'zh': '设置仓库私有',
                'en': 'Model is private',
            },
            'info': {
                'zh': '以私有方式推送魔搭hub',
                'en': 'Set the model as private',
            }
        },
        'push_hub_strategy': {
            'label': {
                'zh': '推送策略',
                'en': 'Push strategy',
            },
            'info': {
                'zh': '设置模型推送策略',
                'en': 'Set the push strategy',
            }
        },
        'hub_token': {
            'label': {
                'zh': '仓库token',
                'en': 'The hub token',
            },
            'info': {
                'zh': '该token可以在www.modelscope.cn找到',
                'en': 'Find the token in www.modelscope.cn',
            }
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='save_param', open=True):
            with gr.Blocks():
                with gr.Row():
                    gr.Textbox(elem_id='save_steps', value='500', lines=1, scale=5)
                    gr.Textbox(elem_id='output_dir', scale=20)
                with gr.Row():
                    gr.Checkbox(elem_id='push_to_hub', scale=20)
                    gr.Textbox(elem_id='hub_model_id', lines=1, scale=20)
                    gr.Checkbox(elem_id='hub_private_repo', scale=20)
                    gr.Dropdown(elem_id='push_hub_strategy', scale=20)
                    gr.Textbox(elem_id='hub_token', lines=1, scale=20)
