import gradio as gr

from swift.ui.base import BaseUI


class SelfCog(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'self_cognition': {
            'label': {
                'zh': '自我认知任务参数',
                'en': 'Self cognition settings'
            },
        },
        'self_cognition_sample': {
            'label': {
                'zh': '数据及采样条数',
                'en': 'Dataset sample size'
            },
            'info': {
                'zh': '设置数据集采样的条数',
                'en': 'Set the dataset sample size'
            }
        },
        'model_name': {
            'label': {
                'zh': '模型认知名称',
                'en': 'Model name'
            },
            'info': {
                'zh': '设置模型应当认知自己的名字',
                'en': 'Set the name of the model think itself of'
            }
        },
        'model_author': {
            'label': {
                'zh': '模型作者',
                'en': 'Model author'
            },
            'info': {
                'zh': '设置模型认知的自己的作者',
                'en': 'Set the author of the model'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: 'BaseUI'):
        with gr.Accordion(elem_id='self_cognition', open=False):
            with gr.Row():
                gr.Textbox(elem_id='self_cognition_sample', scale=20)
                gr.Textbox(elem_id='model_name', scale=20)
                gr.Textbox(elem_id='model_author', scale=20)
