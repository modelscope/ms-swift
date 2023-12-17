import json
import os
from dataclasses import fields

import gradio as gr
import torch

from swift.llm import InferArguments, prepare_model_template, limit_history_length, inference_stream
from swift.ui.base import BaseUI
from swift.ui.llm_infer.generate import Generate
from swift.ui.llm_infer.model import Model


class LLMInfer(BaseUI):

    group = 'llm_infer'

    sub_ui = [
        Model,
        Generate
    ]

    locale_dict = {
        'llm_infer': {
            'label': {
                'zh': 'LLM推理',
                'en': 'LLM Inference',
            }
        },
        'load_alert': {
            'value': {
                'zh':
                    '加载模型中，请等待',
                'en':
                    'Start to load model, please wait'
            }
        },
        'load_checkpoint': {
            'value': {
                'zh':
                    '加载模型',
                'en':
                    'Load model'
            }
        },
        'chatbot': {
            'label': {
                'zh': '对话框',
                'en': 'Chat bot'
            },
        },
        'prompt': {
            'label': {
                'zh': '请输入：',
                'en': 'Input:'
            },
        },
        'clear_history': {
            'label': {
                'zh': '清除对话信息',
                'en': 'Clear history'
            },
        },
        'submit': {
            'label': {
                'zh': '🚀 发送',
                'en': '🚀 Send'
            },
        },
        'gpu_id': {
            'label': {
                'zh': '选择可用GPU',
                'en': 'Choose GPU'
            },
            'info': {
                'zh': '选择训练使用的GPU号，如CUDA不可用只能选择CPU',
                'en': 'Select GPU to train'
            }
        },
    }

    choice_dict = {}
    for f in fields(InferArguments):
        if 'choices' in f.metadata:
            choice_dict[f.name] = f.metadata['choices']

    @classmethod
    def do_build_ui(cls, base_tab: 'BaseUI'):
        with gr.TabItem(elem_id='llm_infer', group='llm_infer', label=''):
            gpu_count = 0
            default_device = 'cpu'
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                default_device = '0'
            with gr.Blocks():
                model_and_template = gr.State([])
                Model.build_ui(base_tab)
                Generate.build_ui(base_tab)
                gr.Dropdown(
                    elem_id='gpu_id',
                    multiselect=True,
                    choices=[str(i) for i in range(gpu_count)] + ['cpu'],
                    value=default_device,
                    scale=8)
                chatbot = gr.Chatbot(elem_id='chatbot', lines=10, elem_classes="control-height")
                prompt = gr.Textbox(elem_id='prompt', lines=2)

                with gr.Row():
                    clear_history = gr.Button(elem_id='clear_history')
                    submit = gr.Button(elem_id='submit')

                submit.click(cls.generate_chat,
                             inputs=[model_and_template, prompt, chatbot, cls.element('max_new_tokens')],
                             outputs=[prompt, chatbot])
                clear_history.click(fn=cls.clear_session,
                                    inputs=[],
                                    outputs=[prompt, chatbot],
                                    queue=False)
                cls.element('load_checkpoint')
                cls.element('load_checkpoint').click(
                    cls.prepare_checkpoint, [], [model_and_template, prompt],
                    show_progress=True)
                cls.element('load_checkpoint').click(cls.clear_session,
                                                  inputs=[],
                                                  outputs=[prompt, chatbot])

    @classmethod
    def prepare_checkpoint(cls):
        global model, tokenizer, template
        args = fields(InferArguments)
        args = {arg.name: arg.type for arg in args}
        kwargs = {}
        more_params = getattr(cls.element('more_params'), 'arg_value', None)
        if more_params:
            more_params = json.loads(more_params)
        else:
            more_params = {}

        elements = cls.elements()
        for e in elements:
            if e in args and getattr(elements[e], 'changed', False) and getattr(elements[e], 'arg_value', None):
                kwargs[e] = elements[e].arg_value
        kwargs.update(more_params)

        devices = getattr(elements['gpu_id'], 'arg_value',
                          ' '.join(elements['gpu_id'].value)).split(' ')
        devices = [d for d in devices if d]
        assert (len(devices) == 1 or 'cpu' not in devices)
        gpus = ','.join(devices)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        args = InferArguments(**kwargs)
        model, template = prepare_model_template(args)
        return [model, template], gr.update(interactive=True)

    @classmethod
    def clear_session(cls):
        return '', None

    @classmethod
    def generate_chat(cls, model_and_template, prompt: str, history, max_new_tokens):
        model, template = model_and_template
        if not cls.element('template_type').last_value.endswith('generation'):
            old_history, history = limit_history_length(template, prompt, history,
                                                        max_new_tokens)
        else:
            old_history = []
            history = []
        gen = inference_stream(model, template, prompt, history)
        for _, history in gen:
            total_history = old_history + history
            yield '', total_history
