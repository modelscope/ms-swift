import os
import re
from typing import Type

import gradio as gr
import json
import torch
from gradio import Accordion, Tab

from swift import snapshot_download
from swift.llm import (InferArguments, inference_stream, limit_history_length,
                       prepare_model_template)
from swift.ui.base import BaseUI
from swift.ui.llm_infer.model import Model


class LLMInfer(BaseUI):

    group = 'llm_infer'

    sub_ui = [Model]

    locale_dict = {
        'generate_alert': {
            'value': {
                'zh': '请先加载模型',
                'en': 'Please load model first',
            }
        },
        'llm_infer': {
            'label': {
                'zh': 'LLM推理',
                'en': 'LLM Inference',
            }
        },
        'load_alert': {
            'value': {
                'zh': '加载模型中，请等待',
                'en': 'Start to load model, please wait'
            }
        },
        'loaded_alert': {
            'value': {
                'zh': '模型加载完成',
                'en': 'Model loaded'
            }
        },
        'chatbot': {
            'value': {
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
            'value': {
                'zh': '清除对话信息',
                'en': 'Clear history'
            },
        },
        'submit': {
            'value': {
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

    choice_dict = BaseUI.get_choices_from_dataclass(InferArguments)

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_infer', label=''):
            gpu_count = 0
            default_device = 'cpu'
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                default_device = '0'
            with gr.Blocks():
                model_and_template = gr.State([])
                Model.build_ui(base_tab)
                gr.Dropdown(
                    elem_id='gpu_id',
                    multiselect=True,
                    choices=[str(i) for i in range(gpu_count)] + ['cpu'],
                    value=default_device,
                    scale=8)
                chatbot = gr.Chatbot(
                    elem_id='chatbot', elem_classes='control-height')
                prompt = gr.Textbox(
                    elem_id='prompt', lines=1, interactive=False)

                with gr.Row():
                    clear_history = gr.Button(elem_id='clear_history')
                    submit = gr.Button(elem_id='submit')

                submit.click(
                    cls.generate_chat,
                    inputs=[
                        model_and_template,
                        cls.element('template_type'), prompt, chatbot,
                        cls.element('max_new_tokens')
                    ],
                    outputs=[prompt, chatbot],
                    queue=True)
                clear_history.click(
                    fn=cls.clear_session, inputs=[], outputs=[prompt, chatbot])
                cls.element('load_checkpoint').click(
                    cls.reset_memory, [], [model_and_template])\
                    .then(cls.reset_loading_button, [], [cls.element('load_checkpoint')]).then(
                        cls.prepare_checkpoint, [
                            value for value in cls.elements().values()
                            if not isinstance(value, (Tab, Accordion))
                        ], [model_and_template]).then(cls.change_interactive, [],
                                                 [prompt]).then( # noqa
                    cls.clear_session,
                    inputs=[],
                    outputs=[prompt, chatbot],
                    queue=True).then(cls.reset_load_button, [], [cls.element('load_checkpoint')])

    @classmethod
    def reset_load_button(cls):
        return gr.update(
            value=cls.locale('load_checkpoint', cls.lang)['value'])

    @classmethod
    def reset_loading_button(cls):
        return gr.update(value=cls.locale('load_alert', cls.lang)['value'])

    @classmethod
    def reset_memory(cls):
        return []

    @classmethod
    def prepare_checkpoint(cls, *args):
        torch.cuda.empty_cache()
        infer_args = cls.get_default_value_from_dataclass(InferArguments)
        kwargs = {}
        kwargs_is_list = {}
        other_kwargs = {}
        more_params = {}
        keys = [
            key for key, value in cls.elements().items()
            if not isinstance(value, (Tab, Accordion))
        ]
        for key, value in zip(keys, args):
            compare_value = infer_args.get(key)
            compare_value_arg = str(compare_value) if not isinstance(
                compare_value, (list, dict)) else compare_value
            compare_value_ui = str(value) if not isinstance(
                value, (list, dict)) else value
            if key in infer_args and compare_value_ui != compare_value_arg and value:
                if isinstance(value, str) and re.fullmatch(
                        cls.int_regex, value):
                    value = int(value)
                elif isinstance(value, str) and re.fullmatch(
                        cls.float_regex, value):
                    value = float(value)
                kwargs[key] = value if not isinstance(
                    value, list) else ' '.join(value)
                kwargs_is_list[key] = isinstance(value, list)
            else:
                other_kwargs[key] = value
            if key == 'more_params' and value:
                more_params = json.loads(value)

        kwargs.update(more_params)
        if kwargs['model_type'] == cls.locale('checkpoint', cls.lang)['value']:
            model_dir = kwargs.pop('model_id_or_path')
            if not os.path.exists(model_dir):
                model_dir = snapshot_download(model_dir)
            kwargs['ckpt_dir'] = model_dir
        if 'ckpt_dir' in kwargs or 'model_id_or_path' in kwargs:
            kwargs.pop('model_type', None)

        devices = other_kwargs['gpu_id']
        devices = [d for d in devices if d]
        assert (len(devices) == 1 or 'cpu' not in devices)
        gpus = ','.join(devices)
        if gpus != 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        infer_args = InferArguments(**kwargs)
        model, template = prepare_model_template(infer_args)
        gr.Info(cls.locale('loaded_alert', cls.lang)['value'])
        return [model, template]

    @classmethod
    def clear_session(cls):
        return '', None

    @classmethod
    def change_interactive(cls):
        return gr.update(interactive=True)

    @classmethod
    def generate_chat(cls, model_and_template, template_type, prompt: str,
                      history, max_new_tokens):
        if not model_and_template:
            gr.Warning(cls.locale('generate_alert', cls.lang)['value'])
            return '', None
        model, template = model_and_template
        if os.environ.get('MODELSCOPE_ENVIRONMENT') == 'studio':
            model.cuda()
        if not template_type.endswith('generation'):
            old_history, history = limit_history_length(
                template, prompt, history, int(max_new_tokens))
        else:
            old_history = []
            history = []
        gen = inference_stream(model, template, prompt, history)
        for _, history in gen:
            total_history = old_history + history
            yield '', total_history
        if os.environ.get('MODELSCOPE_ENVIRONMENT') == 'studio':
            model.cpu()
