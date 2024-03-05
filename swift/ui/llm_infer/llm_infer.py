import collections
import os
import re
import sys
from subprocess import PIPE, STDOUT, Popen
from typing import Type

import gradio as gr
import json
import torch
from gradio import Accordion, Tab

from swift import snapshot_download
from swift.llm import (DeployArguments, InferArguments, XRequestConfig,
                       inference_client)
from swift.ui.base import BaseUI
from swift.ui.llm_infer.model import Model
from swift.ui.llm_infer.runtime import Runtime
from swift.utils import get_logger

logger = get_logger()


class LLMInfer(BaseUI):
    group = 'llm_infer'

    sub_ui = [Model, Runtime]

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
                model_and_template_type = gr.State([])
                Model.build_ui(base_tab)
                Runtime.build_ui(base_tab)
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
                        model_and_template_type,
                        cls.element('template_type'), prompt, chatbot,
                        cls.element('max_new_tokens'),
                        cls.element('system')
                    ],
                    outputs=[prompt, chatbot],
                    queue=True)
                clear_history.click(
                    fn=cls.clear_session, inputs=[], outputs=[prompt, chatbot])

                if os.environ.get('MODELSCOPE_ENVIRONMENT') == 'studio':
                    cls.element('load_checkpoint').click(
                        cls.update_runtime, [],
                        [cls.element('runtime_tab'),
                         cls.element('log')]).then(
                             cls.deploy_studio, [
                                 value for value in cls.elements().values()
                                 if not isinstance(value, (Tab, Accordion))
                             ], [cls.element('log')],
                             queue=True)
                else:
                    cls.element('load_checkpoint').click(
                        cls.reset_memory, [], [model_and_template_type]).then(
                            cls.reset_loading_button, [],
                            [cls.element('load_checkpoint')
                             ]).then(cls.get_model_template_type, [
                                 value for value in cls.elements().values()
                                 if not isinstance(value, (Tab, Accordion))
                             ], [model_and_template_type]).then(
                                 cls.deploy_local, [
                                     value
                                     for value in cls.elements().values()
                                     if not isinstance(value, (Tab, Accordion))
                                 ], []).then(
                                     cls.change_interactive, [],
                                     [prompt]).then(  # noqa
                                         cls.clear_session,
                                         inputs=[],
                                         outputs=[prompt,
                                                  chatbot],
                                         queue=True).then(
                                             cls.reset_load_button, [],
                                             [cls.element('load_checkpoint')])

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
    def clear_session(cls):
        return '', None

    @classmethod
    def change_interactive(cls):
        return gr.update(interactive=True)

    @classmethod
    def generate_chat(cls,
                      model_and_template_type,
                      template_type,
                      prompt: str,
                      history,
                      max_new_tokens,
                      system,
                      seed=42):
        model_type = model_and_template_type[0]
        old_history, history = history, []
        request_config = XRequestConfig(seed=seed)
        request_config.stream = True
        stream_resp_with_history = ''
        if not template_type.endswith('generation'):
            stream_resp = inference_client(
                model_type,
                prompt,
                old_history,
                system=system,
                request_config=request_config)
        else:
            stream_resp = inference_client(
                model_type, prompt, request_config=request_config)
        for chunk in stream_resp:
            stream_resp_with_history += chunk.choices[0].delta.content
            qr_pair = [prompt, stream_resp_with_history]
            total_history = old_history + [qr_pair]
            yield '', total_history

    @classmethod
    def deploy(cls, *args):
        deploy_args = cls.get_default_value_from_dataclass(DeployArguments)
        kwargs = {}
        kwargs_is_list = {}
        other_kwargs = {}
        more_params = {}
        keys = [
            key for key, value in cls.elements().items()
            if not isinstance(value, (Tab, Accordion))
        ]
        for key, value in zip(keys, args):
            compare_value = deploy_args.get(key)
            compare_value_arg = str(compare_value) if not isinstance(
                compare_value, (list, dict)) else compare_value
            compare_value_ui = str(value) if not isinstance(
                value, (list, dict)) else value
            if key in deploy_args and compare_value_ui != compare_value_arg and value:
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
        if 'ckpt_dir' in kwargs or (
                'model_id_or_path' in kwargs
                and not os.path.exists(kwargs['model_id_or_path'])):
            kwargs.pop('model_type', None)
        deploy_args = DeployArguments(
            **{
                key: value.split(' ')
                if key in kwargs_is_list and kwargs_is_list[key] else value
                for key, value in kwargs.items()
            })
        params = ''
        for e in kwargs:
            if e in kwargs_is_list and kwargs_is_list[e]:
                params += f'--{e} {kwargs[e]} '
            else:
                params += f'--{e} "{kwargs[e]}" '
        devices = other_kwargs['gpu_id']
        devices = [d for d in devices if d]
        assert (len(devices) == 1 or 'cpu' not in devices)
        gpus = ','.join(devices)
        cuda_param = ''
        if gpus != 'cpu':
            cuda_param = f'CUDA_VISIBLE_DEVICES={gpus}'

        log_file = os.path.join(os.getcwd(), 'run_deploy.log')
        if sys.platform == 'win32':
            if cuda_param:
                cuda_param = f'set {cuda_param} && '
            run_command = f'{cuda_param}start /b swift deploy {params} > {log_file} 2>&1'
        elif os.environ.get('MODELSCOPE_ENVIRONMENT') == 'studio':
            run_command = f'{cuda_param} swift deploy {params}'
        else:
            run_command = f'{cuda_param} nohup swift deploy {params} > {log_file} 2>&1 &'
        return run_command, deploy_args

    @classmethod
    def deploy_studio(cls, *args):
        run_command, deploy_args = cls.deploy(*args)
        if os.environ.get('MODELSCOPE_ENVIRONMENT') == 'studio':
            lines = collections.deque(
                maxlen=int(os.environ.get('MAX_LOG_LINES', 50)))
            logger.info(f'Run deploying: {run_command}')
            process = Popen(
                run_command, shell=True, stdout=PIPE, stderr=STDOUT)
            with process.stdout:
                for line in iter(process.stdout.readline, b''):
                    line = line.decode('utf-8')
                    lines.append(line)
                    yield '\n'.join(lines)

    @classmethod
    def deploy_local(cls, *args):
        run_command, deploy_args = cls.deploy(*args)
        lines = collections.deque(
            maxlen=int(os.environ.get('MAX_LOG_LINES', 50)))
        logger.info(f'Run deploying: {run_command}')
        process = Popen(run_command, shell=True, stdout=PIPE, stderr=STDOUT)
        with process.stdout:
            for line in iter(process.stdout.readline, b''):
                line = line.decode('utf-8')
                lines.append(line)
                yield '\n'.join(lines)

    @classmethod
    def get_model_template_type(cls, *args):
        run_command, deploy_args = cls.deploy(*args)
        return [deploy_args.model_type, deploy_args.template_type]
