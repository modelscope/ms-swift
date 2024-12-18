# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
import sys
import time
from copy import deepcopy
from datetime import datetime
from functools import partial
from typing import List, Type

import gradio as gr
import json
import torch
from json import JSONDecodeError

from swift.llm import DeployArguments, InferArguments, InferClient, InferRequest, RequestConfig
from swift.ui.base import BaseUI
from swift.ui.llm_infer.model import Model
from swift.ui.llm_infer.runtime import Runtime
from swift.utils import get_logger

logger = get_logger()


class LLMInfer(BaseUI):

    group = 'llm_infer'

    is_gradio_app = False

    is_multimodal = True

    deployed = False

    sub_ui = [Model, Runtime]

    locale_dict = {
        'generate_alert': {
            'value': {
                'zh': '请先部署模型',
                'en': 'Please deploy model first',
            }
        },
        'port': {
            'label': {
                'zh': '端口',
                'en': 'port'
            },
        },
        'llm_infer': {
            'label': {
                'zh': 'LLM推理',
                'en': 'LLM Inference',
            }
        },
        'load_alert': {
            'value': {
                'zh': '部署中，请点击"展示部署状态"查看',
                'en': 'Start to deploy model, '
                'please Click "Show running '
                'status" to view details',
            }
        },
        'loaded_alert': {
            'value': {
                'zh': '模型加载完成',
                'en': 'Model loaded'
            }
        },
        'port_alert': {
            'value': {
                'zh': '该端口已被占用',
                'en': 'The port has been occupied'
            }
        },
        'chatbot': {
            'value': {
                'zh': '对话框',
                'en': 'Chat bot'
            },
        },
        'infer_model_type': {
            'label': {
                'zh': 'Lora模块',
                'en': 'Lora module'
            },
            'info': {
                'zh': '发送给server端哪个LoRA，默认为`default`',
                'en': 'Which LoRA to use on server, default value is `default`'
            }
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
    default_dict = BaseUI.get_default_value_from_dataclass(InferArguments)
    arguments = BaseUI.get_argument_names(InferArguments)

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_infer', label=''):
            gpu_count = 0
            default_device = 'cpu'
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                default_device = '0'
            with gr.Blocks():
                infer_request = gr.State(None)
                if LLMInfer.is_gradio_app:
                    Model.visible = False
                    Runtime.visible = False
                Model.build_ui(base_tab)
                Runtime.build_ui(base_tab)
                with gr.Row(visible=not LLMInfer.is_gradio_app):
                    gr.Dropdown(
                        elem_id='gpu_id',
                        multiselect=True,
                        choices=[str(i) for i in range(gpu_count)] + ['cpu'],
                        value=default_device,
                        scale=8)
                    infer_model_type = gr.Textbox(elem_id='infer_model_type', scale=4)
                    gr.Textbox(elem_id='port', lines=1, value='8000', scale=4)
                chatbot = gr.Chatbot(elem_id='chatbot', elem_classes='control-height')
                with gr.Row():
                    prompt = gr.Textbox(elem_id='prompt', lines=1, interactive=True)
                    with gr.Tabs(visible=not cls.is_gradio_app or cls.is_multimodal):
                        with gr.TabItem(label='Image'):
                            image = gr.Image(type='filepath')
                        with gr.TabItem(label='Video'):
                            video = gr.Video()
                        with gr.TabItem(label='Audio'):
                            audio = gr.Audio(type='filepath')

                with gr.Row():
                    clear_history = gr.Button(elem_id='clear_history')
                    submit = gr.Button(elem_id='submit')

                if not LLMInfer.is_gradio_app:
                    cls.element('load_checkpoint').click(
                        cls.deploy_model, list(base_tab.valid_elements().values()),
                        [cls.element('runtime_tab'), cls.element('running_tasks')])
                submit.click(
                    cls.send_message,
                    inputs=[
                        cls.element('running_tasks'),
                        cls.element('template'), prompt, image, video, audio, infer_request, infer_model_type,
                        cls.element('system'),
                        cls.element('max_new_tokens'),
                        cls.element('temperature'),
                        cls.element('top_k'),
                        cls.element('top_p'),
                        cls.element('repetition_penalty')
                    ],
                    outputs=[prompt, chatbot, image, video, audio, infer_request],
                    queue=True)

                clear_history.click(
                    fn=cls.clear_session, inputs=[], outputs=[prompt, chatbot, image, video, audio, infer_request])

                if not LLMInfer.is_gradio_app:
                    base_tab.element('running_tasks').change(
                        partial(Runtime.task_changed, base_tab=base_tab), [base_tab.element('running_tasks')],
                        list(cls.valid_elements().values()) + [cls.element('log')],
                        cancels=Runtime.log_event)
                    Runtime.element('kill_task').click(
                        Runtime.kill_task,
                        [Runtime.element('running_tasks')],
                        [Runtime.element('running_tasks')] + [Runtime.element('log')],
                        cancels=[Runtime.log_event],
                    )

    @classmethod
    def deploy(cls, *args):
        deploy_args = cls.get_default_value_from_dataclass(DeployArguments)
        kwargs = {}
        kwargs_is_list = {}
        other_kwargs = {}
        more_params = {}
        more_params_cmd = ''
        keys = cls.valid_element_keys()
        for key, value in zip(keys, args):
            compare_value = deploy_args.get(key)
            compare_value_arg = str(compare_value) if not isinstance(compare_value, (list, dict)) else compare_value
            compare_value_ui = str(value) if not isinstance(value, (list, dict)) else value
            if key in deploy_args and compare_value_ui != compare_value_arg and value:
                if isinstance(value, str) and re.fullmatch(cls.int_regex, value):
                    value = int(value)
                elif isinstance(value, str) and re.fullmatch(cls.float_regex, value):
                    value = float(value)
                elif isinstance(value, str) and re.fullmatch(cls.bool_regex, value):
                    value = True if value.lower() == 'true' else False
                kwargs[key] = value if not isinstance(value, list) else ' '.join(value)
                kwargs_is_list[key] = isinstance(value, list) or getattr(cls.element(key), 'is_list', False)
            else:
                other_kwargs[key] = value
            if key == 'more_params' and value:
                try:
                    more_params = json.loads(value)
                except (JSONDecodeError or TypeError):
                    more_params_cmd = value

        kwargs.update(more_params)
        model = kwargs.get('model')
        if os.path.exists(model) and os.path.exists(os.path.join(model, 'args.json')):
            kwargs['ckpt_dir'] = kwargs.pop('model')
            with open(os.path.join(kwargs['ckpt_dir'], 'args.json'), 'r') as f:
                _json = json.load(f)
                kwargs['model_type'] = _json['model_type']
                kwargs['train_type'] = _json['train_type']
        deploy_args = DeployArguments(
            **{
                key: value.split(' ') if key in kwargs_is_list and kwargs_is_list[key] else value
                for key, value in kwargs.items()
            })
        if deploy_args.port in Runtime.get_all_ports():
            raise gr.Error(cls.locale('port_alert', cls.lang)['value'])
        params = ''
        sep = f'{cls.quote} {cls.quote}'
        for e in kwargs:
            if isinstance(kwargs[e], list):
                params += f'--{e} {cls.quote}{sep.join(kwargs[e])}{cls.quote} '
            elif e in kwargs_is_list and kwargs_is_list[e]:
                all_args = [arg for arg in kwargs[e].split(' ') if arg.strip()]
                params += f'--{e} {cls.quote}{sep.join(all_args)}{cls.quote} '
            else:
                params += f'--{e} {cls.quote}{kwargs[e]}{cls.quote} '
        if 'port' not in kwargs:
            params += f'--port "{deploy_args.port}" '
        params += more_params_cmd + ' '
        devices = other_kwargs['gpu_id']
        devices = [d for d in devices if d]
        assert (len(devices) == 1 or 'cpu' not in devices)
        gpus = ','.join(devices)
        cuda_param = ''
        if gpus != 'cpu':
            cuda_param = f'CUDA_VISIBLE_DEVICES={gpus}'
        now = datetime.now()
        time_str = f'{now.year}{now.month}{now.day}{now.hour}{now.minute}{now.second}'
        file_path = f'output/{deploy_args.model_type}-{time_str}'
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        log_file = os.path.join(os.getcwd(), f'{file_path}/run_deploy.log')
        deploy_args.log_file = log_file
        params += f'--log_file "{log_file}" '
        params += '--ignore_args_error true '
        if sys.platform == 'win32':
            if cuda_param:
                cuda_param = f'set {cuda_param} && '
            run_command = f'{cuda_param}start /b swift deploy {params} > {log_file} 2>&1'
        else:
            run_command = f'{cuda_param} nohup swift deploy {params} > {log_file} 2>&1 &'
        return run_command, deploy_args, log_file

    @classmethod
    def deploy_model(cls, *args):
        if cls.is_gradio_app:
            if cls.deployed:
                return gr.update(), gr.update()
        run_command, deploy_args, log_file = cls.deploy(*args)
        logger.info(f'Running deployment command: {run_command}')
        os.system(run_command)
        if not cls.is_gradio_app:
            gr.Info(cls.locale('load_alert', cls.lang)['value'])
            time.sleep(2)
        else:
            from swift.llm.infer.deploy import is_accessible
            logger.info('Begin to check deploy statement...')
            cnt = 0
            while not is_accessible(deploy_args.port):
                time.sleep(1)
                cnt += 1
                if cnt >= 60:
                    logger.warn(f'Deploy costing too much time, please check log file: {log_file}')
            logger.info('Deploy done.')
        cls.deployed = True
        running_task = Runtime.refresh_tasks(log_file)
        if cls.is_gradio_app:
            cls.running_task = running_task['value']
        return gr.update(open=True), running_task

    @classmethod
    def clear_session(cls):
        return '', [], gr.update(value=None), gr.update(value=None), gr.update(value=None), []

    @classmethod
    def _replace_tag_with_media(cls, infer_request: InferRequest):
        total_history = []
        messages = deepcopy(infer_request.messages)
        if messages[0]['role'] == 'system':
            messages.pop(0)
        for i in range(0, len(messages), 2):
            slices = messages[i:i + 2]
            if len(slices) == 2:
                user, assistant = slices
            else:
                user = slices[0]
                assistant = {'role': 'assistant', 'content': None}
            user['content'] = (user['content'] or '').replace('<image>', '').replace('<video>',
                                                                                     '').replace('<audio>', '').strip()
            for media in user['medias']:
                total_history.append([(media, ), None])
            if user['content'] or assistant['content']:
                total_history.append((user['content'], assistant['content']))
        return total_history

    @classmethod
    def agent_type(cls, response):
        if not response:
            return None
        if response.lower().endswith('observation:'):
            return 'react'
        if 'observation:' not in response.lower() and 'action input:' in response.lower():
            return 'toolbench'
        return None

    @classmethod
    def send_message(cls, running_task, template_type, prompt: str, image, video, audio, infer_request: InferRequest,
                     infer_model_type, system, max_new_tokens, temperature, top_k, top_p, repetition_penalty):

        if not infer_request:
            infer_request = InferRequest(messages=[])
        if system:
            if not infer_request.messages or infer_request.messages[0]['role'] != 'system':
                infer_request.messages.insert(0, {'role': 'system', 'content': system})
            else:
                infer_request.messages[0]['content'] = system
        if not infer_request.messages or infer_request.messages[-1]['role'] != 'user':
            infer_request.messages.append({'role': 'user', 'content': '', 'medias': []})
        media = image or video or audio
        media_type = 'images' if image else 'videos' if video else 'audios'
        if media:
            _saved_medias: List = getattr(infer_request, media_type)
            if not _saved_medias or _saved_medias[-1] != media:
                _saved_medias.append(media)
                infer_request.messages[-1]['content'] = infer_request.messages[-1]['content'] + f'<{media_type[:-1]}>'
                infer_request.messages[-1]['medias'].append(media)

        if not prompt:
            yield '', cls._replace_tag_with_media(infer_request), gr.update(value=None), gr.update(
                value=None), gr.update(value=None), infer_request
            return
        else:
            infer_request.messages[-1]['content'] = infer_request.messages[-1]['content'] + prompt

        if cls.is_gradio_app:
            running_task = cls.running_task
        _, args = Runtime.parse_info_from_cmdline(running_task)
        request_config = RequestConfig(
            temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
        request_config.stream = True
        request_config.stop = ['Observation:']
        request_config.max_tokens = max_new_tokens
        stream_resp_with_history = ''
        response = ''
        i = len(infer_request.messages) - 1
        for i in range(len(infer_request.messages) - 1, -1, -1):
            if infer_request.messages[i]['role'] == 'assistant':
                response = infer_request.messages[i]['content']
        agent_type = cls.agent_type(response)
        if i != len(infer_request.messages) - 1 and agent_type == 'toolbench':
            infer_request.messages[i + 1]['role'] = 'tool'

        chat = not template_type.endswith('generation')
        _infer_request = deepcopy(infer_request)
        for m in _infer_request.messages:
            if 'medias' in m:
                m.pop('medias')
        model_kwargs = {}
        if infer_model_type:
            model_kwargs = {'model': infer_model_type}
        stream_resp = InferClient(
            port=args['port'], ).infer(
                infer_requests=[_infer_request], request_config=request_config, **model_kwargs)
        if infer_request.messages[-1]['role'] != 'assistant':
            infer_request.messages.append({'role': 'assistant', 'content': ''})
        for chunk in stream_resp:
            stream_resp_with_history += chunk[0].choices[0].delta.content if chat else chunk.choices[0].text
            infer_request.messages[-1]['content'] = stream_resp_with_history
            yield '', cls._replace_tag_with_media(infer_request), gr.update(value=None), gr.update(
                value=None), gr.update(value=None), infer_request
