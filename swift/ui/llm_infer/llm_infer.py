import os
import re
import sys
import time
from datetime import datetime
from functools import partial
from typing import Type

import gradio as gr
import json
import torch
from gradio import Accordion, Tab
from modelscope import GenerationConfig

from swift import snapshot_download
from swift.llm import (DeployArguments, InferArguments, XRequestConfig, inference_client, inference_stream,
                       limit_history_length, prepare_model_template)
from swift.ui.base import BaseUI
from swift.ui.llm_infer.model import Model
from swift.ui.llm_infer.runtime import Runtime


class LLMInfer(BaseUI):
    group = 'llm_infer'

    sub_ui = [Model, Runtime]

    is_inference = os.environ.get('USE_INFERENCE') == '1' or os.environ.get('MODELSCOPE_ENVIRONMENT') == 'studio'

    locale_dict = {
        'generate_alert': {
            'value': {
                'zh': '请先加载模型' if is_inference else '请先部署模型',
                'en': 'Please load model first' if is_inference else 'Please deploy model first',
            }
        },
        'llm_infer': {
            'label': {
                'zh': 'LLM推理' if is_inference else 'LLM部署',
                'en': 'LLM Inference' if is_inference else 'LLM Deployment',
            }
        },
        'load_alert': {
            'value': {
                'zh':
                '加载中，请等待' if is_inference else '部署中，请点击"展示部署状态"查看',
                'en':
                'Start to load model, please wait' if is_inference else 'Start to deploy model, '
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
                model_and_template = gr.State([])
                Model.build_ui(base_tab)
                Runtime.build_ui(base_tab)
                gr.Dropdown(
                    elem_id='gpu_id',
                    multiselect=True,
                    choices=[str(i) for i in range(gpu_count)] + ['cpu'],
                    value=default_device,
                    scale=8)
                chatbot = gr.Chatbot(elem_id='chatbot', elem_classes='control-height')
                prompt = gr.Textbox(elem_id='prompt', lines=1, interactive=True)

                with gr.Row():
                    clear_history = gr.Button(elem_id='clear_history')
                    submit = gr.Button(elem_id='submit')

                if cls.is_inference:
                    submit.click(
                        cls.generate_chat,
                        inputs=[
                            model_and_template,
                            cls.element('template_type'), prompt, chatbot,
                            cls.element('system'),
                            cls.element('max_new_tokens'),
                            cls.element('temperature'),
                            cls.element('do_sample'),
                            cls.element('top_k'),
                            cls.element('top_p'),
                            cls.element('repetition_penalty')
                        ],
                        outputs=[prompt, chatbot],
                        queue=True)

                    clear_history.click(fn=cls.clear_session, inputs=[], outputs=[prompt, chatbot])

                    cls.element('load_checkpoint').click(
                        cls.reset_memory, [], [model_and_template]) \
                        .then(cls.reset_loading_button, [], [cls.element('load_checkpoint')]).then(
                        cls.prepare_checkpoint, [
                            value for value in cls.elements().values()
                            if not isinstance(value, (Tab, Accordion))
                        ], [model_and_template]).then(cls.change_interactive, [],
                                                      [prompt]).then(  # noqa
                        cls.clear_session,
                        inputs=[],
                        outputs=[prompt, chatbot],
                        queue=True).then(cls.reset_load_button, [], [cls.element('load_checkpoint')])
                else:
                    cls.element('load_checkpoint').click(
                        cls.deploy_model,
                        [value for value in cls.elements().values() if not isinstance(value, (Tab, Accordion))],
                        [cls.element('runtime_tab'),
                         cls.element('running_tasks'), model_and_template])
                    submit.click(
                        cls.send_message,
                        inputs=[
                            cls.element('running_tasks'), model_and_template,
                            cls.element('template_type'), prompt, chatbot,
                            cls.element('system'),
                            cls.element('max_new_tokens'),
                            cls.element('temperature'),
                            cls.element('top_k'),
                            cls.element('top_p'),
                            cls.element('repetition_penalty')
                        ],
                        outputs=[prompt, chatbot],
                        queue=True)

                    clear_history.click(fn=cls.clear_session, inputs=[], outputs=[prompt, chatbot])

                    base_tab.element('running_tasks').change(
                        partial(Runtime.task_changed, base_tab=base_tab), [base_tab.element('running_tasks')],
                        [value for value in base_tab.elements().values() if not isinstance(value, (Tab, Accordion))]
                        + [cls.element('log'), model_and_template],
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
        keys = [key for key, value in cls.elements().items() if not isinstance(value, (Tab, Accordion))]
        for key, value in zip(keys, args):
            compare_value = deploy_args.get(key)
            compare_value_arg = str(compare_value) if not isinstance(compare_value, (list, dict)) else compare_value
            compare_value_ui = str(value) if not isinstance(value, (list, dict)) else value
            if key in deploy_args and compare_value_ui != compare_value_arg and value:
                if isinstance(value, str) and re.fullmatch(cls.int_regex, value):
                    value = int(value)
                elif isinstance(value, str) and re.fullmatch(cls.float_regex, value):
                    value = float(value)
                kwargs[key] = value if not isinstance(value, list) else ' '.join(value)
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

        if 'ckpt_dir' in kwargs:
            with open(os.path.join(kwargs['ckpt_dir'], 'sft_args.json'), 'r') as f:
                _json = json.load(f)
                kwargs['model_type'] = _json['model_type']
                kwargs['sft_type'] = _json['sft_type']
        deploy_args = DeployArguments(
            **{
                key: value.split(' ') if key in kwargs_is_list and kwargs_is_list[key] else value
                for key, value in kwargs.items()
            })
        if deploy_args.port in Runtime.get_all_ports():
            raise gr.Error(cls.locale('port_alert', cls.lang)['value'])
        params = ''
        for e in kwargs:
            if e in kwargs_is_list and kwargs_is_list[e]:
                params += f'--{e} {kwargs[e]} '
            else:
                params += f'--{e} "{kwargs[e]}" '
        if 'port' not in kwargs:
            params += f'--port "{deploy_args.port}" '
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
        run_command, deploy_args, log_file = cls.deploy(*args)
        os.system(run_command)
        gr.Info(cls.locale('load_alert', cls.lang)['value'])
        time.sleep(2)
        return gr.update(open=True), Runtime.refresh_tasks(log_file), [
            deploy_args.model_type, deploy_args.template_type, deploy_args.sft_type
        ]

    @classmethod
    def update_runtime(cls):
        return gr.update(open=True), gr.update(visible=True)

    @classmethod
    def reset_load_button(cls):
        return gr.update(value=cls.locale('load_checkpoint', cls.lang)['value'])

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
        keys = [key for key, value in cls.elements().items() if not isinstance(value, (Tab, Accordion))]
        for key, value in zip(keys, args):
            compare_value = infer_args.get(key)
            compare_value_arg = str(compare_value) if not isinstance(compare_value, (list, dict)) else compare_value
            compare_value_ui = str(value) if not isinstance(value, (list, dict)) else value
            if key in infer_args and compare_value_ui != compare_value_arg and value:
                if isinstance(value, str) and re.fullmatch(cls.int_regex, value):
                    value = int(value)
                elif isinstance(value, str) and re.fullmatch(cls.float_regex, value):
                    value = float(value)
                kwargs[key] = value if not isinstance(value, list) else ' '.join(value)
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
        if 'ckpt_dir' in kwargs or ('model_id_or_path' in kwargs and not os.path.exists(kwargs['model_id_or_path'])):
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
    def send_message(cls, running_task, model_and_template, template_type, prompt: str, history, system, max_new_tokens,
                     temperature, top_k, top_p, repetition_penalty):
        if not model_and_template:
            gr.Warning(cls.locale('generate_alert', cls.lang)['value'])
            return '', None
        _, args = Runtime.parse_info_from_cmdline(running_task)
        model_type, template, sft_type = model_and_template
        old_history, history = history or [], []
        request_config = XRequestConfig(
            temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
        request_config.stream = True
        request_config.stop = ['Observation:']
        stream_resp_with_history = ''
        if not template_type.endswith('generation'):
            stream_resp = inference_client(
                model_type,
                prompt,
                old_history,
                system=system,
                port=args['port'],
                adapter_name='default-lora' if sft_type in ('lora', 'longlora') else None,
                request_config=request_config)
            for chunk in stream_resp:
                stream_resp_with_history += chunk.choices[0].delta.content
                qr_pair = [prompt, stream_resp_with_history]
                total_history = old_history + [qr_pair]
                yield '', total_history
        else:
            request_config.max_tokens = max_new_tokens
            stream_resp = inference_client(
                model_type,
                prompt,
                port=args['port'],
                adapter_name='default-lora' if sft_type in ('lora', 'longlora') else None,
                request_config=request_config)
            for chunk in stream_resp:
                stream_resp_with_history += chunk.choices[0].text
                qr_pair = [prompt, stream_resp_with_history]
                total_history = old_history + [qr_pair]
                yield '', total_history

    @classmethod
    def generate_chat(cls, model_and_template, template_type, prompt: str, history, system, max_new_tokens, temperature,
                      do_sample, top_k, top_p, repetition_penalty):
        if not model_and_template:
            gr.Warning(cls.locale('generate_alert', cls.lang)['value'])
            return '', None
        model, template = model_and_template
        if os.environ.get('MODELSCOPE_ENVIRONMENT') == 'studio':
            model.cuda()
        if not template_type.endswith('generation'):
            old_history, history = limit_history_length(template, prompt, history, int(max_new_tokens))
        else:
            old_history = []
            history = []

        generation_config = GenerationConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=repetition_penalty)
        gen = inference_stream(
            model,
            template,
            prompt,
            history,
            system=system,
            generation_config=generation_config,
            stop_words=['Observation:'])
        for _, history in gen:
            total_history = old_history + history
            yield '', total_history
        if os.environ.get('MODELSCOPE_ENVIRONMENT') == 'studio':
            model.cpu()
