# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
import sys
import time
from copy import deepcopy
from datetime import datetime
from functools import partial
from subprocess import DEVNULL, PIPE, STDOUT, Popen
from typing import Type

import gradio as gr
import json
import torch
from json import JSONDecodeError
from transformers.utils import is_torch_cuda_available, is_torch_npu_available

from swift.llm import DeployArguments, RLHFArguments, RolloutArguments
from swift.ui.base import BaseUI
from swift.ui.llm_grpo.external_runtime import RolloutRuntime
from swift.ui.llm_train.llm_train import run_command_in_background_with_popen
from swift.utils import get_device_count, get_logger

logger = get_logger()


class LLMRollout(BaseUI):

    group = 'llm_grpo'

    is_multimodal = True

    sub_ui = [RolloutRuntime]

    locale_dict = {
        'tensor_parallel_size': {
            'label': {
                'zh': '张量并行大小',
                'en': 'Tensor parallel size'
            },
        },
        'data_parallel_size': {
            'label': {
                'zh': '数据并行大小',
                'en': 'Data parallel size'
            },
        },
        'max_model_len': {
            'label': {
                'zh': '模型支持的最大长度',
                'en': 'Max model len'
            },
        },
        'gpu_memory_utilization': {
            'label': {
                'zh': 'GPU显存利用率',
                'en': 'GPU memory utilization'
            },
        },
        'port': {
            'label': {
                'zh': 'Rollout端口',
                'en': 'Rollout Port'
            },
        },
        'llm_rollout': {
            'label': {
                'zh': '外部rollout模型部署',
                'en': 'External rollout model deployment',
            }
        },
        'rollout': {
            'value': {
                'zh': '开始Rollout',
                'en': 'Start Rollout',
            }
        },
        'load_alert': {
            'value': {
                'zh': 'Rollout中，请点击"展示rollout状态"查看',
                'en': 'Start to rollout, '
                'please Click "Show running '
                'status" to view details',
            }
        },
        'port_alert': {
            'value': {
                'zh': '该端口已被占用',
                'en': 'The port has been occupied'
            }
        },
        'rollout_gpu_id': {
            'label': {
                'zh': '选择用于rollout的GPU',
                'en': 'Choose GPU for rollout'
            }
        },
        'more_roll_params': {
            'label': {
                'zh': '更多rollout参数',
                'en': 'More rollout params'
            },
            'info': {
                'zh': '以json格式或--xxx xxx命令行格式填入',
                'en': 'Fill in with json format or --xxx xxx cmd format'
            }
        }
    }

    choice_dict = BaseUI.get_choices_from_dataclass(RolloutArguments)
    default_dict = BaseUI.get_default_value_from_dataclass(RolloutArguments)
    arguments = BaseUI.get_argument_names(RolloutArguments)

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='llm_rollout', open=False, visible=False):
            default_device = 'cpu'
            device_count = get_device_count()
            if device_count > 0:
                default_device = '0'
            with gr.Blocks():
                with gr.Row():
                    gr.Textbox(elem_id='tensor_parallel_size', lines=1, value='1', scale=4)
                    gr.Textbox(elem_id='data_parallel_size', lines=1, value='1', scale=4)
                    gr.Slider(elem_id='gpu_memory_utilization', minimum=0.0, maximum=1.0, step=0.05, value=0.9, scale=4)
                with gr.Row(equal_height=True):
                    gr.Dropdown(
                        elem_id='rollout_gpu_id',
                        multiselect=True,
                        choices=[str(i) for i in range(device_count)] + ['cpu'],
                        value=default_device,
                        scale=4)
                    gr.Textbox(elem_id='port', lines=1, value='8000', scale=2)
                    gr.Textbox(elem_id='more_roll_params', lines=1, scale=8)
                    gr.Button(elem_id='rollout', scale=2, variant='primary')
                RolloutRuntime.build_ui(base_tab)

                base_tab.element('rollout_running_tasks').change(
                    partial(RolloutRuntime.task_changed, base_tab=base_tab),
                    [base_tab.element('rollout_running_tasks')],
                    list(cls.valid_elements().values()) + [cls.element('rollout_log')])
                RolloutRuntime.element('rollout_kill_task').click(
                    RolloutRuntime.kill_task,
                    [RolloutRuntime.element('rollout_running_tasks')],
                    [RolloutRuntime.element('rollout_running_tasks')] + [RolloutRuntime.element('rollout_log')],
                )

    @classmethod
    def rollout(cls, *args):
        rollout_args = cls.get_default_value_from_dataclass(RolloutArguments)
        kwargs = {}
        kwargs_is_list = {}
        other_kwargs = {}
        more_params = {}
        more_params_cmd = ''
        model_args = args[-3:]
        kwargs['model'] = model_args[0]
        kwargs['model_type'] = model_args[1]
        kwargs['template'] = model_args[2]
        args = args[:-3]
        keys = cls.valid_element_keys()
        for key, value in zip(keys, args):
            compare_value = rollout_args.get(key)
            compare_value_arg = str(compare_value) if not isinstance(compare_value, (list, dict)) else compare_value
            compare_value_ui = str(value) if not isinstance(value, (list, dict)) else value
            if key in rollout_args and compare_value_ui != compare_value_arg and value:
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
            if key == 'more_roll_params' and value:
                try:
                    more_params = json.loads(value)
                except (JSONDecodeError or TypeError):
                    more_params_cmd = value

        kwargs.update(more_params)
        rollout_args = RolloutArguments(
            **{
                key: value.split(' ') if key in kwargs_is_list and kwargs_is_list[key] else value
                for key, value in kwargs.items()
            })
        if rollout_args.port in RolloutRuntime.get_all_ports():
            raise gr.Error(cls.locale('port_alert', cls.lang)['value'])
        params = ''
        command = ['swift', 'rollout']
        sep = f'{cls.quote} {cls.quote}'
        for e in kwargs:
            if isinstance(kwargs[e], list):
                params += f'--{e} {cls.quote}{sep.join(kwargs[e])}{cls.quote} '
                command.extend([f'--{e}'] + kwargs[e])
            elif e in kwargs_is_list and kwargs_is_list[e]:
                all_args = [arg for arg in kwargs[e].split(' ') if arg.strip()]
                params += f'--{e} {cls.quote}{sep.join(all_args)}{cls.quote} '
                command.extend([f'--{e}'] + all_args)
            else:
                params += f'--{e} {cls.quote}{kwargs[e]}{cls.quote} '
                command.extend([f'--{e}', f'{kwargs[e]}'])
        if 'port' not in kwargs:
            params += f'--port "{rollout_args.port}" '
            command.extend(['--port', f'{rollout_args.port}'])
        if more_params_cmd != '':
            params += f'{more_params_cmd.strip()} '
            more_params_cmd = [param.strip() for param in more_params_cmd.split('--')]
            more_params_cmd = [param.split(' ') for param in more_params_cmd if param]
            for param in more_params_cmd:
                command.extend([f'--{param[0]}'] + param[1:])
        devices = other_kwargs['rollout_gpu_id']
        devices = [d for d in devices if d]
        assert (len(devices) == 1 or 'cpu' not in devices)
        gpus = ','.join(devices)
        cuda_param = ''
        all_envs = {}
        if gpus != 'cpu':
            if is_torch_npu_available():
                cuda_param = f'ASCEND_RT_VISIBLE_DEVICES={gpus}'
                all_envs['ASCEND_RT_VISIBLE_DEVICES'] = gpus
            elif is_torch_cuda_available():
                cuda_param = f'CUDA_VISIBLE_DEVICES={gpus}'
                all_envs['CUDA_VISIBLE_DEVICES'] = gpus
            else:
                cuda_param = ''
        output_dir = 'rollout_output'
        now = datetime.now()
        time_str = f'{now.year}{now.month}{now.day}{now.hour}{now.minute}{now.second}'
        file_path = f'{output_dir}/{rollout_args.model_type}-{time_str}'
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        log_file = os.path.join(os.getcwd(), f'{file_path}/run_rollout.log')
        rollout_args.log_file = log_file
        params += f'--log_file "{log_file}" '
        command.extend(['--log_file', f'{log_file}'])
        params += '--ignore_args_error true '
        command.extend(['--ignore_args_error', 'true'])
        if sys.platform == 'win32':
            if cuda_param:
                cuda_param = f'set {cuda_param} && '
            run_command = f'{cuda_param}start /b swift rollout {params} > {log_file} 2>&1'
        else:
            run_command = f'{cuda_param} nohup swift rollout {params} > {log_file} 2>&1 &'
        return command, all_envs, run_command, rollout_args, log_file

    @classmethod
    def rollout_model(cls, *args):
        command, all_envs, run_command, rollout_args, log_file = cls.rollout(*args)
        logger.info(f'Running rollout command: {run_command}')
        run_command_in_background_with_popen(command, all_envs, log_file)
        gr.Info(cls.locale('load_alert', cls.lang)['value'])
        time.sleep(2)
        running_task = RolloutRuntime.refresh_tasks(log_file)
        return gr.update(open=True), running_task

    @classmethod
    def external_rollout_display(cls, mode):
        if mode == 'server':
            return gr.update(visible=True, open=True)
        return gr.update(visible=False)
