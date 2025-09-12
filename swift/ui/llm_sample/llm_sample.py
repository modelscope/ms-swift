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
from json import JSONDecodeError
from transformers.utils import is_torch_cuda_available, is_torch_npu_available

from swift.llm import SamplingArguments
from swift.llm.dataset.register import get_dataset_list
from swift.ui.base import BaseUI
from swift.ui.llm_sample.model import Model
from swift.ui.llm_sample.runtime import SampleRuntime
from swift.ui.llm_sample.sample import Sample
from swift.ui.llm_train.utils import run_command_in_background_with_popen
from swift.utils import get_device_count, get_logger

logger = get_logger()


class LLMSample(BaseUI):

    group = 'llm_sample'

    is_multimodal = True

    sub_ui = [Model, Sample, SampleRuntime]

    locale_dict = {
        'llm_sample': {
            'label': {
                'zh': 'LLM采样',
                'en': 'LLM Sampling',
            }
        },
        'sample': {
            'value': {
                'zh': '开始采样',
                'en': 'Start sampling',
            }
        },
        'load_alert': {
            'value': {
                'zh': '采样中，请点击"展示采样状态"查看',
                'en': 'Start to sample, '
                'please Click "Show running '
                'status" to view details',
            }
        },
        'gpu_id': {
            'label': {
                'zh': '选择可用GPU',
                'en': 'Choose GPU'
            },
            'info': {
                'zh': '选择采样使用的GPU号，如CUDA不可用只能选择CPU',
                'en': 'Select GPU to sample'
            }
        },
        'dataset': {
            'label': {
                'zh': '数据集名称',
                'en': 'Dataset id/path'
            },
            'info': {
                'zh': '选择采样的数据集，支持复选/本地路径',
                'en': 'The dataset(s) to train the models, support multi select and local folder/files'
            }
        },
        'num_sampling_per_gpu_batch_size': {
            'label': {
                'zh': '每次采样的批次大小',
                'en': 'The batch size of sampling'
            }
        },
        'num_sampling_per_gpu_batches': {
            'label': {
                'zh': '采样批次数量',
                'en': 'Num of Sampling batches'
            }
        },
        'output_dir': {
            'label': {
                'zh': '存储目录',
                'en': 'The output dir',
            },
            'info': {
                'zh': '设置采样结果存储在哪个文件夹下',
                'en': 'Set the output folder',
            }
        },
        'envs': {
            'label': {
                'zh': '环境变量',
                'en': 'Extra env vars'
            },
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
    }

    choice_dict = BaseUI.get_choices_from_dataclass(SamplingArguments)
    default_dict = BaseUI.get_default_value_from_dataclass(SamplingArguments)
    arguments = BaseUI.get_argument_names(SamplingArguments)

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_sample', label=''):
            default_device = 'cpu'
            device_count = get_device_count()
            if device_count > 0:
                default_device = '0'
            with gr.Blocks():
                Model.build_ui(base_tab)
                Sample.build_ui(base_tab)
                with gr.Row():
                    gr.Dropdown(
                        elem_id='dataset',
                        multiselect=True,
                        choices=get_dataset_list(),
                        scale=20,
                        allow_custom_value=True)
                    gr.Slider(
                        elem_id='num_sampling_per_gpu_batch_size', minimum=1, maximum=128, step=1, value=1, scale=10)
                    gr.Slider(elem_id='num_sampling_per_gpu_batches', minimum=1, maximum=128, step=1, value=1, scale=10)
                SampleRuntime.build_ui(base_tab)
                with gr.Row(equal_height=True):
                    gr.Dropdown(
                        elem_id='gpu_id',
                        multiselect=True,
                        choices=[str(i) for i in range(device_count)] + ['cpu'],
                        value=default_device,
                        scale=20)
                    gr.Textbox(elem_id='output_dir', value='sample_output', scale=20)
                    gr.Textbox(elem_id='envs', scale=20)
                    gr.Button(elem_id='sample', scale=2, variant='primary')
                with gr.Row():
                    gr.Textbox(elem_id='more_params', lines=4)

                cls.element('sample').click(
                    cls.sample_model, list(base_tab.valid_elements().values()),
                    [cls.element('runtime_tab'), cls.element('running_tasks')])

                base_tab.element('running_tasks').change(
                    partial(SampleRuntime.task_changed, base_tab=base_tab), [base_tab.element('running_tasks')],
                    list(cls.valid_elements().values()) + [cls.element('log')])
                SampleRuntime.element('kill_task').click(
                    SampleRuntime.kill_task,
                    [SampleRuntime.element('running_tasks')],
                    [SampleRuntime.element('running_tasks')] + [SampleRuntime.element('log')],
                )

    @classmethod
    def sample(cls, *args):
        sample_args = cls.get_default_value_from_dataclass(SamplingArguments)
        kwargs = {}
        kwargs_is_list = {}
        other_kwargs = {}
        more_params = {}
        more_params_cmd = ''
        keys = cls.valid_element_keys()
        for key, value in zip(keys, args):
            compare_value = sample_args.get(key)
            compare_value_arg = str(compare_value) if not isinstance(compare_value, (list, dict)) else compare_value
            compare_value_ui = str(value) if not isinstance(value, (list, dict)) else value
            if key in sample_args and compare_value_ui != compare_value_arg and value:
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
            with open(os.path.join(kwargs['ckpt_dir'], 'args.json'), 'r', encoding='utf-8') as f:
                _json = json.load(f)
                kwargs['model_type'] = _json['model_type']
                kwargs['train_type'] = _json['train_type']
        sample_args = SamplingArguments(
            **{
                key: value.split(' ') if key in kwargs_is_list and kwargs_is_list[key] else value
                for key, value in kwargs.items()
            })

        params = ''
        command = ['swift', 'sample']
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
        if more_params_cmd != '':
            params += more_params_cmd + ' '
            more_params_cmd = [param.strip() for param in more_params_cmd.split('--')]
            more_params_cmd = [param.split(' ') for param in more_params_cmd if param]
            for param in more_params_cmd:
                command.extend([f'--{param[0]}'] + param[1:])
        all_envs = {}
        devices = other_kwargs['gpu_id']
        devices = [d for d in devices if d]
        assert (len(devices) == 1 or 'cpu' not in devices)
        gpus = ','.join(devices)
        cuda_param = ''
        if gpus != 'cpu':
            if is_torch_npu_available():
                cuda_param = f'ASCEND_RT_VISIBLE_DEVICES={gpus}'
                all_envs['ASCEND_RT_VISIBLE_DEVICES'] = gpus
            elif is_torch_cuda_available():
                cuda_param = f'CUDA_VISIBLE_DEVICES={gpus}'
                all_envs['CUDA_VISIBLE_DEVICES'] = gpus
            else:
                cuda_param = ''
        now = datetime.now()
        time_str = f'{now.year}{now.month}{now.day}{now.hour}{now.minute}{now.second}'
        file_path = f'output/{sample_args.model_type}-{time_str}'
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        log_file = os.path.join(os.getcwd(), f'{file_path}/run_sample.log')
        sample_args.log_file = log_file
        params += f'--log_file "{log_file}" '
        command.extend(['--log_file', f'{log_file}'])
        params += '--ignore_args_error true '
        command.extend(['--ignore_args_error', 'true'])
        if sys.platform == 'win32':
            if cuda_param:
                cuda_param = f'set {cuda_param} && '
            run_command = f'{cuda_param}start /b swift sample {params} > {log_file} 2>&1'
        else:
            run_command = f'{cuda_param} nohup swift sample {params} > {log_file} 2>&1 &'
        return command, all_envs, run_command, sample_args, log_file

    @classmethod
    def sample_model(cls, *args):
        command, all_envs, run_command, sample_args, log_file = cls.sample(*args)
        logger.info(f'Running sample command: {run_command}')
        run_command_in_background_with_popen(command, all_envs, log_file)
        gr.Info(cls.locale('load_alert', cls.lang)['value'])
        time.sleep(2)
        running_task = SampleRuntime.refresh_tasks(log_file)
        return gr.update(open=True), running_task
