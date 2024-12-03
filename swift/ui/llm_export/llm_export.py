# Copyright (c) Alibaba, Inc. and its affiliates.
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
from json import JSONDecodeError

from swift.llm import ExportArguments
from swift.ui.base import BaseUI
from swift.ui.llm_export.export import Export
from swift.ui.llm_export.model import Model
from swift.ui.llm_export.runtime import ExportRuntime


class LLMExport(BaseUI):
    group = 'llm_export'

    sub_ui = [Model, Export, ExportRuntime]

    locale_dict = {
        'llm_export': {
            'label': {
                'zh': 'LLM导出',
                'en': 'LLM export',
            }
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
        'export': {
            'value': {
                'zh': '开始导出',
                'en': 'Begin Export'
            },
        },
        'gpu_id': {
            'label': {
                'zh': '选择可用GPU',
                'en': 'Choose GPU'
            },
            'info': {
                'zh': '选择使用的GPU号，如CUDA不可用只能选择CPU',
                'en': 'Select GPU to export'
            }
        },
    }

    choice_dict = BaseUI.get_choices_from_dataclass(ExportArguments)
    default_dict = BaseUI.get_default_value_from_dataclass(ExportArguments)
    arguments = BaseUI.get_argument_names(ExportArguments)

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_export', label=''):
            gpu_count = 0
            default_device = 'cpu'
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                default_device = '0'
            with gr.Blocks():
                Model.build_ui(base_tab)
                Export.build_ui(base_tab)
                ExportRuntime.build_ui(base_tab)
                with gr.Row():
                    gr.Textbox(elem_id='more_params', lines=4, scale=20)
                    gr.Button(elem_id='export', scale=2, variant='primary')
                gr.Dropdown(
                    elem_id='gpu_id',
                    multiselect=True,
                    choices=[str(i) for i in range(gpu_count)] + ['cpu'],
                    value=default_device,
                    scale=8)

                cls.element('export').click(
                    cls.export_model, list(base_tab.valid_elements().values()),
                    [cls.element('runtime_tab'), cls.element('running_tasks')])

                base_tab.element('running_tasks').change(
                    partial(ExportRuntime.task_changed, base_tab=base_tab), [base_tab.element('running_tasks')],
                    list(base_tab.valid_elements().values()) + [cls.element('log')],
                    cancels=ExportRuntime.log_event)
                ExportRuntime.element('kill_task').click(
                    ExportRuntime.kill_task,
                    [ExportRuntime.element('running_tasks')],
                    [ExportRuntime.element('running_tasks')] + [ExportRuntime.element('log')],
                    cancels=[ExportRuntime.log_event],
                )

    @classmethod
    def export(cls, *args):
        export_args = cls.get_default_value_from_dataclass(ExportArguments)
        kwargs = {}
        kwargs_is_list = {}
        other_kwargs = {}
        more_params = {}
        more_params_cmd = ''
        keys = cls.valid_element_keys()
        for key, value in zip(keys, args):
            compare_value = export_args.get(key)
            compare_value_arg = str(compare_value) if not isinstance(compare_value, (list, dict)) else compare_value
            compare_value_ui = str(value) if not isinstance(value, (list, dict)) else value
            if key in export_args and compare_value_ui != compare_value_arg and value:
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
        export_args = ExportArguments(
            **{
                key: value.split(' ') if key in kwargs_is_list and kwargs_is_list[key] else value
                for key, value in kwargs.items()
            })
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
        file_path = f'output/{export_args.model_type}-{time_str}'
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        log_file = os.path.join(os.getcwd(), f'{file_path}/run_export.log')
        export_args.log_file = log_file
        params += f'--log_file "{log_file}" '
        params += '--ignore_args_error true '
        additional_param = ''
        if export_args.quant_method == 'gptq':
            additional_param = 'OMP_NUM_THREADS=14'
        if sys.platform == 'win32':
            if cuda_param:
                cuda_param = f'set {cuda_param} && '
            if additional_param:
                additional_param = f'set {additional_param} && '
            run_command = f'{cuda_param}{additional_param}start /b swift export {params} > {log_file} 2>&1'
        else:
            run_command = f'{cuda_param} {additional_param} nohup swift export {params} > {log_file} 2>&1 &'
        return run_command, export_args, log_file

    @classmethod
    def export_model(cls, *args):
        run_command, export_args, log_file = cls.export(*args)
        os.system(run_command)
        time.sleep(2)
        return gr.update(open=True), ExportRuntime.refresh_tasks(log_file), [None]
