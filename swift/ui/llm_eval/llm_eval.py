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
from json import JSONDecodeError
from modelscope import snapshot_download

from swift.llm import EvalArguments
from swift.ui.base import BaseUI
from swift.ui.llm_eval.eval import Eval
from swift.ui.llm_eval.model import Model
from swift.ui.llm_eval.runtime import EvalRuntime


class LLMEval(BaseUI):
    group = 'llm_eval'

    sub_ui = [Model, Eval, EvalRuntime]

    cmd = 'eval'

    locale_dict = {
        'llm_eval': {
            'label': {
                'zh': 'LLM评测',
                'en': 'LLM evaluation',
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
        'evaluate': {
            'value': {
                'zh': '开始评测',
                'en': 'Begin Evaluation'
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

    choice_dict = BaseUI.get_choices_from_dataclass(EvalArguments)
    default_dict = BaseUI.get_default_value_from_dataclass(EvalArguments)
    arguments = BaseUI.get_argument_names(EvalArguments)

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_eval', label=''):
            gpu_count = 0
            default_device = 'cpu'
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                default_device = '0'
            with gr.Blocks():
                model_and_template = gr.State([])
                Model.build_ui(base_tab)
                Eval.build_ui(base_tab)
                EvalRuntime.build_ui(base_tab)
                with gr.Row():
                    gr.Textbox(elem_id='more_params', lines=4, scale=20)
                    gr.Button(elem_id='evaluate', scale=2, variant='primary')
                gr.Dropdown(
                    elem_id='gpu_id',
                    multiselect=True,
                    choices=[str(i) for i in range(gpu_count)] + ['cpu'],
                    value=default_device,
                    scale=8)

                cls.element('evaluate').click(
                    cls.eval_model,
                    [value for value in cls.elements().values() if not isinstance(value, (Tab, Accordion))],
                    [cls.element('runtime_tab'),
                     cls.element('running_tasks'), model_and_template])

                base_tab.element('running_tasks').change(
                    partial(EvalRuntime.task_changed, base_tab=base_tab), [base_tab.element('running_tasks')],
                    [value for value in base_tab.elements().values() if not isinstance(value, (Tab, Accordion))]
                    + [cls.element('log'), model_and_template],
                    cancels=EvalRuntime.log_event)
                EvalRuntime.element('kill_task').click(
                    EvalRuntime.kill_task,
                    [EvalRuntime.element('running_tasks')],
                    [EvalRuntime.element('running_tasks')] + [EvalRuntime.element('log')],
                    cancels=[EvalRuntime.log_event],
                )

    @classmethod
    def eval(cls, *args):
        eval_args = cls.get_default_value_from_dataclass(EvalArguments)
        kwargs = {}
        kwargs_is_list = {}
        other_kwargs = {}
        more_params = {}
        more_params_cmd = ''
        keys = [key for key, value in cls.elements().items() if not isinstance(value, (Tab, Accordion))]
        for key, value in zip(keys, args):
            compare_value = eval_args.get(key)
            compare_value_arg = str(compare_value) if not isinstance(compare_value, (list, dict)) else compare_value
            compare_value_ui = str(value) if not isinstance(value, (list, dict)) else value
            if key in eval_args and compare_value_ui != compare_value_arg and value:
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
        if kwargs['model_type'] == cls.locale('checkpoint', cls.lang)['value']:
            model_dir = kwargs.pop('model_id_or_path')
            if not os.path.exists(model_dir):
                model_dir = snapshot_download(model_dir)
            kwargs['ckpt_dir'] = model_dir

        eval_args = EvalArguments(
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
        file_path = f'output/{eval_args.model_type}-{time_str}'
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        log_file = os.path.join(os.getcwd(), f'{file_path}/run_eval.log')
        eval_args.log_file = log_file
        params += f'--log_file "{log_file}" '
        params += '--ignore_args_error true '
        if sys.platform == 'win32':
            if cuda_param:
                cuda_param = f'set {cuda_param} && '
            run_command = f'{cuda_param}start /b swift eval {params} > {log_file} 2>&1'
        else:
            run_command = f'{cuda_param} nohup swift eval {params} > {log_file} 2>&1 &'
        return run_command, eval_args, log_file

    @classmethod
    def eval_model(cls, *args):
        run_command, eval_args, log_file = cls.eval(*args)
        os.system(run_command)
        time.sleep(2)
        return gr.update(open=True), EvalRuntime.refresh_tasks(log_file), [eval_args.sft_type]
