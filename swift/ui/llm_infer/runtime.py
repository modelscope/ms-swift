import os.path
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Tuple, Type

import gradio as gr
import psutil
import torch

from swift.llm import DeployArguments
from swift.ui.base import BaseUI
from swift.utils import get_logger

logger = get_logger()


class Runtime(BaseUI):
    handlers: Dict[str, Tuple[List, Tuple]] = {}

    group = 'llm_infer'

    locale_dict = {
        'runtime_tab': {
            'label': {
                'zh': '运行时',
                'en': 'Runtime'
            },
        },
        'running_tasks': {
            'label': {
                'zh': '运行中任务',
                'en': 'Running Tasks'
            },
            'info': {
                'zh': '运行中的任务（所有的swift deploy命令）',
                'en': 'All running tasks(started by swift deploy)'
            }
        },
        'show_curl': {
            'label': {
                'zh': 'curl调用方式展示',
                'en': 'Show curl calling method'
            },
            'info': {
                'zh': '仅展示，不可编辑',
                'en': 'Not editable'
            }
        },
        'refresh_tasks': {
            'value': {
                'zh': '刷新运行时任务',
                'en': 'Refresh tasks'
            },
        },
        'kill_task': {
            'value': {
                'zh': '停止任务',
                'en': 'Kill running task'
            },
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='runtime_tab', open=False, visible=True):
            with gr.Blocks():
                with gr.Column():
                    with gr.Row():
                        gr.Dropdown(elem_id='running_tasks', scale=10)
                        gr.Button(elem_id='refresh_tasks', scale=1)
                        gr.Button(elem_id='kill_task', scale=1)
                    gr.Textbox(elem_id='show_curl', interactive=False)
                base_tab.element('refresh_tasks').click(
                    Runtime.refresh_tasks,
                    [base_tab.element('running_tasks')],
                    [base_tab.element('show_curl')]
                    + [base_tab.element('running_tasks')],
                )
                base_tab.element('kill_task').click(
                    Runtime.kill_task,
                    [base_tab.element('running_tasks')],
                    [base_tab.element('running_tasks')]
                    + [base_tab.element('show_curl')],
                )

    @staticmethod
    def refresh_tasks(running_task=None):
        output_dir = running_task if not running_task or 'pid:' not in running_task else None
        process_name = 'swift'
        cmd_name = 'deploy'
        process = []
        selected = None
        for proc in psutil.process_iter():
            try:
                cmdlines = proc.cmdline()
            except (psutil.ZombieProcess, psutil.AccessDenied,
                    psutil.NoSuchProcess):
                cmdlines = []
            if any([process_name in cmdline
                    for cmdline in cmdlines]) and any(  # noqa
                        [cmd_name == cmdline for cmdline in cmdlines]):  # noqa
                process.append(Runtime.construct_running_task(proc))
                if output_dir is not None and any(  # noqa
                    [output_dir == cmdline for cmdline in cmdlines]):  # noqa
                    selected = Runtime.construct_running_task(proc)
        if not selected:
            if running_task and running_task in process:
                selected = running_task
        if not selected and process:
            selected = process[0]
        return Runtime.show_curl(selected), gr.update(
            choices=process, value=selected)

    @staticmethod
    def construct_running_task(proc):
        pid = proc.pid
        ts = time.time()
        create_time = proc.create_time()
        create_time_formatted = datetime.fromtimestamp(create_time).strftime(
            '%Y-%m-%d, %H:%M')

        def format_time(seconds):
            days = int(seconds // (24 * 3600))
            hours = int((seconds % (24 * 3600)) // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)

            if days > 0:
                time_str = f'{days}d {hours}h {minutes}m {seconds}s'
            elif hours > 0:
                time_str = f'{hours}h {minutes}m {seconds}s'
            elif minutes > 0:
                time_str = f'{minutes}m {seconds}s'
            else:
                time_str = f'{seconds}s'

            return time_str

        return f'pid:{pid}/create:{create_time_formatted}' \
               f'/running:{format_time(ts - create_time)}/cmd:{" ".join(proc.cmdline())}'

    @staticmethod
    def kill_task(task):
        if task is None:
            return None, None
        pid = task.split('/')[0].split(':')[1]
        result = subprocess.run(['ps', '--ppid', f'{pid}'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        std_out = result.stdout
        ppid = std_out.split('\n')[1].split(' ')[0]
        os.system(f'kill -9 {pid}')
        os.system(f'kill -9 {ppid}')
        time.sleep(1)
        torch.cuda.empty_cache()
        return None, None

    @staticmethod
    def show_curl(selected_task):

        if selected_task is None:
            return None

        prompt = '浙江 -> 杭州 安徽 -> 合肥 四川 ->'
        content = '晚上睡不着觉怎么办？'
        deploy_cmd_args = selected_task.split('swift deploy')[1].strip().split(
            ' ')
        deploy_cmd_args_dict = {
            k: v
            for k, v in zip(deploy_cmd_args[0::2], deploy_cmd_args[1::2])
        }

        if '--model_id_or_path' not in deploy_cmd_args_dict.keys():
            # ckpt_dir
            if '--ckpt_dir' in deploy_cmd_args_dict.keys():
                deploy_args = DeployArguments(
                    ckpt_dir=deploy_cmd_args_dict['--ckpt_dir'])
                model = deploy_args.model_type
                template_type = deploy_args.template_type
            else:
                return None
        else:
            if '--model_type' not in deploy_cmd_args_dict.keys():
                model = DeployArguments(
                    model_id_or_path=deploy_cmd_args_dict['--model_id_or_path']
                ).model_type
                template_type = deploy_cmd_args_dict['--template_type']
            else:
                # moved model path
                model = deploy_cmd_args_dict['--model_type']
                template_type = deploy_cmd_args_dict['--template_type']

        host = deploy_cmd_args_dict.get('--host', '127.0.0.1')
        port = deploy_cmd_args_dict.get('--port', 8000)
        max_tokens = 32

        if template_type.endswith('generation'):
            curl_cmd = f'curl http://{host}:{port}/v1/completions ' \
                       '-H ' \
                       '"Content-Type: application/json" \\ \n' \
                       '-d ' \
                       "'{" \
                       f'"model": "{model}","prompt": "{prompt}","max_tokens": {max_tokens},' \
                       '"temperature": 0.1,"seed": 42' \
                       "}'"
        else:
            curl_cmd = f'curl http://{host}:{port}/v1/chat/completions ' \
                       '-H ' \
                       '"Content-Type: application/json" \\ \n' \
                       '-d ' \
                       "'{" \
                       f'"model": "{model}","messages": [{{"role": "user", "content": "{content}"}}],' \
                       f'"max_tokens": {max_tokens},"temperature": 0' \
                       "}'"
        return curl_cmd
