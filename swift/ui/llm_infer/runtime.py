import os.path
import time
from datetime import datetime
from typing import Dict, List, Tuple, Type

import gradio as gr
import psutil

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
                    [base_tab.element('running_tasks')],
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
               f'/running:{format_time(ts-create_time)}/cmd:{" ".join(proc.cmdline())}'

    @staticmethod
    def kill_task(task):
        pid = task.split('/')[0].split(':')[1]
        print(f'kill -9 {pid}')
        os.system(f'kill -9 {pid}')
        time.sleep(1)
        return [Runtime.refresh_tasks()]

    @staticmethod
    def show_curl(selected_task):
        if 'swift deploy' not in selected_task:
            return ''
        host = '127.0.0.1'
        port = 8000
        deploy_args = selected_task.split('swift deploy')[1].strip().split(' ')
        key_list, value_list = deploy_args[0::2], deploy_args[1::2]
        for k, v in zip(key_list, value_list):
            if k == '--model_id_or_path':
                model = v.split('/')[-1].lower()
            elif k == '--host':
                host = v
            elif k == '--port':
                port = v
            elif k == '--template_type':
                template_type = v
        if template_type.endswith('generation'):
            prompt = '浙江 -> 杭州\n安徽 -> 合肥\n四川 ->'
        else:
            prompt = '浙江的省会在哪里？'
        curl_cmd = f'curl http://{host}:{port}/v1/completions ' \
                   '-H ' \
                   '"Content-Type: application/json" ' \
                   '-d ' \
                   "'{" \
                   f'"model": "{model}","prompt": "{prompt}","max_tokens": 32,"temperature": 0.1,"seed": 42' \
                   "}'"
        return curl_cmd
