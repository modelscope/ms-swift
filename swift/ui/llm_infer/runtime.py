# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import os.path
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Type

import gradio as gr
import json
import psutil
from packaging import version

from swift.ui.base import BaseUI
from swift.utils import get_logger
from swift.utils.utils import format_time

logger = get_logger()


class Runtime(BaseUI):
    handlers: Dict[str, Tuple[List, Tuple]] = {}

    group = 'llm_infer'

    cmd = 'deploy'

    log_event = None

    locale_dict = {
        'runtime_tab': {
            'label': {
                'zh': '运行时',
                'en': 'Runtime'
            },
        },
        'running_cmd': {
            'label': {
                'zh': '运行命令',
                'en': 'Command line'
            },
            'info': {
                'zh': '执行的实际命令',
                'en': 'The actual command'
            }
        },
        'show_log': {
            'value': {
                'zh': '展示部署状态',
                'en': 'Show running status'
            },
        },
        'stop_show_log': {
            'value': {
                'zh': '停止展示',
                'en': 'Stop showing running status'
            },
        },
        'log': {
            'label': {
                'zh': '日志输出',
                'en': 'Logging content'
            },
            'info': {
                'zh': '如果日志无更新请再次点击"展示日志内容"',
                'en': 'Please press "Show log" if the log content is not updating'
            }
        },
        'running_tasks': {
            'label': {
                'zh': '运行中部署',
                'en': 'Running deployments'
            },
            'info': {
                'zh': '所有的swift deploy命令启动的任务',
                'en': 'Started by swift deploy'
            }
        },
        'refresh_tasks': {
            'value': {
                'zh': '找回部署',
                'en': 'Find deployments'
            },
        },
        'kill_task': {
            'value': {
                'zh': '杀死部署',
                'en': 'Kill running task'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='runtime_tab', open=False, visible=True):
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(elem_id='running_tasks', scale=10, allow_custom_value=True)
                    gr.Button(elem_id='refresh_tasks', scale=1, variant='primary')
                    gr.Button(elem_id='show_log', scale=1, variant='primary')
                    gr.Button(elem_id='stop_show_log', scale=1)
                    gr.Button(elem_id='kill_task', scale=1)
                with gr.Row():
                    gr.Textbox(elem_id='log', lines=6, visible=False)

                concurrency_limit = {}
                if version.parse(gr.__version__) >= version.parse('4.0.0'):
                    concurrency_limit = {'concurrency_limit': 5}
                cls.log_event = base_tab.element('show_log').click(cls.update_log, [], [cls.element('log')]).then(
                    cls.wait, [base_tab.element('running_tasks')], [cls.element('log')], **concurrency_limit)

                base_tab.element('stop_show_log').click(lambda: None, cancels=cls.log_event)

                base_tab.element('refresh_tasks').click(
                    cls.refresh_tasks,
                    [base_tab.element('running_tasks')],
                    [base_tab.element('running_tasks')],
                )

    @classmethod
    def update_log(cls):
        return gr.update(visible=True)

    @classmethod
    def wait(cls, task):
        if not task:
            return [None]
        _, args = cls.parse_info_from_cmdline(task)
        log_file = args['log_file']
        offset = 0
        latest_data = ''
        lines = collections.deque(maxlen=int(os.environ.get('MAX_LOG_LINES', 50)))
        try:
            with open(log_file, 'r', encoding='utf-8') as input:
                input.seek(offset)
                fail_cnt = 0
                while True:
                    try:
                        latest_data += input.read()
                    except UnicodeDecodeError:
                        continue
                    if not latest_data:
                        time.sleep(0.5)
                        fail_cnt += 1
                        if fail_cnt > 50:
                            break

                    if '\n' not in latest_data:
                        continue
                    latest_lines = latest_data.split('\n')
                    if latest_data[-1] != '\n':
                        latest_data = latest_lines[-1]
                        latest_lines = latest_lines[:-1]
                    else:
                        latest_data = ''
                    lines.extend(latest_lines)
                    yield '\n'.join(lines)
        except IOError:
            pass

    @classmethod
    def get_all_ports(cls):
        process_name = 'swift'
        cmd_name = cls.cmd
        ports = set()
        for proc in psutil.process_iter():
            try:
                cmdlines = proc.cmdline()
            except (psutil.ZombieProcess, psutil.AccessDenied, psutil.NoSuchProcess):
                cmdlines = []
            if any([process_name in cmdline for cmdline in cmdlines]) and any(  # noqa
                [cmd_name == cmdline for cmdline in cmdlines]):  # noqa
                try:
                    ports.add(int(cls.parse_info_from_cmdline(cls.construct_running_task(proc))[1].get('port', 8000)))
                except IndexError:
                    pass
        return ports

    @classmethod
    def refresh_tasks(cls, running_task=None):
        log_file = running_task if not running_task or 'pid:' not in running_task else None
        process_name = 'swift'
        negative_name = 'swift.exe'
        cmd_name = cls.cmd
        process = []
        selected = None
        for proc in psutil.process_iter():
            try:
                cmdlines = proc.cmdline()
            except (psutil.ZombieProcess, psutil.AccessDenied, psutil.NoSuchProcess):
                cmdlines = []
            if any([process_name in cmdline
                    for cmdline in cmdlines]) and not any([negative_name in cmdline
                                                           for cmdline in cmdlines]) and any(  # noqa
                                                               [cmd_name == cmdline for cmdline in cmdlines]):  # noqa
                process.append(cls.construct_running_task(proc))
                if log_file is not None and any(  # noqa
                    [log_file == cmdline for cmdline in cmdlines]):  # noqa
                    selected = cls.construct_running_task(proc)
        if not selected:
            if running_task and running_task in process:
                selected = running_task
        if not selected and process:
            selected = process[0]
        return gr.update(choices=process, value=selected)

    @staticmethod
    def construct_running_task(proc):
        pid = proc.pid
        ts = time.time()
        create_time = proc.create_time()
        create_time_formatted = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d, %H:%M')

        return f'pid:{pid}/create:{create_time_formatted}' \
               f'/running:{format_time(ts - create_time)}/cmd:{" ".join(proc.cmdline())}'

    @classmethod
    def parse_info_from_cmdline(cls, task):
        pid = None
        for i in range(3):
            slash = task.find('/')
            if i == 0:
                pid = task[:slash].split(':')[1]
            task = task[slash + 1:]
        args = task.split(f'swift {cls.cmd}')[1]
        args = [arg.strip() for arg in args.split('--') if arg.strip()]
        all_args = {}
        for i in range(len(args)):
            space = args[i].find(' ')
            splits = args[i][:space], args[i][space + 1:]
            all_args[splits[0]] = splits[1]
        return pid, all_args

    @classmethod
    def kill_task(cls, task):
        if task:
            pid, all_args = cls.parse_info_from_cmdline(task)
            log_file = all_args['log_file']
            if sys.platform == 'win32':
                os.system(f'taskkill /f /t /pid "{pid}"')
            else:
                os.system(f'pkill -9 -f {log_file}')
            time.sleep(1)
        return [cls.refresh_tasks()] + [gr.update(value=None)]

    @classmethod
    def task_changed(cls, task, base_tab):
        if task:
            _, all_args = cls.parse_info_from_cmdline(task)
        else:
            all_args = {}
        elements = list(base_tab.valid_elements().values())
        ret = []
        is_custom_path = 'ckpt_dir' in all_args
        for e in elements:
            if e.elem_id in all_args:
                if isinstance(e, gr.Dropdown) and e.multiselect:
                    arg = all_args[e.elem_id].split(' ')
                else:
                    if e.elem_id == 'model':
                        if is_custom_path:
                            arg = all_args['ckpt_dir']
                        else:
                            arg = all_args[e.elem_id]
                    else:
                        arg = all_args[e.elem_id]
                ret.append(gr.update(value=arg))
            else:
                ret.append(gr.update())
        return ret + [gr.update(value=None)]
