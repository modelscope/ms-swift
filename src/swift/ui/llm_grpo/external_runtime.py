# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple, Type

import gradio as gr
import psutil
from packaging import version

from swift.ui.base import BaseUI
from swift.ui.llm_infer.runtime import Runtime
from swift.utils import get_logger

logger = get_logger()


class RolloutRuntime(Runtime):

    group = 'llm_grpo'

    cmd = 'rollout'

    locale_dict = {
        'rollout_runtime_tab': {
            'label': {
                'zh': '运行时',
                'en': 'Runtime'
            },
        },
        'rollout_running_cmd': {
            'label': {
                'zh': '运行命令',
                'en': 'Command line'
            },
            'info': {
                'zh': '执行的实际命令',
                'en': 'The actual command'
            }
        },
        'rollout_show_log': {
            'value': {
                'zh': '展示rollout状态',
                'en': 'Show running status'
            },
        },
        'rollout_stop_show_log': {
            'value': {
                'zh': '停止展示',
                'en': 'Stop showing running status'
            },
        },
        'rollout_log': {
            'label': {
                'zh': '日志输出',
                'en': 'Logging content'
            },
            'info': {
                'zh': '如果日志无更新请再次点击"展示rollout状态"',
                'en': 'Please press "Show running status" if the log content is not updating'
            }
        },
        'rollout_running_tasks': {
            'label': {
                'zh': '运行中rollout',
                'en': 'Running rollouts'
            },
            'info': {
                'zh': '所有的swift rollout命令启动的任务',
                'en': 'Started by swift rollout'
            }
        },
        'rollout_refresh_tasks': {
            'value': {
                'zh': '找回rollout',
                'en': 'Find rollout'
            },
        },
        'rollout_kill_task': {
            'value': {
                'zh': '杀死rollout',
                'en': 'Kill running task'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='rollout_runtime_tab', open=False, visible=True):
            with gr.Blocks():
                with gr.Row(equal_height=True):
                    gr.Dropdown(elem_id='rollout_running_tasks', scale=10, allow_custom_value=True)
                with gr.Row(equal_height=True):
                    gr.Button(elem_id='rollout_refresh_tasks', scale=1, variant='primary')
                    gr.Button(elem_id='rollout_show_log', scale=1, variant='primary')
                    gr.Button(elem_id='rollout_stop_show_log', scale=1)
                    gr.Button(elem_id='rollout_kill_task', scale=1)
                with gr.Row():
                    gr.Textbox(elem_id='rollout_log', lines=6, visible=False)

                concurrency_limit = {}
                if version.parse(gr.__version__) >= version.parse('4.0.0'):
                    concurrency_limit = {'concurrency_limit': 5}
                base_tab.element('rollout_show_log').click(cls.update_log, [], [cls.element('rollout_log')]).then(
                    cls.wait, [base_tab.element('rollout_running_tasks')], [cls.element('rollout_log')],
                    **concurrency_limit)

                base_tab.element('rollout_stop_show_log').click(cls.break_log_event,
                                                                [cls.element('rollout_running_tasks')], [])

                base_tab.element('rollout_refresh_tasks').click(
                    cls.refresh_tasks,
                    [base_tab.element('rollout_running_tasks')],
                    [base_tab.element('rollout_running_tasks')],
                )

    @classmethod
    def kill_task(cls, task):
        if task:
            pid, all_args = cls.parse_info_from_cmdline(task)
            log_file = all_args['log_file']
            parent_process = psutil.Process(int(pid))
            children = parent_process.children(recursive=True)
            commands = []
            if sys.platform == 'win32':
                commands.append(['taskkill', '/f', '/t', '/pid', pid])
                for child in children:
                    commands.append(['taskkill', '/f', '/t', '/pid', f'{str(child.pid)}'])
            else:
                commands.append(['pkill', '-9', '-f', log_file])
                for child in children:
                    commands.append(['kill', '-9', f'{str(child.pid)}'])
            for cmd in commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    assert result.returncode == 0
                except Exception as e:
                    raise e
            cls.break_log_event(task)
        return [cls.refresh_tasks()] + [gr.update(value=None)]
