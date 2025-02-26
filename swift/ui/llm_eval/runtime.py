# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr
from packaging import version

from swift.ui.base import BaseUI
from swift.ui.llm_infer.runtime import Runtime
from swift.utils import get_logger

logger = get_logger()


class EvalRuntime(Runtime):

    group = 'llm_eval'

    cmd = 'eval'

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
                'zh': '展示评测状态',
                'en': 'Show eval status'
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
                'zh': '运行中评测',
                'en': 'Running evaluation'
            },
            'info': {
                'zh': '所有的swift eval命令启动的任务',
                'en': 'All tasks started by swift eval'
            }
        },
        'refresh_tasks': {
            'value': {
                'zh': '找回评测',
                'en': 'Find evaluation'
            },
        },
        'kill_task': {
            'value': {
                'zh': '杀死评测',
                'en': 'Kill evaluation'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='runtime_tab', open=False, visible=True):
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(elem_id='running_tasks', scale=10)
                    gr.Button(elem_id='refresh_tasks', scale=1, variant='primary')
                    gr.Button(elem_id='show_log', scale=1, variant='primary')
                    gr.Button(elem_id='stop_show_log', scale=1)
                    gr.Button(elem_id='kill_task', scale=1, size='lg')
                with gr.Row():
                    gr.Textbox(elem_id='log', lines=6, visible=False)

                concurrency_limit = {}
                if version.parse(gr.__version__) >= version.parse('4.0.0'):
                    concurrency_limit = {'concurrency_limit': 5}
                cls.log_event = base_tab.element('show_log').click(cls.update_log, [], [cls.element('log')]).then(
                    cls.wait, [base_tab.element('running_tasks')], [cls.element('log')], **concurrency_limit)

                base_tab.element('stop_show_log').click(cls.break_log_event, [cls.element('running_tasks')], [])

                base_tab.element('refresh_tasks').click(
                    cls.refresh_tasks,
                    [base_tab.element('running_tasks')],
                    [base_tab.element('running_tasks')],
                )
