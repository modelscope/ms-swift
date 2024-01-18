import collections
import os.path
import time
import webbrowser
from typing import Dict, List, Tuple, Type

import gradio as gr
import psutil
from transformers import is_tensorboard_available

from swift.ui.base import BaseUI
from swift.ui.llm_train.utils import close_loop, run_command_in_subprocess
from swift.utils import get_logger

logger = get_logger()


class Runtime(BaseUI):

    handlers: Dict[str, Tuple[List, Tuple]] = {}

    group = 'llm_train'

    locale_dict = {
        'runtime_tab': {
            'label': {
                'zh': '运行时',
                'en': 'Runtime'
            },
        },
        'tb_not_found': {
            'value': {
                'zh':
                'tensorboard未安装,使用pip install tensorboard进行安装',
                'en':
                'tensorboard not found, install it by pip install tensorboard',
            }
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
                'zh': '展示日志内容',
                'en': 'Show log'
            },
        },
        'logging_dir': {
            'label': {
                'zh': '日志路径',
                'en': 'Logging dir'
            },
            'info': {
                'zh': '支持手动传入文件路径',
                'en': 'Support fill custom path in'
            }
        },
        'log': {
            'label': {
                'zh': '日志输出',
                'en': 'Logging content'
            }
        },
        'tb_url': {
            'label': {
                'zh': 'Tensorboard链接',
                'en': 'Tensorboard URL'
            },
            'info': {
                'zh': '仅展示，不可编辑',
                'en': 'Not editable'
            }
        },
        'start_tb': {
            'value': {
                'zh': '打开TensorBoard',
                'en': 'Start TensorBoard'
            },
        },
        'close_tb': {
            'value': {
                'zh': '关闭TensorBoard',
                'en': 'Close TensorBoard'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='runtime_tab', open=True, visible=False):
            with gr.Blocks():
                with gr.Row():
                    gr.Textbox(
                        elem_id='running_cmd',
                        lines=1,
                        scale=20,
                        interactive=False,
                        max_lines=1)
                    gr.Textbox(
                        elem_id='logging_dir', lines=1, scale=20, max_lines=1)
                    gr.Button(elem_id='show_log', scale=2, variant='primary')
                    gr.Textbox(
                        elem_id='tb_url',
                        lines=1,
                        scale=10,
                        interactive=False,
                        max_lines=1)
                    gr.Button(elem_id='start_tb', scale=2, variant='primary')
                    gr.Button(elem_id='close_tb', scale=2)
                with gr.Row():
                    gr.Textbox(elem_id='log', lines=6, visible=False)

                base_tab.element('show_log').click(
                    Runtime.update_log, [], [cls.element('log')]).then(
                        Runtime.wait, [base_tab.element('logging_dir')],
                        [cls.element('log')])

                base_tab.element('start_tb').click(
                    Runtime.start_tb,
                    [base_tab.element('logging_dir')],
                    [base_tab.element('tb_url')],
                )

                base_tab.element('close_tb').click(
                    Runtime.close_tb,
                    [base_tab.element('logging_dir')],
                    [],
                )

    @classmethod
    def update_log(cls):
        return gr.update(visible=True)

    @classmethod
    def wait(cls, logging_dir):
        log_file = os.path.join(logging_dir, 'run.log')
        offset = 0
        latest_data = ''
        lines = collections.deque(
            maxlen=int(os.environ.get('MAX_LOG_LINES', 50)))
        try:
            with open(log_file, 'r') as input:
                input.seek(offset)
                fail_cnt = 0
                while True:
                    try:
                        latest_data += input.read()
                    except UnicodeDecodeError:
                        continue
                    offset = input.tell()
                    if not latest_data:
                        time.sleep(0.5)
                        fail_cnt += 1
                        if fail_cnt > 20:
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
    def show_log(cls, logging_dir):
        webbrowser.open(
            'file://' + os.path.join(logging_dir, 'run.log'), new=2)

    @classmethod
    def start_tb(cls, logging_dir):
        if not is_tensorboard_available():
            gr.Error(cls.locale('tb_not_found', cls.lang)['value'])
            return ''

        logging_dir = logging_dir.strip()
        logging_dir = logging_dir if not logging_dir.endswith(
            os.sep) else logging_dir[:-1]
        if logging_dir in cls.handlers:
            return cls.handlers[logging_dir][1]

        handler, lines = run_command_in_subprocess(
            'tensorboard', '--logdir', logging_dir, timeout=2)
        localhost_addr = ''
        for line in lines:
            if 'http://localhost:' in line:
                line = line[line.index('http://localhost:'):]
                localhost_addr = line[:line.index(' ')]
        cls.handlers[logging_dir] = (handler, localhost_addr)
        logger.info('===========Tensorboard Log============')
        logger.info('\n'.join(lines))
        webbrowser.open(localhost_addr, new=2)
        return localhost_addr

    @staticmethod
    def close_tb(logging_dir):
        if logging_dir in Runtime.handlers:
            close_loop(Runtime.handlers[logging_dir][0])
            Runtime.handlers.pop(logging_dir)
