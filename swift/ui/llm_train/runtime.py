import os.path
import webbrowser

import gradio as gr
from transformers import is_tensorboard_available

from swift.ui.element import get_elements_by_group
from swift.ui.i18n import get_locale_by_group, add_locale_labels
from swift.ui.llm_train.utils import close_loop, run_command_in_subprocess

_handlers = {}

elements = get_elements_by_group('llm_train')

locales = get_locale_by_group('llm_train')


def runtime():
    add_locale_labels(locale_dict, 'llm_train')
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

            elements['show_log'].click(
                show_log,
                [elements['logging_dir']],
                [],
            )

            elements['start_tb'].click(
                start_tb,
                [elements['logging_dir']],
                [elements['tb_url']],
            )

            elements['close_tb'].click(
                close_tb,
                [elements['logging_dir']],
                [],
            )


def show_log(logging_dir):
    webbrowser.open('file://' + os.path.join(logging_dir, 'run.log'), new=2)


def start_tb(logging_dir):
    if not is_tensorboard_available():
        gr.Error(locales['tb_not_found'])
        return ''

    if logging_dir in _handlers:
        return _handlers[logging_dir][1]

    handler, lines = run_command_in_subprocess(
        'tensorboard', '--logdir', logging_dir, timeout=2)
    localhost_addr = ''
    for line in lines:
        if 'http://localhost:' in line:
            line = line[line.index('http://localhost:'):]
            localhost_addr = line[:line.index(' ')]
    _handlers[logging_dir] = (handler, localhost_addr)
    webbrowser.open(localhost_addr, new=2)
    return localhost_addr


def close_tb(logging_dir):
    if logging_dir in _handlers:
        close_loop(_handlers[logging_dir][0])
        _handlers.pop(logging_dir)


locale_dict = {
    'runtime_tab': {
        'label': {
            'zh': '运行时',
            'en': 'Runtime'
        },
    },
    'tb_not_found': {
        'text': {
            'zh': 'tensorboard未安装,使用pip install tensorboard进行安装',
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
    'logging_content': {
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
