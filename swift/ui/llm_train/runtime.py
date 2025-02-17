# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import os.path
import sys
import time
import webbrowser
from datetime import datetime
from typing import Dict, List, Tuple, Type

import gradio as gr
import json
import matplotlib.pyplot as plt
import psutil
from packaging import version
from transformers import is_tensorboard_available

from swift.ui.base import BaseUI
from swift.ui.llm_train.utils import close_loop, run_command_in_subprocess
from swift.utils import TB_COLOR, TB_COLOR_SMOOTH, get_logger, read_tensorboard_file, tensorboard_smoothing
from swift.utils.utils import format_time

logger = get_logger()


class Runtime(BaseUI):

    handlers: Dict[str, Tuple[List, Tuple]] = {}

    group = 'llm_train'

    all_plots = None

    log_event = {}

    sft_plot = [
        {
            'name': 'train/loss',
            'smooth': 0.9,
        },
        {
            'name': 'train/acc',
            'smooth': None,
        },
        {
            'name': 'train/learning_rate',
            'smooth': None,
        },
        {
            'name': 'eval/loss',
            'smooth': 0.9,
        },
        {
            'name': 'eval/acc',
            'smooth': None,
        },
    ]

    dpo_plot = [
        {
            'name': 'train/loss',
            'smooth': 0.9,
        },
        {
            'name': 'train/rewards/accuracies',
            'smooth': None,
        },
        {
            'name': 'train/rewards/margins',
            'smooth': 0.9,
        },
        {
            'name': 'train/logps/chosen',
            'smooth': 0.9,
        },
        {
            'name': 'train/logps/rejected',
            'smooth': 0.9,
        },
    ]

    kto_plot = [
        {
            'name': 'kl',
            'smooth': None,
        },
        {
            'name': 'rewards/chosen_sum',
            'smooth': 0.9,
        },
        {
            'name': 'logps/chosen_sum',
            'smooth': 0.9,
        },
        {
            'name': 'rewards/rejected_sum',
            'smooth': 0.9,
        },
        {
            'name': 'logps/rejected_sum',
            'smooth': 0.9,
        },
    ]

    orpo_plot = [
        {
            'name': 'train/loss',
            'smooth': 0.9,
        },
        {
            'name': 'train/rewards/accuracies',
            'smooth': None,
        },
        {
            'name': 'train/rewards/margins',
            'smooth': 0.9,
        },
        {
            'name': 'train/rewards/chosen',
            'smooth': 0.9,
        },
        {
            'name': 'train/log_odds_ratio',
            'smooth': 0.9,
        },
    ]

    locale_dict = {
        'runtime_tab': {
            'label': {
                'zh': '运行时',
                'en': 'Runtime'
            },
        },
        'tb_not_found': {
            'value': {
                'zh': 'tensorboard未安装,使用pip install tensorboard进行安装',
                'en': 'tensorboard not found, install it by pip install tensorboard',
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
                'zh': '展示运行状态',
                'en': 'Show running status'
            },
        },
        'stop_show_log': {
            'value': {
                'zh': '停止展示运行状态',
                'en': 'Stop showing running status'
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
            },
            'info': {
                'zh': '如果日志无更新请再次点击"展示日志内容"',
                'en': 'Please press "Show log" if the log content is not updating'
            }
        },
        'running_tasks': {
            'label': {
                'zh': '运行中任务',
                'en': 'Running Tasks'
            },
            'info': {
                'zh': '运行中的任务（所有的swift sft命令）',
                'en': 'All running tasks(started by swift sft)'
            }
        },
        'refresh_tasks': {
            'value': {
                'zh': '找回运行时任务',
                'en': 'Find running tasks'
            },
        },
        'kill_task': {
            'value': {
                'zh': '杀死任务',
                'en': 'Kill running task'
            },
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
        with gr.Accordion(elem_id='runtime_tab', open=False, visible=True):
            with gr.Blocks():
                with gr.Row():
                    gr.Textbox(elem_id='running_cmd', lines=1, scale=20, interactive=False, max_lines=1)
                    gr.Textbox(elem_id='logging_dir', lines=1, scale=20, max_lines=1)
                    gr.Button(elem_id='show_log', scale=2, variant='primary')
                    gr.Button(elem_id='stop_show_log', scale=2)
                    gr.Textbox(elem_id='tb_url', lines=1, scale=10, interactive=False, max_lines=1)
                    gr.Button(elem_id='start_tb', scale=2, variant='primary')
                    gr.Button(elem_id='close_tb', scale=2)
                with gr.Row():
                    gr.Textbox(elem_id='log', lines=6, visible=False)
                with gr.Row():
                    gr.Dropdown(elem_id='running_tasks', scale=10)
                    gr.Button(elem_id='refresh_tasks', scale=1)
                    gr.Button(elem_id='kill_task', scale=1)

                with gr.Row():
                    cls.all_plots = []
                    for idx, k in enumerate(Runtime.sft_plot):
                        name = k['name']
                        cls.all_plots.append(gr.Plot(elem_id=str(idx), label=name))

                concurrency_limit = {}
                if version.parse(gr.__version__) >= version.parse('4.0.0'):
                    concurrency_limit = {'concurrency_limit': 5}
                base_tab.element('show_log').click(
                    Runtime.update_log, [base_tab.element('running_tasks')], [cls.element('log')] + cls.all_plots).then(
                        Runtime.wait, [base_tab.element('logging_dir'),
                                       base_tab.element('running_tasks')], [cls.element('log')] + cls.all_plots,
                        **concurrency_limit)

                base_tab.element('stop_show_log').click(cls.break_log_event, [cls.element('running_tasks')], [])

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

                base_tab.element('refresh_tasks').click(
                    Runtime.refresh_tasks,
                    [base_tab.element('running_tasks')],
                    [base_tab.element('running_tasks')],
                )

    @classmethod
    def get_plot(cls, task):
        if not task or 'swift sft' in task or 'swift pt' in task:
            return cls.sft_plot

        args: dict = cls.parse_info_from_cmdline(task)[1]
        train_type = args.get('rlhf_type', 'dpo')
        if train_type in ('dpo', 'cpo', 'simpo'):
            return cls.dpo_plot
        elif train_type == 'kto':
            return cls.kto_plot
        elif train_type == 'orpo':
            return cls.orpo_plot

    @classmethod
    def update_log(cls, task):
        ret = [gr.update(visible=True)]
        plot = Runtime.get_plot(task)
        for i in range(len(plot)):
            p = plot[i]
            ret.append(gr.update(visible=True, label=p['name']))
        return ret

    @classmethod
    def get_initial(cls, line):
        tqdm_starts = ['Train:', 'Map:', 'Val:', 'Filter:']
        for start in tqdm_starts:
            if line.startswith(start):
                return start
        return None

    @classmethod
    def wait(cls, logging_dir, task):
        if not logging_dir:
            return [None] + Runtime.plot(task)
        log_file = os.path.join(logging_dir, 'run.log')
        cls.log_event[logging_dir] = False
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

                    if cls.log_event.get(logging_dir, False):
                        cls.log_event[logging_dir] = False
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
                    start = cls.get_initial(lines[-1])
                    if start:
                        i = len(lines) - 2
                        while i >= 0:
                            if lines[i].startswith(start):
                                del lines[i]
                                i -= 1
                            else:
                                break
                    yield ['\n'.join(lines)] + Runtime.plot(task)
        except IOError:
            pass

    @classmethod
    def break_log_event(cls, task):
        if not task:
            return
        pid, all_args = Runtime.parse_info_from_cmdline(task)
        cls.log_event[all_args['logging_dir']] = True

    @classmethod
    def show_log(cls, logging_dir):
        webbrowser.open('file://' + os.path.join(logging_dir, 'run.log'), new=2)

    @classmethod
    def start_tb(cls, logging_dir):
        if not is_tensorboard_available():
            gr.Error(cls.locale('tb_not_found', cls.lang)['value'])
            return ''

        logging_dir = logging_dir.strip()
        logging_dir = logging_dir if not logging_dir.endswith(os.sep) else logging_dir[:-1]
        if logging_dir in cls.handlers:
            return cls.handlers[logging_dir][1]

        handler, lines = run_command_in_subprocess('tensorboard', '--logdir', logging_dir, timeout=2)
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

    @staticmethod
    def refresh_tasks(running_task=None):
        output_dir = running_task if not running_task or 'pid:' not in running_task else None
        process_name = 'swift'
        negative_name = 'swift.exe'
        cmd_name = ['pt', 'sft', 'rlhf']
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
                                                               [cmdline in cmd_name for cmdline in cmdlines]):  # noqa
                process.append(Runtime.construct_running_task(proc))
                if output_dir is not None and any(  # noqa
                    [output_dir == cmdline for cmdline in cmdlines]):  # noqa
                    selected = Runtime.construct_running_task(proc)
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
               f'/running:{format_time(ts-create_time)}/cmd:{" ".join(proc.cmdline())}'

    @staticmethod
    def parse_info_from_cmdline(task):
        pid = None
        if '/cmd:' in task:
            for i in range(3):
                slash = task.find('/')
                if i == 0:
                    pid = task[:slash].split(':')[1]
                task = task[slash + 1:]
        if 'swift sft' in task:
            args = task.split('swift sft')[1]
        elif 'swift pt' in task:
            args = task.split('swift pt')[1]
        elif 'swift rlhf' in task:
            args = task.split('swift rlhf')[1]
        else:
            raise ValueError(f'Cannot parse cmd line: {task}')
        args = [arg.strip() for arg in args.split('--') if arg.strip()]
        all_args = {}
        for i in range(len(args)):
            space = args[i].find(' ')
            splits = args[i][:space], args[i][space + 1:]
            all_args[splits[0]] = splits[1]

        output_dir = all_args['output_dir']
        if os.path.exists(os.path.join(output_dir, 'args.json')):
            with open(os.path.join(output_dir, 'args.json'), 'r', encoding='utf-8') as f:
                _json = json.load(f)
            for key in all_args.keys():
                all_args[key] = _json.get(key)
                if isinstance(all_args[key], list):
                    if any([' ' in value for value in all_args[key]]):
                        all_args[key] = [f'"{value}"' for value in all_args[key]]
                    all_args[key] = ' '.join(all_args[key])
        return pid, all_args

    @staticmethod
    def kill_task(task):
        if task:
            pid, all_args = Runtime.parse_info_from_cmdline(task)
            output_dir = all_args['output_dir']
            if sys.platform == 'win32':
                os.system(f'taskkill /f /t /pid "{pid}"')
            else:
                os.system(f'pkill -9 -f {output_dir}')
            time.sleep(1)
            Runtime.break_log_event(task)
        return [Runtime.refresh_tasks()] + [gr.update(value=None)] * (len(Runtime.get_plot(task)) + 1)

    @staticmethod
    def reset():
        return None, 'output'

    @staticmethod
    def task_changed(task, base_tab):
        if task:
            _, all_args = Runtime.parse_info_from_cmdline(task)
        else:
            all_args = {}
        elements = list(base_tab.valid_elements().values())
        ret = []
        for e in elements:
            if e.elem_id in all_args:
                if isinstance(e, gr.Dropdown) and e.multiselect:
                    arg = all_args[e.elem_id].split(' ')
                else:
                    arg = all_args[e.elem_id]
                ret.append(gr.update(value=arg))
            else:
                ret.append(gr.update())
        Runtime.break_log_event(task)
        return ret + [gr.update(value=None)] * (len(Runtime.get_plot(task)) + 1)

    @staticmethod
    def plot(task):
        plot = Runtime.get_plot(task)
        if not task:
            return [None] * len(plot)
        _, all_args = Runtime.parse_info_from_cmdline(task)
        tb_dir = all_args['logging_dir']
        if not os.path.exists(tb_dir):
            return [None] * len(plot)
        fname = [
            fname for fname in os.listdir(tb_dir)
            if os.path.isfile(os.path.join(tb_dir, fname)) and fname.startswith('events.out')
        ]
        if fname:
            fname = fname[0]
        else:
            return [None] * len(plot)
        tb_path = os.path.join(tb_dir, fname)
        data = read_tensorboard_file(tb_path)

        plots = []
        for k in plot:
            name = k['name']
            smooth = k['smooth']
            if name == 'train/acc':
                if 'train/token_acc' in data:
                    name = 'train/token_acc'
                if 'train/seq_acc' in data:
                    name = 'train/seq_acc'
            if name == 'eval/acc':
                if 'eval/token_acc' in data:
                    name = 'eval/token_acc'
                if 'eval/seq_acc' in data:
                    name = 'eval/seq_acc'
            if name not in data:
                plots.append(None)
                continue
            _data = data[name]
            steps = [d['step'] for d in _data]
            values = [d['value'] for d in _data]
            if len(values) == 0:
                continue

            plt.close('all')
            fig = plt.figure()
            ax = fig.add_subplot()
            # _, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 5), dpi=100)
            ax.set_title(name)
            if len(values) == 1:
                ax.scatter(steps, values, color=TB_COLOR_SMOOTH)
            elif smooth is not None:
                ax.plot(steps, values, color=TB_COLOR)
                values_s = tensorboard_smoothing(values, smooth)
                ax.plot(steps, values_s, color=TB_COLOR_SMOOTH)
            else:
                ax.plot(steps, values, color=TB_COLOR_SMOOTH)
            plots.append(fig)
        return plots
