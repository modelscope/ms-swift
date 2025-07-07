# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import gradio as gr

from swift.ui.llm_train.runtime import Runtime
from swift.utils import get_logger

logger = get_logger()


class RLHFRuntime(Runtime):

    group = 'llm_rlhf'

    locale_dict = {
        'runtime_tab': {
            'label': {
                'zh': '运行时',
                'en': 'Runtime'
            },
        },
        'tb_not_found': {
            'value': {
                'zh': 'Tensorboard未安装,使用pip install tensorboard进行安装',
                'en': 'Tensorboard not found, install it by pip install tensorboard',
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
                'zh': '如果日志无更新请再次点击"展示运行状态"',
                'en': 'Please press "Show running status" if the log content is not updating'
            }
        },
        'running_tasks': {
            'label': {
                'zh': '运行中任务',
                'en': 'Running Tasks'
            },
            'info': {
                'zh': '运行中的任务（除`--rlhf_type grpo`之外的所有`swift rlhf`命令）',
                'en': 'All running tasks(started by `swift rlhf` except `--rlhf_type grpo`)'
            }
        },
        'show_running_cmd': {
            'value': {
                'zh': '展示运行命令',
                'en': 'Show running command line'
            },
        },
        'show_sh': {
            'label': {
                'zh': '展示sh命令行',
                'en': 'Show sh command line'
            },
        },
        'cmd_sh': {
            'label': {
                'zh': '训练命令行',
                'en': 'Training command line'
            },
            'info': {
                'zh':
                '如果训练命令行没有展示请再次点击"展示运行命令"，点击下方的"保存训练命令"可以保存sh脚本',
                'en': ('Please press "Show running command line" if the content is none, '
                       'click the "Save training command" below to save the sh script')
            }
        },
        'save_cmd_as_sh': {
            'value': {
                'zh': '保存训练命令',
                'en': 'Save training command'
            }
        },
        'save_cmd_alert': {
            'value': {
                'zh': '训练命令行将被保存在：{}',
                'en': 'The training command line will be saved in: {}'
            }
        },
        'close_cmd_show': {
            'value': {
                'zh': '关闭训练命令展示',
                'en': 'Close training command show'
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
    def save_cmd(cls, cmd):
        if len(cmd) > 0:
            cmd_sh, output_dir = cls.cmd_to_sh_format(cmd)
            os.makedirs(output_dir, exist_ok=True)
            sh_file_path = os.path.join(output_dir, 'rlhf.sh')
            gr.Info(cls.locale('save_cmd_alert', cls.lang)['value'].format(sh_file_path))
            with open(sh_file_path, 'w', encoding='utf-8') as f:
                f.write(cmd_sh)
