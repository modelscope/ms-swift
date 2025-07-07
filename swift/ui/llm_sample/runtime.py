# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.ui.llm_infer.runtime import Runtime
from swift.utils import get_logger

logger = get_logger()


class SampleRuntime(Runtime):

    group = 'llm_sample'

    cmd = 'sample'

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
                'zh': '展示采样状态',
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
                'zh': '如果日志无更新请再次点击"展示采样状态"',
                'en': 'Please press "Show running status" if the log content is not updating'
            }
        },
        'running_tasks': {
            'label': {
                'zh': '运行中采样',
                'en': 'Running sampling'
            },
            'info': {
                'zh': '所有的swift sample命令启动的任务',
                'en': 'Started by swift sample'
            }
        },
        'refresh_tasks': {
            'value': {
                'zh': '找回采样',
                'en': 'Find sampling'
            },
        },
        'kill_task': {
            'value': {
                'zh': '杀死采样',
                'en': 'Kill running task'
            },
        },
    }
