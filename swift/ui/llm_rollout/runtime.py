# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.ui.llm_infer.runtime import Runtime
from swift.utils import get_logger

logger = get_logger()


class RolloutRuntime(Runtime):

    group = 'llm_rollout'

    cmd = 'rollout'

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
                'zh': '展示rollout状态',
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
                'zh': '运行中rollout',
                'en': 'Running rollouts'
            },
            'info': {
                'zh': '所有的swift rollout命令启动的任务',
                'en': 'Started by swift rollout'
            }
        },
        'refresh_tasks': {
            'value': {
                'zh': '找回rollout',
                'en': 'Find rollout'
            },
        },
        'kill_task': {
            'value': {
                'zh': '杀死rollout',
                'en': 'Kill running task'
            },
        },
    }
