import os
import time
from dataclasses import fields

import gradio as gr
import json
import torch

from swift.llm import SftArguments
from swift.ui.element import get_elements_by_group
from swift.ui.i18n import add_locale_labels, get_locale_by_group
from swift.ui.llm_train.advanced import advanced
from swift.ui.llm_train.dataset import dataset
from swift.ui.llm_train.hyper import hyper
from swift.ui.llm_train.lora import lora
from swift.ui.llm_train.model import model
from swift.ui.llm_train.runtime import runtime
from swift.ui.llm_train.save import save
from swift.ui.llm_train.self_cog import self_cognition

elements = get_elements_by_group('llm_train')
locales = get_locale_by_group('llm_train')


def llm_train():
    gpu_count = 0
    default_device = 'cpu'
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        default_device = '0'
    add_locale_labels(locale_dict, 'llm_train')
    with gr.Blocks():
        model()
        dataset()
        runtime()
        with gr.Row():
            gr.Dropdown(elem_id='sft_type', scale=4)
            gr.Textbox(elem_id='seed', scale=4)
            gr.Dropdown(elem_id='dtype', scale=4)
            gr.Checkbox(elem_id='use_ddp', value=False, scale=4)
            gr.Slider(
                elem_id='neftune_alpha',
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                scale=4)
        with gr.Row():
            gr.Dropdown(
                elem_id='gpu_id',
                multiselect=True,
                choices=[str(i) for i in range(gpu_count)] + ['cpu'],
                value=default_device,
                scale=8)
            gr.Textbox(elem_id='gpu_memory_fraction', value='1.0', scale=4)
            gr.Checkbox(elem_id='dry_run', value=False, scale=4)
            gr.Button(elem_id='submit', scale=4, variant='primary')

        save()
        lora()
        hyper()
        self_cognition()
        advanced()

        elements['submit'].click(
            train, [], [
                elements['running_cmd'], elements['logging_dir'],
                elements['runtime_tab']
            ],
            show_progress=True)


def train():
    ignore_elements = ('model_type', 'logging_dir')
    args = fields(SftArguments)
    args = {arg.name: arg.type for arg in args}
    kwargs = {}
    more_params = getattr(elements['more_params'], 'last_value', None)
    if more_params:
        more_params = json.loads(more_params)
    else:
        more_params = {}

    for e in elements:
        if e in args and getattr(elements[e], 'changed', False) and getattr(elements[e], 'last_value', None) \
                and e not in ignore_elements:
            kwargs[e] = elements[e].last_value
    kwargs.update(more_params)
    sft_args = SftArguments(**kwargs)
    params = ''
    output_dir = sft_args.logging_dir.split('runs')[0]
    elements['output_dir'].changed = True
    elements['output_dir'].last_value = output_dir

    for e in elements:
        if e in args and getattr(elements[e], 'changed', False) and getattr(
                elements[e], 'last_value', None) and e not in ignore_elements:
            if getattr(elements[e], 'is_list', False):
                params += f'--{e} {elements[e].last_value} '
            else:
                params += f'--{e} "{elements[e].last_value}" '
    params += '--add_output_dir_suffix False '
    for key, param in more_params.items():
        params += f'--{key} "{param}" '
    ddp_param = ''
    devices = getattr(elements['gpu_id'], 'last_value',
                      ' '.join(elements['gpu_id'].value)).split(' ')
    devices = [d for d in devices if d]
    if getattr(elements['use_ddp'], 'last_value', elements['use_ddp'].value):
        ddp_param = f'NPROC_PER_NODE={len(devices)}'
    assert (len(devices) == 1 or 'cpu' not in devices)
    gpus = ','.join(devices)
    cuda_param = ''
    if gpus != 'cpu':
        cuda_param = f'CUDA_VISIBLE_DEVICES={gpus}'

    log_file = os.path.join(sft_args.logging_dir, 'run.log')
    run_command = f'{cuda_param} {ddp_param} nohup swift sft {params} > {log_file} 2>&1 &'
    if not getattr(elements['dry_run'], 'last_value', False):
        os.makedirs(sft_args.logging_dir, exist_ok=True)
        os.system(run_command)
        time.sleep(1)  # to make sure the log file has been created.
        gr.Info(locales['submit_alert']['value'])
    return run_command, sft_args.logging_dir, gr.update(visible=True)


locale_dict = {
    'submit_alert': {
        'value': {
            'zh':
            '任务已开始，请查看tensorboard或日志记录，关闭本页面不影响训练过程',
            'en':
            'Task started, please check the tensorboard or log file, '
            'closing this page does not affect training'
        }
    },
    'submit': {
        'value': {
            'zh': '🚀 开始训练',
            'en': '🚀 Begin'
        }
    },
    'dry_run': {
        'label': {
            'zh': '仅生成运行命令',
            'en': 'Dry-run'
        },
        'info': {
            'zh': '仅生成运行命令，开发者自行运行',
            'en': 'Generate run command only, for manually running'
        }
    },
    'gpu_id': {
        'label': {
            'zh': '选择可用GPU',
            'en': 'Choose GPU'
        },
        'info': {
            'zh': '选择训练使用的GPU号，如CUDA不可用只能选择CPU',
            'en': 'Select GPU to train'
        }
    },
    'gpu_memory_fraction': {
        'label': {
            'zh': 'GPU显存限制',
            'en': 'GPU memory fraction'
        },
        'info': {
            'zh':
            '设置使用显存的比例，一般用于显存测试',
            'en':
            'Set the memory fraction ratio of GPU, usually used in memory test'
        }
    },
    'sft_type': {
        'label': {
            'zh': '训练方式',
            'en': 'Train type'
        },
        'info': {
            'zh': '选择训练的方式',
            'en': 'Select the training type'
        }
    },
    'seed': {
        'label': {
            'zh': '随机数种子',
            'en': 'Seed'
        },
        'info': {
            'zh': '选择随机数种子',
            'en': 'Select a random seed'
        }
    },
    'dtype': {
        'label': {
            'zh': '训练精度',
            'en': 'Training Precision'
        },
        'info': {
            'zh': '选择训练精度',
            'en': 'Select the training precision'
        }
    },
    'use_ddp': {
        'label': {
            'zh': '使用DDP',
            'en': 'Use DDP'
        },
        'info': {
            'zh': '是否使用数据并行训练',
            'en': 'Use Distributed Data Parallel to train'
        }
    },
    'neftune_alpha': {
        'label': {
            'zh': 'neftune_alpha',
            'en': 'neftune_alpha'
        },
        'info': {
            'zh': '使用neftune提升训练效果',
            'en': 'Use neftune to improve performance'
        }
    }
}
