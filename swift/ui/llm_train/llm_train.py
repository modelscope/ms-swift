import os
import time
from dataclasses import fields
from typing import Dict, Type

import gradio as gr
import json
import torch

from swift.llm import SftArguments
from swift.ui.base import BaseUI
from swift.ui.llm_train.advanced import Advanced
from swift.ui.llm_train.dataset import Dataset
from swift.ui.llm_train.hyper import Hyper
from swift.ui.llm_train.lora import LoRA
from swift.ui.llm_train.model import Model
from swift.ui.llm_train.quantization import Quantization
from swift.ui.llm_train.runtime import Runtime
from swift.ui.llm_train.save import Save
from swift.ui.llm_train.self_cog import SelfCog
from swift.utils import get_logger

logger = get_logger()


class LLMTrain(BaseUI):

    group = 'llm_train'

    sub_ui = [
        Model,
        Dataset,
        Runtime,
        Save,
        LoRA,
        Hyper,
        Quantization,
        SelfCog,
        Advanced,
    ]

    locale_dict: Dict[str, Dict] = {
        'llm_train': {
            'label': {
                'zh': 'LLM训练',
                'en': 'LLM Training',
            }
        },
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

    choice_dict = {}
    for f in fields(SftArguments):
        if 'choices' in f.metadata:
            choice_dict[f.name] = f.metadata['choices']

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_train', label=''):
            gpu_count = 0
            default_device = 'cpu'
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                default_device = '0'
            with gr.Blocks():
                Model.build_ui(base_tab)
                Dataset.build_ui(base_tab)
                Runtime.build_ui(base_tab)
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
                    gr.Textbox(
                        elem_id='gpu_memory_fraction', value='1.0', scale=4)
                    gr.Checkbox(elem_id='dry_run', value=False, scale=4)
                    submit = gr.Button(
                        elem_id='submit', scale=4, variant='primary')

                Save.build_ui(base_tab)
                LoRA.build_ui(base_tab)
                Hyper.build_ui(base_tab)
                Quantization.build_ui(base_tab)
                SelfCog.build_ui(base_tab)
                Advanced.build_ui(base_tab)
                submit.click(
                    cls.train, [], [
                        cls.element('running_cmd'),
                        cls.element('logging_dir'),
                        cls.element('runtime_tab')
                    ],
                    show_progress=True)

    @classmethod
    def train(cls):
        ignore_elements = ('model_type', 'logging_dir')
        args = fields(SftArguments)
        args = {arg.name: arg.type for arg in args}
        kwargs = {}
        more_params = getattr(cls.element('more_params'), 'arg_value', None)
        if more_params:
            more_params = json.loads(more_params)
        else:
            more_params = {}

        elements = cls.elements()

        for e in elements:
            if e in args and getattr(elements[e], 'changed', False) and getattr(elements[e], 'arg_value', None) \
                    and e not in ignore_elements:
                kwargs[e] = elements[e].arg_value
        kwargs.update(more_params)
        sft_args = SftArguments(**kwargs)
        params = ''
        output_dir = sft_args.logging_dir.split('runs')[0]
        elements['output_dir'].changed = True
        elements['output_dir'].arg_value = output_dir

        for e in elements:
            if e in args and getattr(elements[e], 'changed',
                                     False) and getattr(
                                         elements[e], 'arg_value',
                                         None) and e not in ignore_elements:
                if getattr(elements[e], 'is_list', False):
                    params += f'--{e} {elements[e].arg_value} '
                else:
                    params += f'--{e} "{elements[e].arg_value}" '
        params += '--add_output_dir_suffix False '
        for key, param in more_params.items():
            params += f'--{key} "{param}" '
        ddp_param = ''
        devices = getattr(elements['gpu_id'], 'arg_value',
                          ' '.join(elements['gpu_id'].value)).split(' ')
        devices = [d for d in devices if d]
        if getattr(elements['use_ddp'], 'arg_value',
                   elements['use_ddp'].value):
            ddp_param = f'NPROC_PER_NODE={len(devices)}'
        assert (len(devices) == 1 or 'cpu' not in devices)
        gpus = ','.join(devices)
        cuda_param = ''
        if gpus != 'cpu':
            cuda_param = f'CUDA_VISIBLE_DEVICES={gpus}'

        log_file = os.path.join(sft_args.logging_dir, 'run.log')
        run_command = f'{cuda_param} {ddp_param} nohup swift sft {params} > {log_file} 2>&1 &'
        logger.info(f'Run training: {run_command}')
        if not getattr(elements['dry_run'], 'arg_value', False):
            os.makedirs(sft_args.logging_dir, exist_ok=True)
            os.system(run_command)
            time.sleep(1)  # to make sure the log file has been created.
            gr.Info(cls.locale('submit_alert', cls.lang)['value'])
        return run_command, sft_args.logging_dir, gr.update(visible=True)
