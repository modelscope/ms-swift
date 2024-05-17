import collections
import os
import re
import sys
import time
from functools import partial
from subprocess import PIPE, STDOUT, Popen
from typing import Dict, Type

import gradio as gr
import json
import torch
from gradio import Accordion, Tab

from swift.llm import SftArguments
from swift.ui.base import BaseUI
from swift.ui.llm_train.advanced import Advanced
from swift.ui.llm_train.dataset import Dataset
from swift.ui.llm_train.galore import Galore
from swift.ui.llm_train.hyper import Hyper
from swift.ui.llm_train.lisa import Lisa
from swift.ui.llm_train.llamapro import LlamaPro
from swift.ui.llm_train.lora import LoRA
from swift.ui.llm_train.model import Model
from swift.ui.llm_train.quantization import Quantization
from swift.ui.llm_train.runtime import Runtime
from swift.ui.llm_train.save import Save
from swift.ui.llm_train.self_cog import SelfCog
from swift.utils import get_logger

logger = get_logger()

is_spaces = True if 'SPACE_ID' in os.environ else False
if is_spaces:
    is_shared_ui = True if 'modelscope/swift' in os.environ['SPACE_ID'] else False
else:
    is_shared_ui = False


class LLMTrain(BaseUI):

    group = 'llm_train'

    is_studio = os.environ.get('MODELSCOPE_ENVIRONMENT') == 'studio'

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
        'dataset_alert': {
            'value': {
                'zh': '请选择或填入一个数据集',
                'en': 'Please input or select a dataset'
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
                'zh': '设置使用显存的比例，一般用于显存测试',
                'en': 'Set the memory fraction ratio of GPU, usually used in memory test'
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
        'ddp_num': {
            'label': {
                'zh': 'DDP分片数量',
                'en': 'Number of DDP sharding'
            },
            'info': {
                'zh': '启用多少进程的数据并行',
                'en': 'The data parallel size of DDP'
            }
        },
        'neftune_noise_alpha': {
            'label': {
                'zh': 'neftune_noise_alpha',
                'en': 'neftune_noise_alpha'
            },
            'info': {
                'zh': '使用neftune提升训练效果, 一般设置为5或者10',
                'en': 'Use neftune to improve performance, normally the value should be 5 or 10'
            }
        },
        'tuner_backend': {
            'label': {
                'zh': 'Tuner backend',
                'en': 'Tuner backend'
            },
            'info': {
                'zh': 'tuner实现框架，建议peft或者unsloth',
                'en': 'The tuner backend, suggest to use peft or unsloth'
            }
        },
        'sequence_parallel_size': {
            'label': {
                'zh': '序列并行分段',
                'en': 'Sequence parallel size'
            },
            'info': {
                'zh': 'DDP条件下的序列并行（减小显存），需要安装ms-swift[seq_parallel]',
                'en': 'Sequence parallel when ddp, need to install ms-swift[seq_parallel]'
            }
        },
    }

    choice_dict = BaseUI.get_choices_from_dataclass(SftArguments)
    default_dict = BaseUI.get_default_value_from_dataclass(SftArguments)
    arguments = BaseUI.get_argument_names(SftArguments)

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
                    gr.Dropdown(elem_id='tuner_backend', scale=4)
                    gr.Textbox(elem_id='sequence_parallel_size', scale=4)
                    gr.Textbox(elem_id='seed', scale=4)
                    gr.Dropdown(elem_id='dtype', scale=4)
                    gr.Checkbox(elem_id='use_ddp', value=False, scale=4)
                    gr.Textbox(elem_id='ddp_num', value='2', scale=4)
                    gr.Slider(elem_id='neftune_noise_alpha', minimum=0.0, maximum=20.0, step=0.5, scale=4)
                with gr.Row():
                    gr.Dropdown(
                        elem_id='gpu_id',
                        multiselect=True,
                        choices=[str(i) for i in range(gpu_count)] + ['cpu'],
                        value=default_device,
                        scale=8)
                    gr.Textbox(elem_id='gpu_memory_fraction', scale=4)
                    if is_shared_ui:
                        gr.Checkbox(elem_id='dry_run', value=True, interactive=False, scale=4)
                    else:
                        gr.Checkbox(elem_id='dry_run', value=False, scale=4)
                    submit = gr.Button(elem_id='submit', scale=4, variant='primary')

                Save.build_ui(base_tab)
                LoRA.build_ui(base_tab)
                Hyper.build_ui(base_tab)
                Galore.build_ui(base_tab)
                Lisa.build_ui(base_tab)
                LlamaPro.build_ui(base_tab)
                Quantization.build_ui(base_tab)
                SelfCog.build_ui(base_tab)
                Advanced.build_ui(base_tab)
                if cls.is_studio:
                    submit.click(
                        cls.update_runtime, [],
                        [cls.element('runtime_tab'), cls.element('log')]).then(
                            cls.train_studio,
                            [value for value in cls.elements().values() if not isinstance(value, (Tab, Accordion))],
                            [cls.element('log')] + Runtime.all_plots + [cls.element('running_cmd')],
                            queue=True)
                else:
                    submit.click(
                        cls.train_local,
                        [value for value in cls.elements().values() if not isinstance(value, (Tab, Accordion))], [
                            cls.element('running_cmd'),
                            cls.element('logging_dir'),
                            cls.element('runtime_tab'),
                            cls.element('running_tasks'),
                        ],
                        queue=True)
                if not cls.is_studio:
                    base_tab.element('running_tasks').change(
                        partial(Runtime.task_changed, base_tab=base_tab), [base_tab.element('running_tasks')],
                        [value for value in base_tab.elements().values() if not isinstance(value, (Tab, Accordion))]
                        + [cls.element('log')] + Runtime.all_plots,
                        cancels=Runtime.log_event)
                    Runtime.element('kill_task').click(
                        Runtime.kill_task,
                        [Runtime.element('running_tasks')],
                        [Runtime.element('running_tasks')] + [Runtime.element('log')] + Runtime.all_plots,
                        cancels=[Runtime.log_event],
                    ).then(Runtime.reset, [], [Runtime.element('logging_dir')] + [Save.element('output_dir')])

    @classmethod
    def update_runtime(cls):
        return gr.update(open=True), gr.update(visible=True)

    @classmethod
    def train(cls, *args):
        ignore_elements = ('model_type', 'logging_dir', 'more_params')
        sft_args = cls.get_default_value_from_dataclass(SftArguments)
        kwargs = {}
        kwargs_is_list = {}
        other_kwargs = {}
        more_params = {}
        keys = [key for key, value in cls.elements().items() if not isinstance(value, (Tab, Accordion))]
        model_type = None
        for key, value in zip(keys, args):
            compare_value = sft_args.get(key)
            compare_value_arg = str(compare_value) if not isinstance(compare_value, (list, dict)) else compare_value
            compare_value_ui = str(value) if not isinstance(value, (list, dict)) else value

            if isinstance(value, str) and re.fullmatch(cls.int_regex, value):
                value = int(value)
            elif isinstance(value, str) and re.fullmatch(cls.float_regex, value):
                value = float(value)

            if key not in ignore_elements and key in sft_args and compare_value_ui != compare_value_arg and value:
                kwargs[key] = value if not isinstance(value, list) else ' '.join(value)
                kwargs_is_list[key] = isinstance(value, list) or getattr(cls.element(key), 'is_list', False)
            else:
                other_kwargs[key] = value
            if key == 'more_params' and value:
                more_params = json.loads(value)

            if key == 'model_type':
                model_type = value

        if os.path.exists(kwargs['model_id_or_path']):
            kwargs['model_type'] = model_type

        kwargs.update(more_params)
        if 'dataset' not in kwargs and 'custom_train_dataset_path' not in kwargs:
            raise gr.Error(cls.locale('dataset_alert', cls.lang)['value'])

        sft_args = SftArguments(
            **{
                key: value.split(' ') if kwargs_is_list.get(key, False) and isinstance(value, str) else value
                for key, value in kwargs.items()
            })
        params = ''

        for e in kwargs:
            if e in kwargs_is_list and kwargs_is_list[e]:
                params += f'--{e} {kwargs[e]} '
            else:
                params += f'--{e} "{kwargs[e]}" '
        params += f'--add_output_dir_suffix False --output_dir {sft_args.output_dir} ' \
                  f'--logging_dir {sft_args.logging_dir} '
        ddp_param = ''
        devices = other_kwargs['gpu_id']
        devices = [d for d in devices if d]
        if other_kwargs['use_ddp']:
            assert int(other_kwargs['ddp_num']) > 0
            ddp_param = f'NPROC_PER_NODE={int(other_kwargs["ddp_num"])}'
        assert (len(devices) == 1 or 'cpu' not in devices)
        gpus = ','.join(devices)
        cuda_param = ''
        if gpus != 'cpu':
            cuda_param = f'CUDA_VISIBLE_DEVICES={gpus}'

        log_file = os.path.join(sft_args.logging_dir, 'run.log')
        if sys.platform == 'win32':
            if cuda_param:
                cuda_param = f'set {cuda_param} && '
            if ddp_param:
                ddp_param = f'set {ddp_param} && '
            run_command = f'{cuda_param}{ddp_param}start /b swift sft {params} > {log_file} 2>&1'
        elif cls.is_studio:
            run_command = f'{cuda_param} {ddp_param} swift sft {params}'
        else:
            run_command = f'{cuda_param} {ddp_param} nohup swift sft {params} > {log_file} 2>&1 &'
        logger.info(f'Run training: {run_command}')
        return run_command, sft_args, other_kwargs

    @classmethod
    def train_studio(cls, *args):
        run_command, sft_args, other_kwargs = cls.train(*args)
        if not other_kwargs['dry_run']:
            lines = collections.deque(maxlen=int(os.environ.get('MAX_LOG_LINES', 50)))
            process = Popen(run_command, shell=True, stdout=PIPE, stderr=STDOUT)
            with process.stdout:
                for line in iter(process.stdout.readline, b''):
                    line = line.decode('utf-8')
                    lines.append(line)
                    yield ['\n'.join(lines)] + Runtime.plot(run_command) + [run_command]
        else:
            yield [
                'Current is dryrun mode so you can only view the training cmd, please duplicate this space to '
                'do training or use with inference.'
            ] + [None] * len(Runtime.sft_plot) + [run_command]

    @classmethod
    def train_local(cls, *args):
        run_command, sft_args, other_kwargs = cls.train(*args)
        if not other_kwargs['dry_run']:
            os.makedirs(sft_args.logging_dir, exist_ok=True)
            os.system(run_command)
            time.sleep(1)  # to make sure the log file has been created.
            gr.Info(cls.locale('submit_alert', cls.lang)['value'])
        return run_command, sft_args.logging_dir, gr.update(open=True), Runtime.refresh_tasks(sft_args.output_dir)
