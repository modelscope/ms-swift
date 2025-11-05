# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import os
import re
import sys
import time
from copy import deepcopy
from functools import partial
from subprocess import DEVNULL, PIPE, STDOUT, Popen
from typing import Dict, Type

import gradio as gr
import json
from json import JSONDecodeError
from transformers.utils import is_torch_cuda_available, is_torch_npu_available

from swift.llm import ExportArguments, RLHFArguments
from swift.llm.argument.base_args.base_args import get_supported_tuners
from swift.ui.base import BaseUI
from swift.ui.llm_train.advanced import Advanced
from swift.ui.llm_train.dataset import Dataset
from swift.ui.llm_train.hyper import Hyper
from swift.ui.llm_train.model import Model
from swift.ui.llm_train.optimizer import Optimizer
from swift.ui.llm_train.quantization import Quantization
from swift.ui.llm_train.report_to import ReportTo
from swift.ui.llm_train.runtime import Runtime
from swift.ui.llm_train.save import Save
from swift.ui.llm_train.self_cog import SelfCog
from swift.ui.llm_train.task import Task
from swift.ui.llm_train.tuner import Tuner
from swift.ui.llm_train.utils import run_command_in_background_with_popen
from swift.utils import get_device_count, get_logger

logger = get_logger()


class LLMTrain(BaseUI):
    group = 'llm_train'

    sub_ui = [
        Model,
        Dataset,
        Runtime,
        Save,
        Optimizer,
        Task,
        Tuner,
        Hyper,
        Quantization,
        SelfCog,
        Advanced,
        ReportTo,
    ]

    locale_dict: Dict[str, Dict] = {
        'llm_train': {
            'label': {
                'zh': 'LLM预训练/微调',
                'en': 'LLM PT/SFT',
            }
        },
        'train_stage': {
            'label': {
                'zh': '训练Stage',
                'en': 'Train Stage'
            },
            'info': {
                'zh': '请注意选择与此匹配的数据集',
                'en': 'Please choose matched dataset'
            }
        },
        'submit_alert': {
            'value': {
                'zh':
                '任务已开始，请查看tensorboard或日志记录，请勿关闭终端，否则训练过程将被打断',
                'en':
                'Task started, please check the tensorboard or log file, '
                'do not close the terminal, otherwise the training process will be interrupted'
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
        'train_type': {
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
        'torch_dtype': {
            'label': {
                'zh': '训练精度',
                'en': 'Training Precision'
            },
            'info': {
                'zh': '选择训练精度',
                'en': 'Select the training precision'
            }
        },
        'envs': {
            'label': {
                'zh': '环境变量',
                'en': 'Extra env vars'
            },
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
        'use_liger_kernel': {
            'label': {
                'zh': '使用Liger kernel',
                'en': 'Use Liger kernel'
            },
            'info': {
                'zh': 'Liger kernel可以有效降低显存使用',
                'en': 'Liger kernel can reduce memory usage'
            }
        },
        'sequence_parallel_size': {
            'label': {
                'zh': '序列并行大小',
                'en': 'Sequence parallel size',
            },
            'info': {
                'zh': '当前支持CPT/SFT/DPO/GRPO',
                'en': 'Currently supports CPT/SFT/DPO/GRPO',
            }
        },
        'deepspeed': {
            'label': {
                'zh': 'DeepSpeed',
                'en': 'DeepSpeed',
            },
            'info': {
                'zh': '可以选择下拉列表，也支持传入路径',
                'en': 'Choose from the dropbox or fill in a valid path',
            }
        },
        'resume_checkpoint_alert': {
            'value': {
                'zh': '检测到"args.json"在{}中，将从此检查点开始断点续训',
                'en': 'Detected that "args.json" is in {}, will start breakpoint resume training from this checkpoint'
            }
        },
        'resume_only_model_alert': {
            'value': {
                'zh':
                '检测到"args.json"在{}中，但未检测到优化器参数，将仅加载模型参数开始断点续训',
                'en':
                '"args.json" is detected in {}, but optimizer parameters are not detected. '
                'Only model parameters will be loaded to start breakpoint continuation training'
            }
        },
        'more_params': {
            'label': {
                'zh': '其他高级参数',
                'en': 'Other params'
            },
            'info': {
                'zh': '以json格式或--xxx xxx命令行格式填入',
                'en': 'Fill in with json format or --xxx xxx cmd format'
            }
        },
        'extra_params': {
            'label': {
                'zh': '其他参数设置',
                'en': 'Extra settings'
            },
        },
        'train_param': {
            'label': {
                'zh': '训练参数设置',
                'en': 'Train settings'
            },
        },
    }

    choice_dict = BaseUI.get_choices_from_dataclass(RLHFArguments)
    default_dict = BaseUI.get_default_value_from_dataclass(RLHFArguments)
    arguments = BaseUI.get_argument_names(RLHFArguments)

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_train', label=''):
            default_device = 'cpu'
            device_count = get_device_count()
            if device_count > 0:
                default_device = '0'
            with gr.Blocks():
                Model.build_ui(base_tab)
                Dataset.build_ui(base_tab)
                with gr.Accordion(elem_id='train_param', open=True):
                    with gr.Row():
                        gr.Dropdown(elem_id='train_stage', choices=['pt', 'sft'], value='sft', scale=4)
                        gr.Dropdown(elem_id='train_type', scale=4, choices=list(get_supported_tuners()))
                        gr.Textbox(elem_id='seed', scale=4)
                        gr.Dropdown(elem_id='torch_dtype', scale=4)
                        gr.Checkbox(elem_id='use_liger_kernel', scale=4)
                    with gr.Row():
                        gr.Dropdown(
                            elem_id='gpu_id',
                            multiselect=True,
                            choices=[str(i) for i in range(device_count)] + ['cpu'],
                            value=default_device,
                            scale=4)
                        gr.Checkbox(elem_id='use_ddp', value=False, scale=4)
                        gr.Textbox(elem_id='ddp_num', value='1', scale=4)
                        gr.Dropdown(
                            elem_id='deepspeed',
                            scale=4,
                            allow_custom_value=True,
                            value=None,
                            choices=['zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'])
                        gr.Textbox(elem_id='sequence_parallel_size', lines=1, scale=4)
                Hyper.build_ui(base_tab)
                Runtime.build_ui(base_tab)
                with gr.Row(equal_height=True):
                    gr.Textbox(elem_id='envs', scale=12)
                    gr.Checkbox(elem_id='dry_run', value=False, scale=4)
                    submit = gr.Button(elem_id='submit', scale=4, variant='primary')

                Tuner.build_ui(base_tab)
                Optimizer.build_ui(base_tab)
                Task.build_ui(base_tab)
                with gr.Accordion(elem_id='extra_params', open=False):
                    with gr.Tabs():
                        Advanced.build_ui(base_tab)
                        Quantization.build_ui(base_tab)
                        SelfCog.build_ui(base_tab)
                        Save.build_ui(base_tab)
                        ReportTo.build_ui(base_tab)
                    with gr.Row():
                        gr.Textbox(elem_id='more_params', lines=4, scale=20)

                cls.element('train_type').change(
                    Hyper.update_lr, inputs=[base_tab.element('train_type')], outputs=[cls.element('learning_rate')])

                submit.click(cls.train_local, list(cls.valid_elements().values()), [
                    cls.element('running_cmd'),
                    cls.element('logging_dir'),
                    cls.element('runtime_tab'),
                    cls.element('running_tasks'),
                    cls.element('train_record'),
                ])

                base_tab.element('gpu_id').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))
                base_tab.element('use_ddp').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))
                base_tab.element('running_tasks').change(
                    partial(Runtime.task_changed, base_tab=base_tab), [base_tab.element('running_tasks')],
                    list(base_tab.valid_elements().values()) + [cls.element('log')] + Runtime.all_plots)
                Runtime.element('kill_task').click(
                    Runtime.kill_task,
                    [Runtime.element('running_tasks')],
                    [Runtime.element('running_tasks')] + [Runtime.element('log')] + Runtime.all_plots,
                ).then(Runtime.reset, [], [Runtime.element('logging_dir')] + [Hyper.element('output_dir')])

    @classmethod
    def update_runtime(cls):
        return gr.update(open=True), gr.update(visible=True)

    @classmethod
    def train(cls, *args):
        ignore_elements = ('logging_dir', 'more_params', 'train_stage', 'envs')
        default_args = cls.get_default_value_from_dataclass(RLHFArguments)
        extra_default_args = cls.get_default_value_from_dataclass(ExportArguments)
        kwargs = {}
        kwargs_is_list = {}
        other_kwargs = {}
        more_params = {}
        more_params_cmd = ''
        keys = cls.valid_element_keys()
        if cls.group in ('llm_grpo', 'llm_rlhf'):
            train_stage = 'rlhf'
        else:
            train_stage = 'sft'
        for key, value in zip(keys, args):
            compare_value = default_args.get(key) if key != 'hub_private_repo' else extra_default_args.get(key)
            if isinstance(value, str) and re.fullmatch(cls.int_regex, value):
                value = int(value)
            elif isinstance(value, str) and re.fullmatch(cls.float_regex, value):
                value = float(value)
            elif isinstance(value, str) and re.fullmatch(cls.bool_regex, value):
                value = True if value.lower() == 'true' else False
                if compare_value in ('true', 'false'):
                    value = str(value).lower()
            if key not in ignore_elements and key in default_args and compare_value != value and (value or value
                                                                                                  in (0, False)):
                kwargs[key] = value if not isinstance(value, list) else ' '.join(value)
                kwargs_is_list[key] = isinstance(value, list) or getattr(cls.element(key), 'is_list', False)
            else:
                other_kwargs[key] = value
            if key == 'more_params' and value:
                try:
                    more_params = json.loads(value)
                except (JSONDecodeError or TypeError):
                    more_params_cmd = value

            if key == 'train_stage':
                train_stage = value

        model = kwargs.get('model')
        if '-merged' not in model and os.path.exists(model) and os.path.exists(os.path.join(model, 'args.json')):
            ckpt_dir = kwargs.pop('model')
            with open(os.path.join(ckpt_dir, 'args.json'), 'r', encoding='utf-8') as f:
                _json = json.load(f)
                kwargs['model'] = _json['model_dir']
                kwargs['model_type'] = _json['model_type']
                kwargs['template'] = _json['template']
                if os.path.exists(os.path.join(ckpt_dir, 'scheduler.pt')):
                    kwargs['resume_from_checkpoint'] = ckpt_dir
                    gr.Info(cls.locale('resume_checkpoint_alert', cls.lang)['value'].format(ckpt_dir))
                else:
                    kwargs['resume_from_checkpoint'] = ckpt_dir
                    kwargs['resume_only_model'] = True
                    gr.Info(cls.locale('resume_only_model_alert', cls.lang)['value'].format(ckpt_dir))

        model = kwargs.get('model')
        kwargs.update(more_params)
        if 'dataset' not in kwargs and 'custom_train_dataset_path' not in kwargs:
            raise gr.Error(cls.locale('dataset_alert', cls.lang)['value'])

        cmd = train_stage
        if kwargs.get('deepspeed'):
            more_params_cmd += f' --deepspeed {kwargs.pop("deepspeed")} '
        use_liger_kernel = kwargs.get('use_liger_kernel', None)
        if use_liger_kernel:
            kwargs.pop('use_liger_kernel')
        if other_kwargs.get('use_muon'):
            kwargs['use_muon'] = other_kwargs.pop('use_muon')

        # filter kwargs
        tabs_relation_dict = cls.prepare_sub_to_filter()
        cls.remove_useless_args(kwargs, tabs_relation_dict)
        use_muon = kwargs.pop('use_muon', None)
        if cls.group == 'llm_rlhf':
            cls.filter_rlhf_args(kwargs)
        try:
            sft_args = RLHFArguments(
                **{
                    key: value.split(' ') if kwargs_is_list.get(key, False) and isinstance(value, str) else value
                    for key, value in kwargs.items()
                })
        except Exception as e:
            raise e
        params = ''
        command = ['swift', cmd]
        if cls.group == 'llm_grpo' and sys.platform != 'win32':
            params += f'--rlhf_type {cls.quote}grpo{cls.quote} '
            command.extend(['--rlhf_type', 'grpo'])

        sep = f'{cls.quote} {cls.quote}'
        for e in kwargs:
            if isinstance(kwargs[e], list):
                params += f'--{e} {cls.quote}{sep.join(kwargs[e])}{cls.quote} '
                command.extend([f'--{e}'] + kwargs[e])
            elif e in kwargs_is_list and kwargs_is_list[e]:
                all_args = [arg for arg in kwargs[e].split(' ') if arg.strip()]
                params += f'--{e} {cls.quote}{sep.join(all_args)}{cls.quote} '
                command.extend([f'--{e}'] + all_args)
            else:
                params += f'--{e} {cls.quote}{kwargs[e]}{cls.quote} '
                command.extend([f'--{e}', f'{kwargs[e]}'])
        if use_liger_kernel:
            params += f'--use_liger_kernel {cls.quote}{use_liger_kernel}{cls.quote} '
            command.extend(['--use_liger_kernel', f'{use_liger_kernel}'])
        if use_muon:
            params += f'--optimizer {cls.quote}muon{cls.quote} '
            command.extend(['--optimizer', 'muon'])
        more_params_cmd = more_params_cmd.strip()
        if more_params_cmd != '':
            params += f'{more_params_cmd} '
            more_params_cmd = [param.strip() for param in more_params_cmd.split('--')]
            more_params_cmd = [param.split(' ') for param in more_params_cmd if param]
            for param in more_params_cmd:
                command.extend([f'--{param[0]}'] + param[1:])
        params += f'--add_version False --output_dir {sft_args.output_dir} ' \
                  f'--logging_dir {sft_args.logging_dir} --ignore_args_error True'
        command.extend([
            '--add_version', 'False', '--output_dir', f'{sft_args.output_dir}', '--logging_dir',
            f'{sft_args.logging_dir}', '--ignore_args_error', 'True'
        ])
        all_envs = {}
        ddp_param = ''
        devices = other_kwargs['gpu_id']
        envs = other_kwargs['envs'] or ''
        envs = envs.strip()
        devices = [d for d in devices if d]
        if other_kwargs['use_ddp']:
            assert int(other_kwargs['ddp_num']) > 0
            ddp_param = f'NPROC_PER_NODE={int(other_kwargs["ddp_num"])}'
            all_envs['NPROC_PER_NODE'] = str(other_kwargs['ddp_num'])
        assert (len(devices) == 1 or 'cpu' not in devices)
        gpus = ','.join(devices)
        cuda_param = ''
        if gpus != 'cpu':
            if is_torch_npu_available():
                cuda_param = f'ASCEND_RT_VISIBLE_DEVICES={gpus}'
                all_envs['ASCEND_RT_VISIBLE_DEVICES'] = gpus
            elif is_torch_cuda_available():
                cuda_param = f'CUDA_VISIBLE_DEVICES={gpus}'
                all_envs['CUDA_VISIBLE_DEVICES'] = gpus
            else:
                cuda_param = ''
        if envs:
            envs = envs.split(' ')
            for env in envs:
                k, v = env.split('=')
                all_envs[k] = v
        log_file = os.path.join(sft_args.logging_dir, 'run.log')
        if sys.platform == 'win32':
            if cuda_param:
                cuda_param = f'set {cuda_param} && '
            if ddp_param:
                ddp_param = f'set {ddp_param} && '
            if envs:
                envs = [env.strip() for env in envs.split(' ') if env.strip()]
                _envs = ''
                for env in envs:
                    _envs += f'set {env} && '
                envs = _envs
            run_command = f'{cuda_param}{ddp_param}{envs}start /b swift sft {params} > {log_file} 2>&1'
        else:
            run_command = f'{cuda_param} {ddp_param} {envs} nohup swift {cmd} {params} > {log_file} 2>&1 &'
        logger.info(f'Run training: {run_command}')
        if model:
            record = {}
            for key, value in zip(keys, args):
                if key in default_args or key in ('more_params', 'train_stage', 'use_ddp', 'ddp_num', 'gpu_id', 'envs'):
                    record[key] = value or None
            cls.save_cache(model, record)
        return command, all_envs, log_file, run_command, sft_args, other_kwargs

    @classmethod
    def train_studio(cls, *args):
        command, all_envs, log_file, run_command, sft_args, other_kwargs = cls.train(*args)
        if not other_kwargs['dry_run']:
            lines = collections.deque(maxlen=int(os.environ.get('MAX_LOG_LINES', 50)))
            env = deepcopy(os.environ)
            if len(all_envs) > 0:
                for k, v in all_envs.items():
                    env[k] = v
            process = Popen(command, env=env, stdout=PIPE, stderr=STDOUT)
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
        command, all_envs, log_file, run_command, sft_args, other_kwargs = cls.train(*args)
        if cls.group == 'llm_grpo' and sft_args.vllm_mode == 'server':
            host = sft_args.vllm_server_host if sft_args.vllm_server_host else '127.0.0.1'
            port = sft_args.vllm_server_port if sft_args.vllm_server_port else '8000'
            try:
                import requests
                headers = {'Accept': 'application/json'}
                url = f'http://{host}:{port}/health/'
                response = requests.get(url, headers=headers)
                res = response.json()
                assert res['status'] == 'ok', 'statue must be ok'
            except Exception as err:
                gr.Info(cls.locale('external_alert', cls.lang)['value'].format(err))
                return [None] * 2 + [gr.update(open=False)] + [None] * 2
        if not other_kwargs['dry_run']:
            os.makedirs(sft_args.logging_dir, exist_ok=True)
            run_command_in_background_with_popen(command, all_envs, log_file)
            time.sleep(1)  # to make sure the log file has been created.
            gr.Info(cls.locale('submit_alert', cls.lang)['value'])
        return run_command, sft_args.logging_dir, gr.update(open=True), Runtime.refresh_tasks(
            sft_args.output_dir, cls.group), gr.update(choices=cls.list_cache(sft_args.model))

    @classmethod
    def prepare_sub_to_filter(cls):
        tabs_relation_dict = {
            key: val
            for key, val in zip(['train_type', 'optimizer', 'task_type'],
                                [Tuner.tabs_to_filter, Optimizer.tabs_to_filter, Task.tabs_to_filter])
        }
        return tabs_relation_dict

    @classmethod
    def remove_useless_args(cls, uncleaned_kwargs, tabs_relation_dict):
        for target, tabs_to_filter in tabs_relation_dict.items():
            target_value = uncleaned_kwargs.get(target)
            if target == 'train_type' and target_value is None:
                target_value = 'lora'
            elif target == 'vllm_mode' and target_value is None:
                target_value = 'colocate'
            elif target == 'optimizer':
                if uncleaned_kwargs.get('use_galore'):
                    target_value = 'galore'
                if uncleaned_kwargs.get('lorap_lr_ratio'):
                    target_value = 'lorap'
                if uncleaned_kwargs.get('vit_lr') or uncleaned_kwargs.get('aligner_lr'):
                    target_value = 'multimodal'
                if uncleaned_kwargs.get('use_muon'):
                    target_value = 'muon'

            for tab_key in tabs_to_filter.keys():
                if tab_key == 'lora' and target_value in ('longlora', 'adalora'):
                    continue
                if tab_key == 'lisa' and target_value == 'full' and uncleaned_kwargs.get('lisa_activated_layers'):
                    continue
                if tab_key == 'lora_ga' and target_value == 'lora' and uncleaned_kwargs.get(
                        'init_weights') == 'lora-ga':
                    continue
                if tab_key != target_value:
                    for arg in tabs_to_filter[tab_key]:
                        if uncleaned_kwargs.get(arg) is not None:
                            uncleaned_kwargs.pop(arg)
