# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Dict, Type

import gradio as gr

from swift.llm.argument.base_args.base_args import get_supported_tuners
from swift.ui.base import BaseUI
from swift.ui.llm_rlhf.advanced import RLHFAdvanced
from swift.ui.llm_rlhf.dataset import RLHFDataset
from swift.ui.llm_rlhf.hyper import RLHFHyper
from swift.ui.llm_rlhf.model import RLHFModel
from swift.ui.llm_rlhf.optimizer import RLHFOptimizer
from swift.ui.llm_rlhf.quantization import RLHFQuantization
from swift.ui.llm_rlhf.report_to import RLHFReportTo
from swift.ui.llm_rlhf.rlhf import RLHF
from swift.ui.llm_rlhf.runtime import RLHFRuntime
from swift.ui.llm_rlhf.save import RLHFSave
from swift.ui.llm_rlhf.tuner import RLHFTuner
from swift.ui.llm_train.llm_train import LLMTrain
from swift.utils import get_device_count, get_logger

logger = get_logger()


class LLMRLHF(LLMTrain):
    group = 'llm_rlhf'

    sub_ui = [
        RLHFModel,
        RLHFDataset,
        RLHFHyper,
        RLHFRuntime,
        RLHFTuner,
        RLHFOptimizer,
        RLHF,
        RLHFQuantization,
        RLHFSave,
        RLHFReportTo,
        RLHFAdvanced,
    ]

    locale_dict: Dict[str, Dict] = {
        'llm_rlhf': {
            'label': {
                'zh': 'LLM人类对齐',
                'en': 'LLM RLHF',
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
        'rlhf_type': {
            'label': {
                'zh': '人类对齐算法类型',
                'en': 'RLHF type'
            },
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

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_rlhf', label=''):
            default_device = 'cpu'
            device_count = get_device_count()
            if device_count > 0:
                default_device = '0'
            with gr.Blocks():
                RLHFModel.build_ui(base_tab)
                RLHFDataset.build_ui(base_tab)
                with gr.Accordion(elem_id='train_param', open=True):
                    with gr.Row():
                        gr.Dropdown(elem_id='rlhf_type', scale=2)
                        gr.Dropdown(elem_id='train_type', scale=2, choices=list(get_supported_tuners()))
                        gr.Textbox(elem_id='seed', scale=2)
                        gr.Dropdown(elem_id='torch_dtype', scale=2)
                        gr.Checkbox(elem_id='use_liger_kernel', scale=2)
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
                RLHFHyper.build_ui(base_tab)
                RLHFRuntime.build_ui(base_tab)
                with gr.Row(equal_height=True):
                    gr.Textbox(elem_id='envs', scale=12)
                    gr.Checkbox(elem_id='dry_run', value=False, scale=4)
                    submit = gr.Button(elem_id='submit', scale=4, variant='primary')

                RLHFTuner.build_ui(base_tab)
                RLHFOptimizer.build_ui(base_tab)
                RLHF.build_ui(base_tab)
                with gr.Accordion(elem_id='extra_params', open=False):
                    with gr.Tabs():
                        RLHFAdvanced.build_ui(base_tab)
                        RLHFQuantization.build_ui(base_tab)
                        RLHFSave.build_ui(base_tab)
                        RLHFReportTo.build_ui(base_tab)
                    with gr.Row():
                        gr.Textbox(elem_id='more_params', lines=4, scale=20)

                base_tab.element('gpu_id').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))
                base_tab.element('use_ddp').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))
                cls.element('train_type').change(
                    RLHFHyper.update_lr,
                    inputs=[base_tab.element('train_type')],
                    outputs=[cls.element('learning_rate')])
                cls.element('rlhf_type').change(
                    RLHF.update_beta, inputs=[base_tab.element('rlhf_type')], outputs=[base_tab.element('beta')])

                submit.click(
                    cls.train_local,
                    list(cls.valid_elements().values()), [
                        cls.element('running_cmd'),
                        cls.element('logging_dir'),
                        cls.element('runtime_tab'),
                        cls.element('running_tasks'),
                        cls.element('train_record'),
                    ],
                    queue=True)

                base_tab.element('running_tasks').change(
                    partial(RLHFRuntime.task_changed, base_tab=base_tab), [base_tab.element('running_tasks')],
                    list(base_tab.valid_elements().values()) + [cls.element('log')] + RLHFRuntime.all_plots)
                RLHFRuntime.element('kill_task').click(
                    RLHFRuntime.kill_task,
                    [RLHFRuntime.element('running_tasks')],
                    [RLHFRuntime.element('running_tasks')] + [RLHFRuntime.element('log')] + RLHFRuntime.all_plots,
                ).then(RLHFRuntime.reset, [], [RLHFRuntime.element('logging_dir')] + [RLHFHyper.element('output_dir')])

    @classmethod
    def prepare_sub_to_filter(cls):
        tabs_relation_dict = {
            key: val
            for key, val in zip(['train_type', 'optimizer'], [RLHFTuner.tabs_to_filter, RLHFOptimizer.tabs_to_filter])
        }
        return tabs_relation_dict

    @classmethod
    def filter_rlhf_args(cls, uncleaned_kwargs):
        cur_rlhf_type = uncleaned_kwargs.get('rlhf_type', 'dpo')
        cur_selected = RLHF.rlhf_args_dict.pop(cur_rlhf_type, None)
        for _, vals in RLHF.rlhf_args_dict.items():
            for rlhf_arg in vals:
                if uncleaned_kwargs.get(rlhf_arg) and (cur_selected is None or rlhf_arg not in cur_selected):
                    uncleaned_kwargs.pop(rlhf_arg)
