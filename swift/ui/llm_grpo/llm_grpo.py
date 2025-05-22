# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Dict, Type

import gradio as gr

from swift.llm import RLHFArguments
from swift.llm.argument.base_args.base_args import get_supported_tuners
from swift.ui.base import BaseUI
from swift.ui.llm_grpo.grpo_advanced import GRPOAdvanced
from swift.ui.llm_grpo.model import Model
from swift.ui.llm_grpo.ref_model import RefModel
from swift.ui.llm_grpo.reward import Reward
from swift.ui.llm_grpo.rollout import Rollout
from swift.ui.llm_train.advanced import Advanced
from swift.ui.llm_train.dataset import Dataset
from swift.ui.llm_train.hyper import Hyper
from swift.ui.llm_train.llm_train import LLMTrain
from swift.ui.llm_train.lora import LoRA
from swift.ui.llm_train.quantization import Quantization
from swift.ui.llm_train.report_to import ReportTo
from swift.ui.llm_train.runtime import Runtime
from swift.ui.llm_train.save import Save
from swift.utils import get_device_count, get_logger

logger = get_logger()


class LLMGRPO(LLMTrain):
    group = 'llm_grpo'

    sub_ui = [
        Model,
        Dataset,
        Reward,
        Runtime,
        Rollout,
        Save,
        LoRA,
        Hyper,
        Quantization,
        Advanced,
        RefModel,
        GRPOAdvanced,
        ReportTo,
    ]

    locale_dict: Dict[str, Dict] = {
        'llm_grpo': {
            'label': {
                'zh': 'LLM GRPO',
                'en': 'LLM GRPO',
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
        'tuner_backend': {
            'label': {
                'zh': 'Tuner backend',
                'en': 'Tuner backend'
            },
            'info': {
                'zh': 'tuner实现框架',
                'en': 'The tuner backend'
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
        'train_param': {
            'label': {
                'zh': '训练参数设置',
                'en': 'Train settings'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_grpo', label=''):
            default_device = 'cpu'
            device_count = get_device_count()
            if device_count > 0:
                default_device = '0'
            with gr.Blocks():
                Model.build_ui(base_tab)
                Dataset.build_ui(base_tab)
                Reward.build_ui(base_tab)
                with gr.Accordion(elem_id='train_param', open=True):
                    with gr.Row():
                        gr.Dropdown(elem_id='train_type', scale=4, choices=list(get_supported_tuners()))
                        gr.Dropdown(elem_id='tuner_backend', scale=4)
                        gr.Textbox(elem_id='seed', scale=4)
                        gr.Dropdown(elem_id='torch_dtype', scale=4)
                    with gr.Row():
                        gr.Checkbox(elem_id='use_liger_kernel', scale=4)
                        gr.Checkbox(elem_id='use_ddp', value=False, scale=4)
                        gr.Textbox(elem_id='ddp_num', value='2', scale=4)
                Hyper.build_ui(base_tab)
                Runtime.build_ui(base_tab)
                with gr.Row():
                    gr.Dropdown(
                        elem_id='gpu_id',
                        multiselect=True,
                        choices=[str(i) for i in range(device_count)] + ['cpu'],
                        value=default_device,
                        scale=8)
                    gr.Textbox(elem_id='envs', scale=8)
                    gr.Checkbox(elem_id='dry_run', value=False, scale=4)
                    submit = gr.Button(elem_id='submit', scale=4, variant='primary')

                Rollout.build_ui(base_tab)
                LoRA.build_ui(base_tab)
                RefModel.build_ui(base_tab)
                Quantization.build_ui(base_tab)
                Save.build_ui(base_tab)
                ReportTo.build_ui(base_tab)
                GRPOAdvanced.build_ui(base_tab)
                Advanced.build_ui(base_tab)

                cls.element('train_type').change(
                    Hyper.update_lr, inputs=[base_tab.element('train_type')], outputs=[cls.element('learning_rate')])

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
                    partial(Runtime.task_changed, base_tab=base_tab), [base_tab.element('running_tasks')],
                    list(base_tab.valid_elements().values()) + [cls.element('log')] + Runtime.all_plots)
                Runtime.element('kill_task').click(
                    Runtime.kill_task,
                    [Runtime.element('running_tasks')],
                    [Runtime.element('running_tasks')] + [Runtime.element('log')] + Runtime.all_plots,
                ).then(Runtime.reset, [], [Runtime.element('logging_dir')] + [Hyper.element('output_dir')])
