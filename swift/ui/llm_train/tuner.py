# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI
from swift.ui.llm_train.lora import LoRA
from swift.ui.llm_train.target import Target


class Tuner(BaseUI):

    group = 'llm_train'

    sub_ui = [LoRA, Target]

    locale_dict = {
        'adalora_tab': {
            'label': {
                'zh': 'AdaLoRA参数设置',
                'en': 'AdaLoRA settings'
            },
        },
        'adalora_target_r': {
            'label': {
                'zh': 'AdaLoRA的平均秩',
                'en': 'Average rank of AdaLoRA'
            },
        },
        'adalora_init_r': {
            'label': {
                'zh': 'AdaLoRA的初始秩',
                'en': 'Initial rank of AdaLoRA'
            },
        },
        'adalora_tinit': {
            'label': {
                'zh': 'AdaLoRA初始微调预热的步数',
                'en': 'Initial fine-tuning warmup steps of AdaLoRA'
            },
        },
        'adalora_tfinal': {
            'label': {
                'zh': 'AdaLoRA最终微调的步数',
                'en': 'Final fine-tuning steps of AdaLoRA'
            },
        },
        'adalora_deltaT': {
            'label': {
                'zh': 'AdaLoRA两次预算分配间隔',
                'en': 'Internval of AdaLoRA two budget allocations'
            },
        },
        'adalora_beta1': {
            'label': {
                'zh': 'AdaLoRA的EMA参数',
                'en': 'AdaLoRA EMA parameters'
            },
        },
        'adalora_beta2': {
            'label': {
                'zh': 'AdaLoRA的EMA参数',
                'en': 'AdaLoRA EMA parameters'
            },
        },
        'adalora_orth_reg_weight': {
            'label': {
                'zh': 'AdaLoRA的正交正则化参数',
                'en': 'Coefficient of AdaLoRA orthogonal regularization'
            },
        },
        'lora_ga_tab': {
            'label': {
                'zh': 'LoRA-GA参数设置',
                'en': 'LoRA-GA settings'
            },
        },
        'lora_ga_batch_size': {
            'label': {
                'zh': 'LoRA-GA初始化批处理大小',
                'en': 'LoRA-GA initialization batch size'
            },
        },
        'lora_ga_iters': {
            'label': {
                'zh': 'LoRA-GA初始化迭代次数',
                'en': 'LoRA-GA initialization iters'
            },
        },
        'lora_ga_max_length': {
            'label': {
                'zh': 'LoRA-GA初始化最大输入长度',
                'en': 'LoRA-GA initialization max length'
            },
        },
        'lora_ga_direction': {
            'label': {
                'zh': 'LoRA-GA初始化的初始方向',
                'en': 'LoRA-GA initialization direction'
            },
        },
        'lora_ga_scale': {
            'label': {
                'zh': 'LoRA-GA初始化缩放方式',
                'en': 'LoRA-GA initialization scaling method'
            },
        },
        'lora_ga_stable_gamma': {
            'label': {
                'zh': 'Gamma参数值',
                'en': 'Gamma value'
            },
            'info': {
                'zh': '当初始化时选择stable缩放时的gamma值',
                'en': 'Select the gamma value for stable scaling',
            }
        },
        'longlora': {
            'label': {
                'zh': 'LongLoRA参数设置',
                'en': 'LongLoRA settings'
            },
        },
        'reft_tab': {
            'label': {
                'zh': 'ReFT参数设置',
                'en': 'ReFT settings'
            },
        },
        'reft_layers': {
            'label': {
                'zh': '应用ReFT的层',
                'en': 'ReFT layers'
            },
        },
        'reft_rank': {
            'label': {
                'zh': 'ReFT矩阵的秩',
                'en': 'Rank of the ReFT matrix'
            },
        },
        'reft_intervention_type': {
            'label': {
                'zh': 'ReFT的类型',
                'en': 'ReFT intervention type'
            },
        },
        'vera_tab': {
            'label': {
                'zh': 'VeRA参数设置',
                'en': 'VeRA settings'
            },
        },
        'vera_rank': {
            'label': {
                'zh': 'VeRA注意力维度',
                'en': 'VeRA rank'
            },
        },
        'vera_projection_prng_key': {
            'label': {
                'zh': 'VeRA PRNG初始化key',
                'en': 'VeRA PRNG initialisation key'
            },
        },
        'vera_dropout': {
            'label': {
                'zh': 'VeRA的丢弃概率',
                'en': 'VeRA dropout'
            },
        },
        'vera_d_initial': {
            'label': {
                'zh': 'VeRA的d矩阵初始值',
                'en': 'Initial value of d matrix'
            },
        },
        'boft_tab': {
            'label': {
                'zh': 'BOFT参数设置',
                'en': 'BOFT settings'
            },
        },
        'boft_block_size': {
            'label': {
                'zh': 'BOFT块大小',
                'en': 'BOFT block size'
            },
        },
        'boft_block_num': {
            'label': {
                'zh': 'BOFT块数量',
                'en': 'Number of BOFT blocks'
            },
            'info': {
                'zh': '不能和boft_block_size同时使用',
                'en': 'Cannot be used with boft_block_size',
            }
        },
        'boft_dropout': {
            'label': {
                'zh': 'BOFT丢弃概率',
                'en': 'Dropout value of BOFT'
            },
        },
        'fourierft_tab': {
            'label': {
                'zh': 'FourierFT参数设置',
                'en': 'FourierFT settings'
            },
        },
        'fourier_n_frequency': {
            'label': {
                'zh': 'FourierFT频率数量',
                'en': 'Num of FourierFT frequencies'
            },
        },
        'fourier_scaling': {
            'label': {
                'zh': 'W矩阵缩放值',
                'en': 'W matrix scaling value'
            },
        },
        'llamapro_tab': {
            'label': {
                'zh': 'LLaMA Pro参数设置',
                'en': 'LLaMA Pro Settings'
            },
        },
        'llamapro_num_new_blocks': {
            'label': {
                'zh': 'LLaMA Pro插入层数',
                'en': 'LLaMA Pro new layers'
            },
        },
        'llamapro_num_groups': {
            'label': {
                'zh': 'LLaMA Pro对原模型的分组数',
                'en': 'LLaMA Pro groups of model'
            }
        },
        'lisa_tab': {
            'label': {
                'zh': 'LISA参数设置',
                'en': 'LISA settings'
            },
        },
        'lisa_activated_layers': {
            'label': {
                'zh': 'LISA激活层数',
                'en': 'Num of LISA activated layers'
            },
            'info': {
                'zh': 'LISA每次训练的模型层数，调整为正整数代表使用LISA',
                'en': 'Num of layers activated each time, a positive value means using LISA'
            }
        },
        'lisa_step_interval': {
            'label': {
                'zh': 'LISA切换层间隔',
                'en': 'The interval of LISA layers switching'
            }
        },
        'tuner_params': {
            'label': {
                'zh': 'Tuner参数',
                'en': 'Tuner params'
            }
        },
    }

    tabs_to_filter = {
        'lora': ['lora_rank', 'lora_alpha', 'lora_dropout', 'lora_dtype', 'use_rslora', 'use_dora'],
        'llamapro': ['llamapro_num_new_blocks', 'llamapro_num_groups'],
        'lisa': ['lisa_activated_layers', 'lisa_step_interval'],
        'adalora': [
            'adalora_target_r', 'adalora_init_r', 'adalora_tinit', 'adalora_tfinal', 'adalora_deltaT', 'adalora_beta1',
            'adalora_beta2', 'adalora_orth_reg_weight'
        ],
        'lora_ga': [
            'lora_ga_batch_size', 'lora_ga_iters', 'lora_ga_max_length', 'lora_ga_direction', 'lora_ga_scale',
            'lora_ga_stable_gamma'
        ],
        'reft': ['reft_layers', 'reft_rank', 'reft_intervention_type'],
        'vera': ['vera_rank', 'vera_projection_prng_key', 'vera_dropout', 'vera_d_initial'],
        'boft': ['boft_block_size', 'boft_block_num', 'boft_dropout'],
        'fourierft': ['fourier_n_frequency', 'fourier_scaling']
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='tuner_params', open=False):
            with gr.Tabs():
                LoRA.set_lang(cls.lang)
                LoRA.build_ui(base_tab)
                with gr.TabItem(elem_id='llamapro_tab'):
                    with gr.Blocks():
                        with gr.Row():
                            gr.Textbox(elem_id='llamapro_num_new_blocks', scale=2)
                            gr.Textbox(elem_id='llamapro_num_groups', scale=2)
                with gr.TabItem(elem_id='lisa_tab'):
                    with gr.Blocks():
                        with gr.Row():
                            gr.Textbox(elem_id='lisa_activated_layers', value='0', scale=2)
                            gr.Textbox(elem_id='lisa_step_interval', value='20', scale=2)
                with gr.TabItem(elem_id='adalora_tab'):
                    with gr.Blocks():
                        with gr.Row():
                            gr.Textbox(elem_id='adalora_target_r', value='8', scale=2)
                            gr.Slider(elem_id='adalora_init_r', value=12, minimum=1, maximum=512, step=4, scale=2)
                            gr.Textbox(elem_id='adalora_tinit', value='0', scale=2)
                            gr.Textbox(elem_id='adalora_tfinal', value='0', scale=2)
                        with gr.Row():
                            gr.Textbox(elem_id='adalora_deltaT', value='1', scale=2)
                            gr.Textbox(elem_id='adalora_beta1', value='0.85', scale=2)
                            gr.Textbox(elem_id='adalora_beta2', value='0.85', scale=2)
                            gr.Textbox(elem_id='adalora_orth_reg_weight', value='0.5', scale=2)
                with gr.TabItem(elem_id='lora_ga_tab'):
                    with gr.Blocks():
                        with gr.Row():
                            gr.Slider(elem_id='lora_ga_batch_size', value=2, minimum=1, maximum=256, step=1, scale=20)
                            gr.Textbox(elem_id='lora_ga_iters', value='2', scale=20)
                            gr.Textbox(elem_id='lora_ga_max_length', value='2048', scale=20)
                            gr.Dropdown(
                                elem_id='lora_ga_direction',
                                scale=20,
                                value='ArB2r',
                                choices=['ArBr', 'A2rBr', 'ArB2r', 'random'])
                            gr.Dropdown(
                                elem_id='lora_ga_scale',
                                scale=20,
                                value='stable',
                                choices=['gd', 'unit', 'stable', 'weights'])
                            gr.Textbox(elem_id='lora_ga_stable_gamma', value='16', scale=20)
                with gr.TabItem(elem_id='reft_tab'):
                    with gr.Blocks():
                        with gr.Row():
                            gr.Textbox(elem_id='reft_layers', scale=2)
                            gr.Slider(elem_id='reft_rank', value=4, minimum=1, maximum=512, step=4, scale=2)
                            gr.Dropdown(
                                elem_id='reft_intervention_type',
                                scale=2,
                                value='LoreftIntervention',
                                choices=[
                                    'NoreftIntervention', 'LoreftIntervention', 'ConsreftIntervention',
                                    'LobireftIntervention', 'DireftIntervention', 'NodireftIntervention'
                                ])
                with gr.TabItem(elem_id='vera_tab'):
                    with gr.Blocks():
                        with gr.Row():
                            gr.Slider(elem_id='vera_rank', value=256, minimum=1, maximum=512, step=4, scale=2)
                            gr.Textbox(elem_id='vera_projection_prng_key', value='0', scale=2)
                            gr.Textbox(elem_id='vera_dropout', value='0.0', scale=2)
                            gr.Textbox(elem_id='vera_d_initial', value='0.1', scale=2)
                with gr.TabItem(elem_id='boft_tab'):
                    with gr.Blocks():
                        with gr.Row():
                            gr.Textbox(elem_id='boft_block_size', value='4', scale=2)
                            gr.Textbox(elem_id='boft_block_num', scale=2)
                            gr.Textbox(elem_id='boft_dropout', value='0.0', scale=2)
                with gr.TabItem(elem_id='fourierft_tab'):
                    with gr.Blocks():
                        with gr.Row():
                            gr.Textbox(elem_id='fourier_n_frequency', value='2000', scale=2)
                            gr.Textbox(elem_id='fourier_scaling', value='300.0', scale=2)
            Target.set_lang(cls.lang)
            Target.build_ui(base_tab)
