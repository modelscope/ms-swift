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
        'lora_tab': {
            'label': {
                'zh': 'LoRA参数设置',
                'en': 'LoRA settings'
            },
        },
        'adalora_tab': {
            'label': {
                'zh': 'adalora参数设置',
                'en': 'adalora settings'
            },
        },
        'adalora_target_r': {
            'label': {
                'zh': 'adalora平均rank',
                'en': 'Average rank of adalora'
            },
        },
        'adalora_init_r': {
            'label': {
                'zh': 'adalora初始rank',
                'en': 'Initial rank of adalora'
            },
        },
        'adalora_tinit': {
            'label': {
                'zh': 'adalora初始warmup',
                'en': 'Initial warmup of adalora'
            },
        },
        'adalora_tfinal': {
            'label': {
                'zh': 'adalora的final warmup',
                'en': 'Final warmup of adalora'
            },
        },
        'adalora_deltaT': {
            'label': {
                'zh': 'adalora的step间隔',
                'en': 'Adalora step interval'
            },
        },
        'adalora_beta1': {
            'label': {
                'zh': 'adalora的EMA参数',
                'en': 'adalora EMA parameters'
            },
        },
        'adalora_beta2': {
            'label': {
                'zh': 'adalora的EMA参数',
                'en': 'adalora EMA parameters'
            },
        },
        'adalora_orth_reg_weight': {
            'label': {
                'zh': 'adalora的正则化参数',
                'en': 'Regularization parameter of adalora'
            },
        },
        'lora_ga_tab': {
            'label': {
                'zh': 'lora_ga参数设置',
                'en': 'lora_ga settings'
            },
        },
        'lora_ga_batch_size': {
            'label': {
                'zh': 'lora_ga批处理大小',
                'en': 'lora_ga batch size'
            },
        },
        'lora_ga_iters': {
            'label': {
                'zh': 'lora_ga迭代次数',
                'en': 'lora_ga iters'
            },
        },
        'lora_ga_max_length': {
            'label': {
                'zh': 'lora_ga最大输入长度',
                'en': 'lora_ga max length'
            },
        },
        'lora_ga_direction': {
            'label': {
                'zh': 'lora_ga初始方向',
                'en': 'lora_ga initial direction'
            },
        },
        'lora_ga_scale': {
            'label': {
                'zh': 'lora_ga缩放方式',
                'en': 'lora_ga scaling method'
            },
        },
        'lora_ga_stable_gamma': {
            'label': {
                'zh': 'gamma参数值',
                'en': 'gamma parameter value'
            },
            'info': {
                'zh': '当初始化时选择stable缩放时的gamma值',
                'en': 'select the gamma value for stable scaling',
            }
        },
        'longlora': {
            'label': {
                'zh': 'longlora参数设置',
                'en': 'longlora settings'
            },
        },
        'reft_tab': {
            'label': {
                'zh': 'reft参数设置',
                'en': 'reft settings'
            },
        },
        'reft_layers': {
            'label': {
                'zh': '应用ReFT的层',
                'en': 'reft layers'
            },
        },
        'reft_rank': {
            'label': {
                'zh': 'ReFT矩阵的rank',
                'en': 'Rank of the ReFT matrix'
            },
        },
        'reft_intervention_type': {
            'label': {
                'zh': 'ReFT的类型',
                'en': 'reft intervention type'
            },
        },
        'vera_tab': {
            'label': {
                'zh': 'vera参数设置',
                'en': 'vera settings'
            },
        },
        'vera_rank': {
            'label': {
                'zh': ' Vera Attention的尺寸',
                'en': 'vera rank'
            },
        },
        'vera_projection_prng_key': {
            'label': {
                'zh': '存储Vera映射矩阵',
                'en': 'store the Vera mapping matrix'
            },
        },
        'vera_dropout': {
            'label': {
                'zh': 'Vera的dropout值',
                'en': 'vera dropout'
            },
        },
        'vera_d_initial': {
            'label': {
                'zh': 'Vera的d矩阵的初始值',
                'en': 'Initial value of d matrix'
            },
        },
        'boft_tab': {
            'label': {
                'zh': 'boft参数设置',
                'en': 'boft settings'
            },
        },
        'boft_block_size': {
            'label': {
                'zh': 'BOFT块尺寸',
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
                'zh': 'boft的dropout值',
                'en': 'dropout value of Boft'
            },
        },
        'fourierft_tab': {
            'label': {
                'zh': 'fourierft参数设置',
                'en': 'fourierft settings'
            },
        },
        'fourier_n_frequency': {
            'label': {
                'zh': 'fourierft频率数量',
                'en': 'Fourierft frequency quantity'
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
                'zh': 'LLAMAPRO参数设置',
                'en': 'LLAMAPRO Settings'
            },
        },
        'llamapro_num_new_blocks': {
            'label': {
                'zh': 'LLAMAPRO插入层数',
                'en': 'LLAMAPRO new layers'
            },
        },
        'llamapro_num_groups': {
            'label': {
                'zh': 'LLAMAPRO对原模型的分组数',
                'en': 'LLAMAPRO groups of model'
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
                'en': 'LoRA activated layers'
            },
            'info': {
                'zh': 'LISA每次训练的模型层数，调整为正整数代表使用LISA',
                'en': 'Num of layers activated each time, a positive value means using lisa'
            }
        },
        'lisa_step_interval': {
            'label': {
                'zh': 'LISA切换layers间隔',
                'en': 'The interval of lisa layers switching'
            }
        },
        'target_params': {
            'label': {
                'zh': 'target模块参数',
                'en': 'Tuner modules params'
            }
        },
        'freeze_llm': {
            'label': {
                'zh': '冻结llm',
                'en': 'freeze llm'
            },
        },
        'freeze_aligner': {
            'label': {
                'zh': '冻结aligner',
                'en': 'freeze aligner'
            },
        },
        'freeze_vit': {
            'label': {
                'zh': '冻结vit',
                'en': 'freeze vit'
            },
        },
        'target_modules': {
            'label': {
                'zh': 'Tuner参数',
                'en': 'Tuner params'
            }
        },
        'target_regex': {
            'label': {
                'zh': 'Tuner参数',
                'en': 'Tuner params'
            }
        },
        'modules_to_save': {
            'label': {
                'zh': 'Tuner参数',
                'en': 'Tuner params'
            }
        },
        'init_weights': {
            'label': {
                'zh': 'lora初始化方法',
                'en': 'init lora weights'
            },
            'info': {
                'zh': 'gaussian/pissa/pissa_niter_[n]/olora/loftq/true/false',
                'en': 'gaussian/pissa/pissa_niter_[n]/olora/loftq/true/false',
            }
        },
        'tuner_params': {
            'label': {
                'zh': 'tuner参数',
                'en': 'tuner params'
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
                with gr.TabItem(elem_id='lora_tab'):
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
            Target.build_ui(base_tab)
