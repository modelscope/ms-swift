# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Type

import gradio as gr

from swift.llm import BaseArguments, ModelType
from swift.llm.model.register import get_all_models
from swift.ui.base import BaseUI


class GrpoAdvanced(BaseUI):
    group = 'llm_grpo'

    locale_dict = {
        'grpo_advanced_tab': {
            'label': {
                'zh': 'GRPO高级参数设置',
                'en': 'GRPO advanced settings'
            },
        },
        'loss_type': {
            'label': {
                'zh': '损失归一化类型',
                'en': 'Loss normalization type'
            }
        },
        'epsilon': {
            'label': {
                'zh': 'Clip系数',
                'en': 'Clip coefficient'
            }
        },
        'epsilon_high': {
            'label': {
                'zh': 'Upper clip系数',
                'en': 'Upper clip coefficient'
            }
        },
        'move_model_batches': {
            'label': {
                'zh': '模型参数移动批次数',
                'en': 'Batches of model params moving'
            },
            'info': {
                'zh':
                '在模型向vLLM等推理框架移动参数时，将模型分为多少个批次',
                'en': ('How many batches to divide the model into '
                       'when moving parameters to an inference framework such as vLLM')
            }
        },
        'multi_turn_scheduler': {
            'label': {
                'zh': '多轮调度器',
                'en': 'Multi turn Scheduler'
            },
            'info': {
                'zh': '多轮GRPO参数, 传入对应的plugin名称',
                'en': 'Multi turn of GRPO parameters, pass in the corresponding plugin name'
            }
        },
        'max_turns': {
            'label': {
                'zh': '多轮轮数上限',
                'en': 'Max num of multi turn'
            }
        },
        'dynamic_sample': {
            'label': {
                'zh': '动态采样',
                'en': 'Dynamic sampling'
            },
            'info': {
                'zh': '筛除group内奖励标准差为0的数据，额外采样新数据',
                'en': 'Filter out data with a reward standard deviation of 0 within the group and sample new data'
            }
        },
        'max_resample_times': {
            'label': {
                'zh': '最大重采样次数',
                'en': 'Max num of resampling times'
            },
            'info': {
                'zh': '动态采样设置下限制重采样次数',
                'en': 'Limit the number of resampling times when dynamic_sample is set'
            }
        },
        'overlong_filter': {
            'label': {
                'zh': '跳过超长样本',
                'en': 'Skip overlong samples'
            },
            'info': {
                'zh': '跳过超长截断的样本，不参与损失计算',
                'en': 'Skip overlong truncated samples and exclude them from loss calculation'
            }
        },
        'reward_model': {
            'label': {
                'zh': '奖励模型id或路径',
                'en': 'Reward Model id or path'
            },
            'info': {
                'zh': '实际的模型id',
                'en': 'The actual model id or model path'
            }
        },
        'reward_model_type': {
            'label': {
                'zh': '奖励模型类型',
                'en': 'Select Reward Model Type'
            },
            'info': {
                'zh': 'SWIFT已支持的模型类型',
                'en': 'Base model type supported by SWIFT'
            }
        },
        'ref_model_type': {
            'label': {
                'zh': 'Ref模型类型',
                'en': 'Ref model type'
            },
            'info': {
                'zh': 'SWIFT已支持的模型类型',
                'en': 'Model type supported by SWIFT'
            }
        },
        'ref_model': {
            'label': {
                'zh': 'Ref模型id或路径',
                'en': 'Ref model id or path'
            },
            'info': {
                'zh': '实际的模型id或路径',
                'en': 'The actual model id or path'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='grpo_advanced_tab'):
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(elem_id='loss_type', choices=['grpo', 'bnpo', 'dr_grpo'], value='grpo', scale=20)
                    gr.Textbox(elem_id='epsilon', value=0.2, lines=1, scale=20)
                    gr.Textbox(elem_id='epsilon_high', value=None, lines=1, scale=20)
                    gr.Textbox(elem_id='move_model_batches', lines=1, scale=20)
                with gr.Row():
                    gr.Textbox(elem_id='multi_turn_scheduler', lines=1, scale=20)
                    gr.Textbox(elem_id='max_turns', lines=1, scale=20)
                    gr.Checkbox(elem_id='dynamic_sample', scale=20)
                    gr.Slider(elem_id='max_resample_times', minimum=1, maximum=16, step=1, value=3, scale=20)
                    gr.Checkbox(elem_id='overlong_filter', scale=20)

            with gr.Row():
                gr.Dropdown(elem_id='reward_model', multiselect=True, choices=get_all_models(), scale=20)
                gr.Dropdown(
                    elem_id='reward_model_type',
                    multiselect=True,
                    choices=ModelType.get_model_name_list(),
                    allow_custom_value=True,
                    scale=20)
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(
                        elem_id='ref_model', scale=20, value=None, choices=get_all_models(), allow_custom_value=True)
                    gr.Dropdown(elem_id='ref_model_type', choices=ModelType.get_model_name_list(), value=None, scale=20)

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('ref_model').change(
            partial(cls.update_input_model, allow_keys=['ref_model_type'], has_record=False, is_ref_model=True),
            inputs=[cls.element('ref_model')],
            outputs=[cls.element('ref_model_type')])
        cls.element('reward_model').change(
            partial(cls.update_input_models, allow_keys=['reward_model_type'], is_reward_model=True, has_record=False),
            inputs=[cls.element('reward_model')],
            outputs=[cls.element('reward_model_type')])

    @classmethod
    def update_input_models(cls,
                            models,
                            allow_keys=None,
                            has_record=False,
                            arg_cls=BaseArguments,
                            is_reward_model=False):
        if models is None:
            return gr.update()
        rm_type_str = ''
        for model in models:
            rm_type_str = ' '.join([
                rm_type_str,
                cls.update_input_model(
                    model,
                    allow_keys=allow_keys,
                    has_record=has_record,
                    arg_cls=arg_cls,
                    is_reward_model=is_reward_model)['value']
            ])

        return gr.update(value=rm_type_str.strip())
