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
        'beta': {
            'label': {
                'zh': 'KL正则项系数',
                'en': 'KL regularization coefficient'
            }
        },
        'vllm_enable_prefix_caching': {
            'label': {
                'zh': '开启前缀缓存',
                'en': 'Enable prefix cache'
            },
            'info': {
                'zh': 'Colocate模式中vLLM透传参数',
                'en': 'vLLM transparent transmission parameters in colocate mode'
            }
        },
        'log_completions': {
            'label': {
                'zh': '记录生成内容',
                'en': 'Record generated content'
            },
            'info': {
                'zh': '是否记录训练中的模型生成内容',
                'en': 'Whether to record the model generation content during training'
            }
        },
        'num_iterations': {
            'label': {
                'zh': '每个批次更新次数',
                'en': 'Num of updates per batch'
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
        'reward_model_plugin': {
            'label': {
                'zh': '奖励模型逻辑',
                'en': 'Reward model logic'
            },
            'info': {
                'zh': '利用reward_model_plugin自定义奖励模型的处理逻辑',
                'en': 'Use reward_model_plugin to customize the processing logic of the reward model'
            }
        },
        'external_plugins': {
            'label': {
                'zh': '外部插件文件',
                'en': 'External plugin file'
            },
            'info': {
                'zh': '外部插件文件列表，将被注册进插件模块中',
                'en': 'List of external plugin files that will be registered into the plugin module'
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
                    gr.Dropdown(elem_id='loss_type', choices=['grpo', 'bnpo', 'dr_grpo'], value='grpo', scale=4)
                    gr.Textbox(elem_id='epsilon', value=0.2, lines=1, scale=4)
                    gr.Textbox(elem_id='epsilon_high', value=None, lines=1, scale=4)
                    gr.Textbox(elem_id='beta', value=0.04, lines=1, scale=4)
                    gr.Textbox(elem_id='num_iterations', lines=1, scale=4)
                with gr.Row():
                    gr.Textbox(elem_id='move_model_batches', lines=1, scale=4)
                    gr.Checkbox(elem_id='dynamic_sample', scale=4)
                    gr.Slider(elem_id='max_resample_times', minimum=1, maximum=16, step=1, value=3, scale=4)
                    gr.Checkbox(elem_id='overlong_filter', scale=4)
                    gr.Checkbox(elem_id='vllm_enable_prefix_caching', scale=4)
                with gr.Row():
                    gr.Checkbox(elem_id='log_completions', scale=4)
                    gr.Textbox(elem_id='multi_turn_scheduler', lines=1, scale=4)
                    gr.Textbox(elem_id='max_turns', lines=1, scale=4)
                    gr.Textbox(elem_id='external_plugins', lines=1, scale=8)

            with gr.Row():
                gr.Textbox(elem_id='reward_model_plugin', lines=1, scale=8)
                gr.Dropdown(elem_id='reward_model', multiselect=True, choices=get_all_models(), scale=8)
                gr.Dropdown(
                    elem_id='reward_model_type',
                    multiselect=True,
                    choices=ModelType.get_model_name_list(),
                    allow_custom_value=True,
                    scale=4)
            with gr.Blocks():
                with gr.Row():
                    gr.Dropdown(
                        elem_id='ref_model', scale=12, value=None, choices=get_all_models(), allow_custom_value=True)
                    gr.Dropdown(elem_id='ref_model_type', choices=ModelType.get_model_name_list(), value=None, scale=8)

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
