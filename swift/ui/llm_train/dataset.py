import os
from typing import Type

import gradio as gr

from swift.llm import DATASET_MAPPING
from swift.ui.base import BaseUI


class Dataset(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'dataset': {
            'label': {
                'zh': '数据集名称',
                'en': 'Dataset Code'
            },
            'info': {
                'zh': '选择训练的数据集，支持复选',
                'en': 'The dataset(s) to train the models'
            }
        },
        'max_length': {
            'label': {
                'zh': '句子最大长度',
                'en': 'The max length',
            },
            'info': {
                'zh': '设置输入模型的最大长度',
                'en': 'Set the max length input to the model',
            }
        },
        'custom_train_dataset_path': {
            'label': {
                'zh': '自定义训练数据集路径',
                'en': 'Custom train dataset path'
            },
            'info': {
                'zh': '输入自定义的训练数据集路径，空格分隔',
                'en': 'Extra train files, split by blank'
            }
        },
        'custom_val_dataset_path': {
            'label': {
                'zh': '自定义校验数据集路径',
                'en': 'Custom val dataset path'
            },
            'info': {
                'zh': '输入自定义的校验数据集路径，逗号分隔',
                'en': 'Extra val files, split by comma'
            }
        },
        'dataset_test_ratio': {
            'label': {
                'zh': '验证集拆分比例',
                'en': 'Split ratio of eval dataset'
            },
            'info': {
                'zh': '表示将总数据的多少拆分到验证集中',
                'en': 'Split the datasets by this ratio for eval'
            }
        },
        'train_dataset_sample': {
            'label': {
                'zh': '训练集采样数量',
                'en': 'The sample size from the train dataset'
            },
            'info': {
                'zh': '从训练集中采样一定行数进行训练',
                'en': 'Train with the sample size from the dataset',
            }
        },
        'val_dataset_sample': {
            'label': {
                'zh': '验证集采样数量',
                'en': 'The sample size from the val dataset'
            },
            'info': {
                'zh': '从验证集中采样一定行数进行训练',
                'en': 'Validate with the sample size from the dataset',
            }
        },
        'truncation_strategy': {
            'label': {
                'zh': '数据集超长策略',
                'en': 'Dataset truncation strategy'
            },
            'info': {
                'zh': '如果token超长该如何处理',
                'en': 'How to deal with the rows exceed the max length'
            }
        },
        'custom_dataset_info': {
            'label': {
                'zh': '外部数据集配置',
                'en': 'Custom dataset config'
            },
            'info': {
                'zh': '注册外部数据集的配置文件',
                'en': 'An extra dataset config to register your own datasets'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row():
            gr.Dropdown(elem_id='dataset', multiselect=True, choices=list(DATASET_MAPPING.keys()), scale=20)
            gr.Textbox(elem_id='custom_dataset_info', is_list=False, scale=20)
            gr.Textbox(elem_id='custom_train_dataset_path', is_list=True, scale=20)
            gr.Textbox(elem_id='custom_val_dataset_path', is_list=True, scale=20)
        with gr.Row():
            gr.Slider(elem_id='dataset_test_ratio', minimum=0.0, maximum=1.0, step=0.05, scale=20)
            gr.Slider(elem_id='max_length', minimum=32, maximum=8192, step=32, scale=20)
            gr.Textbox(elem_id='train_dataset_sample', scale=20)
            gr.Textbox(elem_id='val_dataset_sample', scale=20)
            gr.Dropdown(elem_id='truncation_strategy', scale=20)
