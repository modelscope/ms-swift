# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.llm.dataset.register import get_dataset_list
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
                'zh': '选择训练的数据集，支持复选/本地路径',
                'en': 'The dataset(s) to train the models, support multi select and local folder/files'
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
        'split_dataset_ratio': {
            'label': {
                'zh': '验证集拆分比例',
                'en': 'Split ratio of eval dataset'
            },
            'info': {
                'zh': '表示将总数据的多少拆分到验证集中',
                'en': 'Split the datasets by this ratio for eval'
            }
        },
        'padding_free': {
            'label': {
                'zh': '无填充批处理',
                'en': 'Padding-free batching'
            },
            'info': {
                'zh': '将一个batch中的数据进行展平而避免数据padding',
                'en': 'Flatten the data in a batch to avoid data padding'
            }
        },
        'dataset_param': {
            'label': {
                'zh': '数据集设置',
                'en': 'Dataset settings'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='dataset_param', open=True):
            with gr.Row():
                gr.Dropdown(
                    elem_id='dataset', multiselect=True, choices=get_dataset_list(), scale=20, allow_custom_value=True)
                gr.Slider(elem_id='split_dataset_ratio', minimum=0.0, maximum=1.0, step=0.05, scale=10)
                gr.Slider(elem_id='max_length', minimum=32, maximum=32768, value=1024, step=1, scale=10)
                gr.Checkbox(elem_id='padding_free', scale=10)
