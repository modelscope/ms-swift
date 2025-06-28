# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Rollout(BaseUI):

    group = 'llm_rollout'

    locale_dict = {
        'temperature': {
            'label': {
                'zh': 'temperature',
                'en': 'temperature'
            },
        },
        'top_k': {
            'label': {
                'zh': 'top_k',
                'en': 'top_k'
            },
        },
        'top_p': {
            'label': {
                'zh': 'top_p',
                'en': 'top_p'
            },
        },
        'repetition_penalty': {
            'label': {
                'zh': '重复惩罚项',
                'en': 'repetition penalty'
            },
        },
        'enable_prefix_caching': {
            'label': {
                'zh': '开启前缀缓存',
                'en': 'enable prefix cache'
            },
        },
        'tensor_parallel_size': {
            'label': {
                'zh': '张量并行大小',
                'en': 'tensor parallel size'
            },
        },
        'data_parallel_size': {
            'label': {
                'zh': '数据并行大小',
                'en': 'data parallel size'
            },
        },
        'pipeline_parallel_size': {
            'label': {
                'zh': 'pipeline并行大小',
                'en': 'pipeline parallel size'
            },
        },
        'max_model_len': {
            'label': {
                'zh': '模型支持的最大长度',
                'en': 'max model len'
            },
        },
        'gpu_memory_utilization': {
            'label': {
                'zh': 'GPU显存利用率',
                'en': 'GPU memory utilization'
            },
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row():
            gr.Slider(elem_id='temperature', minimum=0.0, maximum=10, step=0.1, value=1, scale=4)
            gr.Slider(elem_id='top_k', minimum=1, maximum=100, step=5, value=80, scale=4)
            gr.Slider(elem_id='top_p', minimum=0.0, maximum=1.0, step=0.05, value=1, scale=4)
            gr.Slider(elem_id='repetition_penalty', minimum=0.0, maximum=10, step=0.05, value=1.05, scale=4)
        with gr.Row():
            gr.Checkbox(elem_id='enable_prefix_caching', value=True, scale=4)
            gr.Textbox(elem_id='tensor_parallel_size', lines=1, value='1', scale=4)
            gr.Textbox(elem_id='data_parallel_size', lines=1, value='1', scale=4)
            gr.Textbox(elem_id='pipeline_parallel_size', lines=1, value='', scale=4)
            gr.Textbox(elem_id='max_model_len', lines=1, value='', scale=4)
            gr.Slider(elem_id='gpu_memory_utilization', minimum=0.0, maximum=1.0, step=0.05, value=0.9, scale=4)
