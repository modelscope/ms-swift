# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Task(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'embed_tab': {
            'label': {
                'zh': '文本嵌入',
                'en': 'Embedding'
            },
        },
        'loss_type': {
            'label': {
                'zh': 'Loss类型',
                'en': 'Loss type'
            }
        },
        'seq_cls_tab': {
            'label': {
                'zh': '序列分类',
                'en': 'Sequence Classification'
            },
        },
        'num_labels': {
            'label': {
                'zh': '标签数量',
                'en': 'Number of labels'
            }
        },
        'use_chat_template': {
            'label': {
                'zh': '使用对话模板',
                'en': 'use chat template'
            },
            'info': {
                'zh': '使用对话模板或生成模板',
                'en': 'Use the chat template or generation template'
            }
        },
        'task_type': {
            'label': {
                'zh': '任务类型',
                'en': 'Task type'
            },
        },
        'task_params': {
            'label': {
                'zh': '任务参数',
                'en': 'Task params'
            },
        }
    }

    tabs_to_filter = {'embedding': ['loss_type'], 'seq_cls': ['num_labels', 'use_chat_template']}

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='task_params', open=False):
            gr.Dropdown(elem_id='task_type', choices=['causal_lm', 'seq_cls', 'embedding'])
            with gr.Tabs():
                with gr.TabItem(elem_id='embed_tab'):
                    with gr.Row():
                        gr.Dropdown(
                            elem_id='loss_type',
                            choices=['cosine_similarity', 'contrastive', 'online_contrastive', 'infonce'])
                with gr.TabItem(elem_id='seq_cls_tab'):
                    with gr.Row():
                        gr.Textbox(elem_id='num_labels', scale=4)
                        gr.Checkbox(elem_id='use_chat_template', value=True, scale=4)
