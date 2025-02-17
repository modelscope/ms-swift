# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class LlamaPro(BaseUI):

    group = 'llm_train'

    locale_dict = {
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
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='llamapro_tab', open=False):
            with gr.Blocks():
                with gr.Row():
                    gr.Textbox(elem_id='llamapro_num_new_blocks')
                    gr.Textbox(elem_id='llamapro_num_groups')
