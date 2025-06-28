# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI
from swift.ui.llm_rlhf.lora import RLHFLoRA
from swift.ui.llm_rlhf.target import RLHFTarget
from swift.ui.llm_train.tuner import Tuner


class RLHFTuner(Tuner):

    group = 'llm_rlhf'

    sub_ui = [RLHFLoRA, RLHFTarget]

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='tuner_params', open=False):
            with gr.Tabs():
                RLHFLoRA.build_ui(base_tab)
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
            RLHFTarget.build_ui(base_tab)
